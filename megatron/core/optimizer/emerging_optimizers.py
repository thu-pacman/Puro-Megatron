# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Emerging optimizer registry.

To add a new emerging optimizer:
  1. Define its optimizer class (or import it).
  2. Write its ``_<name>_init_state_fn`` and ``_<name>_config_to_kwargs``.
  3. Add an ``EmergingOptimizerEntry`` to ``_EMERGING_OPTIMIZERS`` at the bottom.
"""

import inspect
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size, log_single_rank

from .optimizer_config import ParamKey, ParamPredicate, ParamWithNamePredicate

try:
    from emerging_optimizers.orthogonalized_optimizers import (
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT, newton_schulz_tp

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object


logger = logging.getLogger(__name__)


def get_supported_coefficient_types() -> tuple[str, ...]:
    """Return the coefficient types supported by the installed emerging_optimizers.

    Reads the members of the ``NSCoeffT`` Literal type so that new types
    added upstream are automatically available without code changes here.
    """
    assert (
        HAVE_EMERGING_OPTIMIZERS
    ), "emerging_optimizers >= 0.2 is required for NSCoeffT. Please install or upgrade it."
    return get_args(NSCoeffT)


def validate_coefficient_type(coefficient_type: str) -> None:
    """Raise ``ValueError`` if *coefficient_type* is not supported."""
    supported = get_supported_coefficient_types()
    if coefficient_type not in supported:
        raise ValueError(
            f"Unsupported muon coefficient type '{coefficient_type}'. "
            f"Supported types: {supported}"
        )


_PARTITION_HYPERBALL_STATE_VERSION = 1
_PARTITION_HYPERBALL_LAYOUT_CODES = {"qkv": 1, "swiglu_fc1": 2}


def format_hyperball_diagnostic_metrics(
    stats_by_tag: dict[str, dict[str, float]], eps: float = 1e-12
) -> dict[str, float]:
    """Convert reduced Hyperball diagnostic stats into scalar log metrics."""
    metrics: dict[str, float] = {}
    for tag, stats in stats_by_tag.items():
        count = max(float(stats.get("count", 0.0)), 1.0)
        weight_norm = math.sqrt(max(float(stats.get("weight_norm_sq", 0.0)), 0.0))
        update_norm = math.sqrt(max(float(stats.get("update_norm_sq", 0.0)), 0.0))
        radius = math.sqrt(max(float(stats.get("radius_sq", 0.0)), 0.0))
        rel_update = update_norm / max(weight_norm, eps)
        lr_sum = float(stats.get("lr_sum", 0.0))
        metrics[f"hyperball/{tag}/weight_norm"] = weight_norm
        metrics[f"hyperball/{tag}/update_norm"] = update_norm
        metrics[f"hyperball/{tag}/relative_update"] = rel_update
        metrics[f"hyperball/{tag}/effective_lr"] = rel_update
        metrics[f"hyperball/{tag}/radius"] = radius
        metrics[f"hyperball/{tag}/param_count"] = count
        metrics[f"hyperball/{tag}/optimizer_lr"] = lr_sum / count
        if "actual_update_norm_sq" in stats:
            actual_norm = math.sqrt(max(float(stats["actual_update_norm_sq"]), 0.0))
            before_norm = math.sqrt(max(float(stats["weight_before_norm_sq"]), 0.0))
            metrics[f"hyperball/{tag}/actual_relative_update"] = actual_norm / max(before_norm, eps)
    return metrics


# ===========================================================================
# Registry dataclass and public API
# ===========================================================================


def _eopt_init_state_fn(opt, config=None):
    """Initialize emerging optimizer state for torch_dist checkpoint format."""
    for group in opt.param_groups:
        # Checkpoint init needs state for all parameters, including those without grads yet.
        opt._init_group(group, skip_non_grad_params=False)


def _default_param_overrides_factory() -> Dict[ParamKey, Dict[str, Any]]:
    """Default param overrides: route non-linear/embedding params to Adam."""
    return {
        ParamKey(
            predicate=ParamPredicate(name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding)
        ): {'optimizer': 'adam'}
    }


def _scalar_param_overrides_factory(scalar_optimizer: str) -> Dict[ParamKey, Dict[str, Any]]:
    """Route embeddings and non-matrix parameters to the scalar optimizer."""
    return {
        ParamKey(
            predicate=ParamPredicate(name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding)
        ): {'optimizer': scalar_optimizer}
    }


def _muon_hyperball_param_overrides_factory(
    scalar_optimizer: str,
) -> Dict[ParamKey, Dict[str, Any]]:
    """Keep selected matrices on MuonH and route all other parameters to AdamW."""
    hyperball_matrix = ParamWithNamePredicate(name="hyperball_matrix", fn=_is_hyperball_matrix)
    overrides = {
        ParamKey(with_name_predicate=hyperball_matrix): {'wd_mult': 0.0},
        ParamKey(attr="is_feedback_parameter"): {'optimizer': 'adam', 'wd_mult': 0.0},
        ParamKey(
            predicate=ParamPredicate(name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding)
        ): {'optimizer': scalar_optimizer},
        ParamKey(attr="is_attention_output_gate"): {'optimizer': scalar_optimizer},
        ParamKey(name=("*.router.weight", "*.router.bias", "*.gate_weight")): {
            'optimizer': scalar_optimizer
        },
    }
    return overrides


@dataclass
class EmergingOptimizerEntry:
    """Everything needed to create and configure an emerging optimizer.

    Attributes:
        optimizer_cls: The torch optimizer class.
        init_state_fn: Lazily initialises optimizer state (needed for checkpoint formats).
        config_to_kwargs: ``(config, model_chunks, pg_collection) -> dict`` of constructor kwargs.
        default_param_overrides: Per-parameter config overrides applied automatically
            (e.g. route non-linear params to Adam).
    """

    optimizer_cls: type
    init_state_fn: Callable = _eopt_init_state_fn
    config_to_kwargs: Callable | None = None
    default_param_overrides: Dict[ParamKey, Dict[str, Any]] = field(
        default_factory=_default_param_overrides_factory
    )


def _create_emerging_optimizer(config, param_groups, eopt_name, model_chunks, pg_collection):
    """Instantiate an emerging optimizer and return it with its init_state_fn."""
    entry = _EMERGING_OPTIMIZERS[eopt_name]
    if entry.config_to_kwargs is None:
        raise ValueError(f"No configuration adapter registered for optimizer '{eopt_name}'.")
    eopt_kwargs = entry.config_to_kwargs(config, model_chunks, pg_collection)
    optimizer = entry.optimizer_cls(param_groups, **eopt_kwargs)
    return optimizer, entry.init_state_fn


def get_emerging_optimizer_param_overrides(
    eopt_name: str, config, entry: Optional[EmergingOptimizerEntry] = None
) -> Dict[ParamKey, Dict[str, Any]]:
    """Return parameter-routing overrides for an emerging optimizer.

    Puro-Megatron keeps parameter routing deliberately narrow: Muon-family optimizers
    handle selected matrix weights and AdamW handles the remaining parameters.
    """
    if eopt_name == 'muon':
        return _scalar_param_overrides_factory(config.muon_scalar_optimizer)
    if eopt_name == 'muon_hyperball':
        return _muon_hyperball_param_overrides_factory(config.muon_hyperball_scalar_optimizer)
    if entry is None:
        entry = _EMERGING_OPTIMIZERS[eopt_name]
    return entry.default_param_overrides


# ===========================================================================
# Shared helpers
# ===========================================================================


def _is_nonlinear_or_embedding(param):
    """True for parameters that should NOT use the emerging optimizer."""
    return getattr(param, 'is_embedding_or_output_parameter', False) or len(param.shape) != 2


def _is_hyperball_matrix(param, name: str) -> bool:
    """Return whether a parameter follows the fixed-radius MuonH update."""
    if len(param.shape) != 2:
        return False
    if getattr(param, 'is_embedding_or_output_parameter', False):
        return False
    if getattr(param, 'is_attention_output_gate', False):
        return False
    if name.endswith(".bias"):
        return False
    if ".router." in name or name.endswith(".gate_weight"):
        return False
    return True


def _get_qkv_split_shapes(model_cfg) -> List[int]:
    """Compute QKV split shapes from model config."""
    return [
        model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels,
        model_cfg.kv_channels,
        model_cfg.kv_channels,
    ]


# ===========================================================================
# Registry – populated below only when emerging_optimizers is installed.
# ===========================================================================

_EMERGING_OPTIMIZERS: Dict[str, EmergingOptimizerEntry] = {}


class _TensorParallelHyperballMixin:
    """Shared TP-aware norm helpers for Hyperball-style optimizers.

    Hyperball normalization should use the norm of the *global* TP parameter,
    not the local shard norm. For duplicated or non-TP parameters, the local
    norm is already the correct one and no collective is needed.
    """

    pg_collection: Optional[ProcessGroupCollection]
    hyperball_eps: float

    def _get_hyperball_tp_group(self, p: torch.Tensor) -> Optional[torch.distributed.ProcessGroup]:
        """Return the TP process group relevant to ``p`` if one exists."""
        if self.pg_collection is None or not torch.distributed.is_initialized():
            return None
        if getattr(p, "expert_tp", False):
            return getattr(self.pg_collection, "expt_tp", None)
        return getattr(self.pg_collection, "tp", None)

    def _uses_tp_global_norm(self, p: torch.Tensor) -> bool:
        """Whether Hyperball norm for ``p`` must be reduced across TP ranks."""
        tp_group = self._get_hyperball_tp_group(p)
        if tp_group is None or get_pg_size(tp_group) <= 1:
            return False
        partition_dim = getattr(p, "partition_dim", None)
        return partition_dim is not None and partition_dim != -1

    def _hyperball_vector_norm(
        self, p: torch.Tensor, tensor: torch.Tensor, detach: bool = True
    ) -> torch.Tensor:
        """Compute the Frobenius norm of ``tensor`` using TP-global semantics."""
        value = tensor.detach() if detach else tensor
        norm_sq = value.float().pow(2).sum()
        if self._uses_tp_global_norm(p):
            tp_group = self._get_hyperball_tp_group(p)
            torch.distributed.all_reduce(norm_sq, group=tp_group)
        return norm_sq.sqrt().clamp_min(self.hyperball_eps)

    def _hyperball_vector_norm_sq(
        self, p: torch.Tensor, tensor: torch.Tensor, detach: bool = True
    ) -> torch.Tensor:
        """Compute an unclamped squared Frobenius norm with TP-global semantics."""
        value = tensor.detach() if detach else tensor
        norm_sq = value.float().pow(2).sum()
        if self._uses_tp_global_norm(p):
            tp_group = self._get_hyperball_tp_group(p)
            torch.distributed.all_reduce(norm_sq, group=tp_group)
        return norm_sq


# ===========================================================================
# Muon
# ===========================================================================


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    diagnostic_family = "muon"

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool | None = None,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        is_swiglu_fc1_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_ns_mode: Literal["fused", "separate"] | None = None,
        qkv_norm_mode: Literal["fused", "separate"] = "fused",
        swiglu_ns_mode: Literal["fused", "separate"] = "fused",
        swiglu_norm_mode: Literal["fused", "separate"] = "fused",
        qkv_split_shapes: Sequence[int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        effective_lr_mult: float | None = None,
        strict_effective_lr: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if effective_lr_mult is not None and effective_lr_mult <= 0.0:
            raise ValueError(f"Invalid Muon effective LR multiplier: {effective_lr_mult}")
        if strict_effective_lr and effective_lr_mult is None:
            raise ValueError("Muon strict effective LR requires an effective LR multiplier")
        if strict_effective_lr and not use_decoupled_weight_decay:
            raise ValueError("Muon strict effective LR requires decoupled weight decay")
        valid_matrix_modes = ("fused", "separate")
        if qkv_ns_mode is None:
            qkv_ns_mode = "separate" if split_qkv else "fused"
        elif qkv_ns_mode not in valid_matrix_modes:
            raise ValueError(f"Invalid QKV NS mode: {qkv_ns_mode}")
        for mode_name, mode in (
            ("qkv_norm_mode", qkv_norm_mode),
            ("swiglu_ns_mode", swiglu_ns_mode),
            ("swiglu_norm_mode", swiglu_norm_mode),
        ):
            if mode not in valid_matrix_modes:
                raise ValueError(f"Invalid {mode_name}: {mode}")
        if qkv_norm_mode == "separate" and qkv_split_shapes is not None:
            if len(qkv_split_shapes) != 3:
                raise ValueError(
                    "Separate QKV norm mode requires exactly Q/K/V split shapes; "
                    "gated attention is not supported yet"
                )

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, '
                f'{coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim is not None:
                size[partition_dim] *= get_pg_size(tp_group)
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="duplicated" if tp_mode == "blockwise" else tp_mode,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.qkv_ns_mode = qkv_ns_mode
        self.qkv_norm_mode = qkv_norm_mode
        self.swiglu_ns_mode = swiglu_ns_mode
        self.swiglu_norm_mode = swiglu_norm_mode
        self.split_qkv = qkv_ns_mode == "separate"
        self.is_qkv_fn = is_qkv_fn
        self.is_swiglu_fc1_fn = is_swiglu_fc1_fn
        self.qkv_split_shapes = qkv_split_shapes
        self.effective_lr_mult = effective_lr_mult
        self.strict_effective_lr = strict_effective_lr
        self._effective_lr_eps = 1e-12
        self._diagnostic_enabled = False
        self._diagnostic_stats: dict[str, dict[str, float]] = {}
        self._current_lr_by_param: dict[int, float] = {}
        self._current_weight_decay_by_param: dict[int, float] = {}

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        OrthogonalizedOptimizer.__init__(
            self,
            params,
            lr,
            momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    def _matrix_partition_kind(self, p: torch.Tensor) -> str | None:
        """Return the fused matrix family represented by ``p``, if any."""
        if self.is_qkv_fn is not None and self.is_qkv_fn(p):
            return "qkv"
        if self.is_swiglu_fc1_fn is not None and self.is_swiglu_fc1_fn(p):
            return "swiglu_fc1"
        return None

    def _uses_separate_ns(self, p: torch.Tensor) -> bool:
        kind = self._matrix_partition_kind(p)
        return (kind == "qkv" and self.qkv_ns_mode == "separate") or (
            kind == "swiglu_fc1" and self.swiglu_ns_mode == "separate"
        )

    def _uses_separate_norm(self, p: torch.Tensor) -> bool:
        kind = self._matrix_partition_kind(p)
        return (kind == "qkv" and self.qkv_norm_mode == "separate") or (
            kind == "swiglu_fc1" and self.swiglu_norm_mode == "separate"
        )

    def _matrix_partition_parts(
        self, p: torch.Tensor, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Expose logical projection views without changing the fused layout."""
        kind = self._matrix_partition_kind(p)
        if kind == "qkv":
            if self.qkv_split_shapes is None or len(self.qkv_split_shapes) != 3:
                raise ValueError("Muon QKV partitioning requires Q/K/V split shapes")
            partition_width = sum(self.qkv_split_shapes)
            if tensor.ndim != 2 or tensor.shape[0] % partition_width != 0:
                raise ValueError(
                    f"Invalid fused QKV shape {tuple(tensor.shape)} for partition shapes "
                    f"{tuple(self.qkv_split_shapes)}"
                )
            num_query_groups = tensor.shape[0] // partition_width
            grouped = tensor.view(num_query_groups, partition_width, tensor.shape[-1])
            parts = torch.split(grouped, self.qkv_split_shapes, dim=1)
            return list(zip(("linear_q", "linear_k", "linear_v"), parts))

        if kind == "swiglu_fc1":
            if tensor.ndim != 2 or tensor.shape[0] % 2 != 0:
                raise ValueError(
                    f"Invalid fused SwiGLU FC1 shape {tuple(tensor.shape)}; "
                    "the output dimension must be even"
                )
            gate, up = torch.chunk(tensor, 2, dim=0)
            return [("mlp_gate", gate), ("mlp_up", up)]

        return [(self._get_param_module_tag(p), tensor)]

    @staticmethod
    def _partition_as_matrix(part: torch.Tensor) -> torch.Tensor:
        return part.reshape(-1, part.shape[-1])

    def _merge_matrix_partitions(
        self, p: torch.Tensor, matrices: Sequence[torch.Tensor], original_shape: torch.Size
    ) -> torch.Tensor:
        """Restore independently processed matrices to their fused physical layout."""
        kind = self._matrix_partition_kind(p)
        if kind == "qkv":
            assert self.qkv_split_shapes is not None
            num_query_groups = original_shape[0] // sum(self.qkv_split_shapes)
            grouped = [
                matrix.view(num_query_groups, width, original_shape[-1])
                for matrix, width in zip(matrices, self.qkv_split_shapes)
            ]
            return torch.cat(grouped, dim=1).reshape(original_shape)
        if kind == "swiglu_fc1":
            return torch.cat(list(matrices), dim=0).reshape(original_shape)
        if len(matrices) != 1:
            raise ValueError(f"Expected one matrix for unfused parameter, got {len(matrices)}")
        return matrices[0].reshape(original_shape)

    def _get_muon_tp_group(self, p: torch.Tensor) -> Optional[torch.distributed.ProcessGroup]:
        """Return the TP process group relevant to a parameter."""
        if self.pg_collection is None or not torch.distributed.is_initialized():
            return None
        if getattr(p, "expert_tp", False):
            return getattr(self.pg_collection, "expt_tp", None)
        return getattr(self.pg_collection, "tp", None)

    def _uses_muon_tp_global_rms(self, p: torch.Tensor) -> bool:
        tp_group = self._get_muon_tp_group(p)
        if tp_group is None or get_pg_size(tp_group) <= 1:
            return False
        partition_dim = getattr(p, "partition_dim", None)
        return partition_dim is not None and partition_dim != -1

    def _muon_tensor_rms(
        self, p: torch.Tensor, tensor: torch.Tensor, detach: bool = True
    ) -> torch.Tensor:
        """Compute local or TP-global RMS in FP32."""
        value = tensor.detach() if detach else tensor
        norm_sq = value.float().pow(2).sum()
        numel = torch.tensor(float(value.numel()), device=value.device, dtype=torch.float32)
        if self._uses_muon_tp_global_rms(p):
            stats = torch.stack((norm_sq, numel))
            torch.distributed.all_reduce(stats, group=self._get_muon_tp_group(p))
            norm_sq, numel = stats[0], stats[1]
        return (norm_sq / numel.clamp_min(1.0)).sqrt()

    def _get_effective_lr_scale(
        self, p: torch.Tensor, update: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Scale an update so ||lr * update|| / ||weight|| equals lr times M."""
        assert self.effective_lr_mult is not None
        weight_rms = self._muon_tensor_rms(p, p if weight is None else weight)
        update_rms = self._muon_tensor_rms(p, update, detach=False)
        return (
            float(self.effective_lr_mult)
            * weight_rms.clamp_min(self._effective_lr_eps)
            / update_rms.clamp_min(self._effective_lr_eps)
        )

    def _tensor_geometry(
        self, p: torch.Tensor, update: torch.Tensor, weight: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return TP-global weight/update squared norms and inner product in FP32."""
        weight = (p if weight is None else weight).detach().float()
        update_fp32 = update.detach().float()
        stats = torch.stack(
            (weight.pow(2).sum(), update_fp32.pow(2).sum(), (weight * update_fp32).sum())
        )
        if self._uses_muon_tp_global_rms(p):
            torch.distributed.all_reduce(stats, group=self._get_muon_tp_group(p))
        return stats[0], stats[1], stats[2]

    def _get_strict_effective_lr_scale(
        self,
        p: torch.Tensor,
        update: torch.Tensor,
        base_lr: float,
        weight_decay: float,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve the update scale for the target normalized pre/post weight distance."""
        assert self.effective_lr_mult is not None
        weight_norm_sq, update_norm_sq, weight_update_dot = self._tensor_geometry(
            p, update, weight=weight
        )
        eps = torch.tensor(self._effective_lr_eps, device=p.device, dtype=torch.float32)
        if base_lr == 0.0:
            return torch.zeros((), device=p.device, dtype=torch.float32)

        weight_norm = weight_norm_sq.clamp_min(0.0).sqrt().clamp_min(eps)
        safe_weight_norm_sq = weight_norm.square()
        radial = weight_update_dot / safe_weight_norm_sq
        tangential = (
            (update_norm_sq / safe_weight_norm_sq - radial.square())
            .clamp_min(0.0)
            .sqrt()
            .clamp_min(eps)
        )

        max_chord = 2.0 - torch.finfo(torch.float32).eps
        target = torch.tensor(
            float(base_lr) * float(self.effective_lr_mult), device=p.device, dtype=torch.float32
        ).clamp(min=0.0, max=max_chord)
        cosine = 1.0 - 0.5 * target.square()
        sine = target * (1.0 - 0.25 * target.square()).clamp_min(0.0).sqrt()
        denominator = (cosine * tangential + sine * (float(weight_decay) + radial)).clamp_min(eps)
        actual_lr = sine / denominator
        return actual_lr / float(base_lr)

    @staticmethod
    def _get_param_module_tag(p: torch.Tensor) -> str:
        if getattr(p, "is_qkv", False):
            return "linear_qkv"
        name = getattr(p, "param_name", "")
        if "self_attention.linear_proj.weight" in name:
            return "attention_proj"
        if "mlp.linear_fc1.weight" in name:
            return "mlp_fc1"
        if "mlp.linear_fc2.weight" in name:
            return "mlp_fc2"
        return "matrix"

    def set_diagnostic_context(self, iteration: int, interval: int) -> None:
        """Enable effective-LR diagnostics for the selected optimizer step."""
        self._diagnostic_enabled = interval > 0 and iteration % interval == 0
        self._diagnostic_stats = {}
        if self._diagnostic_enabled:
            self._refresh_param_group_values()

    def pop_diagnostic_stats(self) -> dict[str, dict[str, float]]:
        """Return and clear the accumulated optimizer diagnostics."""
        stats = self._diagnostic_stats
        self._diagnostic_stats = {}
        return stats

    def _refresh_param_group_values(self) -> None:
        self._current_lr_by_param = {}
        self._current_weight_decay_by_param = {}
        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group.get("weight_decay", 0.0))
            for p in group["params"]:
                self._current_lr_by_param[id(p)] = lr
                self._current_weight_decay_by_param[id(p)] = weight_decay

    def _record_update(
        self,
        p: torch.Tensor,
        update: torch.Tensor,
        base_lr: float,
        applied_lr: float,
        partition_tag: str | None = None,
        weight: torch.Tensor | None = None,
    ) -> None:
        tag = partition_tag or self._get_param_module_tag(p)
        stats = self._diagnostic_stats.setdefault(
            tag, {"count": 0.0, "weight_norm_sq": 0.0, "update_norm_sq": 0.0, "lr_sum": 0.0}
        )
        weight_tensor = p if weight is None else weight
        weight_norm = float(weight_tensor.detach().float().norm().item())
        update_norm = abs(float(base_lr)) * float(update.detach().float().norm().item())
        stats["count"] += 1.0
        stats["weight_norm_sq"] += weight_norm * weight_norm
        stats["update_norm_sq"] += update_norm * update_norm
        stats["lr_sum"] += float(applied_lr)

    def _apply_weight_decay_inplace(
        self, p: torch.Tensor, grad: torch.Tensor, lr: float, weight_decay: float
    ) -> None:
        """Defer aligned Muon weight decay until the per-tensor LR is known."""
        if (
            self.effective_lr_mult is not None
            and getattr(self, "weight_decay_method", "l2") == "decoupled"
        ):
            return
        super()._apply_weight_decay_inplace(p, grad, lr, weight_decay)

    def _apply_deferred_weight_decay(
        self, p: torch.Tensor, lr: float, target: torch.Tensor | None = None
    ) -> None:
        if getattr(self, "weight_decay_method", "l2") != "decoupled":
            return
        weight_decay = self._current_weight_decay_by_param.get(id(p), 0.0)
        if weight_decay:
            weight = p if target is None else target
            weight.add_(weight, alpha=(-float(weight_decay) * float(lr)))

    def pre_weight_update_fn_inplace(self, p: torch.Tensor, update: torch.Tensor) -> None:
        """Apply optional relative-update alignment before the Muon weight update."""
        base_lr = self._current_lr_by_param.get(id(p), 0.0)
        applied_lr = base_lr
        if self.effective_lr_mult is not None:
            weight_decay = self._current_weight_decay_by_param.get(id(p), 0.0)
            if self._uses_separate_norm(p):
                weight_parts = self._matrix_partition_parts(p, p)
                update_parts = self._matrix_partition_parts(p, update)
                for (tag, weight_part), (update_tag, update_part) in zip(
                    weight_parts, update_parts
                ):
                    if tag != update_tag:
                        raise RuntimeError(
                            f"Matrix partition mismatch: weight={tag}, update={update_tag}"
                        )
                    if self.strict_effective_lr:
                        scale = self._get_strict_effective_lr_scale(
                            p, update_part, base_lr, weight_decay, weight=weight_part
                        )
                    else:
                        scale = self._get_effective_lr_scale(p, update_part, weight=weight_part)
                    update_part.mul_(scale.to(device=update.device, dtype=update.dtype))
                    partition_lr = base_lr * float(scale.detach().float().item())
                    self._apply_deferred_weight_decay(p, partition_lr, target=weight_part)
                    if self._diagnostic_enabled:
                        self._record_update(
                            p,
                            update_part,
                            base_lr,
                            partition_lr,
                            partition_tag=tag,
                            weight=weight_part,
                        )
                return
            if self.strict_effective_lr:
                scale = self._get_strict_effective_lr_scale(p, update, base_lr, weight_decay)
            else:
                scale = self._get_effective_lr_scale(p, update)
            update.mul_(scale.to(device=update.device, dtype=update.dtype))
            applied_lr = base_lr * float(scale.detach().float().item())
            self._apply_deferred_weight_decay(p, applied_lr)
        if self._diagnostic_enabled:
            if self._uses_separate_norm(p):
                for (tag, weight_part), (update_tag, update_part) in zip(
                    self._matrix_partition_parts(p, p), self._matrix_partition_parts(p, update)
                ):
                    if tag != update_tag:
                        raise RuntimeError(
                            f"Matrix partition mismatch: weight={tag}, update={update_tag}"
                        )
                    self._record_update(
                        p, update_part, base_lr, applied_lr, partition_tag=tag, weight=weight_part
                    )
            else:
                self._record_update(p, update, base_lr, applied_lr)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Apply one update using the current parameter-group learning rates."""
        if not self._current_lr_by_param and (
            self.effective_lr_mult is not None or self._diagnostic_enabled
        ):
            self._refresh_param_group_values()
        try:
            return super().step(closure)
        finally:
            self._current_lr_by_param = {}
            self._current_weight_decay_by_param = {}

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to
                momentum because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        if self._uses_separate_ns(p):
            partitions = self._matrix_partition_parts(p, grad)
            partition_grads = [
                self.scaled_orthogonalize_fn(
                    self._partition_as_matrix(part), tp_group, partition_dim
                )
                for _, part in partitions
            ]
            grad = self._merge_matrix_partitions(p, partition_grads, grad.shape)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad


class TensorParallelMuonHyperball(_TensorParallelHyperballMixin, TensorParallelMuon):
    """Tensor-parallel Muon with Hyperball-style norm-preserving matrix updates."""

    diagnostic_family = "hyperball"

    checkpoint_scalar_state_keys = (
        "hyperball_radius",
        "hyperball_partition_radii",
        "hyperball_partition_state_version",
        "hyperball_partition_layout_code",
    )

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool | None = None,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        is_swiglu_fc1_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_ns_mode: Literal["fused", "separate"] | None = None,
        qkv_norm_mode: Literal["fused", "separate"] = "fused",
        swiglu_ns_mode: Literal["fused", "separate"] = "fused",
        swiglu_norm_mode: Literal["fused", "separate"] = "fused",
        qkv_split_shapes: Sequence[int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
        hyperball_eps: float = 1e-8,
        hyperball_radius: float | None = None,
        hyperball_rms: float | None = None,
        lr_mult: float = 1.0,
    ) -> None:
        if hyperball_eps <= 0.0:
            raise ValueError(f"Invalid hyperball epsilon value: {hyperball_eps}")
        if hyperball_radius is not None and hyperball_radius <= 0.0:
            raise ValueError(f"Invalid hyperball radius value: {hyperball_radius}")
        if hyperball_rms is not None and hyperball_rms <= 0.0:
            raise ValueError(f"Invalid hyperball RMS value: {hyperball_rms}")
        if hyperball_radius is not None and hyperball_rms is not None:
            raise ValueError("hyperball_radius and hyperball_rms are mutually exclusive")
        if (
            qkv_norm_mode == "separate" or swiglu_norm_mode == "separate"
        ) and hyperball_radius is not None:
            raise ValueError(
                "hyperball_radius is not supported with separate matrix norm modes; "
                "use hyperball_rms or per-partition initial norms"
            )
        if lr_mult <= 0.0:
            raise ValueError(f"Invalid hyperball learning-rate multiplier: {lr_mult}")

        self.hyperball_eps = hyperball_eps
        self.hyperball_radius = hyperball_radius
        self.hyperball_rms = hyperball_rms
        self.hyperball_lr_mult = lr_mult
        self._hyperball_diag_enabled = False
        self._hyperball_diag_stats: dict[str, dict[str, float]] = {}
        self._hyperball_current_lr_by_param: dict[int, float] = {}
        self._hyperball_weight_before: torch.Tensor | None = None
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            split_qkv=split_qkv,
            is_qkv_fn=is_qkv_fn,
            is_swiglu_fc1_fn=is_swiglu_fc1_fn,
            qkv_ns_mode=qkv_ns_mode,
            qkv_norm_mode=qkv_norm_mode,
            swiglu_ns_mode=swiglu_ns_mode,
            swiglu_norm_mode=swiglu_norm_mode,
            qkv_split_shapes=qkv_split_shapes,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
        )

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if self._uses_separate_norm(p):
                        partitions = self._matrix_partition_parts(p, p)
                        partition_norms_sq = [
                            self._hyperball_vector_norm_sq(p, part) for _, part in partitions
                        ]
                        for (tag, _), norm_sq in zip(partitions, partition_norms_sq):
                            if norm_sq.item() == 0:
                                raise ValueError(
                                    "MuonHyperball requires every matrix partition to have "
                                    f"non-zero norm. Found zero norm in {tag}."
                                )
                        partition_norms = [
                            norm_sq.sqrt().clamp_min(self.hyperball_eps)
                            for norm_sq in partition_norms_sq
                        ]
                        partition_radii = self._ensure_partition_hyperball_radius_state(p)
                        if self._uses_fixed_hyperball_target():
                            for (_, part), norm, radius in zip(
                                partitions, partition_norms, partition_radii
                            ):
                                part.mul_(radius / norm.clamp_min(self.hyperball_eps))
                        self._set_hyperball_param_module_tag(p)
                        continue
                    p_norm_sq = self._hyperball_vector_norm_sq(p, p)
                    if p_norm_sq.item() == 0:
                        raise ValueError(
                            "MuonHyperball requires all hyperball parameters to have "
                            "non-zero norm. Found parameter with zero norm."
                        )
                    p_norm = p_norm_sq.sqrt().clamp_min(self.hyperball_eps)
                    if self._uses_fixed_hyperball_target():
                        p.mul_(self._make_hyperball_radius(p) / p_norm)
                    self._ensure_hyperball_radius_state(p)
                    self._set_hyperball_param_module_tag(p)

    @staticmethod
    def _get_hyperball_param_module_tag(p: torch.Tensor) -> str:
        """Return a compact module tag for Hyperball diagnostics."""
        return getattr(p, "hyperball_module_tag", "matrix")

    @staticmethod
    def _set_hyperball_param_module_tag(p: torch.Tensor) -> None:
        """Infer a stable module tag from model-parameter attributes."""
        if getattr(p, "is_qkv", False):
            tag = "linear_qkv"
        else:
            name = getattr(p, "param_name", "")
            if "self_attention.linear_proj.weight" in name:
                tag = "attention_proj"
            elif "mlp.linear_fc1.weight" in name:
                tag = "mlp_fc1"
            elif "mlp.linear_fc2.weight" in name:
                tag = "mlp_fc2"
            else:
                tag = "matrix"
        p.hyperball_module_tag = tag

    def set_diagnostic_context(self, iteration: int, interval: int) -> None:
        """Enable optimizer-local diagnostics for the next step when on a diag interval."""
        self._hyperball_diag_enabled = interval > 0 and iteration % interval == 0
        if self._hyperball_diag_enabled:
            self._hyperball_diag_stats = {}
            self._hyperball_refresh_lr_map()

    def pop_diagnostic_stats(self) -> dict[str, dict[str, float]]:
        """Return and clear raw module-wise Hyperball diagnostic stats.

        The caller is expected to reduce these sum-like fields across ranks before
        deriving norms and ratios. Returning raw stats avoids logging rank-local
        norms as if they were global values.
        """
        if not self._hyperball_diag_stats:
            return {}

        stats = {
            tag: {key: float(value) for key, value in tag_stats.items()}
            for tag, tag_stats in self._hyperball_diag_stats.items()
        }

        self._hyperball_diag_stats = {}
        return stats

    def pop_diagnostic_metrics(self) -> dict[str, float]:
        """Return rank-local Hyperball diagnostics for non-aggregating callers."""
        stats = self.pop_diagnostic_stats()
        return format_hyperball_diagnostic_metrics(stats, eps=self.hyperball_eps)

    def _hyperball_refresh_lr_map(self) -> None:
        """Cache effective per-parameter LR before the step mutates any groups."""
        self._hyperball_current_lr_by_param = {}
        for group in self.param_groups:
            lr = float(group["lr"]) * self.hyperball_lr_mult
            for p in group["params"]:
                self._hyperball_current_lr_by_param[id(p)] = lr

    def _record_hyperball_update_stats(
        self,
        p: torch.Tensor,
        update_norm: torch.Tensor,
        radius: torch.Tensor,
        lr: float,
        partition_tag: str | None = None,
    ) -> None:
        """Accumulate per-module update stats for later logging."""
        tag = partition_tag or self._get_hyperball_param_module_tag(p)
        stats = self._hyperball_diag_stats.setdefault(
            tag,
            {
                "count": 0.0,
                "weight_norm_sq": 0.0,
                "update_norm_sq": 0.0,
                "radius_sq": 0.0,
                "lr_sum": 0.0,
            },
        )
        update_value = float(update_norm.detach().float().item())
        radius_value = float(radius.detach().float().item())
        stats["count"] += 1.0
        stats["weight_norm_sq"] += radius_value * radius_value
        stats["update_norm_sq"] += (lr * update_value) * (lr * update_value)
        stats["radius_sq"] += radius_value * radius_value
        stats["lr_sum"] += lr

    @torch.no_grad()
    def _init_group(self, group: Dict[str, Any], skip_non_grad_params: bool = True) -> None:
        """Initialize both Muon momentum state and fixed Hyperball radius state."""
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.data)
            if self._uses_separate_norm(p):
                self._ensure_partition_hyperball_radius_state(p)
            else:
                self._ensure_hyperball_radius_state(p)

    def _uses_fixed_hyperball_target(self) -> bool:
        """Whether construction should rescale parameters to a user-specified target."""
        return self.hyperball_radius is not None or self.hyperball_rms is not None

    def _hyperball_global_numel(self, p: torch.Tensor) -> int:
        """Return tensor-parallel global element count for ``p``."""
        return self._hyperball_tensor_global_numel(p, p)

    def _hyperball_tensor_global_numel(self, p: torch.Tensor, tensor: torch.Tensor) -> int:
        """Return tensor-parallel global element count for a physical tensor or partition."""
        numel = tensor.numel()
        if self._uses_tp_global_norm(p):
            tp_group = self._get_hyperball_tp_group(p)
            numel *= get_pg_size(tp_group)
        return numel

    def _make_hyperball_radius(self, p: torch.Tensor) -> torch.Tensor:
        """Create the fixed Hyperball radius tensor for ``p``."""
        if self.hyperball_radius is None:
            if self.hyperball_rms is not None:
                radius = self.hyperball_rms * math.sqrt(self._hyperball_global_numel(p))
                return torch.tensor(radius, device=p.device, dtype=torch.float32).clamp_min(
                    self.hyperball_eps
                )
            return self._hyperball_vector_norm(p, p)
        return torch.tensor(self.hyperball_radius, device=p.device, dtype=torch.float32).clamp_min(
            self.hyperball_eps
        )

    def _canonicalize_hyperball_radius(self, p: torch.Tensor, radius: Any) -> torch.Tensor:
        """Move loaded radius state to the parameter device in canonical dtype."""
        if torch.is_tensor(radius):
            radius = radius.detach().to(device=p.device, dtype=torch.float32)
        else:
            radius = torch.tensor(float(radius), device=p.device, dtype=torch.float32)
        return radius.clamp_min(self.hyperball_eps)

    def _make_partition_hyperball_radii(self, p: torch.Tensor) -> torch.Tensor:
        """Create one fixed radius for each matrix partition in a fused parameter."""
        radii = []
        for _, part in self._matrix_partition_parts(p, p):
            if self.hyperball_rms is None:
                radius = self._hyperball_vector_norm(p, part)
            else:
                radius = torch.tensor(
                    self.hyperball_rms * math.sqrt(self._hyperball_tensor_global_numel(p, part)),
                    device=p.device,
                    dtype=torch.float32,
                )
            radii.append(radius.clamp_min(self.hyperball_eps))
        return torch.stack(radii)

    def _ensure_partition_hyperball_radius_state(self, p: torch.Tensor) -> torch.Tensor:
        """Return versioned per-partition radii for a fused parameter.

        The explicit layout and version fields intentionally make a future cross-mode
        checkpoint converter straightforward. Cross-mode optimizer checkpoint loading is
        not currently guaranteed.
        """
        kind = self._matrix_partition_kind(p)
        if kind is None:
            raise ValueError("Partitioned Hyperball state requested for an unfused parameter")

        state = self.state[p]
        radii = state.get("hyperball_partition_radii")
        if radii is None:
            radii = self._make_partition_hyperball_radii(p)
        radii = self._canonicalize_hyperball_radius(p, radii).reshape(-1)

        expected_count = len(self._matrix_partition_parts(p, p))
        if radii.numel() != expected_count:
            raise ValueError(
                f"Partitioned Hyperball radius count mismatch for "
                f"{getattr(p, 'param_name', kind)}: "
                f"checkpoint has {radii.numel()}, layout requires {expected_count}"
            )

        loaded_version = state.get("hyperball_partition_state_version")
        if loaded_version is not None and int(loaded_version) != _PARTITION_HYPERBALL_STATE_VERSION:
            raise ValueError(
                f"Unsupported partitioned Hyperball state version {loaded_version}; "
                f"expected {_PARTITION_HYPERBALL_STATE_VERSION}"
            )
        expected_layout_code = _PARTITION_HYPERBALL_LAYOUT_CODES[kind]
        loaded_layout_code = state.get("hyperball_partition_layout_code")
        if loaded_layout_code is not None and int(loaded_layout_code) != expected_layout_code:
            raise ValueError(
                "Partitioned Hyperball layout mismatch: "
                f"checkpoint code={loaded_layout_code}, model={kind}"
            )

        state["hyperball_partition_radii"] = radii
        state["hyperball_partition_state_version"] = _PARTITION_HYPERBALL_STATE_VERSION
        state["hyperball_partition_layout_code"] = expected_layout_code
        return radii

    def _ensure_hyperball_radius_state(self, p: torch.Tensor) -> torch.Tensor:
        """Return the fixed Hyperball radius stored in optimizer state."""
        state = self.state[p]
        radius = state.get("hyperball_radius")
        if radius is None and "hyperball_R" in state:
            radius = state.pop("hyperball_R")
        if radius is None:
            radius = self._make_hyperball_radius(p)
        radius = self._canonicalize_hyperball_radius(p, radius)
        state["hyperball_radius"] = radius
        return radius

    def _ensure_all_hyperball_radius_states(self) -> None:
        """Ensure every Hyperball parameter has checkpointable radius state."""
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if self._uses_separate_norm(p):
                        self._ensure_partition_hyperball_radius_state(p)
                    else:
                        self._ensure_hyperball_radius_state(p)

    def pre_weight_update_fn_inplace(self, p: torch.Tensor, update: torch.Tensor) -> None:
        """Normalize the Muon update to the fixed parameter radius."""
        if self._uses_separate_norm(p):
            radii = self._ensure_partition_hyperball_radius_state(p)
            update_parts = self._matrix_partition_parts(p, update)
            for index, ((tag, update_part), radius) in enumerate(zip(update_parts, radii)):
                update_norm = self._hyperball_vector_norm(p, update_part, detach=False)
                update_part.mul_(radius / update_norm.clamp_min(self.hyperball_eps))
                if self._hyperball_diag_enabled:
                    self._record_hyperball_update_stats(
                        p,
                        self._hyperball_vector_norm(p, update_part, detach=False),
                        radius,
                        self._hyperball_current_lr_by_param.get(id(p), 0.0),
                        partition_tag=tag,
                    )
            return

        radius = self._ensure_hyperball_radius_state(p)
        update_norm = self._hyperball_vector_norm(p, update, detach=False)
        update.mul_(radius / update_norm)
        if self._hyperball_diag_enabled:
            self._record_hyperball_update_stats(
                p,
                self._hyperball_vector_norm(p, update, detach=False),
                radius,
                self._hyperball_current_lr_by_param.get(id(p), 0.0),
            )

    def _apply_weight_decay_inplace(self, p, grad, lr, weight_decay):
        # Capture a single matrix before the update, including any weight decay.
        self._hyperball_weight_before = (
            p.detach().float().clone() if self._hyperball_diag_enabled else None
        )
        return super()._apply_weight_decay_inplace(p, grad, lr, weight_decay)

    def _record_actual_hyperball_update(self, p):
        before = self._hyperball_weight_before
        self._hyperball_weight_before = None
        if before is None:
            return
        if self._uses_separate_norm(p):
            old_parts = self._matrix_partition_parts(p, before)
            new_parts = self._matrix_partition_parts(p, p)
        else:
            tag = self._get_hyperball_param_module_tag(p)
            old_parts, new_parts = [(tag, before)], [(tag, p)]
        for (tag, old), (_, new) in zip(old_parts, new_parts):
            stats = self._hyperball_diag_stats[tag]
            stats['weight_before_norm_sq'] = stats.get('weight_before_norm_sq', 0.0) + float(
                self._hyperball_vector_norm_sq(p, old)
            )
            stats['actual_update_norm_sq'] = stats.get('actual_update_norm_sq', 0.0) + float(
                self._hyperball_vector_norm_sq(p, new.float() - old)
            )

    def post_weight_update_fn_inplace(self, p: torch.Tensor) -> None:
        """Project the updated weight back to its fixed radius."""
        if self._uses_separate_norm(p):
            radii = self._ensure_partition_hyperball_radius_state(p)
            for (_, weight_part), radius in zip(self._matrix_partition_parts(p, p), radii):
                weight_norm = self._hyperball_vector_norm(p, weight_part, detach=False)
                weight_part.mul_(radius / weight_norm.clamp_min(self.hyperball_eps))
            self._record_actual_hyperball_update(p)
            return

        radius = self._ensure_hyperball_radius_state(p)
        p_norm = self._hyperball_vector_norm(p, p, detach=False)
        p.mul_(radius / p_norm)
        self._record_actual_hyperball_update(p)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Apply a Hyperball-only LR multiplier without changing param-group identity."""
        if self._hyperball_diag_enabled and not self._hyperball_current_lr_by_param:
            self._hyperball_refresh_lr_map()
        if self.hyperball_lr_mult == 1.0:
            return super().step(closure)

        for group in self.param_groups:
            group["lr"] *= self.hyperball_lr_mult
        try:
            return super().step(closure)
        finally:
            for group in self.param_groups:
                group["lr"] /= self.hyperball_lr_mult

    def load_state_dict(self, state_dict):
        """Reload optimizer state and ensure fixed Hyperball radii are present."""
        super().load_state_dict(state_dict)
        self._ensure_all_hyperball_radius_states()


def _kwargs_from_config(optimizer_cls: type, prefix: str, config) -> Dict[str, Any]:
    """Match ``optimizer_cls.__init__`` parameters to config attributes.

    For each init parameter, looks for ``{prefix}_{name}`` on *config* first,
    then falls back to ``{name}`` (unprefixed).  ``self`` and ``params`` are
    always skipped.
    """
    skip_params = {"self", "params"}
    sig = inspect.signature(optimizer_cls.__init__)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters:
        if name in skip_params:
            continue
        prefixed = f"{prefix}_{name}"
        if hasattr(config, prefixed):
            kwargs[name] = getattr(config, prefixed)
        elif hasattr(config, name):
            kwargs[name] = getattr(config, name)
    return kwargs


def _muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelMuon constructor kwargs."""
    model_config = model_chunks[0].config
    kwargs = _kwargs_from_config(TensorParallelMuon, "muon", config)
    kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
    kwargs["is_swiglu_fc1_fn"] = lambda p: (
        model_config.gated_linear_unit
        and p.ndim == 2
        and "linear_fc1.weight" in getattr(p, "param_name", "")
    )
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(model_config)
    kwargs["pg_collection"] = pg_collection
    return kwargs


def _muon_hyperball_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelMuonHyperball constructor kwargs."""
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs.pop("effective_lr_mult", None)
    kwargs.pop("strict_effective_lr", None)
    kwargs.update(_kwargs_from_config(TensorParallelMuonHyperball, "muon_hyperball", config))
    kwargs["hyperball_eps"] = config.muon_hyperball_eps
    kwargs["hyperball_radius"] = config.muon_hyperball_radius
    kwargs["hyperball_rms"] = config.muon_hyperball_rms
    return kwargs


# -----------------------------------------------------------------------
# Register emerging optimizers
# -----------------------------------------------------------------------
_EMERGING_OPTIMIZERS.update(
    {
        'muon': EmergingOptimizerEntry(
            optimizer_cls=TensorParallelMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muon_config_to_kwargs,
        ),
        'muon_hyperball': EmergingOptimizerEntry(
            optimizer_cls=TensorParallelMuonHyperball,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muon_hyperball_config_to_kwargs,
        ),
    }
)
