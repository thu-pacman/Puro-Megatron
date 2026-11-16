# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import math
import warnings
from collections import defaultdict
from itertools import chain
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedStateDict
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
    optim_state_to_sharding_state,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_rank, get_pg_size, is_te_min_version

from .clip_grads import count_zeros_fp32, get_grad_norm_fp32
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig

logger = logging.getLogger(__name__)

try:
    from transformer_engine.pytorch.optimizers import FusedAdam as _MegatronAdam

    _USING_PYTORCH_ADAM = False
except ImportError:
    try:
        from apex.optimizers import FusedAdam as _MegatronAdam

        _USING_PYTORCH_ADAM = False
    except ImportError:
        warnings.warn(
            "Transformer Engine and Apex are not installed. Falling back to Torch optimizers."
        )
        _MegatronAdam = torch.optim.AdamW
        _USING_PYTORCH_ADAM = True


def _pad_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple) if value else 0


def build_variable_flat_shard_ranges(
    total_size: int, desired_sizes: List[float], alignment: int
) -> List[range]:
    """Build deterministic variable-size contiguous flat ranges.

    Boundaries are aligned where possible, and the final range always ends at
    ``total_size``. Unlike ``shard_buffer``, ranges are intentionally not equal
    sized: callers pass desired sizes that compensate for non-Adam memory already
    assigned to each rank.
    """

    if not desired_sizes:
        return []
    if total_size == 0:
        return [range(0, 0) for _ in desired_sizes]

    total_desired = sum(desired_sizes)
    if total_desired <= 0:
        desired_sizes = [float(total_size) / len(desired_sizes)] * len(desired_sizes)
    else:
        desired_sizes = [size * float(total_size) / total_desired for size in desired_sizes]

    boundaries = [0]
    cumulative = 0.0
    for size in desired_sizes[:-1]:
        cumulative += size
        boundary = int(round(cumulative / alignment) * alignment)
        boundary = max(boundaries[-1], min(total_size, boundary))
        boundaries.append(boundary)
    boundaries.append(total_size)

    return [range(boundaries[i], boundaries[i + 1]) for i in range(len(desired_sizes))]


def build_param_range_map_from_world_range(
    param_index_map: Dict[torch.Tensor, tuple[int, int, int]],
    world_range: range,
    bucket_offset: int = 0,
) -> Dict[torch.Tensor, Dict[str, range]]:
    """Map a flat world range to per-parameter sub-ranges.

    This mirrors the standard DistributedOptimizer range-map idea while allowing
    variable world ranges instead of equal-sized DP shards.
    """

    param_range_map = {}
    for param, (param_world_start, param_world_end, _bucket_id) in param_index_map.items():
        local_start = max(0, param_world_start - world_range.start)
        local_end = min(world_range.stop - world_range.start, param_world_end - world_range.start)
        if local_end <= local_start:
            continue
        world_start = world_range.start + local_start
        world_end = world_range.start + local_end
        param_start = max(0, world_range.start - param_world_start)
        param_range_map[param] = {
            "flat_world": range(world_start, world_end),
            "flat_world_in_bucket": range(world_start - bucket_offset, world_end - bucket_offset),
            "flat_local": range(local_start, local_end),
            "param": range(param_start, param_start + (local_end - local_start)),
        }
    return param_range_map


def _optimizer_class_name(optimizer) -> str:
    return type(optimizer).__name__.lower()


def _param_log_name(param: torch.Tensor) -> str:
    return getattr(param, "param_name", f"unnamed_param_{id(param)}")


def _format_mem(num_bytes: float) -> str:
    mib = num_bytes / (1024.0**2)
    gib = num_bytes / (1024.0**3)
    if gib >= 1.0:
        return f"{gib:.3f} GiB"
    return f"{mib:.3f} MiB"


def _dtype_num_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _optimizer_state_dtype(config: OptimizerConfig) -> torch.dtype:
    if config.use_precision_aware_optimizer:
        return config.main_params_dtype
    return torch.float32


def _has_optimizer_main_param(config: OptimizerConfig) -> bool:
    return config.fp16 or config.bf16 or config.use_precision_aware_optimizer


def _adam_flat_cost_multiplier(config: OptimizerConfig) -> int:
    if config.use_precision_aware_optimizer:
        return (
            _dtype_num_bytes(config.main_params_dtype)
            + _dtype_num_bytes(config.exp_avg_dtype)
            + _dtype_num_bytes(config.exp_avg_sq_dtype)
        )
    return 3 * _dtype_num_bytes(torch.float32)


def _muon_param_optimizer_state_cost(
    param: torch.Tensor, optimizer, config: OptimizerConfig
) -> int:
    state_dtype = _optimizer_state_dtype(config)
    state_bytes = _dtype_num_bytes(state_dtype)
    cost = param.numel() * state_bytes  # momentum_buffer
    if _has_optimizer_main_param(config):
        cost += param.numel() * _dtype_num_bytes(config.main_params_dtype)

    moment2_method = getattr(optimizer, "moment2_method", None)
    if moment2_method == "adamuon":
        cost += param.numel() * state_bytes
    elif moment2_method == "normuon":
        if param.ndim == 2:
            moment2_numel = min(param.shape[-2], param.shape[-1])
            cost += moment2_numel * state_bytes
        else:
            cost += param.numel() * state_bytes

    if "hyperball" in _optimizer_class_name(optimizer):
        cost += _dtype_num_bytes(torch.float32)
    return cost


def _whole_param_optimizer_state_cost(
    param: torch.Tensor, optimizer, config: OptimizerConfig
) -> int:
    if _is_muon_optimizer(optimizer):
        return _muon_param_optimizer_state_cost(param, optimizer, config)
    state_dtype = _optimizer_state_dtype(config)
    cost = param.numel() * _dtype_num_bytes(state_dtype)
    if _has_optimizer_main_param(config):
        cost += param.numel() * _dtype_num_bytes(config.main_params_dtype)
    return cost


def _is_adam_optimizer(optimizer) -> bool:
    name = _optimizer_class_name(optimizer)
    return "adam" in name


def _is_muon_optimizer(optimizer) -> bool:
    name = _optimizer_class_name(optimizer)
    return "muon" in name


class FlatShardedAdamFallback(Float16OptimizerWithFloat16Params):
    """Variable flat-sharded AdamW fallback for Muon layer-wise distributed optimizer.

    Model parameters keep their original shapes. This helper owns only local
    flat shards of Adam main parameters and moment tensors, then all-gathers
    updated shards back into the full model parameters after each optimizer step.
    """

    PARAM_START_ALIGNMENT = 64
    SHARD_BOUNDARY_ALIGNMENT = 128

    def __init__(
        self,
        param_groups: List[dict],
        config: OptimizerConfig,
        process_group: torch.distributed.ProcessGroup,
        base_cost_by_rank: List[float],
        adam_cost_multiplier: Optional[int] = None,
    ) -> None:
        self.config = config
        self.process_group = process_group
        self.rank = get_pg_rank(process_group)
        self.world_size = get_pg_size(process_group)
        self.adam_cost_multiplier = (
            _adam_flat_cost_multiplier(config)
            if adam_cost_multiplier is None
            else adam_cost_multiplier
        )
        self._param_groups = [self._clone_param_group(group) for group in param_groups]
        self._param_index_map: Dict[torch.Tensor, tuple[int, int, int]] = {}
        self._param_to_group_index: Dict[torch.Tensor, int] = {}
        self._rank_ranges: List[range] = []
        self._local_segments_by_group: List[List[dict]] = []
        self._main_shards: List[torch.nn.Parameter] = []
        self._float16_groups: List[List[torch.Tensor]] = []
        self._state: Dict[torch.nn.Parameter, dict] = {}
        self._optimizer_group_by_group_index: Dict[int, dict] = {}
        self._step = 0
        self._total_size = 0

        self._build_flat_layout(base_cost_by_rank)
        self._init_local_shards()

        optimizer = self._build_megatron_adam_optimizer()
        super().__init__(optimizer, config, None, None)
        self._state = optimizer.state
        self._init_megatron_adam_state()
        self.is_stub_optimizer = len(self._main_shards) == 0
        self.grad_stats_parallel_group = None

    @staticmethod
    def _clone_param_group(group: dict) -> dict:
        return {key: (list(value) if key == "params" else value) for key, value in group.items()}

    @property
    def param_groups(self) -> List[dict]:
        """Return the public parameter groups controlled by the scheduler."""
        return self._param_groups

    @param_groups.setter
    def param_groups(self, value) -> None:
        self._param_groups = value

    @property
    def state(self) -> Dict[torch.nn.Parameter, dict]:
        """Return local Adam state."""
        return self._state

    @state.setter
    def state(self, value) -> None:
        self._state = value
        self.optimizer.state = value

    def _build_megatron_adam_optimizer(self):
        flat_param_groups = []
        for group_index, group in enumerate(self._param_groups):
            main_shard = group.get("_flat_shard_param")
            if main_shard is None:
                continue
            flat_group = {
                key: value
                for key, value in group.items()
                if key not in ("params", "_flat_shard_param")
            }
            flat_group["params"] = [main_shard]
            flat_param_groups.append(flat_group)
            self._optimizer_group_by_group_index[group_index] = flat_group

        if not flat_param_groups:
            return SimpleNamespace(param_groups=[], state={})

        kwargs = {
            "params": flat_param_groups,
            "lr": self.config.lr,
            "weight_decay": self.config.weight_decay,
            "betas": (self.config.adam_beta1, self.config.adam_beta2),
            "eps": self.config.adam_eps,
        }
        if _USING_PYTORCH_ADAM:
            adam_cls = torch.optim.AdamW if self.config.decoupled_weight_decay else torch.optim.Adam
        else:
            kwargs["adam_w_mode"] = self.config.decoupled_weight_decay
            adam_cls = _MegatronAdam

        if self.config.use_precision_aware_optimizer:
            kwargs.update(
                {
                    "exp_avg_dtype": self.config.exp_avg_dtype,
                    "exp_avg_sq_dtype": self.config.exp_avg_sq_dtype,
                }
            )
            if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                kwargs.update(
                    {
                        "master_weights": True,
                        "use_decoupled_grad": True,
                        "master_weight_dtype": self.config.main_params_dtype,
                    }
                )
            if is_te_min_version("2.1.0.dev0"):
                kwargs.update({"store_param_remainders": self.config.store_param_remainders})
        return adam_cls(**kwargs)

    def _init_megatron_adam_state(self) -> None:
        for group_index, group in enumerate(self._param_groups):
            main_shard = group.get("_flat_shard_param")
            if main_shard is None:
                continue
            state = self._state[main_shard]
            if len(state) == 0:
                if self.config.use_precision_aware_optimizer and hasattr(
                    self.optimizer, "initialize_state"
                ):
                    self.optimizer.initialize_state(main_shard)
                else:
                    state["exp_avg"] = torch.zeros_like(main_shard.data)
                    state["exp_avg_sq"] = torch.zeros_like(main_shard.data)

    def _sync_megatron_adam_group_options(self) -> None:
        for group_index, optimizer_group in self._optimizer_group_by_group_index.items():
            source_group = self._param_groups[group_index]
            for key, value in source_group.items():
                if key not in ("params", "_flat_shard_param"):
                    optimizer_group[key] = value

    def _get_local_group_step(self, group_index: int, device: torch.device):
        optimizer_group = self._optimizer_group_by_group_index.get(group_index)
        if optimizer_group is not None and "step" in optimizer_group:
            return torch.tensor(int(optimizer_group["step"]), dtype=torch.int64, device=device)

        main_shard = self._param_groups[group_index].get("_flat_shard_param")
        if main_shard is None:
            return None
        step = self._state[main_shard].get("step", None)
        if step is None:
            return None
        if torch.is_tensor(step):
            return step.to(device=device, dtype=torch.int64)
        return torch.tensor(int(step), dtype=torch.int64, device=device)

    def _set_local_group_step(self, group_index: int, step: torch.Tensor) -> None:
        main_shard = self._param_groups[group_index].get("_flat_shard_param")
        if main_shard is not None:
            self._state[main_shard]["step"] = step.to(main_shard.device, dtype=torch.int64)

        optimizer_group = self._optimizer_group_by_group_index.get(group_index)
        if optimizer_group is not None and not _USING_PYTORCH_ADAM:
            optimizer_group["step"] = int(step.item())

    def _build_flat_layout(self, base_cost_by_rank: List[float]) -> None:
        offset = 0
        for group_index, group in enumerate(self._param_groups):
            for param in group["params"]:
                offset = _pad_to_multiple(offset, self.PARAM_START_ALIGNMENT)
                start = offset
                end = start + param.numel()
                self._param_index_map[param] = (start, end, 0)
                self._param_to_group_index[param] = group_index
                offset = end

        bucket_alignment = math.lcm(self.world_size, self.SHARD_BOUNDARY_ALIGNMENT)
        total_size = _pad_to_multiple(offset, bucket_alignment)
        self._total_size = total_size
        adam_total_cost = total_size * self.adam_cost_multiplier
        target_total_cost = (sum(base_cost_by_rank) + adam_total_cost) / self.world_size
        desired_adam_cost = [
            max(0.0, target_total_cost - float(base_cost_by_rank[rank]))
            for rank in range(self.world_size)
        ]
        desired_adam_numel = [cost / self.adam_cost_multiplier for cost in desired_adam_cost]
        self._rank_ranges = build_variable_flat_shard_ranges(
            total_size, desired_adam_numel, self.SHARD_BOUNDARY_ALIGNMENT
        )

    def _init_local_shards(self) -> None:
        self._local_segments_by_group = [[] for _ in self._param_groups]
        if not self._rank_ranges:
            return

        local_range = self._rank_ranges[self.rank]
        range_map = build_param_range_map_from_world_range(self._param_index_map, local_range)
        group_flat_parts: List[List[torch.Tensor]] = [[] for _ in self._param_groups]

        for param, ranges in range_map.items():
            group_index = self._param_to_group_index[param]
            param_range = ranges["param"]
            flat_world = ranges["flat_world"]
            part = param.detach().view(-1)[param_range.start : param_range.stop].float().clone()
            local_offset = sum(t.numel() for t in group_flat_parts[group_index])
            group_flat_parts[group_index].append(part)
            self._local_segments_by_group[group_index].append(
                {
                    "param": param,
                    "param_range": param_range,
                    "flat_world": flat_world,
                    "local_offset": local_offset,
                    "numel": part.numel(),
                }
            )

        for group_index, parts in enumerate(group_flat_parts):
            if parts:
                main_shard = torch.nn.Parameter(torch.cat(parts), requires_grad=True)
                self._main_shards.append(main_shard)
                self._float16_groups.append([])
                self._param_groups[group_index]["_flat_shard_param"] = main_shard
            else:
                self._float16_groups.append([])

    def get_parameters(self) -> List[torch.nn.Parameter]:
        return self._main_shards

    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        return [param.grad for param in self._main_shards if param.grad is not None]

    def get_loss_scale(self) -> torch.Tensor:
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        return torch.tensor([1.0], dtype=torch.float32, device=device)

    def zero_grad(self, set_to_none: bool = True):
        for group in self._param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.zero_()
        for param in self._main_shards:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()

    def reload_model_params(self, state_dict=None):
        self._copy_model_params_to_main_params(state_dict=state_dict)

    def _copy_model_params_to_main_params(self, state_dict=None):
        assert state_dict is None, "Initialize main params from state dict is not supported"
        for group_index, segments in enumerate(self._local_segments_by_group):
            main_shard = self._param_groups[group_index].get("_flat_shard_param")
            if main_shard is None:
                continue
            parts = []
            for segment in segments:
                param = segment["param"]
                param_range = segment["param_range"]
                parts.append(param.detach().view(-1)[param_range.start : param_range.stop].float())
            if parts:
                main_shard.data.copy_(torch.cat(parts))

    @torch.no_grad()
    def _copy_model_grads_to_main_grads(self) -> None:
        for group_index, segments in enumerate(self._local_segments_by_group):
            main_shard = self._param_groups[group_index].get("_flat_shard_param")
            if main_shard is None:
                continue
            grad_parts = []
            for segment in segments:
                param = segment["param"]
                param_range = segment["param_range"]
                if (
                    self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
                    and hasattr(param, "decoupled_grad")
                    and param.decoupled_grad is not None
                ):
                    grad = param.decoupled_grad
                elif hasattr(param, "main_grad"):
                    grad = param.main_grad
                else:
                    grad = param.grad

                if grad is None:
                    grad_parts.append(torch.zeros(segment["numel"], device=param.device))
                else:
                    grad_parts.append(
                        grad.detach().view(-1)[param_range.start : param_range.stop].float()
                    )
            main_shard.grad = torch.cat(grad_parts) if grad_parts else None

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        self._sync_megatron_adam_group_options()
        return super().step_with_ready_grads()

    @torch.no_grad()
    def _copy_main_params_to_model_params(self) -> None:
        self._copy_main_shards_to_model_params()

    @torch.no_grad()
    def _copy_main_shards_to_model_params(self) -> None:
        for group_index, segments in enumerate(self._local_segments_by_group):
            main_shard = self._param_groups[group_index].get("_flat_shard_param")
            if main_shard is None:
                continue
            for segment in segments:
                start = segment["local_offset"]
                end = start + segment["numel"]
                param = segment["param"]
                param_range = segment["param_range"]
                param.view(-1)[param_range.start : param_range.stop].copy_(
                    main_shard.data[start:end].to(dtype=param.dtype)
                )

    @torch.no_grad()
    def allgather_params(self) -> None:
        """Gather updated Adam shards into the replicated model parameters."""
        for group_index, group in enumerate(self._param_groups):
            params = group["params"]
            if not params:
                continue
            dtype = params[0].dtype
            device = params[0].device
            local_range = self._rank_ranges[self.rank] if self._rank_ranges else range(0, 0)
            local_tensor = self._build_local_model_flat(group_index, local_range, dtype, device)
            sizes = [
                sum(
                    max(0, min(r.stop, end) - max(r.start, start))
                    for start, end, bucket_id in [self._param_index_map[p] for p in group["params"]]
                )
                for r in self._rank_ranges
            ]
            if not sizes or max(sizes) == 0:
                continue
            gather_list = [
                (
                    local_tensor
                    if rank == self.rank
                    else torch.empty(sizes[rank], device=device, dtype=dtype)
                )
                for rank in range(self.world_size)
            ]
            torch.distributed.all_gather(gather_list, local_tensor, group=self.process_group)
            for rank, tensor in enumerate(gather_list):
                if rank == self.rank or tensor.numel() == 0:
                    continue
                self._copy_gathered_model_flat_to_params(
                    group_index, self._rank_ranges[rank], tensor
                )

    def _build_local_model_flat(
        self, group_index: int, local_range: range, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        parts = []
        for segment in self._local_segments_by_group[group_index]:
            param = segment["param"]
            param_range = segment["param_range"]
            parts.append(param.detach().view(-1)[param_range.start : param_range.stop].to(dtype))
        return torch.cat(parts) if parts else torch.empty(0, device=device, dtype=dtype)

    def _copy_gathered_model_flat_to_params(
        self, group_index: int, rank_range: range, tensor: torch.Tensor
    ) -> None:
        offset = 0
        for param in self._param_groups[group_index]["params"]:
            start, end, _bucket_id = self._param_index_map[param]
            local_start = max(0, start - rank_range.start)
            local_end = min(rank_range.stop - rank_range.start, end - rank_range.start)
            if local_end <= local_start:
                continue
            length = local_end - local_start
            param_start = max(0, rank_range.start - start)
            param.view(-1)[param_start : param_start + length].copy_(
                tensor[offset : offset + length].to(dtype=param.dtype)
            )
            offset += length

    def flat_param_slices_for_bucket(
        self, bucket_params: List[torch.Tensor], rank: int
    ) -> List[dict]:
        """Return Adam flat-shard slices owned by ``rank`` for params in one DDP bucket."""
        if not self._rank_ranges:
            return []

        rank_range = self._rank_ranges[rank]
        flat_slices = []
        for param in bucket_params:
            if param not in self._param_index_map:
                continue
            param_world_start, param_world_end, _bucket_id = self._param_index_map[param]
            flat_local_start = max(0, param_world_start - rank_range.start)
            flat_local_end = min(
                rank_range.stop - rank_range.start, param_world_end - rank_range.start
            )
            if flat_local_end <= flat_local_start:
                continue

            param_start = max(0, rank_range.start - param_world_start)
            numel = flat_local_end - flat_local_start
            flat_slices.append(
                {
                    "param": param,
                    "param_range": range(param_start, param_start + numel),
                    "numel": numel,
                }
            )
        return flat_slices

    def _group_range_numel(self, group_index: int, rank_range: range) -> int:
        return sum(
            max(0, min(rank_range.stop, end) - max(rank_range.start, start))
            for start, end, _bucket_id in [
                self._param_index_map[param] for param in self._param_groups[group_index]["params"]
            ]
        )

    def _group_local_checkpoint_flat(
        self, group_index: int, tensor_name: str, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        main_shard = self._param_groups[group_index].get("_flat_shard_param")
        if main_shard is None:
            return torch.empty(0, device=device, dtype=dtype)
        if tensor_name == "fp32":
            return main_shard.detach().to(dtype=dtype, device=device)
        return self._state[main_shard][tensor_name].detach().to(dtype=dtype, device=device)

    def _copy_checkpoint_flat_to_group_tensors(
        self,
        group_index: int,
        rank_range: range,
        flat_tensor: torch.Tensor,
        full_tensors_by_param: Dict[torch.Tensor, torch.Tensor],
    ) -> None:
        offset = 0
        for param in self._param_groups[group_index]["params"]:
            start, end, _bucket_id = self._param_index_map[param]
            local_start = max(0, start - rank_range.start)
            local_end = min(rank_range.stop - rank_range.start, end - rank_range.start)
            if local_end <= local_start:
                continue
            length = local_end - local_start
            param_start = max(0, rank_range.start - start)
            full_tensors_by_param[param].view(-1)[param_start : param_start + length].copy_(
                flat_tensor[offset : offset + length].to(dtype=full_tensors_by_param[param].dtype)
            )
            offset += length

    def _gather_full_group_tensors(self, group_index: int, tensor_name: str) -> List[torch.Tensor]:
        """Gather variable flat shards and reconstruct per-parameter tensors.

        The torch_dist optimizer checkpoint format is per-parameter. Reconstructing
        this view lets checkpoints saved by the memory-balanced path reload through
        the same metadata as old ping-pong layer-wise Adam checkpoints, while the
        runtime optimizer still owns only local variable flat shards.
        """

        params = self._param_groups[group_index]["params"]
        if not params:
            return []
        device = params[0].device
        dtype = torch.float32
        local_tensor = self._group_local_checkpoint_flat(group_index, tensor_name, dtype, device)
        sizes = [self._group_range_numel(group_index, r) for r in self._rank_ranges]
        if not sizes or max(sizes) == 0:
            return [torch.zeros_like(param, dtype=dtype, device=device) for param in params]
        gather_list = [
            (
                local_tensor
                if rank == self.rank
                else torch.empty(sizes[rank], device=device, dtype=dtype)
            )
            for rank in range(self.world_size)
        ]
        torch.distributed.all_gather(gather_list, local_tensor, group=self.process_group)

        full_tensors = [torch.empty_like(param, dtype=dtype, device=device) for param in params]
        full_tensors_by_param = dict(zip(params, full_tensors))
        for rank, tensor in enumerate(gather_list):
            if tensor.numel() == 0:
                continue
            self._copy_checkpoint_flat_to_group_tensors(
                group_index, self._rank_ranges[rank], tensor, full_tensors_by_param
            )
        return full_tensors

    def _group_step_tensor(self, group_index: int, device: torch.device) -> torch.Tensor:
        local_step_tensor = self._get_local_group_step(group_index, device)
        local_step = int(local_step_tensor.item()) if local_step_tensor is not None else None
        all_steps = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(all_steps, local_step, group=self.process_group)
        step = next((step for step in all_steps if step is not None), 0)
        return torch.tensor(step, dtype=torch.int64, device=device)

    def state_dict(self):
        optimizer_state = {"state": {}, "param_groups": []}
        fp32_from_fp16_params = []
        next_param_id = 0

        for group_index, group in enumerate(self._param_groups):
            params = group["params"]
            fp32_params = self._gather_full_group_tensors(group_index, "fp32")
            exp_avg_params = self._gather_full_group_tensors(group_index, "exp_avg")
            exp_avg_sq_params = self._gather_full_group_tensors(group_index, "exp_avg_sq")
            param_ids = list(range(next_param_id, next_param_id + len(params)))
            next_param_id += len(params)

            group_state = {
                k: v for k, v in group.items() if k not in ("params", "_flat_shard_param")
            }
            group_state["params"] = param_ids
            optimizer_state["param_groups"].append(group_state)
            fp32_from_fp16_params.append(fp32_params)

            step = self._group_step_tensor(group_index, params[0].device) if params else None
            for param_id, exp_avg, exp_avg_sq in zip(param_ids, exp_avg_params, exp_avg_sq_params):
                optimizer_state["state"][param_id] = {
                    "step": step.clone(),
                    "exp_avg": exp_avg,
                    "exp_avg_sq": exp_avg_sq,
                }

        return {"optimizer": optimizer_state, "fp32_from_fp16_params": fp32_from_fp16_params}

    def load_state_dict(self, state_dict):
        if "optimizer" in state_dict and "fp32_from_fp16_params" in state_dict:
            self._load_from_float16_optimizer_state_dict(state_dict)
            return

        shard_states = iter(state_dict.get("state", []))
        for group_index, group in enumerate(self._param_groups):
            main_shard = group.get("_flat_shard_param")
            if main_shard is None:
                continue
            shard_state = next(shard_states)
            self._set_local_group_step(group_index, shard_state["step"])
            self._state[main_shard]["exp_avg"].copy_(shard_state["exp_avg"].to(main_shard.device))
            self._state[main_shard]["exp_avg_sq"].copy_(
                shard_state["exp_avg_sq"].to(main_shard.device)
            )

    @staticmethod
    def _unwrap_checkpoint_value(value):
        if hasattr(value, "unwrap"):
            return value.unwrap()
        return value

    @staticmethod
    def _lookup_saved_param_state(saved_state: dict, saved_id):
        candidate_ids = [saved_id]
        if torch.is_tensor(saved_id) and saved_id.numel() == 1:
            candidate_ids.append(int(saved_id.item()))
        if isinstance(saved_id, str):
            try:
                candidate_ids.append(int(saved_id))
            except ValueError:
                pass
        else:
            candidate_ids.append(str(saved_id))

        for candidate_id in candidate_ids:
            try:
                if candidate_id in saved_state:
                    return saved_state[candidate_id]
            except TypeError:
                continue
        return {}

    def _load_from_float16_optimizer_state_dict(self, state_dict) -> None:
        """Migrate old layer-wise Adam checkpoint state into local flat shards.

        Old ping-pong layer-wise Adam checkpoints store full per-parameter fp32
        main weights and Adam moments. This helper slices those tensors according
        to the new variable flat ranges owned by this rank.
        """

        optimizer_state = state_dict["optimizer"]
        saved_param_groups = self._unwrap_checkpoint_value(optimizer_state["param_groups"])
        saved_fp32_groups = self._unwrap_checkpoint_value(state_dict["fp32_from_fp16_params"])
        if isinstance(saved_fp32_groups, dict):
            saved_fp32_groups = [saved_fp32_groups[i] for i in sorted(saved_fp32_groups)]
        saved_state = self._unwrap_checkpoint_value(optimizer_state["state"])
        saved_step = saved_state.get("common_step", None) if isinstance(saved_state, dict) else None

        src_global_rank = torch.distributed.get_global_rank(self.process_group, 0)
        is_checkpoint_source = self.rank == 0
        segments_by_group_param: List[Dict[torch.Tensor, List[dict]]] = []
        for segments in self._local_segments_by_group:
            segments_by_param = defaultdict(list)
            for segment in segments:
                segments_by_param[segment["param"]].append(segment)
            segments_by_group_param.append(segments_by_param)

        group_buffers = {}
        for group_index, segments in enumerate(self._local_segments_by_group):
            main_shard = self._param_groups[group_index].get("_flat_shard_param")
            if main_shard is None:
                continue
            group_buffers[group_index] = {
                "main": torch.empty_like(main_shard.data),
                "exp_avg": torch.empty_like(self._state[main_shard]["exp_avg"]),
                "exp_avg_sq": torch.empty_like(self._state[main_shard]["exp_avg_sq"]),
                "step": None,
            }

        for group_index, current_group in enumerate(self._param_groups):
            saved_param_ids = []
            saved_fp32_params = []
            if (
                is_checkpoint_source
                and group_index < len(saved_param_groups)
                and group_index < len(saved_fp32_groups)
            ):
                saved_group = saved_param_groups[group_index]
                saved_param_ids = self._unwrap_checkpoint_value(saved_group.get("params", []))
                saved_fp32_params = self._unwrap_checkpoint_value(saved_fp32_groups[group_index])
                if (
                    saved_param_ids is None
                    or saved_fp32_params is None
                    or isinstance(saved_param_ids, bool)
                    or isinstance(saved_fp32_params, bool)
                ):
                    saved_param_ids = []
                    saved_fp32_params = []

            for param_index, param in enumerate(current_group["params"]):
                device = param.device
                if is_checkpoint_source and param_index < len(saved_param_ids):
                    saved_id = saved_param_ids[param_index]
                    param_state = self._lookup_saved_param_state(saved_state, saved_id)
                    saved_fp32 = self._unwrap_checkpoint_value(saved_fp32_params[param_index])
                    fp32_tensor = saved_fp32.to(device=device, dtype=torch.float32)
                    exp_avg_tensor = self._unwrap_checkpoint_value(param_state.get("exp_avg", None))
                    exp_avg_sq_tensor = self._unwrap_checkpoint_value(
                        param_state.get("exp_avg_sq", None)
                    )
                    exp_avg_tensor = (
                        torch.zeros_like(fp32_tensor)
                        if exp_avg_tensor is None
                        else exp_avg_tensor.to(device=device, dtype=torch.float32)
                    )
                    exp_avg_sq_tensor = (
                        torch.zeros_like(fp32_tensor)
                        if exp_avg_sq_tensor is None
                        else exp_avg_sq_tensor.to(device=device, dtype=torch.float32)
                    )
                    step_tensor = self._unwrap_checkpoint_value(param_state.get("step", saved_step))
                    if step_tensor is None:
                        step_tensor = torch.tensor(0, dtype=torch.int64, device=device)
                    elif not torch.is_tensor(step_tensor):
                        step_tensor = torch.tensor(step_tensor, dtype=torch.int64, device=device)
                    else:
                        step_tensor = step_tensor.to(device=device, dtype=torch.int64)
                else:
                    fp32_tensor = torch.empty_like(param, dtype=torch.float32, device=device)
                    exp_avg_tensor = torch.empty_like(fp32_tensor)
                    exp_avg_sq_tensor = torch.empty_like(fp32_tensor)
                    step_tensor = torch.empty((), dtype=torch.int64, device=device)

                torch.distributed.broadcast(
                    fp32_tensor, src=src_global_rank, group=self.process_group
                )
                torch.distributed.broadcast(
                    exp_avg_tensor, src=src_global_rank, group=self.process_group
                )
                torch.distributed.broadcast(
                    exp_avg_sq_tensor, src=src_global_rank, group=self.process_group
                )
                torch.distributed.broadcast(
                    step_tensor, src=src_global_rank, group=self.process_group
                )

                buffers = group_buffers.get(group_index)
                if buffers is None:
                    continue
                if buffers["step"] is None:
                    buffers["step"] = step_tensor
                for segment in segments_by_group_param[group_index].get(param, []):
                    param_range = segment["param_range"]
                    local_start = segment["local_offset"]
                    local_end = local_start + segment["numel"]
                    buffers["main"][local_start:local_end].copy_(
                        fp32_tensor.view(-1)[param_range.start : param_range.stop]
                    )
                    buffers["exp_avg"][local_start:local_end].copy_(
                        exp_avg_tensor.view(-1)[param_range.start : param_range.stop]
                    )
                    buffers["exp_avg_sq"][local_start:local_end].copy_(
                        exp_avg_sq_tensor.view(-1)[param_range.start : param_range.stop]
                    )

        for group_index, buffers in group_buffers.items():
            main_shard = self._param_groups[group_index].get("_flat_shard_param")
            main_shard.data.copy_(buffers["main"])
            self._state[main_shard]["exp_avg"].copy_(buffers["exp_avg"])
            self._state[main_shard]["exp_avg_sq"].copy_(buffers["exp_avg_sq"])
            if buffers["step"] is not None:
                self._set_local_group_step(group_index, buffers["step"])
        self._copy_main_shards_to_model_params()

    @torch.no_grad()
    def step(self):
        self.prepare_grads()
        return self.step_with_ready_grads(), self.get_grad_norm(), self.count_zeros()

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False, **kwargs
    ):
        state_dict = self.state_dict()
        if self.rank != 0:
            state_dict["fp32_from_fp16_params"] = [[] for _ in self._param_groups]
            state_dict["optimizer"]["state"] = {}
            for group in state_dict["optimizer"]["param_groups"]:
                group["params"] = []
            return state_dict

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict,
            chain.from_iterable(group["params"] for group in self._param_groups),
        )

        state_dict["fp32_from_fp16_params"] = [
            [
                make_sharded_optimizer_tensor(
                    id_to_sharded_param_map[param_id],
                    fp32_param,
                    prefix="optimizer.state.fp32_param",
                )
                for param_id, fp32_param in zip(state_group["params"], fp32_group)
            ]
            for fp32_group, state_group in zip(
                state_dict["fp32_from_fp16_params"], state_dict["optimizer"]["param_groups"]
            )
        ]
        step = next(
            (
                param_state["step"]
                for param_state in state_dict["optimizer"]["state"].values()
                if isinstance(param_state, dict) and "step" in param_state
            ),
            None,
        )
        optim_state_to_sharding_state(
            state_dict["optimizer"],
            id_to_sharded_param_map,
            exclude_keys=(
                "step",
                "hyperball_radius",
                "hyperball_partition_radii",
                "hyperball_partition_state_version",
                "hyperball_partition_layout_code",
            ),
        )
        if step is not None:
            state_dict["optimizer"]["state"]["common_step"] = self._checkpoint_common_step(step)
        return state_dict

    def memory_assignment_summary(self, label: str) -> tuple[list[str], list[float]]:
        """Describe the estimated memory assigned to each optimizer owner."""
        local_range = self._rank_ranges[self.rank] if self._rank_ranges else range(0, 0)
        local_cost = float(local_range.stop - local_range.start) * self.adam_cost_multiplier
        rank_costs = [
            float(rank_range.stop - rank_range.start) * self.adam_cost_multiplier
            for rank_range in self._rank_ranges
        ]
        lines = [
            (
                f"  [{label}] adam_flat_shard range=[{local_range.start}, {local_range.stop}) "
                f"numel={local_range.stop - local_range.start} "
                f"estimated_optimizer_state={_format_mem(local_cost)}"
            )
        ]
        for group_index, segments in enumerate(self._local_segments_by_group):
            for segment in segments:
                param = segment["param"]
                param_range = segment["param_range"]
                cost = float(segment["numel"]) * self.adam_cost_multiplier
                lines.append(
                    "    "
                    f"optimizer=adam param={_param_log_name(param)} "
                    f"shape={tuple(param.shape)} "
                    f"slice=[{param_range.start}, {param_range.stop}) "
                    f"numel={segment['numel']} "
                    f"estimated_optimizer_state={_format_mem(cost)}"
                )
        return lines, rank_costs


class LayerWiseDistributedOptimizer(ChainedOptimizer):
    """Layer-wise distributed optimizer for Megatron-core models.

    Experimental distributed optimizer wrapper that distributes weight to DP ranks by layer.
    Implemented as ChainedOptimizer to support multiple optimizers (e.g. muon + adamW)
    When using, keep all megatron distributed-optimizer related options OFF.

    How LayerWiseDistributedOptimizer work:
    1. weights are splited into lists and each rank only keep its shard in its optimizer
    2. Megatron DDP handle allreduce grad, note that each rank have full model and grad
    3. optimizer is already modified so only param belong to this DP rank is updated
    4. grad_norm and zero counting will reduce metrics globally in step function
    5. Do regular update with chained optimizers, modified optimizer only update shard
    6. allgather updated params to every rank
    """

    def __init__(
        self,
        optimizers: List[MegatronOptimizer],
        config: OptimizerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        init_state_fn_list: Optional[List[Callable]] = None,
        model_chunks: Optional[List] = None,
        async_allgather: bool = False,
    ) -> None:
        """
        Initialize LayerWiseDistributedOptimizer.

        Args:
            optimizers: List of MegatronOptimizers.
            config: OptimizerConfig.
            pg_collection: ProcessGroupCollection.
            init_state_fn_list: List of init state functions.
            model_chunks: DDP-wrapped model chunks (needed for async_allgather).
            async_allgather: If True, defer param all-gather to forward pre-hooks.
        """

        self.pg_collection = pg_collection
        self.config = config
        self._flat_adam_fallback_optimizers: List[FlatShardedAdamFallback] = []
        self._flat_adam_fallback_optimizer_map: Dict[int, List[FlatShardedAdamFallback]] = (
            defaultdict(list)
        )
        self.shard_params(optimizers)

        # Set up async all-gather using DDP bucket infrastructure.
        self.async_allgather = async_allgather
        self.model_chunks = model_chunks
        if self.async_allgather:
            assert (
                model_chunks is not None
            ), "model_chunks must be provided if async_allgather is True"
            self.set_bucket_layerwise_params_list(model_chunks)

        if init_state_fn_list:
            assert len(init_state_fn_list) == len(
                optimizers
            ), "init_state_fn_list must be the same length as optimizers if provided"

        # Wrap base torch optimizers with Float16 for bf16 training.
        # Callers pass base optimizers; wrapping happens here *after*
        # shard_params so master weights are only created for the local shard.
        if init_state_fn_list:
            new_optimizers = []
            new_init_state_fn_list = []
            for optimizer_index, (optimizer, init_fn) in enumerate(
                zip(optimizers, init_state_fn_list)
            ):
                if getattr(optimizer, "_layerwise_remove_optimizer", False):
                    replacements = self._flat_adam_fallback_optimizer_map.get(optimizer_index, [])
                    new_optimizers.extend(replacements)
                    new_init_state_fn_list.extend([None for _ in replacements])
                else:
                    new_optimizers.append(optimizer)
                    new_init_state_fn_list.append(init_fn)
            optimizers[:] = new_optimizers
            init_state_fn_list = new_init_state_fn_list
        else:
            new_optimizers = []
            for optimizer_index, optimizer in enumerate(optimizers):
                if getattr(optimizer, "_layerwise_remove_optimizer", False):
                    new_optimizers.extend(
                        self._flat_adam_fallback_optimizer_map.get(optimizer_index, [])
                    )
                else:
                    new_optimizers.append(optimizer)
            optimizers[:] = new_optimizers
        if config.bf16:
            for i in range(len(optimizers)):
                opt = optimizers[i]
                if isinstance(opt, MegatronOptimizer):
                    continue
                if isinstance(opt, (Float16OptimizerWithFloat16Params, FP32Optimizer)):
                    raise TypeError(
                        'LayerWiseDistributedOptimizer expects base torch optimizers, '
                        f'got {type(opt).__name__}. Do not pre-wrap with Megatron optimizers.'
                    )
                optimizers[i] = Float16OptimizerWithFloat16Params(
                    opt, config, None, init_state_fn_list[i] if init_state_fn_list else None
                )

        super().__init__(optimizers)

        # TODO(kunlun, deyuf): potential future perf optimization
        # since allreduce is unchanged and handled by megatron DDP, they're already in
        # contiguous gbuf. So instead of shard param by layer randomly, we can shard by
        # buf range but keep some "extras" to keep boundary weight not sharded.
        # This way each rank do some duplicated work but allgather_v is no longer needed
        # All current distopt optimization can also be potentially applied

    def shard_params(self, optimizers):
        """Shard all params into lists by rank."""
        if self.config.layerwise_optimizer_memory_balance:
            return self._shard_params_memory_optimized(optimizers)

        # list of parameter are sorted by numel and assigned to ranks in ping-pong style
        # example of 4 ranks and 10 parameters p0-p9 after sorting, then dp_cp_params_list will be
        # [[p0, p7, p8], [p1, p6, p9], [p2, p5], [p3, p4]]

        # simplify when dp_cp group size is 1
        if get_pg_size(self.pg_collection.dp_cp) == 1:
            self.dp_cp_params_list = None
            self.expt_dp_params_list = None
            return

        dp_cp_idx, expt_dp_idx = 0, 0
        dp_cp_size = get_pg_size(self.pg_collection.dp_cp)
        expt_dp_size = get_pg_size(self.pg_collection.expt_dp)
        # create ping-pong style loop so memory is more balanced
        dp_cp_loop = list(range(dp_cp_size)) + list(range(dp_cp_size))[::-1]
        expt_dp_loop = list(range(expt_dp_size)) + list(range(expt_dp_size))[::-1]
        self.dp_cp_params_list = [[] for _ in range(dp_cp_size)]
        self.expt_dp_params_list = [[] for _ in range(expt_dp_size)]
        # get all param groups
        param_groups = []
        for optimizer in optimizers:
            param_groups += optimizer.param_groups

        # sort param in all groups by param numel and assign to each rank evenly
        param_list = []
        for group_index, group in enumerate(param_groups):
            for p in group["params"]:
                param_list.append((p, group_index))
        param_list.sort(key=lambda x: x[0].numel())
        param_groups_this_rank = [[] for g in param_groups]

        # assign params to rank in ping-pong style loop
        for p, group_index in param_list:
            if param_groups[group_index].get("is_expert_parallel", False):
                if expt_dp_loop[expt_dp_idx] == get_pg_rank(self.pg_collection.expt_dp):
                    param_groups_this_rank[group_index].append(p)
                self.expt_dp_params_list[expt_dp_loop[expt_dp_idx]].append(p)
                expt_dp_idx = (expt_dp_idx + 1) % len(expt_dp_loop)
            else:
                if dp_cp_loop[dp_cp_idx] == get_pg_rank(self.pg_collection.dp_cp):
                    param_groups_this_rank[group_index].append(p)
                self.dp_cp_params_list[dp_cp_loop[dp_cp_idx]].append(p)
                dp_cp_idx = (dp_cp_idx + 1) % len(dp_cp_loop)

        # now we modify the group to only handle local params
        for groups, params in zip(param_groups, param_groups_this_rank):
            groups["params"] = params

        # Optimizers are constructed before layer-wise sharding, and some optimizers
        # eagerly initialize state in their constructors. Keep only state attached to
        # params that remain in this rank's param groups; otherwise torch.optim
        # state_dict() cannot map those tensor keys back to a param-group index.
        for optimizer in optimizers:
            self._prune_optimizer_state_to_local_params(optimizer)

        # simplify when expt_dp group size is 1 or expert parallel is off
        if expt_dp_size == 1 or len(self.expt_dp_params_list[0]) == 0:
            self.expt_dp_params_list = None

    def _shard_params_memory_optimized(self, optimizers):
        """Shard Muon-family tensors whole and Adam fallback tensors as flat ranges."""

        if get_pg_size(self.pg_collection.dp_cp) == 1:
            self.dp_cp_params_list = None
            self.expt_dp_params_list = None
            return

        dp_cp_size = get_pg_size(self.pg_collection.dp_cp)
        expt_dp_size = get_pg_size(self.pg_collection.expt_dp)
        dp_cp_rank = get_pg_rank(self.pg_collection.dp_cp)
        expt_dp_rank = get_pg_rank(self.pg_collection.expt_dp)
        self.dp_cp_params_list = [[] for _ in range(dp_cp_size)]
        self.expt_dp_params_list = [[] for _ in range(expt_dp_size)]
        dp_cp_costs = [0.0 for _ in range(dp_cp_size)]
        expt_dp_costs = [0.0 for _ in range(expt_dp_size)]

        param_groups = []
        group_optimizer = []
        optimizer_index_by_id = {}
        for optimizer_index, optimizer in enumerate(optimizers):
            optimizer_index_by_id[id(optimizer)] = optimizer_index
            for group in optimizer.param_groups:
                param_groups.append(group)
                group_optimizer.append(optimizer)

        param_groups_this_rank = [[] for _ in param_groups]
        adam_groups_by_optimizer_index = defaultdict(lambda: {"dense": [], "expert": []})
        muon_items = []
        other_items = []
        local_assignment_lines = []

        for group_index, group in enumerate(param_groups):
            optimizer = group_optimizer[group_index]
            is_expert = group.get("is_expert_parallel", False)
            if _is_adam_optimizer(optimizer):
                cloned_group = FlatShardedAdamFallback._clone_param_group(group)
                optimizer_index = optimizer_index_by_id[id(optimizer)]
                if is_expert:
                    adam_groups_by_optimizer_index[optimizer_index]["expert"].append(cloned_group)
                else:
                    adam_groups_by_optimizer_index[optimizer_index]["dense"].append(cloned_group)
                group["params"] = []
                setattr(optimizer, "_layerwise_remove_optimizer", True)
                continue

            for order, param in enumerate(group["params"]):
                item = (param, group_index, order)
                if _is_muon_optimizer(optimizer):
                    muon_items.append(item)
                else:
                    other_items.append(item)

        def assign_whole_tensors(items):
            for param, group_index, order in sorted(
                items,
                key=lambda x: (
                    -_whole_param_optimizer_state_cost(x[0], group_optimizer[x[1]], self.config),
                    x[1],
                    x[2],
                ),
            ):
                group = param_groups[group_index]
                is_expert = group.get("is_expert_parallel", False)
                optimizer_name = _optimizer_class_name(group_optimizer[group_index])
                cost = float(
                    _whole_param_optimizer_state_cost(
                        param, group_optimizer[group_index], self.config
                    )
                )
                if is_expert:
                    rank = min(range(expt_dp_size), key=lambda r: (expt_dp_costs[r], r))
                    self.expt_dp_params_list[rank].append(param)
                    expt_dp_costs[rank] += cost
                    if rank == expt_dp_rank:
                        param_groups_this_rank[group_index].append(param)
                        local_assignment_lines.append(
                            "  "
                            f"[expt_dp] optimizer={optimizer_name} "
                            f"param={_param_log_name(param)} shape={tuple(param.shape)} "
                            f"numel={param.numel()} "
                            f"estimated_optimizer_state={_format_mem(cost)}"
                        )
                else:
                    rank = min(range(dp_cp_size), key=lambda r: (dp_cp_costs[r], r))
                    self.dp_cp_params_list[rank].append(param)
                    dp_cp_costs[rank] += cost
                    if rank == dp_cp_rank:
                        param_groups_this_rank[group_index].append(param)
                        local_assignment_lines.append(
                            "  "
                            f"[dp_cp] optimizer={optimizer_name} "
                            f"param={_param_log_name(param)} shape={tuple(param.shape)} "
                            f"numel={param.numel()} "
                            f"estimated_optimizer_state={_format_mem(cost)}"
                        )

        assign_whole_tensors(muon_items)
        assign_whole_tensors(other_items)

        for group, params in zip(param_groups, param_groups_this_rank):
            if group["params"]:
                group["params"] = params

        for optimizer in optimizers:
            if not getattr(optimizer, "_layerwise_remove_optimizer", False):
                self._prune_optimizer_state_to_local_params(optimizer)

        for optimizer_index in sorted(adam_groups_by_optimizer_index):
            adam_groups = adam_groups_by_optimizer_index[optimizer_index]
            if adam_groups["dense"]:
                flat_optimizer = FlatShardedAdamFallback(
                    adam_groups["dense"], self.config, self.pg_collection.dp_cp, dp_cp_costs
                )
                self._flat_adam_fallback_optimizers.append(flat_optimizer)
                self._flat_adam_fallback_optimizer_map[optimizer_index].append(flat_optimizer)
            if adam_groups["expert"]:
                flat_optimizer = FlatShardedAdamFallback(
                    adam_groups["expert"], self.config, self.pg_collection.expt_dp, expt_dp_costs
                )
                self._flat_adam_fallback_optimizers.append(flat_optimizer)
                self._flat_adam_fallback_optimizer_map[optimizer_index].append(flat_optimizer)

        if expt_dp_size == 1 or len(self.expt_dp_params_list[0]) == 0:
            self.expt_dp_params_list = None

        self._log_memory_optimized_assignment(
            local_assignment_lines, dp_cp_costs, expt_dp_costs, dp_cp_rank, expt_dp_rank
        )

    def _log_memory_optimized_assignment(
        self,
        local_assignment_lines: List[str],
        dp_cp_costs: List[float],
        expt_dp_costs: List[float],
        dp_cp_rank: int,
        expt_dp_rank: int,
    ) -> None:
        dense_adam_costs = [0.0 for _ in dp_cp_costs]
        expert_adam_costs = [0.0 for _ in expt_dp_costs]
        flat_lines = []
        for flat_optimizer in self._flat_adam_fallback_optimizers:
            label = (
                "dp_cp" if flat_optimizer.process_group == self.pg_collection.dp_cp else "expt_dp"
            )
            lines, rank_costs = flat_optimizer.memory_assignment_summary(label)
            flat_lines.extend(lines)
            if label == "dp_cp":
                dense_adam_costs = rank_costs
            else:
                expert_adam_costs = rank_costs

        dp_cp_total = [
            float(muon_cost) + float(adam_cost)
            for muon_cost, adam_cost in zip(dp_cp_costs, dense_adam_costs)
        ]
        expt_dp_total = [
            float(muon_cost) + float(adam_cost)
            for muon_cost, adam_cost in zip(expt_dp_costs, expert_adam_costs)
        ]
        local_dense_total = dp_cp_total[dp_cp_rank] if dp_cp_total else 0.0
        local_expert_total = expt_dp_total[expt_dp_rank] if expt_dp_total else 0.0
        local_total = local_dense_total + local_expert_total

        def _rank_cost_summary(costs: List[float]) -> str:
            return ", ".join(f"rank{rank}={_format_mem(cost)}" for rank, cost in enumerate(costs))

        def _safe_parallel_rank(get_rank_fn) -> int:
            try:
                return int(get_rank_fn())
            except Exception:
                return -1

        local_payload = {
            "dp_cp_rank": dp_cp_rank,
            "expt_dp_rank": expt_dp_rank,
            "local_dense_total": local_dense_total,
            "local_expert_total": local_expert_total,
            "local_total": local_total,
            "lines": local_assignment_lines + flat_lines,
        }
        group_payloads = [None for _ in range(get_pg_size(self.pg_collection.dp_cp))]
        torch.distributed.all_gather_object(
            group_payloads, local_payload, group=self.pg_collection.dp_cp
        )

        # Print once per dense optimizer sharding group. With PP=2, TP=1, CP=1 this
        # produces two blocks: one for each pipeline stage's DP/CP optimizer group.
        if dp_cp_rank != 0:
            return

        group_payloads = sorted(group_payloads, key=lambda payload: payload["dp_cp_rank"])
        header = [
            "[layerwise memory optimization] optimizer assignment group",
            (
                f"  printer_global_rank={torch.distributed.get_rank()} "
                f"tensor_model_parallel_rank="
                f"{_safe_parallel_rank(parallel_state.get_tensor_model_parallel_rank)} "
                f"pipeline_model_parallel_rank="
                f"{_safe_parallel_rank(parallel_state.get_pipeline_model_parallel_rank)} "
                f"expert_model_parallel_rank="
                f"{_safe_parallel_rank(parallel_state.get_expert_model_parallel_rank)}"
            ),
            f"  dp_cp_group_size={get_pg_size(self.pg_collection.dp_cp)}",
            f"  dp_cp_muon_family_by_rank: {_rank_cost_summary(dp_cp_costs)}",
            f"  dp_cp_adam_by_rank: {_rank_cost_summary(dense_adam_costs)}",
            f"  dp_cp_total_by_rank: {_rank_cost_summary(dp_cp_total)}",
        ]
        if any(expt_dp_total):
            header.extend(
                [
                    f"  expt_dp_muon_family_by_rank: {_rank_cost_summary(expt_dp_costs)}",
                    f"  expt_dp_adam_by_rank: {_rank_cost_summary(expert_adam_costs)}",
                    f"  expt_dp_total_by_rank: {_rank_cost_summary(expt_dp_total)}",
                ]
            )
        header.append("  rank_assignments:")
        for payload in group_payloads:
            header.extend(
                [
                    (
                        f"  dp_cp_rank={payload['dp_cp_rank']} "
                        f"expt_dp_rank={payload['expt_dp_rank']} "
                        f"dense_total={_format_mem(payload['local_dense_total'])} "
                        f"expert_total={_format_mem(payload['local_expert_total'])} "
                        f"total={_format_mem(payload['local_total'])}"
                    )
                ]
            )
            if payload["lines"]:
                header.extend(payload["lines"])
        print("\n".join(header), flush=True)  # pylint: disable=bad-builtin

    @staticmethod
    def _prune_optimizer_state_to_local_params(optimizer) -> None:
        """Drop optimizer state for parameters removed by layer-wise sharding."""
        if not getattr(optimizer, "state", None):
            return

        local_param_ids = {
            id(param) for group in optimizer.param_groups for param in group.get("params", [])
        }
        if isinstance(optimizer.state, defaultdict):
            pruned_state = defaultdict(optimizer.state.default_factory)
        else:
            pruned_state = {}
        for key, value in optimizer.state.items():
            if not isinstance(key, torch.Tensor) or id(key) in local_param_ids:
                pruned_state[key] = value
        optimizer.state = pruned_state

    def set_bucket_layerwise_params_list(self, model_chunks):
        """Map sharded params to DDP buckets for async all-gather.

        For each bucket in each model chunk's bucket groups, build per-rank param lists
        by cross-referencing the layer-wise sharded param lists with the bucket's params.

        Args:
            model_chunks: DDP-wrapped model chunks with bucket_groups.
        """
        for model_chunk in model_chunks:
            for group in model_chunk.bucket_groups:
                for bucket in group.buckets:
                    bucket_params_list = [[] for _ in range(get_pg_size(self.pg_collection.dp_cp))]
                    full_params_by_rank = self.dp_cp_params_list
                    if full_params_by_rank is None:
                        # A size-one DP group does not shard parameters, but the overlap
                        # path still needs explicit bucket ownership metadata.
                        full_params_by_rank = [bucket.params_list]
                    for bucket_list, full_params_list in zip(
                        bucket_params_list, full_params_by_rank
                    ):
                        for param in full_params_list:
                            if param in bucket.params:
                                bucket_list.append(param)
                    flat_slices_by_rank = self._flat_adam_slices_for_bucket(
                        bucket.params_list, self.pg_collection.dp_cp
                    )
                    bucket.set_layerwise_params_list(bucket_params_list, flat_slices_by_rank)
            # Do the same for expert parallel bucket groups.
            if self.expt_dp_params_list is not None:
                for group in model_chunk.expert_parallel_bucket_groups:
                    for bucket in group.buckets:
                        bucket_params_list = [
                            [] for _ in range(get_pg_size(self.pg_collection.expt_dp))
                        ]
                        for bucket_list, full_params_list in zip(
                            bucket_params_list, self.expt_dp_params_list
                        ):
                            for param in full_params_list:
                                if param in bucket.params:
                                    bucket_list.append(param)
                        flat_slices_by_rank = self._flat_adam_slices_for_bucket(
                            bucket.params_list, self.pg_collection.expt_dp
                        )
                        bucket.set_layerwise_params_list(bucket_params_list, flat_slices_by_rank)

    def _flat_adam_slices_for_bucket(
        self, bucket_params: List[torch.Tensor], process_group: torch.distributed.ProcessGroup
    ) -> List[List[dict]]:
        """Build per-rank Adam flat slice metadata for one DDP bucket."""
        flat_slices_by_rank = [[] for _ in range(get_pg_size(process_group))]
        for flat_optimizer in self._flat_adam_fallback_optimizers:
            if flat_optimizer.process_group != process_group:
                continue
            for rank in range(get_pg_size(process_group)):
                flat_slices_by_rank[rank].extend(
                    flat_optimizer.flat_param_slices_for_bucket(bucket_params, rank)
                )
        return flat_slices_by_rank

    @torch.no_grad()
    def allgather_params(self) -> None:
        """All-gather updated params from all ranks."""

        # helper function to flatten local params, all-gather,
        # unflatten and copy to model params
        def _allgather_helper(params_list, group):
            first_nonempty_params = next((params for params in params_list if params), None)
            if first_nonempty_params is None:
                return
            device = first_nonempty_params[0].device
            dtype = first_nonempty_params[0].dtype
            rank = get_pg_rank(group)
            dp_size = get_pg_size(group)
            # Flatten this rank's params.
            src = (
                _flatten_dense_tensors(params_list[rank])
                if len(params_list[rank]) > 0
                else torch.empty(0, device=device, dtype=dtype)
            )
            flat_sizes = [sum(p.numel() for p in params) for params in params_list]
            if max(flat_sizes) == 0:
                return

            # Allocate per-rank receive buffers with actual sizes (no padding).
            # PyTorch's NCCL backend handles uneven sizes in all_gather via
            # grouped send/recv internally. Reuse src for local rank's slot.
            gather_list = []
            for i in range(dp_size):
                if i == rank:
                    gather_list.append(src)
                else:
                    gather_list.append(torch.empty(flat_sizes[i], device=device, dtype=dtype))

            torch.distributed.all_gather(gather_list, src, group=group)

            # Unflatten and copy gathered params for each rank.
            for idx, params in enumerate(params_list):
                if len(params) == 0 or idx == rank:
                    continue
                updated_params = _unflatten_dense_tensors(gather_list[idx], params)
                for updated_p, model_p in zip(updated_params, params):
                    model_p.data.copy_(updated_p)

        if self.pg_collection is None:
            return
        if self.dp_cp_params_list:
            _allgather_helper(self.dp_cp_params_list, self.pg_collection.dp_cp)
        if self.expt_dp_params_list:
            _allgather_helper(self.expt_dp_params_list, self.pg_collection.expt_dp)
        for optimizer in self.chained_optimizers:
            if isinstance(optimizer, FlatShardedAdamFallback):
                optimizer.allgather_params()

    @torch.no_grad()
    def broadcast_params(self):
        """All rank broadcast updated local params."""
        # Broadcast linear layer weights to all other ranks. Kept as reference test.
        if self.dp_cp_params_list is None:
            return
        for i, params in enumerate(self.dp_cp_params_list):
            src_global_rank = torch.distributed.get_global_rank(self.pg_collection.dp_cp, i)
            for p in params:
                torch.distributed.broadcast(p, src_global_rank, self.pg_collection.dp_cp)
        if self.expt_dp_params_list is None:
            return
        for i, params in enumerate(self.expt_dp_params_list):
            src_global_rank = torch.distributed.get_global_rank(self.pg_collection.expt_dp, i)
            for p in params:
                torch.distributed.broadcast(p, src_global_rank, self.pg_collection.expt_dp)

    @torch.no_grad()
    def get_grad_norm(self):
        # similar to dist opt, always aggregate globally
        grads_for_norm = []
        for optimizer in self.chained_optimizers:
            grads_for_norm += optimizer.get_main_grads_for_grad_norm()
        grad_norm = get_grad_norm_fp32(grads_for_norm, grad_stats_parallel_group=None)
        return grad_norm

    @torch.no_grad()
    def count_zeros(self):
        params = []
        for optimizer in self.chained_optimizers:
            params += optimizer.get_parameters()
        return count_zeros_fp32(
            params,
            grad_stats_parallel_group=None,
            use_decoupled_grad=self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8,
        )

    @torch.no_grad()
    def step(self):  # type: ignore[no-untyped-def]
        """step function for layer-wise optimizer."""
        update_successful, grad_norm, num_zeros_in_grad = super().step()

        # All gather updated params. If async_allgather is True, the allgather
        # is deferred to the forward pre-hooks via DDP bucket infrastructure.
        if not self.async_allgather:
            self.allgather_params()
        else:
            if self._flat_adam_fallback_optimizers:
                for model_chunk in self.model_chunks:
                    bucket_groups = (
                        model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups
                    )
                    if bucket_groups:
                        # Pre-dispatch the chain entry after optimizer step. Forward
                        # pre-hooks wait/copy it and dispatch following buckets.
                        bucket_groups[-1].start_param_sync()

        return update_successful, grad_norm, num_zeros_in_grad

    # TODO(deyuf): need to improve dist checkpointing design to properly handle this
    # fp32_from_fp16_params is list, each sub list could be empty if group is empty
    # this breaks dist checkpointing assumption since extract_sharded_base drop list structure
    # for now, we convert it to dict with index as key and convert back in load_state_dict
    def load_state_dict(self, state_dict):
        if len(self.chained_optimizers) == 1:
            wrapped_state_dict = {1: state_dict}
        else:
            wrapped_state_dict = state_dict
        for sd in wrapped_state_dict.values():
            if 'fp32_from_fp16_params' in sd and isinstance(sd['fp32_from_fp16_params'], dict):
                logger.info('[layerwise] converting fp32_from_fp16_params from dict to list')
                sd['fp32_from_fp16_params'] = [
                    v for k, v in sorted(sd['fp32_from_fp16_params'].items())
                ]
        super().load_state_dict(state_dict)

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False, **kwargs
    ):
        """
        Sharded state dict for torch_dist format checkpointing.
        For fixed DP usage only, set replica_id to 0 for all ShardedTensor.
        """
        sharded_state_dict = super().sharded_state_dict(
            model_sharded_state_dict, is_loading, **kwargs
        )

        # for fixed DP usage only
        for sh_base in nested_values(sharded_state_dict):
            if hasattr(sh_base, 'replica_id'):
                assert (
                    isinstance(sh_base.replica_id, int) or len(sh_base.replica_id) == 3
                ), f'Expected replica_id as int or (PP, TP, DP), got: {sh_base}'
                sh_base.replica_id = (
                    0 if isinstance(sh_base.replica_id, int) else (*sh_base.replica_id[:2], 0)
                )

        # later code assume list but chained optimizer fallback to non-list if there's only one
        if len(self.chained_optimizers) == 1:
            wrapped_sharded_state_dict = {1: sharded_state_dict}
        else:
            wrapped_sharded_state_dict = sharded_state_dict

        # Adjust dict rank 0 output correct global metadata into common_dict
        for sd in wrapped_sharded_state_dict.values():
            if 'optimizer' not in sd:
                continue
            # wrap empty containers into LocalNonpersistentObject so it won't be saved/loaded
            # params is already wrapped, we only need to handle fp32_from_fp16_params and state
            # more details in load_state_dict comment
            if 'fp32_from_fp16_params' in sd:
                sd['fp32_from_fp16_params'][:] = [
                    group if group else LocalNonpersistentObject(group)
                    for group in sd['fp32_from_fp16_params']
                ]
                sd['fp32_from_fp16_params'] = {
                    i: v for i, v in enumerate(sd['fp32_from_fp16_params'])
                }
            # state is a single dict and will be empty if optimizer is fully empty
            if not sd['optimizer']['state']:
                sd['optimizer']['state'] = LocalNonpersistentObject(sd['optimizer']['state'])
            # group keys(e.g. 'step') might be missing or not updated
            for i, group in enumerate(sd['optimizer']['param_groups']):
                # keep local param tensor so we only gather metadata
                local_params = group.pop('params')
                unwrapped_local_params = (
                    local_params.unwrap() if hasattr(local_params, 'unwrap') else local_params
                )
                # save whether this group is empty, so we can use non-empty rank for metadata
                group['params'] = bool(unwrapped_local_params)
                all_rank_groups = [None for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(all_rank_groups, group)
                # find first non-empty group if it exists
                nonempty_rank_group = next((g for g in all_rank_groups if g['params']), group)
                nonempty_rank_group['params'] = local_params
                sd['optimizer']['param_groups'][i] = nonempty_rank_group
        return sharded_state_dict

    def save_state_dict_to_file(self, filename: str) -> None:
        """Save the parameter state of the optimizer. For torch format only.
        Args:
            filename: The filename to save the parameter state.
        """
        torch.save(super().state_dict(), filename)

    def load_state_dict_from_file(self, filename: str) -> None:
        """Load the parameter state of the optimizer. For torch format only."""
        super().load_state_dict(torch.load(filename))
