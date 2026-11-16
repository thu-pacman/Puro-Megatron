# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Causal, document-local feedback for the fixed two-pass GPT objective."""

from copy import copy
from typing import Optional

import torch
from torch import Tensor, nn

from megatron.core.packed_seq_params import PackedSeqParams


class FeedbackRMSNorm(nn.Module):
    """RMS normalization in FP32 with the residual stream's output dtype."""

    def __init__(self, hidden_size, eps, *, device, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, inputs: Tensor) -> Tensor:
        """Normalize the last dimension and apply the learned scale."""
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(inputs.dtype) * self.weight


def segment_feedback_inputs(
    hidden: Tensor, packed_seq_params: Optional[PackedSeqParams] = None
) -> tuple[Tensor, Tensor]:
    """Shift [S, B, H] by one token within each sample and packed document.

    Packed THD boundaries index the batch-major token order. Sample boundaries
    remain barriers even when the supplied document metadata spans a whole batch.
    The returned mask selects document starts, which retain the original input.
    """
    sequence_length, batch_size, hidden_size = hidden.shape
    token_count = sequence_length * batch_size
    flat_hidden = hidden.transpose(0, 1).reshape(token_count, hidden_size)
    boundaries = torch.arange(
        0, token_count + 1, sequence_length, device=hidden.device, dtype=torch.long
    )
    if packed_seq_params is not None:
        if packed_seq_params.qkv_format != "thd":
            raise ValueError("Latent feedback requires THD packed sequence metadata")
        packed_boundaries = packed_seq_params.cu_seqlens_q_padded
        if packed_boundaries is None:
            packed_boundaries = packed_seq_params.cu_seqlens_q
        if packed_boundaries is None:
            raise ValueError("Packed latent feedback requires cu_seqlens_q")
        packed_boundaries = packed_boundaries.to(device=hidden.device, dtype=torch.long)
        if (
            packed_boundaries.ndim != 1
            or packed_boundaries.numel() < 2
            or packed_boundaries[0].item() != 0
            or packed_boundaries[-1].item() != token_count
            or not torch.all(packed_boundaries[1:] > packed_boundaries[:-1]).item()
        ):
            raise ValueError("Packed boundaries must strictly partition all input tokens")
        boundaries = torch.unique(torch.cat((boundaries, packed_boundaries)), sorted=True)
    positions = torch.arange(token_count, device=hidden.device)
    segments = torch.bucketize(positions, boundaries[1:], right=True)
    offsets = positions - boundaries[segments]
    plain_prefix = offsets == 0
    shifted = torch.cat((torch.zeros_like(flat_hidden[:1]), flat_hidden[:-1]), dim=0)
    shifted = shifted.masked_fill(plain_prefix.unsqueeze(-1), 0)
    shifted = shifted.reshape(batch_size, sequence_length, hidden_size).transpose(0, 1)
    plain_prefix = plain_prefix.reshape(batch_size, sequence_length).transpose(0, 1)
    return shifted, plain_prefix


class LatentFeedbackFusion(nn.Module):
    """Residual-GLU with an identity-preserving, zero-initialized value projection."""

    def __init__(self, config):
        super().__init__()
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        kwargs = {"device": device, "dtype": config.params_dtype}
        # Construction/initialization order matches the original training implementation.
        self.hidden_projection = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False, **kwargs
        )
        self.token_projection = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False, **kwargs
        )
        self.embedding_norm = FeedbackRMSNorm(
            config.hidden_size, config.layernorm_epsilon, **kwargs
        )
        self.input_norm = FeedbackRMSNorm(config.hidden_size, config.layernorm_epsilon, **kwargs)
        if config.perform_initialization:
            with torch.no_grad():
                self.hidden_projection.weight.zero_()
            config.init_method(self.token_projection.weight)

    def forward(
        self, previous_hidden: Tensor, embeddings: Tensor, plain_prefix: Optional[Tensor] = None
    ) -> Tensor:
        """Add gated feedback, preserving the original input at document starts."""
        value = self.hidden_projection(self.input_norm(previous_hidden))
        gate = torch.sigmoid(self.token_projection(self.embedding_norm(embeddings)))
        fused = embeddings + value * gate
        if plain_prefix is not None:
            fused = torch.where(plain_prefix.unsqueeze(-1), embeddings, fused)
        return fused

    def feedback_sequence(self, hidden, embeddings, packed_seq_params=None):
        """Fuse the preceding token's representation without detaching its gradient."""
        previous_hidden, plain_prefix = segment_feedback_inputs(hidden, packed_seq_params)
        return self(previous_hidden, embeddings, plain_prefix)


def feedback_checkpoint_args(args):
    """Serialize fixed feedback semantics understood by the original training fork.

    These are checkpoint compatibility fields, not additional training modes.
    Copy the namespace so saving cannot modify live training arguments.
    """
    if not getattr(args, "full_bandwidth", False):
        return args
    saved = copy(args)
    for name, value in {
        "feedback_loss_weight": 1.0,
        "feedback_jitter": 0.0,
        "feedback_prefix_mixin": False,
        "feedback_pass_policy": "fixed",
        "feedback_input_scale": None,
        "middle_layer_feedback": False,
        "middle_layer_loop": False,
    }.items():
        if getattr(args, name, value) != value:
            raise ValueError(f"Unsupported feedback checkpoint setting: {name}")
        setattr(saved, name, value)
    return saved


def check_feedback_checkpoint_args(args, checkpoint_args):
    """Reject resumes that reinterpret feedback architecture or objective semantics."""
    active = any(getattr(ns, "full_bandwidth", False) for ns in (args, checkpoint_args))
    if not active:
        return
    contract = {
        "full_bandwidth": False,
        "feedback_passes": 2,
        "feedback_fusion": "glu",
        "feedback_loss_reduction": "first_plus_feedback_mean",
        "feedback_pass_policy": "fixed",
        "feedback_jitter": 0.0,
        "feedback_prefix_mixin": False,
        "feedback_loss_weight": 1.0,
        "feedback_input_scale": None,
        "middle_layer_feedback": False,
        "middle_layer_loop": False,
    }
    release_defaults = dict(
        contract, feedback_fusion="residual_glu", feedback_loss_reduction="mean"
    )
    for name, default in contract.items():
        actual = getattr(args, name, release_defaults[name])
        saved = getattr(checkpoint_args, name, default)
        if actual != saved:
            raise ValueError(
                f"Feedback checkpoint mismatch for {name}: saved={saved}, requested={actual}"
            )
