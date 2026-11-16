# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Bound the token dimension of training logits without changing per-token loss."""

import torch
from torch.utils.checkpoint import checkpoint


def chunked_output_loss(
    hidden_states,
    labels,
    output_layer,
    compute_loss,
    chunk_size,
    *,
    weight=None,
    runtime_gather_output=None,
):
    """Project [S,B,H] in token chunks, returning the original [B,S] losses.

    Each chunk uses a non-reentrant activation checkpoint so its vocabulary-sized
    intermediates are released before the next chunk. Backward reconstructs only
    that chunk, preserving arbitrary upstream token weights and the output
    layer's existing fused main_grad accumulation / DDP bookkeeping.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    sequence, batch, hidden = hidden_states.shape
    if labels.shape != (batch, sequence):
        raise ValueError("labels must match the [batch, sequence] hidden-state layout")
    flat_hidden = hidden_states.reshape(sequence * batch, 1, hidden)
    flat_labels = labels.transpose(0, 1).reshape(1, sequence * batch)

    def project_and_loss(chunk_hidden, chunk_labels):
        logits, _ = output_layer(
            chunk_hidden, weight=weight, runtime_gather_output=runtime_gather_output
        )
        return compute_loss(chunk_labels, logits)

    losses = []
    for start in range(0, sequence * batch, chunk_size):
        inputs = (
            flat_hidden[start : start + chunk_size],
            flat_labels[:, start : start + chunk_size],
        )
        if torch.is_grad_enabled():
            loss = checkpoint(
                project_and_loss, *inputs, use_reentrant=False, preserve_rng_state=False
            )
        else:
            loss = project_and_loss(*inputs)
        losses.append(loss)
    return torch.cat(losses, dim=1).reshape(sequence, batch).transpose(0, 1).contiguous()
