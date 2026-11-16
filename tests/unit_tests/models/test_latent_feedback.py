# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Document-local feedback, fixed-K2 objective, and shared-weight gradients."""

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.common.chunked_output import chunked_output_loss
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.latent_feedback import (
    LatentFeedbackFusion,
    check_feedback_checkpoint_args,
    segment_feedback_inputs,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig


def configuration(**kwargs):
    options = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        use_cpu_initialization=True,
        full_bandwidth=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        normalization="RMSNorm",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        gradient_accumulation_fusion=False,
    )
    options.update(kwargs)
    return TransformerConfig(**options)


def test_batch_major_document_shift_and_gradient():
    hidden = torch.arange(12.0).reshape(3, 2, 2).requires_grad_()
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=torch.tensor([0, 2, 3, 6]))
    shifted, prefix = segment_feedback_inputs(hidden, packed)
    assert prefix.tolist() == [[True, True], [False, False], [True, False]]
    torch.testing.assert_close(shifted[1, 0], hidden[0, 0], rtol=0, atol=0)
    assert shifted[0, 1].count_nonzero() == shifted[2, 0].count_nonzero() == 0
    shifted.sum().backward()
    assert hidden.grad[1, 0].count_nonzero() == 0
    assert hidden.grad[0, 0].count_nonzero() == 2


def test_sample_boundaries_remain_barriers_with_coarse_metadata():
    hidden = torch.randn(4, 2, 8)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=torch.tensor([0, 8]))
    shifted, prefix = segment_feedback_inputs(hidden, packed)
    assert prefix[0].all()
    assert shifted[0].count_nonzero() == 0
    torch.testing.assert_close(shifted[1:], hidden[:-1], rtol=0, atol=0)


@pytest.mark.parametrize("boundaries", [[1, 6], [0, 2, 2, 6], [0, 7], [0, 5], [0]])
def test_invalid_packed_boundaries_rejected(boundaries):
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=torch.tensor(boundaries))
    with pytest.raises(ValueError, match="partition"):
        segment_feedback_inputs(torch.randn(3, 2, 4), packed)


def test_padded_boundaries_take_precedence():
    hidden = torch.randn(3, 2, 4)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 2, 4]),
        cu_seqlens_q_padded=torch.tensor([0, 3, 6]),
    )
    shifted, prefix = segment_feedback_inputs(hidden, packed)
    assert prefix[0].all() and not prefix[1:].any()
    torch.testing.assert_close(shifted[1:], hidden[:-1], rtol=0, atol=0)


def test_fusion_initialization_and_cross_pass_gradient():
    fusion = LatentFeedbackFusion(configuration())
    hidden = torch.randn(5, 2, 64, requires_grad=True)
    embeddings = torch.randn_like(hidden, requires_grad=True)
    torch.testing.assert_close(
        fusion.feedback_sequence(hidden, embeddings), embeddings, rtol=0, atol=0
    )
    with torch.no_grad():
        fusion.hidden_projection.weight.normal_(std=0.02)
    shifted, prefix = segment_feedback_inputs(hidden)
    actual = fusion.feedback_sequence(hidden, embeddings)
    expected = embeddings + fusion.hidden_projection(fusion.input_norm(shifted)) * torch.sigmoid(
        fusion.token_projection(fusion.embedding_norm(embeddings))
    )
    expected = torch.where(prefix.unsqueeze(-1), embeddings, expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual[3].sum().backward()
    assert hidden.grad[2].abs().sum() > 0
    assert hidden.grad[3:].count_nonzero() == 0
    assert all(parameter.grad is not None for parameter in fusion.parameters())


@pytest.mark.parametrize(
    "override",
    [
        dict(feedback_passes=1),
        dict(feedback_loss_reduction="last"),
        dict(feedback_fusion="glu"),
        dict(pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16),
        dict(tensor_model_parallel_size=2),
        dict(context_parallel_size=2),
        dict(fp8="e4m3"),
    ],
)
def test_unsupported_feedback_modes_rejected(override):
    with pytest.raises(ValueError):
        configuration(**override)


@pytest.mark.parametrize(
    "name,value",
    [
        ("feedback_loss_reduction", "last"),
        ("feedback_fusion", "glu"),
        ("feedback_passes", 3),
        ("feedback_pass_policy", "uniform_1_2"),
        ("feedback_jitter", 0.02),
        ("feedback_prefix_mixin", True),
        ("middle_layer_loop", True),
        ("full_bandwidth", False),
    ],
)
def test_checkpoint_semantics_cannot_change(name, value):
    args = SimpleNamespace(
        full_bandwidth=True,
        feedback_passes=2,
        feedback_fusion="residual_glu",
        feedback_loss_reduction="mean",
    )
    saved = copy.copy(args)
    check_feedback_checkpoint_args(args, saved)
    setattr(saved, name, value)
    with pytest.raises(ValueError, match=name):
        check_feedback_checkpoint_args(args, saved)


def test_chunked_head_preserves_layout_and_arbitrary_token_gradients():
    class Head(torch.nn.Linear):
        def forward(self, inputs, **kwargs):
            return super().forward(inputs), None

    torch.manual_seed(3)
    head = Head(8, 13, bias=False)
    reference_head = copy.deepcopy(head)
    hidden = torch.randn(5, 2, 8, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    labels = torch.randint(0, 13, (2, 5))
    weights = torch.rand(2, 5)

    def loss_fn(target, logits):
        return torch.nn.functional.cross_entropy(
            logits.transpose(0, 1).reshape(-1, 13), target.reshape(-1), reduction="none"
        ).reshape_as(target)

    loss = chunked_output_loss(hidden, labels, head, loss_fn, 3)
    expected = loss_fn(labels, reference_head(reference_hidden)[0])
    torch.testing.assert_close(loss, expected, rtol=0, atol=1e-6)
    (loss * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=1e-6)
    torch.testing.assert_close(head.weight.grad, reference_head.weight.grad, rtol=0, atol=1e-6)


@pytest.fixture(scope="module")
def distributed_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native GPT integration tests")
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory() as directory:
        torch.distributed.init_process_group(
            "nccl", init_method=f"file://{Path(directory) / 'rendezvous'}", rank=0, world_size=1
        )
        parallel_state.initialize_model_parallel()
        model_parallel_cuda_manual_seed(1234)
        with patch("megatron.training.global_vars._GLOBAL_ARGS", SimpleNamespace(diag_interval=0)):
            yield
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def build_model(**kwargs):
    config = configuration(
        bf16=True, params_dtype=torch.bfloat16, attention_backend=AttnBackend.unfused, **kwargs
    )
    return (
        GPTModel(
            config,
            get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=32,
            position_embedding_type="rope",
            share_embeddings_and_output_weights=False,
        )
        .cuda()
        .bfloat16()
    )


@pytest.mark.parametrize("chunk_size", [0, 5])
def test_native_k2_outputs_and_shared_gradients_match_explicit_reference(
    distributed_gpu, chunk_size
):
    model = build_model(output_loss_chunk_size=chunk_size).train()
    with torch.no_grad():
        model.latent_feedback.hidden_projection.weight.normal_(std=0.02)
    reference = build_model(output_loss_chunk_size=chunk_size).train()
    reference.load_state_dict(model.state_dict())
    tokens = torch.arange(16, device="cuda").reshape(2, 8)
    positions = torch.arange(8, device="cuda").expand_as(tokens)
    mask = torch.ones(1, 1, 8, 8, device="cuda", dtype=torch.bool).triu(1)
    actual = model(tokens, positions, mask, labels=tokens)
    embeddings = reference.embedding(input_ids=tokens, position_ids=positions)
    first, hidden = reference._forward_single(
        tokens, positions, mask, decoder_input=embeddings, labels=tokens, return_hidden_states=True
    )
    shifted, prefix = segment_feedback_inputs(hidden)
    fused = reference.latent_feedback(shifted, embeddings, prefix)
    second = reference._forward_single(tokens, positions, mask, decoder_input=fused, labels=tokens)
    expected = (first + second) / 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)
    torch.testing.assert_close(model.feedback_losses[0], first, rtol=0, atol=1e-4)
    torch.testing.assert_close(model.feedback_losses[1], second, rtol=0, atol=1e-4)
    actual.float().mean().backward()
    expected.float().mean().backward()
    for (name, param), (other_name, other) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert name == other_name and param.grad is not None and other.grad is not None
        torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=1e-4, msg=name)


def test_one_pass_matches_unmodified_path_and_has_no_future_leakage(distributed_gpu):
    model = build_model().eval()
    baseline = build_model(full_bandwidth=False).eval()
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("latent_feedback.")}
    baseline.load_state_dict(state)
    tokens = torch.arange(16, device="cuda").reshape(2, 8)
    positions = torch.arange(8, device="cuda").expand_as(tokens)
    mask = torch.ones(1, 1, 8, 8, device="cuda", dtype=torch.bool).triu(1)
    with torch.no_grad():
        torch.testing.assert_close(
            model(tokens, positions, mask, feedback_passes=1),
            baseline(tokens, positions, mask),
            rtol=0,
            atol=0,
        )
        model.latent_feedback.hidden_projection.weight.normal_(std=0.02)
        first = model(tokens, positions, mask, feedback_passes=2)
        changed = tokens.clone()
        changed[:, 5:] += 32
        second = model(changed, positions, mask, feedback_passes=2)
    torch.testing.assert_close(first[:, :5], second[:, :5], rtol=0, atol=0)


def test_chunked_shared_head_with_fused_gradients_and_ddp(distributed_gpu):
    model = build_model(
        output_loss_chunk_size=5,
        gradient_accumulation_fusion=True,
        recompute_granularity='full',
        recompute_method='block',
        recompute_num_layers=1,
    ).train()
    with torch.no_grad():
        model.latent_feedback.hidden_projection.weight.normal_(std=0.02)
    wrapped = DistributedDataParallel(
        config=model.config,
        ddp_config=DistributedDataParallelConfig(
            overlap_grad_reduce=True, grad_reduce_in_fp32=True
        ),
        module=model,
    )
    wrapped.zero_grad_buffer()
    tokens = torch.arange(16, device='cuda').reshape(2, 8)
    positions = torch.arange(8, device='cuda').expand_as(tokens)
    mask = torch.ones(1, 1, 8, 8, device='cuda', dtype=torch.bool).triu(1)
    wrapped(tokens, positions, mask, labels=tokens).float().mean().backward()
    wrapped.finish_grad_sync()
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter.main_grad).all(), name
        assert parameter.main_grad.abs().sum() > 0, name
