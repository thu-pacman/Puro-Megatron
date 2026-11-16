# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Strict Qwen3 initialization and grouped QKV conversion on CPU."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import SafetensorError
from safetensors.torch import save_file

from megatron.core.models.gpt.hf_checkpoint import load_qwen3_safetensors


def register(root, name, shape):
    parts = name.split('.')
    for part in parts[:-1]:
        if not hasattr(root, part):
            root.add_module(part, torch.nn.Module())
        root = getattr(root, part)
    root.register_parameter(parts[-1], torch.nn.Parameter(torch.empty(shape)))


def model(indices, pp, pre, post):
    result = torch.nn.Module()
    result.config = SimpleNamespace(
        num_layers=4,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
        layernorm_epsilon=1e-6,
        add_bias_linear=False,
        qk_layernorm=True,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp,
    )
    result.vocab_size = 16
    result.share_embeddings_and_output_weights = False
    if pre:
        register(result, 'embedding.word_embeddings.weight', (16, 8))
    if post:
        register(result, 'output_layer.weight', (16, 8))
    result.decoder = torch.nn.Module()
    result.decoder.layers = torch.nn.ModuleList()
    if post:
        register(result, 'decoder.final_layernorm.weight', (8,))
    for index in indices:
        layer = torch.nn.Module()
        layer.layer_number = index + 1
        for name, shape in {
            'self_attention.linear_qkv.weight': (16, 8),
            'self_attention.linear_qkv.layer_norm_weight': (8,),
            'self_attention.linear_proj.weight': (8, 8),
            'self_attention.q_layernorm.weight': (2,),
            'self_attention.k_layernorm.weight': (2,),
            'mlp.linear_fc1.weight': (32, 8),
            'mlp.linear_fc1.layer_norm_weight': (8,),
            'mlp.linear_fc2.weight': (8, 16),
        }.items():
            register(layer, name, shape)
        result.decoder.layers.append(layer)
    return result


@pytest.fixture
def checkpoint(tmp_path):
    metadata = dict(
        model_type='qwen3',
        num_hidden_layers=4,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        vocab_size=16,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=False,
    )
    (tmp_path / 'config.json').write_text(json.dumps(metadata))
    tensors = {
        'model.embed_tokens.weight': torch.randn(16, 8),
        'lm_head.weight': torch.randn(16, 8),
        'model.norm.weight': torch.randn(8),
    }
    for i in range(4):
        for suffix, shape in {
            'self_attn.q_proj.weight': (8, 8),
            'self_attn.k_proj.weight': (4, 8),
            'self_attn.v_proj.weight': (4, 8),
            'self_attn.o_proj.weight': (8, 8),
            'self_attn.q_norm.weight': (2,),
            'self_attn.k_norm.weight': (2,),
            'mlp.gate_proj.weight': (16, 8),
            'mlp.up_proj.weight': (16, 8),
            'mlp.down_proj.weight': (8, 16),
            'input_layernorm.weight': (8,),
            'post_attention_layernorm.weight': (8,),
        }.items():
            tensors[f'model.layers.{i}.{suffix}'] = torch.randn(shape)
    save_file(tensors, str(tmp_path / 'model.safetensors'))
    return tmp_path, tensors


def test_grouped_qkv_and_gate_up_layout(checkpoint):
    path, tensors = checkpoint
    target = model(range(4), 1, True, True)
    register(target, 'latent_feedback.hidden_projection.weight', (8, 8))
    target.latent_feedback.hidden_projection.weight.data.zero_()
    result = load_qwen3_safetensors(target, path, rotary_base=10000)
    assert result['new_feedback_tensors'] == ['latent_feedback.hidden_projection.weight']
    assert not target.latent_feedback.hidden_projection.weight.count_nonzero()
    for i, layer in enumerate(target.decoder.layers):
        prefix = f'model.layers.{i}.'
        q, k, v = [tensors[prefix + f'self_attn.{name}_proj.weight'] for name in ('q', 'k', 'v')]
        expected = torch.cat((q[:4], k[:2], v[:2], q[4:], k[2:], v[2:]))
        torch.testing.assert_close(layer.self_attention.linear_qkv.weight, expected, rtol=0, atol=0)
        expected_fc1 = torch.cat(
            (tensors[prefix + 'mlp.gate_proj.weight'], tensors[prefix + 'mlp.up_proj.weight'])
        )
        torch.testing.assert_close(layer.mlp.linear_fc1.weight, expected_fc1, rtol=0, atol=0)
    torch.testing.assert_close(
        target.embedding.word_embeddings.weight,
        tensors['model.embed_tokens.weight'],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize('failure', ['metadata', 'extra', 'missing', 'shape'])
def test_invalid_hf_checkpoint_is_rejected(checkpoint, failure):
    path, tensors = checkpoint
    if failure == 'metadata':
        metadata = json.loads((path / 'config.json').read_text())
        metadata['num_hidden_layers'] = 3
        (path / 'config.json').write_text(json.dumps(metadata))
    elif failure == 'extra':
        tensors['unrecognized.weight'] = torch.ones(1)
    elif failure == 'missing':
        del tensors['lm_head.weight']
    else:
        tensors['model.norm.weight'] = torch.ones(7)
    save_file(tensors, str(path / 'model.safetensors'))
    with pytest.raises((ValueError, SafetensorError)):
        load_qwen3_safetensors(model(range(4), 1, True, True), path, rotary_base=10000)
