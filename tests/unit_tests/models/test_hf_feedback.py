# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Exercise the published Transformers adapter without a GPU or Megatron runtime."""

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

from examples.puro.hf_feedback.configuration_puro_feedback import PuroFeedbackConfig
from examples.puro.hf_feedback.modeling_puro_feedback import (
    PuroFeedbackCache,
    PuroFeedbackForCausalLM,
)
from examples.puro.hf_feedback.package_checkpoint import package, sha256


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        config = PuroFeedbackConfig(
            vocab_size=73,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=72,
        )
        config._attn_implementation = "sdpa"
        result = PuroFeedbackForCausalLM(config).eval()
        # Nonzero feedback is essential: a zero projection hides a shift/cache bug.
        with torch.no_grad():
            result.latent_feedback.hidden_projection.weight.normal_(std=0.15)
        yield result


@pytest.mark.parametrize("passes", [1, 2])
@torch.inference_mode()
def test_chunked_cache_equals_full_prefix_with_left_padding(model, passes):
    tokens = torch.tensor([[0, 0, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]])
    mask = (tokens != 0).long()
    full = model(tokens, attention_mask=mask, feedback_passes=passes, use_cache=False).logits
    cache = None
    for start, end in [(0, 4), (4, 6), (6, 7)]:
        output = model(
            tokens[:, start:end],
            attention_mask=mask[:, :end],
            past_key_values=cache,
            feedback_passes=passes,
            use_cache=True,
        )
        cache = output.past_key_values
        assert isinstance(cache, PuroFeedbackCache)
        assert cache.get_seq_length() == end
        if passes == 2:
            assert cache.first_pass.get_seq_length() == end
        # Padded query outputs are undefined; compare only real tokens.
        valid = mask[:, start:end].bool()
        torch.testing.assert_close(
            output.logits[valid], full[:, start:end][valid], rtol=2e-5, atol=2e-6
        )


@pytest.mark.parametrize("passes", [1, 2])
@torch.inference_mode()
def test_generate_cache_matches_no_cache_and_batch(model, passes):
    tokens = torch.tensor([[0, 0, 2, 3], [4, 5, 6, 7]])
    mask = (tokens != 0).long()
    kwargs = dict(
        max_new_tokens=5,
        do_sample=False,
        eos_token_id=None,
        feedback_passes=passes,
        attention_mask=mask,
    )
    cached = model.generate(tokens, use_cache=True, **kwargs)
    full = model.generate(tokens, use_cache=False, **kwargs)
    assert torch.equal(cached, full)
    for row in range(2):
        prompt = tokens[row, mask[row].bool()][None]
        single = model.generate(
            prompt, max_new_tokens=5, do_sample=False, eos_token_id=None, feedback_passes=passes
        )
        assert torch.equal(single[:, -5:], cached[row : row + 1, -5:])


@torch.inference_mode()
def test_two_pass_matches_explicit_reference_and_is_causal(model):
    tokens = torch.tensor([[2, 3, 4, 5, 6]])
    embeddings = model.model.embed_tokens(tokens)
    first = model.model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
    previous = torch.cat((torch.zeros_like(first[:, :1]), first[:, :-1]), dim=1)
    f = model.latent_feedback

    # Independent spelling of the trained FP32-normalization / Residual-GLU formula.
    def norm(x, weight):
        return (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + model.config.rms_norm_eps)
        ).to(x.dtype) * weight

    value = torch.nn.functional.linear(
        norm(previous, f.input_norm.weight), f.hidden_projection.weight
    )
    gate = torch.sigmoid(
        torch.nn.functional.linear(
            norm(embeddings, f.embedding_norm.weight), f.token_projection.weight
        )
    )
    fused = embeddings + value * gate
    fused[:, :1] = embeddings[:, :1]
    expected = model.lm_head(model.model(inputs_embeds=fused, use_cache=False).last_hidden_state)
    actual = model(tokens, feedback_passes=2, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    changed = tokens.clone()
    changed[:, -1] += 1
    torch.testing.assert_close(
        actual[:, :-1], model(changed, use_cache=False).logits[:, :-1], rtol=0, atol=0
    )
    with torch.no_grad():
        f.hidden_projection.weight.zero_()
    torch.testing.assert_close(
        model(tokens, feedback_passes=1, use_cache=False).logits,
        model(tokens, feedback_passes=2, use_cache=False).logits,
        rtol=0,
        atol=0,
    )


def test_label_loss_is_mean_with_connected_gradients(model):
    tokens = torch.tensor([[2, 3, 4, 5]])
    output = model(tokens, labels=tokens, use_cache=False)
    k1 = model(tokens, labels=tokens, use_cache=False, feedback_passes=1)
    shifted_labels = tokens[:, 1:].reshape(-1)
    k2 = torch.nn.functional.cross_entropy(output.logits[:, :-1].reshape(-1, 73), shifted_labels)
    torch.testing.assert_close(output.k1_loss, k1.loss)
    torch.testing.assert_close(output.k2_loss, k2)
    torch.testing.assert_close(output.loss, (k1.loss + k2) / 2)
    output.loss.backward()
    assert model.latent_feedback.hidden_projection.weight.grad.abs().sum() > 0
    assert model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


@torch.inference_mode()
def test_auto_model_round_trip_requires_custom_architecture(model, tmp_path):
    model.save_pretrained(tmp_path, safe_serialization=True)
    config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=True)
    assert config.model_type == "puro_feedback"
    assert config.feedback_passes == 2
    assert json.loads((tmp_path / 'config.json').read_text())['auto_map']['AutoModelForCausalLM']
    restored, info = AutoModelForCausalLM.from_pretrained(
        tmp_path, trust_remote_code=True, attn_implementation="sdpa", output_loading_info=True
    )
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys']
    tokens = torch.tensor([[2, 3, 4]])
    torch.testing.assert_close(
        model(tokens, use_cache=False).logits,
        restored(tokens, use_cache=False).logits,
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError):
        AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=False)


@torch.inference_mode()
def test_beam_and_sampling_generation(model):
    tokens = torch.tensor([[2, 3, 4]])
    options = dict(max_new_tokens=4, do_sample=False, num_beams=3, eos_token_id=None)
    assert torch.equal(
        model.generate(tokens, use_cache=True, **options),
        model.generate(tokens, use_cache=False, **options),
    )
    torch.manual_seed(29)
    sample = model.generate(tokens, max_new_tokens=4, do_sample=True, top_p=0.9, temperature=0.7)
    assert sample.shape[0] == 1 and sample.shape[1] > tokens.shape[1]


@pytest.mark.parametrize("eos", [70, 72])
@torch.inference_mode()
def test_both_eos_ids_stop_generation(model, eos):
    from transformers import LogitsProcessor, LogitsProcessorList

    class ForceEOS(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores.fill_(-torch.inf)
            scores[:, eos] = 0
            return scores

    tokens = torch.tensor([[2, 3, 4]])
    generated = model.generate(
        tokens,
        max_new_tokens=5,
        eos_token_id=[70, 72],
        logits_processor=LogitsProcessorList([ForceEOS()]),
    )
    assert generated.shape[1] == 4 and generated[0, -1] == eos


@torch.inference_mode()
def test_cache_rejects_switching_passes_or_legacy_conversion(model):
    cache = model(torch.tensor([[2, 3]]), use_cache=True).past_key_values
    with pytest.raises(ValueError, match="change K1/K2"):
        model(torch.tensor([[4]]), past_key_values=cache, feedback_passes=1)
    with pytest.raises(ValueError, match="legacy"):
        cache.to_legacy_cache()
    with pytest.raises(ValueError, match="rollback"):
        cache.crop(1)
    cache.crop(0)
    assert cache.get_seq_length() == cache.first_pass.get_seq_length() == 0


def test_configuration_rejects_incorrect_feedback_or_rope():
    for kwargs in [
        dict(feedback_passes=3),
        dict(feedback_shift=0),
        dict(rope_theta=1e6, rope_parameters=dict(rope_theta=10000)),
    ]:
        with pytest.raises(ValueError):
            PuroFeedbackConfig(**kwargs)


@pytest.fixture
def legacy_export(model, tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    state = model.state_dict()
    base = {name: value for name, value in state.items() if not name.startswith("latent_feedback.")}
    feedback = {
        name.removeprefix("latent_feedback."): value
        for name, value in state.items()
        if name.startswith("latent_feedback.")
    }
    save_file(base, str(source / "model.safetensors"), metadata={"format": "pt"})
    save_file(feedback, str(source / "feedback.safetensors"))
    config = model.config.to_dict()
    config['model_type'] = 'qwen3'
    (source / 'config.json').write_text(json.dumps(config))
    for name in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json']:
        (source / name).write_text('{}')
    report = dict(
        iteration=7,
        parameters=sum(value.numel() for value in state.values()),
        weights_sha256={
            name: sha256(source / name) for name in ['model.safetensors', 'feedback.safetensors']
        },
    )
    (source / 'export.json').write_text(json.dumps(report))
    return source


@torch.inference_mode()
def test_manual_shard_package_round_trip_preserves_all_weights(model, legacy_export, tmp_path):
    output = tmp_path / 'packaged'
    manifest = package(legacy_export, output, expected_step=7)
    assert manifest['files_sha256']['model-00001-of-00002.safetensors'] == sha256(
        legacy_export / 'model.safetensors'
    )
    loaded, info = AutoModelForCausalLM.from_pretrained(
        output, trust_remote_code=True, torch_dtype=torch.float32, output_loading_info=True
    )
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys']
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
    tokens = torch.tensor([[2, 3, 4]])
    torch.testing.assert_close(
        model(tokens, use_cache=False).logits,
        loaded(tokens, use_cache=False).logits,
        rtol=0,
        atol=0,
    )


def test_packaging_rejects_wrong_step_or_corrupt_source_hash(legacy_export, tmp_path):
    with pytest.raises(ValueError, match='Expected step'):
        package(legacy_export, tmp_path / 'wrong_step', expected_step=8)
    report = json.loads((legacy_export / 'export.json').read_text())
    report['weights_sha256']['feedback.safetensors'] = '0' * 64
    (legacy_export / 'export.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='hash mismatch'):
        package(legacy_export, tmp_path / 'bad_hash', expected_step=7)
    assert not (tmp_path / 'bad_hash').exists()
