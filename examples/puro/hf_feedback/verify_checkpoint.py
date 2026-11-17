# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Verify a standalone HF package, its dual cache, and optional legacy export."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--legacy-export", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--new-tokens", default=8, type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    manifest = json.loads((args.model / "checkpoint_manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        with (args.model / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, name
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding_side="left", local_files_only=True
    )
    model, info = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        output_loading_info=True,
    )
    assert not any(
        info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ), info
    model.to(args.device).eval()
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == manifest["parameters"]
    assert model.config.rope_theta == 10000
    assert model.generation_config.eos_token_id == [151643, 151645]
    prompts = ["The capital of France is", "def fibonacci(n):\n"]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(args.device)
    reference = feedback = None
    if args.legacy_export is not None:
        config = AutoConfig.from_pretrained(args.legacy_export, local_files_only=True)
        config.rope_theta = model.config.rope_theta
        reference = (
            AutoModelForCausalLM.from_pretrained(
                args.legacy_export,
                config=config,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .to(args.device)
            .eval()
        )
        feedback = {
            name: value.to(args.device)
            for name, value in load_file(str(args.legacy_export / 'feedback.safetensors')).items()
        }
        for name, value in model.latent_feedback.state_dict().items():
            assert torch.equal(value, feedback[name]), name

    def legacy_fuse(previous, embeddings, valid):
        def norm(value, weight):
            return (
                value.float()
                * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + config.rms_norm_eps)
            ).to(value.dtype) * weight

        value = torch.nn.functional.linear(
            norm(previous, feedback['input_norm.weight']), feedback['hidden_projection.weight']
        )
        gate = torch.sigmoid(
            torch.nn.functional.linear(
                norm(embeddings, feedback['embedding_norm.weight']),
                feedback['token_projection.weight'],
            )
        )
        return torch.where(valid[..., None], embeddings + value * gate, embeddings)

    def legacy_logits(passes):
        embeddings = reference.model.embed_tokens(inputs.input_ids)
        positions = (inputs.attention_mask.cumsum(-1) - 1).clamp_min(0)
        kwargs = dict(attention_mask=inputs.attention_mask, position_ids=positions, use_cache=False)
        first = reference.model(inputs_embeds=embeddings, **kwargs).last_hidden_state
        hidden = first
        if passes == 2:
            previous = torch.cat((torch.zeros_like(first[:, :1]), first[:, :-1]), dim=1)

            valid = torch.zeros_like(inputs.attention_mask, dtype=torch.bool)
            valid[:, 1:] = (
                inputs.attention_mask[:, 1:].bool() & inputs.attention_mask[:, :-1].bool()
            )
            fused = legacy_fuse(previous, embeddings, valid)
            hidden = reference.model(inputs_embeds=fused, **kwargs).last_hidden_state
        return reference.lm_head(hidden)

    result = dict(
        passed=False,
        device=args.device,
        dtype="bfloat16",
        parameters=parameters,
        packaged_hashes_verified=True,
        loading_keys_exact=True,
        passes=[],
        generations=[],
    )
    valid = inputs.attention_mask.bool()
    final_logits = {}
    for passes in (1, 2):
        full = model(**inputs, use_cache=False, feedback_passes=passes).logits
        final_logits[passes] = full[:, -1].float()
        check = dict(feedback_passes=passes)
        if reference is not None:
            expected = legacy_logits(passes)
            torch.testing.assert_close(full[valid], expected[valid], rtol=0, atol=0)
            check['legacy_export_logits_max_absolute_error'] = float(
                (full[valid] - expected[valid]).abs().max()
            )
        cache = None
        legacy_first_cache, legacy_second_cache = DynamicCache(), DynamicCache()
        legacy_previous_hidden = legacy_previous_valid = None
        cached_reference_error = 0.0
        width = inputs.input_ids.shape[1]
        for start, end in [(0, width - 2), (width - 2, width - 1), (width - 1, width)]:
            output = model(
                inputs.input_ids[:, start:end],
                attention_mask=inputs.attention_mask[:, :end],
                past_key_values=cache,
                use_cache=True,
                feedback_passes=passes,
            )
            cache = output.past_key_values
            if reference is not None:
                # Ordinary Qwen3 caches provide a reference independently of the
                # new dual-cache class, including its BF16 chunking numerics.
                embedding = reference.model.embed_tokens(inputs.input_ids[:, start:end])
                mask = inputs.attention_mask[:, :end]
                positions = (mask.cumsum(-1) - 1).clamp_min(0)[:, start:end]
                first = reference.model(
                    inputs_embeds=embedding,
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=legacy_first_cache,
                    use_cache=True,
                ).last_hidden_state
                hidden = first
                if passes == 2:
                    previous = (
                        torch.zeros_like(first[:, :1]) if start == 0 else legacy_previous_hidden
                    )
                    previous_valid = (
                        torch.zeros_like(mask[:, :1], dtype=torch.bool)
                        if start == 0
                        else legacy_previous_valid
                    )
                    shifted = torch.cat((previous, first[:, :-1]), dim=1)
                    current_valid = mask[:, start:end].bool()
                    valid_feedback = current_valid & torch.cat(
                        (previous_valid, current_valid[:, :-1]), dim=1
                    )
                    fused = legacy_fuse(shifted, embedding, valid_feedback)
                    hidden = reference.model(
                        inputs_embeds=fused,
                        attention_mask=mask,
                        position_ids=positions,
                        past_key_values=legacy_second_cache,
                        use_cache=True,
                    ).last_hidden_state
                    legacy_previous_hidden = first[:, -1:]
                    legacy_previous_valid = current_valid[:, -1:]
                expected_cached = reference.lm_head(hidden)
                torch.testing.assert_close(output.logits, expected_cached, rtol=0, atol=0)
                cached_reference_error = max(
                    cached_reference_error, float((output.logits - expected_cached).abs().max())
                )
        left, right = output.logits[:, -1].float(), full[:, -1].float()
        kl = (right.log_softmax(-1).exp() * (right.log_softmax(-1) - left.log_softmax(-1))).sum(-1)
        check['cached_last_logit_max_absolute_error'] = float((left - right).abs().max())
        check['cached_last_logit_max_kl'] = float(kl.max())
        if reference is not None:
            check['legacy_export_cached_logits_max_absolute_error'] = cached_reference_error
        assert check['cached_last_logit_max_kl'] < 0.003, check
        check['cache_lengths'] = dict(
            final=cache.get_seq_length(),
            first=cache.first_pass.get_seq_length() if passes == 2 else None,
        )
        result['passes'].append(check)
        options = dict(max_new_tokens=args.new_tokens, do_sample=False, feedback_passes=passes)
        cached = model.generate(**inputs, use_cache=True, **options)
        uncached = model.generate(**inputs, use_cache=False, **options)
        assert torch.equal(
            cached, uncached
        ), "Cached greedy generation differs from full-prefix generation"
        result['generations'].append(
            dict(
                feedback_passes=passes,
                cached_equals_full_prefix=True,
                generated_token_ids=cached[:, width:].tolist(),
                completions=tokenizer.batch_decode(cached[:, width:], skip_special_tokens=True),
            )
        )
    result['k1_vs_k2_last_logit_max_absolute_difference'] = float(
        (final_logits[1] - final_logits[2]).abs().max()
    )
    assert (
        result['k1_vs_k2_last_logit_max_absolute_difference'] > 0
    ), 'Expected trained feedback to affect logits'
    result['passed'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
