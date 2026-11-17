# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Test a local/HF Puro feedback checkpoint with standard Transformers generate."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt", action="append", help="Raw continuation prompt; repeat for a padded batch"
    )
    parser.add_argument("--prompt-file", type=Path, help="Read one raw prompt from a UTF-8 file")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--feedback-passes", choices=["1", "2", "both"], default="2")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32", "float16"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-cache", action="store_true", help="Full-prefix reference decoding")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    if args.sample and (args.temperature <= 0 or not 0 < args.top_p <= 1):
        parser.error("Sampling requires temperature > 0 and 0 < top-p <= 1")
    if not args.interactive and not args.prompt and not args.prompt_file:
        parser.error("Provide --prompt, --prompt-file, or --interactive")
    torch.set_num_threads(args.cpu_threads)
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, info = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=getattr(torch, args.dtype),
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise ValueError(f"Checkpoint loading {key}: {info[key]}")
    model.to(device).eval()
    modes = [1, 2] if args.feedback_passes == "both" else [int(args.feedback_passes)]

    @torch.inference_mode()
    def complete(prompts):
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        width = inputs.input_ids.shape[1]
        if width + args.max_new_tokens > model.config.max_position_embeddings:
            raise ValueError(
                "Prompt plus requested completion exceeds the trained 4096-token context"
            )
        options = dict(
            max_new_tokens=args.max_new_tokens, do_sample=args.sample, use_cache=not args.no_cache
        )
        if args.sample:
            options.update(temperature=args.temperature, top_p=args.top_p)
        for passes in modes:
            torch.manual_seed(args.seed)
            output = model.generate(**inputs, feedback_passes=passes, **options)
            for prompt, tokens in zip(prompts, output[:, width:]):
                print(
                    json.dumps(
                        dict(
                            feedback_passes=passes,
                            prompt=prompt,
                            completion=tokenizer.decode(tokens, skip_special_tokens=True),
                            generated_token_ids=tokens.tolist(),
                        ),
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    prompts = list(args.prompt or [])
    if args.prompt_file:
        prompts.append(args.prompt_file.read_text())
    if prompts:
        complete(prompts)
    if args.interactive:
        while True:
            try:
                prompt = input("prompt (/quit to exit)> ")
            except EOFError:
                break
            if prompt.strip() == "/quit":
                break
            if prompt.strip():
                complete([prompt])


if __name__ == "__main__":
    main()
