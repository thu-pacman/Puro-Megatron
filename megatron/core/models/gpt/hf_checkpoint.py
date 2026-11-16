# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

import json
from pathlib import Path

import torch


def load_qwen3_safetensors(model, directory, *, rotary_base):
    """Load Qwen3 weights strictly, allowing only newly initialized feedback tensors."""
    from safetensors import safe_open

    directory = Path(directory)
    metadata = json.loads((directory / "config.json").read_text())
    config = model.config
    if metadata.get("model_type") != "qwen3":
        raise ValueError("Only dense Qwen3 safetensors initialization is supported")
    if config.tensor_model_parallel_size != 1:
        raise ValueError("HF initialization requires TP=1")
    checks = {
        "num_hidden_layers": config.num_layers,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.ffn_hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_query_groups,
        "head_dim": config.kv_channels,
        "vocab_size": model.vocab_size,
        "rms_norm_eps": config.layernorm_epsilon,
        "tie_word_embeddings": model.share_embeddings_and_output_weights,
        "attention_bias": config.add_bias_linear,
    }
    for name, expected in checks.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"HF architecture mismatch for {name}: {metadata.get(name)} != {expected}"
            )
    rope = metadata.get("rope_parameters", {})
    if rope.get("rope_theta", metadata.get("rope_theta", 10000)) != rotary_base:
        raise ValueError("HF rotary base does not match the model")
    if metadata.get("use_sliding_window", False) or metadata.get("rope_scaling"):
        raise ValueError("Sliding-window attention and scaled RoPE require a separate conversion")
    if not config.qk_layernorm or not config.gated_linear_unit:
        raise ValueError("Qwen3 conversion requires QK normalization and gated SiLU")
    if config.activation_func is not torch.nn.functional.silu:
        raise ValueError("Qwen3 conversion requires SiLU")
    if model.share_embeddings_and_output_weights:
        raise ValueError("This conversion currently requires untied Qwen3 weights")
    parameters = dict(model.named_parameters())
    copied = set()
    consumed = set()
    with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as source:

        def read(name):
            consumed.add(name)
            return source.get_tensor(name)

        def assign(name, value):
            if name not in parameters or parameters[name].shape != value.shape:
                actual = parameters[name].shape if name in parameters else None
                raise ValueError(f"HF tensor shape mismatch: {name}: {actual} != {value.shape}")
            with torch.no_grad():
                parameters[name].copy_(value)
            copied.add(name)

        for target_name, source_name in (
            ("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            ("output_layer.weight", "lm_head.weight"),
            ("decoder.final_layernorm.weight", "model.norm.weight"),
        ):
            if target_name in parameters:
                assign(target_name, read(source_name))
        global_layers = []
        for local_index, layer in enumerate(model.decoder.layers):
            # layer_number is global and one-based, including asymmetric PP layouts.
            layer_index = layer.layer_number - 1
            if not 0 <= layer_index < config.num_layers or layer_index in global_layers:
                raise ValueError(
                    f"Invalid global layer number in HF conversion: {layer.layer_number}"
                )
            global_layers.append(layer_index)
            source_prefix = f"model.layers.{layer_index}."
            target_prefix = f"decoder.layers.{local_index}."
            groups = config.num_query_groups
            heads_per_group = config.num_attention_heads // groups
            head_size = config.kv_channels
            query = read(source_prefix + "self_attn.q_proj.weight").reshape(
                groups, heads_per_group * head_size, config.hidden_size
            )
            key = read(source_prefix + "self_attn.k_proj.weight").reshape(
                groups, head_size, config.hidden_size
            )
            value = read(source_prefix + "self_attn.v_proj.weight").reshape(
                groups, head_size, config.hidden_size
            )
            assign(
                target_prefix + "self_attention.linear_qkv.weight",
                torch.cat((query, key, value), dim=1).reshape(-1, config.hidden_size),
            )
            assign(
                target_prefix + "mlp.linear_fc1.weight",
                torch.cat(
                    (
                        read(source_prefix + "mlp.gate_proj.weight"),
                        read(source_prefix + "mlp.up_proj.weight"),
                    ),
                    dim=0,
                ),
            )
            for target_name, source_name in (
                ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
                ("self_attention.linear_proj.weight", "self_attn.o_proj.weight"),
                ("self_attention.q_layernorm.weight", "self_attn.q_norm.weight"),
                ("self_attention.k_layernorm.weight", "self_attn.k_norm.weight"),
                ("mlp.linear_fc1.layer_norm_weight", "post_attention_layernorm.weight"),
                ("mlp.linear_fc2.weight", "mlp.down_proj.weight"),
            ):
                assign(target_prefix + target_name, read(source_prefix + source_name))
        missing = set(parameters) - copied
        if any(not name.startswith("latent_feedback.") for name in missing):
            raise ValueError(f"Uninitialized base parameters after HF conversion: {missing}")
        expected_source = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
        layer_suffixes = (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        )
        expected_source.update(
            f"model.layers.{i}.{suffix}"
            for i in range(config.num_layers)
            for suffix in layer_suffixes
        )
        unexpected = set(source.keys()) - expected_source
        if unexpected:
            raise ValueError(f"Unconsumed HF checkpoint tensors: {unexpected}")
        missing_source = expected_source - set(source.keys())
        if missing_source:
            raise ValueError(f"Missing HF checkpoint tensors: {missing_source}")
        if config.pipeline_model_parallel_size == 1 and consumed != expected_source:
            raise ValueError("PP1 HF initialization did not consume the complete checkpoint")
    return {
        "source": str(directory.resolve()),
        "copied_parameter_tensors": len(copied),
        "new_feedback_tensors": sorted(missing),
        "global_layer_indices": global_layers,
        "pipeline_model_parallel_size": config.pipeline_model_parallel_size,
        "optimizer_state_loaded": False,
    }
