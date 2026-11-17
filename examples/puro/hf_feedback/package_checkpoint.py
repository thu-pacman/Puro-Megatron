# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Package a verified Qwen3 + feedback evaluation export as one HF model.

The large decoder shard is copied byte-for-byte. Only the four feedback tensor
names acquire their latent_feedback prefix. No optimizer or training data is
copied. This command prepares local files and never uploads a model.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

FEEDBACK_NAMES = {
    "hidden_projection.weight",
    "token_projection.weight",
    "embedding_norm.weight",
    "input_norm.weight",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package(source, output, *, expected_step=13892, reference_validation=None):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((source / "export.json").read_text())
    if report["iteration"] != expected_step:
        raise ValueError(f"Expected step {expected_step}, found {report['iteration']}")
    source_hashes = {}
    for name in ("model.safetensors", "feedback.safetensors"):
        source_hashes[name] = sha256(source / name)
        if source_hashes[name] != report["weights_sha256"][name]:
            raise ValueError(f"Source export hash mismatch: {name}")
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        raise ValueError("Expected the original Qwen3 evaluation export")
    rope_theta = (config.get("rope_parameters") or {}).get(
        "rope_theta", config.get("rope_theta", 10000)
    )
    if config.get("rope_theta", rope_theta) != rope_theta or rope_theta != 10000:
        raise ValueError("The Puro continuation requires RoPE base 10000")
    feedback = load_file(str(source / "feedback.safetensors"))
    if set(feedback) != FEEDBACK_NAMES:
        raise ValueError(f"Unexpected feedback tensors: {set(feedback)}")
    hidden = config["hidden_size"]
    for name, value in feedback.items():
        shape = (hidden, hidden) if "projection" in name else (hidden,)
        if tuple(value.shape) != shape:
            raise ValueError(f"Invalid feedback shape: {name}: {value.shape}")
    weights_map = {}
    parameters = total_bytes = 0
    with safe_open(str(source / "model.safetensors"), framework="pt", device="cpu") as tensors:
        for name in tensors.keys():
            value = tensors.get_tensor(name)
            weights_map[name] = "model-00001-of-00002.safetensors"
            parameters += value.numel()
            total_bytes += value.numel() * value.element_size()
    renamed = {"latent_feedback." + name: value for name, value in feedback.items()}
    for name, value in renamed.items():
        if name in weights_map:
            raise ValueError(f"Duplicate tensor: {name}")
        weights_map[name] = "model-00002-of-00002.safetensors"
        parameters += value.numel()
        total_bytes += value.numel() * value.element_size()
    if parameters != report["parameters"]:
        raise ValueError(f"Parameter count differs from source export: {parameters}")
    output.parent.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix=output.name + ".tmp-", dir=output.parent) as temporary:
        destination = Path(temporary)
        shutil.copy2(source / "model.safetensors", destination / "model-00001-of-00002.safetensors")
        save_file(
            renamed,
            str(destination / "model-00002-of-00002.safetensors"),
            metadata={"format": "pt"},
        )
        config.update(
            model_type="puro_feedback",
            architectures=["PuroFeedbackForCausalLM"],
            auto_map={
                "AutoConfig": "configuration_puro_feedback.PuroFeedbackConfig",
                "AutoModelForCausalLM": "modeling_puro_feedback.PuroFeedbackForCausalLM",
            },
            feedback_passes=2,
            feedback_fusion="residual_glu",
            feedback_shift=1,
            feedback_loss_reduction="mean",
            rope_theta=rope_theta,
            torch_dtype="bfloat16",
            transformers_version="4.57.1",
        )
        (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (destination / "model.safetensors.index.json").write_text(
            json.dumps(
                dict(metadata=dict(total_size=total_bytes), weight_map=weights_map), indent=2
            )
            + "\n"
        )
        for name in (
            "configuration_puro_feedback.py",
            "modeling_puro_feedback.py",
            "generate.py",
            "verify_checkpoint.py",
            "requirements.txt",
            "README.md",
            "LICENSE-MODEL",
        ):
            shutil.copy2(code / name, destination / name)
        for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            shutil.copy2(source / name, destination / name)
        for pattern in ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"):
            if (source / pattern).exists():
                shutil.copy2(source / pattern, destination / pattern)
        # Preserve the repository's source-code license independently of the model card.
        license_path = code.parents[2] / "LICENSE"
        if license_path.exists():
            shutil.copy2(license_path, destination / "LICENSE-CODE")
        if reference_validation is not None:
            validation = json.loads(Path(reference_validation).read_text())
            if not validation.get("passed"):
                raise ValueError("The supplied native/HF export check did not pass")
            (destination / "native_export_validation.json").write_text(
                json.dumps(validation, indent=2) + "\n"
            )
        manifest = dict(
            source_iteration=expected_step,
            source_base_model="thu-pacman/Puro-2B-Base",
            parameters=parameters,
            source_export_weights_sha256=source_hashes,
            weight_format="BF16 safetensors; 2 shards, one shared decoder plus feedback",
            decoder_layers=config["num_hidden_layers"],
            default_inference_passes=2,
            trained_objective="(NLL_K1 + NLL_K2) / 2",
            feedback="full-layer shift-1 residual_glu; no detach",
            rope_theta=rope_theta,
            files_sha256={p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file()},
        )
        if (
            manifest["files_sha256"]["model-00001-of-00002.safetensors"]
            != source_hashes["model.safetensors"]
        ):
            raise ValueError("Decoder shard changed during copying")
        (destination / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        destination.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=13892)
    parser.add_argument("--reference-validation", type=Path)
    args = parser.parse_args()
    manifest = package(
        args.source,
        args.output,
        expected_step=args.expected_step,
        reference_validation=args.reference_validation,
    )
    print(
        json.dumps(
            dict(
                output=str(args.output),
                parameters=manifest["parameters"],
                iteration=manifest["source_iteration"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
