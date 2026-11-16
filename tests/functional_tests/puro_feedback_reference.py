# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Deterministic BF16 training probe, runnable against either checkout via PYTHONPATH.

This is a small-model migration check. The separate full-size FP8 training check
must also pass before publishing a claim about the production configuration.
"""

import argparse
import dataclasses
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--baseline', action='store_true')
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    with tempfile.TemporaryDirectory() as directory:
        torch.distributed.init_process_group(
            'nccl', init_method=f'file://{directory}/rendezvous', rank=0, world_size=1
        )
        parallel_state.initialize_model_parallel()
        model_parallel_cuda_manual_seed(1234)
        fields = {field.name for field in dataclasses.fields(TransformerConfig)}
        kwargs = dict(
            num_layers=2,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=32,
            full_bandwidth=not options.baseline,
            feedback_passes=2,
            feedback_fusion='residual_glu',
            feedback_loss_reduction='mean',
            feedback_jitter=0.0,
            feedback_prefix_mixin=False,
            normalization='RMSNorm',
            qk_layernorm=True,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=False,
            attention_backend=AttnBackend.unfused,
            output_loss_chunk_size=0 if options.baseline else 16,
        )
        config = TransformerConfig(**{key: value for key, value in kwargs.items() if key in fields})
        with patch('megatron.training.global_vars._GLOBAL_ARGS', SimpleNamespace(diag_interval=0)):
            model = (
                GPTModel(
                    config,
                    get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
                    vocab_size=128,
                    max_sequence_length=32,
                    position_embedding_type='rope',
                    share_embeddings_and_output_weights=False,
                )
                .cuda()
                .bfloat16()
                .train()
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.95))
            start = 0
            if options.resume:
                state = torch.load(options.resume, weights_only=False, map_location='cpu')
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
                start = state['iteration']
                torch.set_rng_state(state['rng'])
                torch.cuda.set_rng_state(state['cuda_rng'])
                get_cuda_rng_tracker().set_states(state['tracker'])
            for step in range(start, options.steps):
                generator = torch.Generator(device='cuda').manual_seed(9000 + step)
                tokens = torch.randint(0, 128, (2, 17), device='cuda', generator=generator)
                positions = torch.arange(16, device='cuda').expand(2, -1)
                mask = torch.ones(1, 1, 16, 16, device='cuda', dtype=torch.bool).triu(1)
                optimizer.zero_grad(set_to_none=True)
                losses = model(tokens[:, :-1], positions, mask, labels=tokens[:, 1:])
                objective = losses.float().mean()
                if options.baseline:
                    k1, k2 = objective.detach().item(), None
                else:
                    k1, k2 = [value.float().mean().item() for value in model.feedback_losses]
                objective.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()
                record = dict(
                    iteration=step + 1,
                    k1_nll=k1,
                    k2_nll=k2,
                    objective_nll=objective.item(),
                    grad_norm=grad_norm,
                )
                with (options.output / 'training.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
            torch.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    iteration=options.steps,
                    rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(),
                    tracker=get_cuda_rng_tracker().get_states(),
                ),
                options.output / 'state.pt',
            )
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
