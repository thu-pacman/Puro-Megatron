# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Run native pretraining with migration diagnostics and unmodified model math.

Set PURO_PARITY_DIR and PURO_PARITY_END. All other arguments are normal
pretrain_gpt.py arguments. Each step records token-weighted K1/K2/mean NLL,
the exact CPU input fingerprint on every rank, and synchronized batch wall time.
"""

import hashlib
import importlib.metadata
import inspect
import json
import os
import time
from functools import wraps
from pathlib import Path

import torch

import gpt_builders
import megatron.training.training as training
from megatron.core.optimizer.layer_wise_optimizer import FlatShardedAdamFallback
from megatron.training import get_args

OUTPUT = Path(os.environ['PURO_PARITY_DIR'])
END = int(os.environ['PURO_PARITY_END'])
SYNC_CHECKPOINT = os.environ.get('PURO_PARITY_SYNC_CHECKPOINT', '0') == '1'
INPUT_HASH = hashlib.sha256()
INPUT_ROWS = 0
FIRST_STEP = True


def tensor_digest(tensor):
    """Hash every element, retaining dtype and shape in the digest."""
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def append(filename, record):
    """Write rank-zero records without affecting model RNG."""
    if torch.distributed.get_rank() == 0:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with (OUTPUT / filename).open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')


def snapshot(kind, iteration, model, optimizer, scheduler):
    """Save owned master weights/state and rank-zero parameters for exact diagnostics."""
    rank = torch.distributed.get_rank()
    directory = OUTPUT / 'states'
    directory.mkdir(parents=True, exist_ok=True)
    children = getattr(optimizer, 'chained_optimizers', [optimizer])
    groups = []
    for child in children:
        native = child.optimizer
        current = []
        for group in native.param_groups:
            options = group
            flat_step = None
            if isinstance(child, FlatShardedAdamFallback):
                index = next(
                    i
                    for i, value in child._optimizer_group_by_group_index.items()
                    if value is group
                )
                options = child.param_groups[index]
                value = child._get_local_group_step(index, group['params'][0].device)
                flat_step = 0 if value is None else int(value)
            parameters = []
            for parameter in group['params']:
                state = native.state.get(parameter, {})
                state = {
                    key: value for key, value in state.items() if flat_step is None or key != 'step'
                }
                parameters.append(
                    dict(
                        parameter=parameter.detach().cpu().clone(),
                        gradient=(
                            None
                            if parameter.grad is None
                            else parameter.grad.detach().cpu().clone()
                        ),
                        state={
                            key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                            for key, value in state.items()
                        },
                    )
                )
            scalars = {
                key: float(options[key])
                for key in ('lr', 'eps', 'weight_decay', 'step')
                if key in options and options[key] is not None
            }
            if flat_step is not None:
                scalars['step'] = flat_step
            current.append(dict(parameters=parameters, scalars=scalars))
        groups.append(dict(optimizer=type(native).__name__, groups=current))
    result = dict(iteration=iteration, optimizer=groups, scheduler=scheduler.state_dict())
    if rank == 0:
        result['model'] = {
            f'{i}.{name}': parameter.detach().cpu().clone()
            for i, chunk in enumerate(model)
            for name, parameter in chunk.named_parameters()
        }
    if kind == 'initial':

        def summarize(value):
            if torch.is_tensor(value):
                return tensor_digest(value)
            if isinstance(value, dict):
                return {key: summarize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [summarize(item) for item in value]
            return value

        (directory / f'{kind}_{iteration:07d}_rank{rank:02d}.json').write_text(
            json.dumps(summarize(result), indent=2) + '\n'
        )
    else:
        torch.save(result, directory / f'{kind}_{iteration:07d}_rank{rank:02d}.pt')
    torch.distributed.barrier()


original_builder = gpt_builders.gpt_builder


@wraps(original_builder)
def builder(*args, **kwargs):
    model = original_builder(*args, **kwargs)
    if torch.distributed.get_rank() == 0:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        config = get_args()
        selected = [
            'num_layers',
            'hidden_size',
            'ffn_hidden_size',
            'num_attention_heads',
            'num_query_groups',
            'kv_channels',
            'padded_vocab_size',
            'seq_length',
            'full_bandwidth',
            'feedback_passes',
            'feedback_loss_reduction',
            'feedback_fusion',
            'micro_batch_size',
            'global_batch_size',
            'tensor_model_parallel_size',
            'pipeline_model_parallel_size',
            'context_parallel_size',
            'fp8',
            'fp8_recipe',
            'bf16',
            'recompute_granularity',
            'recompute_method',
            'recompute_num_layers',
            'output_loss_chunk_size',
            'optimizer',
            'muon_momentum',
            'muon_hyperball_lr_mult',
            'muon_hyperball_eps',
            'adam_beta1',
            'adam_beta2',
            'adam_eps',
            'weight_decay',
            'lr',
            'min_lr',
            'lr_decay_style',
            'lr_decay_iters',
            'lr_warmup_iters',
            'overlap_grad_reduce',
            'overlap_param_gather',
            'seed',
            'attention_backend',
            'eos_token_id',
        ]
        metadata = dict(
            world_size=torch.distributed.get_world_size(),
            checkpoint_sync_enforced=SYNC_CHECKPOINT,
            config={key: str(getattr(config, key)) for key in selected},
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            initial_parameters={
                name: tensor_digest(parameter) for name, parameter in model.named_parameters()
            },
            packages={
                key: importlib.metadata.version(key)
                for key in [
                    'torch',
                    'transformer-engine',
                    'flash-attn',
                    'emerging-optimizers',
                    'numpy',
                ]
            },
            environment={
                key: os.environ.get(key)
                for key in [
                    'NVTE_FLASH_ATTN',
                    'NVTE_FUSED_ATTN',
                    'NVTE_ALLOW_NONDETERMINISTIC_ALGO',
                    'NCCL_NET',
                    'NCCL_IB_DISABLE',
                    'NCCL_IB_HCA',
                ]
            },
        )
        (OUTPUT / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return model


original_step = training.train_step
step_signature = inspect.signature(original_step)

original_save = training.save_checkpoint_and_time
save_signature = inspect.signature(original_save)


@wraps(original_save)
def save_checkpoint(*args, **kwargs):
    """Optionally align short reference saves with its production validation saves.

    The research fork's layer-wise optimizer guard omits the mandatory parameter
    all-gather before standalone saves. Its normal validation immediately before
    production checkpoints happens to synchronize the model. Record this opt-in
    workaround for isolated short-run checkpoints; do not change training math.
    """
    config = get_args()
    if (
        SYNC_CHECKPOINT
        and config.use_layer_wise_distributed_optimizer
        and config.overlap_param_gather
        and not training.should_disable_forward_pre_hook(config)
    ):
        bound = save_signature.bind(*args, **kwargs)
        training.force_param_sync(bound.arguments['model'])
    return original_save(*args, **kwargs)


@wraps(original_step)
def step(*args, **kwargs):
    global INPUT_HASH, INPUT_ROWS, FIRST_STEP
    bound = step_signature.bind(*args, **kwargs)
    iteration = bound.arguments['iteration']
    model, optimizer = bound.arguments['model'], bound.arguments['optimizer']
    scheduler = bound.arguments['opt_param_scheduler']
    if FIRST_STEP:
        snapshot('initial', iteration, model, optimizer, scheduler)
        FIRST_STEP = False
    INPUT_HASH, INPUT_ROWS = hashlib.sha256(), 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = original_step(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    times = torch.tensor(
        [elapsed, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
        dtype=torch.float64,
        device='cuda',
    )
    torch.distributed.all_reduce(times, op=torch.distributed.ReduceOp.MAX)
    data = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(data, dict(sha256=INPUT_HASH.hexdigest(), rows=INPUT_ROWS))
    config = get_args()
    assert sum(item['rows'] for item in data) == config.global_batch_size, data
    metrics = {key: float(value.item()) for key, value in result[0].items()}
    k1, k2, objective = (
        metrics[key] for key in ['feedback loss/pass1', 'feedback loss/pass2', 'fbt objective']
    )
    assert abs(objective - (k1 + k2) / 2) <= 2e-6, metrics
    assert not result[1], 'Skipped optimizer update'
    append(
        'training.jsonl',
        dict(
            iteration=iteration + 1,
            k1_nll=k1,
            k2_nll=k2,
            objective_nll=objective,
            grad_norm=float(result[5]),
            data=data,
            batch_seconds=float(times[0]),
            input_tokens=config.global_batch_size * config.seq_length,
            peak_allocated_bytes=int(times[1]),
            peak_reserved_bytes=int(times[2]),
        ),
    )
    if iteration + 1 == END:
        snapshot('final', iteration + 1, model, optimizer, scheduler)
    return result


gpt_builders.gpt_builder = builder
training.train_step = step
training.save_checkpoint_and_time = save_checkpoint

# Import the entrypoint once, then invoke the same native setup as its __main__.
import pretrain_gpt as entry

original_getitem = entry.CustomGPTDataset.__getitem__


def getitem(self, index):
    global INPUT_ROWS
    result = original_getitem(self, index)
    INPUT_HASH.update(str(index).encode())
    for key in sorted(result):
        INPUT_HASH.update(key.encode())
        INPUT_HASH.update(result[key].numpy().tobytes())
    INPUT_ROWS += 1
    return result


def main():
    """Keep checkpoint writer spawn workers from re-entering distributed training."""
    entry.CustomGPTDataset.__getitem__ = getitem
    entry.set_startup_timestamps(program_start=entry._PROGRAM_START_TIME, main_entry=time.time())
    entry.train_valid_test_datasets_provider.is_distributed = True
    pretrain, store = entry.inprocess_restart.maybe_wrap_for_inprocess_restart(entry.pretrain)
    pretrain(
        entry.custom_train_valid_test_datasets_provider,
        entry.partial(entry.model_provider, entry.gpt_builder),
        entry.ModelType.encoder_or_decoder,
        entry.forward_step,
        dummy_train_step_func=entry.consume_dummy_train_batch,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=entry.add_modelopt_args if entry.has_nvidia_modelopt else None,
        store=store,
        get_embedding_ranks=entry.get_embedding_ranks,
    )


if __name__ == '__main__':
    main()
