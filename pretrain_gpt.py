# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT GPT."""

# Capture the true program start time BEFORE any heavy imports.
import time
_PROGRAM_START_TIME = time.time()

import json
import sys

# Suppress warnings on all ranks but rank 0.
import os
import warnings
rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from functools import partial
from typing import List, Optional, Tuple

import torch

from gpt_builders import gpt_builder
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_attr_wrapped_model, get_thd_batch_on_this_cp_rank, get_batch_on_this_hybrid_cp_rank, StragglerDetector
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank, get_mtp_ranks
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    custom_get_batch_on_this_tp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from model_provider import model_provider
from megatron.core.packed_seq_params import PackedSeqParams

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()


def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch."""
    args = get_args()
    config = core_transformer_config_from_args(args)
    needs_model_input = (
        is_first_or_last_pipeline_stage(vp_stage)
        or mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
    )
    # Packed THD attention needs cu_seqlens/rotary metadata on every PP stage.
    # Dense attention can keep the cheaper legacy path and skip middle-stage data.
    if not needs_model_input and args.attention_type != "thd":
        return None, None, None, None, None, None

    # Keep the upstream TP-rank batching path for reference, but use the
    # custom path in this fork to match the custom dataset output format.
    use_custom_tp_batch = True
    if use_custom_tp_batch:
        batch = custom_get_batch_on_this_tp_rank(data_iterator)
    else:
        batch = get_batch_on_this_tp_rank(
            data_iterator,
            mtp_on_this_rank=mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage),
        )

    cu_seqlens = batch.pop('cu_seqlens', None)
    cu_seqlens_padded = batch.pop('cu_seqlens_padded', None)
    max_seqlen = batch.pop('max_seqlen', None)
    local_cp_size = batch.pop('local_cp_size', None)
    if local_cp_size is not None:
        local_cp_size = int(local_cp_size.item())

    if cu_seqlens is None and local_cp_size is None:
        if args.attention_type == "thd":
            # CustomGPTDataset path: build packed_seq_params from EOS tokens at runtime
            batch = get_batch_on_this_cp_rank(batch)
            eos_id = args.eos_token_id
            tokens_list = [batch["tokens"][i].clone() for i in range(len(batch["tokens"]))]
            for t in tokens_list:
                t[-1] = eos_id
            tokens_flat = torch.cat(tokens_list, dim=0)
            boundaries = (tokens_flat == eos_id).nonzero().flatten() + 1
            cu_seqlens_built = torch.cat([
                torch.tensor([0], dtype=torch.int32, device=torch.cuda.current_device()),
                boundaries.to(torch.int32)
            ])
            max_seq_len = int((cu_seqlens_built[1:] - cu_seqlens_built[:-1]).max().item())
            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens_built,
                cu_seqlens_kv=cu_seqlens_built,
                max_seqlen_q=max_seq_len,
                max_seqlen_kv=max_seq_len,
                qkv_format='thd',
            )
        else:
            # Standard path: slice batch along sequence dimension for context parallelism
            batch = get_batch_on_this_cp_rank(batch)
            packed_seq_params = None
    elif local_cp_size is None:  # Packed THD format (dataset provides cu_seqlens)
        assert max_seqlen.dim() == 1
        batch, packed_seq_params = get_thd_batch_on_this_cp_rank(batch, cu_seqlens, cu_seqlens_padded, max_seqlen)
    else:  # Hybrid CP format
        batch, packed_seq_params = get_batch_on_this_hybrid_cp_rank(batch, local_cp_size)

    if not needs_model_input:
        return None, None, None, None, None, packed_seq_params

    return (*batch.values(), packed_seq_params)


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        pass_losses = ()
        if model is not None and getattr(args, 'full_bandwidth', False):
            pass_losses = get_attr_wrapped_model(model, 'feedback_losses')
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}
        if pass_losses:
            report['fbt objective'] = report['lm loss']
            for pass_index, pass_loss in enumerate(pass_losses, start=1):
                masked_pass_loss = (pass_loss.reshape(-1).float() * loss_mask).sum()
                report[f'feedback loss/pass{pass_index}'] = torch.cat(
                    [masked_pass_loss.detach().view(1), num_tokens.view(1)]
                )
            # Preserve the frozen run's public K1 metric; name the optimized mean explicitly.
            report['lm loss'] = report['feedback loss/pass1']

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator, vp_stage)
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask,
                    packed_seq_params=packed_seq_params
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, packed_seq_params=packed_seq_params
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def consume_dummy_train_batch(data_iterator):
    """Consume one batch using the GPT entrypoint's custom batch path."""
    get_batch(data_iterator)


def is_dataset_built_on_rank(vp_stage=None):
    args = get_args()
    config = core_transformer_config_from_args(args)
    needs_model_input = (
        is_first_or_last_pipeline_stage(vp_stage)
        or mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
    )
    return (
        needs_model_input
        or args.attention_type == "thd"
    ) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    data_args = {
        "random_seed": args.seed,
        "sequence_length": args.seq_length,
        "blend": blend,
        "blend_per_split": blend_per_split,
        "split": args.split,
        "multiple_validation_sets": args.multiple_validation_sets,
        "full_validation": args.full_validation,
        "num_dataset_builder_threads": args.num_dataset_builder_threads,
        "path_to_cache": args.data_cache_path,
        "mmap_bin_files": args.mmap_bin_files,
        "tokenizer": tokenizer,
        "reset_position_ids": args.reset_position_ids,
        "reset_attention_mask": args.reset_attention_mask,
        "eod_mask_loss": args.eod_mask_loss,
        "create_attention_mask": args.create_attention_mask_in_dataloader,
        "object_storage_cache_path": args.object_storage_cache_path,
        "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
        "allow_ambiguous_pad_tokens": args.allow_ambiguous_pad_tokens,
        "fast_cache_load": args.dataloader_fast_cache_load,
        "sequences_per_dataset": sequences_per_dataset,
        "defer_npy_index_mmap": args.dataloader_defer_npy_index_mmap,
        "context_parallel_size": args.context_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "sequence_parallel_size": args.tensor_model_parallel_size*args.sequence_parallel,
        "hybrid_context_parallel": args.hybrid_context_parallel,
    }

    # add FIM args to the config
    if args.fim_data:
        extra_tokens = {
            "prefix": args.fim_prefix_token,
            "middle": args.fim_middle_token,
            "suffix": args.fim_suffix_token,
            "pad": args.fim_pad_token,
            "eod": args.fim_eod_token,
        }
        data_args.update(
            {
                "fim_rate": args.fim_rate,
                "fim_spm_rate": args.fim_spm_rate,
                "fim_extra_tokens": extra_tokens,
                "fim_split_sample": args.fim_split_sample,
                "fim_fragment_rate": args.fim_fragment_rate,
                "fim_no_prefix": args.fim_no_prefix,
            }
        )
        return GPTFIMDatasetConfig(**data_args)

    return GPTDatasetConfig(**data_args)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        elif args.fim_data:
            dataset_type = GPTFIMDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    is_dataset_built = partial(is_dataset_built_on_rank, vp_stage=vp_stage)
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

class CustomGPTDataset:
    def __init__(self, dirname):
        self.dirname = dirname
        metadata_filename = os.path.join(dirname, "metadata.json")
        with open(metadata_filename, "r") as f:
            metadata = json.load(f)
        self.real_size = int(metadata["size"])
        self.sample_index_offset = int(
            metadata.get("sample_index_offset", metadata.get("logical_start_index", 0))
        )
        self.size = self.real_size + self.sample_index_offset
        self.file_list = metadata["file_list"]
        self.file_len_list = metadata["file_len"]
        self.file_len_cumsum = [0]
        for i in self.file_len_list:
            self.file_len_cumsum.append(self.file_len_cumsum[-1] + i)
        assert self.file_len_cumsum[-1] == self.real_size, (
            f"metadata size ({self.real_size}) does not match file_len sum "
            f"({self.file_len_cumsum[-1]})"
        )

        self.cur_file_idx = -1

    def __len__(self):
        return self.size

    def _logical_to_real_idx(self, idx):
        idx = int(idx)
        if self.sample_index_offset > 0 and idx >= self.sample_index_offset:
            idx -= self.sample_index_offset
        return idx % self.real_size

    def _get_sequential_sample(self, idx):
        logical_idx = idx
        idx = self._logical_to_real_idx(idx)
        new_cur_file_idx = self.cur_file_idx
        while idx >= self.file_len_cumsum[new_cur_file_idx + 1]:
            new_cur_file_idx += 1
        while idx < self.file_len_cumsum[new_cur_file_idx]:
            new_cur_file_idx -= 1

        if new_cur_file_idx != self.cur_file_idx:
            self.cur_file_idx = new_cur_file_idx
            file_path = os.path.join(self.dirname, self.file_list[self.cur_file_idx])
            print_rank_0(
                f"=== {self.__class__.__name__} loading file {file_path} "
                f"for logical sample {logical_idx} (real sample {idx}) ==="
            )
            self.cur_data = np.load(file_path, mmap_mode='r')

        local_idx = idx - self.file_len_cumsum[new_cur_file_idx]
        text = self.cur_data[local_idx]
        return text

    def __getitem__(self, idx):
        text = self._get_sequential_sample(idx)
        tokens = torch.from_numpy(text[:-1]).long().contiguous()
        labels = torch.from_numpy(text[1:]).long().contiguous()
        loss_mask = torch.ones(len(text) - 1).float()

        result = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
        }
        return result


class CustomValidGPTDataset(CustomGPTDataset):
    def __len__(self):
        return sys.maxsize  # effectively infinite length while fitting index-sized integer

    def __getitem__(self, idx):
        text = self._get_sequential_sample(self.sample_index_offset + (idx % self.real_size))
        tokens = torch.from_numpy(text[:-1]).long().contiguous()
        labels = torch.from_numpy(text[1:]).long().contiguous()
        loss_mask = torch.ones(len(text) - 1).float()

        result = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
        }
        return result

def custom_train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets."""
    args = get_args()
    train_ds = CustomGPTDataset(args.train_data_path[0])
    eval_ds = CustomValidGPTDataset(args.valid_data_path[0])

    return train_ds, eval_ds, None

def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = core_transformer_config_from_args(args)
        mtp_ranks = get_mtp_ranks(pp_ranks, config)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        custom_train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        dummy_train_step_func=consume_dummy_train_batch,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
