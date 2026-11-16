# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Native iterator construction and split runs must preserve the model RNG."""

from types import SimpleNamespace

import pytest
import torch

from megatron.training.datasets import data_samplers


@pytest.fixture(autouse=True)
def preserve_cpu_rng():
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


@pytest.mark.parametrize('rank,world_size', [(0, 1), (3, 4)])
def test_iterator_reconstruction_preserves_rng_and_sample_order(monkeypatch, rank, world_size):
    args = SimpleNamespace(
        dataloader_type='single',
        hybrid_context_parallel=False,
        micro_batch_size=2,
        global_batch_size=8,
        num_workers=0,
        exit_signal_handler=False,
        full_validation=False,
        seed=1234,
    )
    monkeypatch.setattr(data_samplers, 'get_args', lambda: args)
    monkeypatch.setattr(data_samplers.mpu, 'get_data_parallel_rank', lambda: rank)
    monkeypatch.setattr(data_samplers.mpu, 'get_data_parallel_world_size', lambda: world_size)
    original_loader = torch.utils.data.DataLoader

    def cpu_loader(*loader_args, **kwargs):
        # Exercise the real iterator and its RNG behavior without a CUDA context.
        kwargs['pin_memory'] = False
        return original_loader(*loader_args, **kwargs)

    monkeypatch.setattr(torch.utils.data, 'DataLoader', cpu_loader)
    dataset = torch.utils.data.TensorDataset(torch.arange(64))
    torch.random.default_generator.manual_seed(91)
    checkpoint_rng = torch.get_rng_state().clone()
    loader = data_samplers.build_pretraining_data_loader(dataset, 0)
    iterator = iter(loader)
    valid_iterator = iter(data_samplers.build_pretraining_data_loader(dataset, 0))
    next(valid_iterator)
    assert loader.generator.initial_seed() == 1234 + rank
    next(iterator)
    next(iterator)
    expected = [next(iterator)[0], next(iterator)[0]]
    assert torch.equal(torch.get_rng_state(), checkpoint_rng)
    torch.rand(3)  # Model/setup work before restoring a checkpoint.
    torch.set_rng_state(checkpoint_rng)
    resumed = iter(data_samplers.build_pretraining_data_loader(dataset, 4 * world_size))
    for batch in expected:
        torch.testing.assert_close(next(resumed)[0], batch, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), checkpoint_rng)
