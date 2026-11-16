# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Blockwise FP8 may reset TE runtime state; strict parameter loading must remain."""
import pytest
import torch
from transformer_engine.pytorch import RMSNorm

from megatron.training.checkpointing import _load_model_state_without_te_runtime


class CustomMetadata(torch.nn.Module):
    def get_extra_state(self):
        return {'must_restore': 17}

    def set_extra_state(self, state):
        self.restored = state


def model():
    result = torch.nn.Module()
    result.norm = RMSNorm(4, device='cpu', dtype=torch.float32)
    result.latent_feedback = torch.nn.Linear(4, 4, bias=False)
    result.custom = CustomMetadata()
    return result


def checkpoint_without_te_runtime(module):
    state = module.state_dict()
    # The checkpoint contains all non-TE metadata; only TE runtime is reset.
    state.pop('norm._extra_state')
    return state


def test_te_runtime_can_be_reset_without_relaxing_weight_loading():
    original = model()
    restored = model()
    state = original.state_dict()
    result = _load_model_state_without_te_runtime(restored, state, True)
    assert result.missing_keys == ['norm._extra_state']
    assert torch.equal(restored.norm.weight, original.norm.weight)
    assert torch.equal(restored.latent_feedback.weight, original.latent_feedback.weight)
    assert restored.custom.restored == {'must_restore': 17}


@pytest.mark.parametrize('key', ['norm.weight', 'latent_feedback.weight', 'custom._extra_state'])
def test_missing_parameters_and_non_te_metadata_still_fail(key):
    original = model()
    state = checkpoint_without_te_runtime(original)
    state.pop(key)
    with pytest.raises(RuntimeError, match='Invalid model state'):
        _load_model_state_without_te_runtime(model(), state, True)


def test_unexpected_parameters_still_fail():
    state = checkpoint_without_te_runtime(model())
    state['latent_feedback.unexpected'] = torch.zeros(4)
    with pytest.raises(RuntimeError, match='unexpected'):
        _load_model_state_without_te_runtime(model(), state, True)


def test_bad_shape_still_fails():
    state = checkpoint_without_te_runtime(model())
    state['norm.weight'] = torch.zeros(5)
    with pytest.raises(RuntimeError, match='size mismatch'):
        _load_model_state_without_te_runtime(model(), state, True)
