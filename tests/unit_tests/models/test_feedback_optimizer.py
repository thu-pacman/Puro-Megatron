# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Feedback routing and checkpoint persistence of MuonH's fixed radii."""

import copy

import torch

from megatron.core.optimizer.emerging_optimizers import (
    TensorParallelMuonHyperball,
    _muon_hyperball_param_overrides_factory,
)


def make_optimizer(parameter):
    return TensorParallelMuonHyperball(
        [parameter],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0,
        qkv_ns_mode='separate',
        qkv_norm_mode='separate',
        is_qkv_fn=lambda param: True,
        qkv_split_shapes=(4, 2, 2),
        hyperball_eps=1e-12,
    )


def test_feedback_zero_projection_uses_adam_without_weight_decay():
    parameter = torch.nn.Parameter(torch.zeros(4, 4))
    parameter.is_feedback_parameter = True
    overrides = _muon_hyperball_param_overrides_factory('adam')
    matched = [
        value
        for key, value in overrides.items()
        if key.matches(parameter, 'latent_feedback.hidden_projection.weight')
    ]
    assert any(value.get('optimizer') == 'adam' for value in matched)
    assert any(value.get('wd_mult') == 0 for value in matched)


def test_partition_radii_and_updates_survive_checkpoint():
    torch.manual_seed(7)
    initial = torch.randn(8, 4)
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = make_optimizer(parameter)
    expected_radius = optimizer.state[parameter]['hyperball_partition_radii'].clone()
    generator = torch.Generator().manual_seed(13)
    gradients = [torch.randn(8, 4, generator=generator) for _ in range(6)]
    for gradient in gradients[:3]:
        parameter.grad = gradient.clone()
        optimizer.step()
    saved_parameter = parameter.detach().clone()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    for gradient in gradients[3:]:
        parameter.grad = gradient.clone()
        optimizer.step()
    resumed = torch.nn.Parameter(saved_parameter)
    resumed_optimizer = make_optimizer(resumed)
    resumed_optimizer.load_state_dict(saved_optimizer)
    torch.testing.assert_close(
        resumed_optimizer.state[resumed]['hyperball_partition_radii'],
        expected_radius,
        rtol=0,
        atol=0,
    )
    for gradient in gradients[3:]:
        resumed.grad = gradient.clone()
        resumed_optimizer.step()
    torch.testing.assert_close(resumed, parameter, rtol=0, atol=0)
    torch.testing.assert_close(
        resumed_optimizer.state[resumed]['momentum_buffer'],
        optimizer.state[parameter]['momentum_buffer'],
        rtol=0,
        atol=0,
    )
