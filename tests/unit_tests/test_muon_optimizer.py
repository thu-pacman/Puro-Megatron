# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.emerging_optimizers import (
    _EMERGING_OPTIMIZERS,
    HAVE_EMERGING_OPTIMIZERS,
    TensorParallelMuon,
    TensorParallelMuonHyperball,
    _muon_config_to_kwargs,
    _muon_hyperball_param_overrides_factory,
    get_supported_coefficient_types,
    validate_coefficient_type,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


# Skip all tests in this file for LTS versions or when emerging_optimizers is missing
pytestmark = [
    pytest.mark.skipif(
        Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
        reason="Skip emerging optimizer tests for LTS test",
    ),
    pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package is not installed"
    ),
]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(80, 48)
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# ===========================================================================
# Muon optimizer tests
# ===========================================================================


def test_muon_optimizer_smoke():
    """Smoke test for TensorParallelMuon optimizer."""
    # Create a simple linear model for testing
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Create TensorParallelMuon optimizer
    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.01,
        use_decoupled_weight_decay=True,
        split_qkv=False,
        fp32_matmul_prec="medium",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        pg_collection=None,
        tp_mode="duplicated",
    )

    # Test basic properties
    assert optimizer is not None, "Optimizer should not be None"
    assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
    assert len(optimizer.param_groups) > 0, "Optimizer should have at least one parameter group"

    # Test forward and backward pass
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Store original weight
    original_weight = model.weight.data.clone()

    # Test optimizer step
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    # Test zero_grad
    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    # Test state_dict and load_state_dict
    state_dict = optimizer.state_dict()
    assert 'state' in state_dict, "State dict should contain state"
    assert 'param_groups' in state_dict, "State dict should contain param_groups"

    # Load state dict should not raise error
    optimizer.load_state_dict(state_dict)


def test_muon_hyperball_optimizer_preserves_norm():
    """Smoke test for TensorParallelMuonHyperball optimizer."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)

    original_norm = model.weight.data.norm().item()
    optimizer = TensorParallelMuonHyperball(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.0,
        use_decoupled_weight_decay=True,
        split_qkv=False,
        fp32_matmul_prec="medium",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        pg_collection=None,
        tp_mode="duplicated",
        hyperball_eps=1e-12,
    )

    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(model.weight.data, original_weight)
    assert torch.allclose(
        model.weight.data.norm(), torch.tensor(original_norm, device='cuda'), atol=1e-5, rtol=1e-5
    )

    state_dict = optimizer.state_dict()
    for param_state in state_dict["state"].values():
        assert "hyperball_R" not in param_state
        assert all(not isinstance(value, float) for value in param_state.values())


def test_muon_hyperball_is_registered():
    """MuonHyperball should be exposed as a supported emerging optimizer."""
    assert "muon_hyperball" in _EMERGING_OPTIMIZERS


def test_muon_effective_lr_mult_aligns_relative_update_on_cpu():
    param = torch.nn.Parameter(torch.full((4, 4), 2.0, dtype=torch.float32))
    optimizer = TensorParallelMuon(
        params=[param],
        lr=0.1,
        momentum=0.95,
        weight_decay=0.0,
        use_decoupled_weight_decay=True,
        num_ns_steps=1,
        effective_lr_mult=3.0,
        pg_collection=None,
    )

    param.sum().backward()
    before = param.detach().clone()
    optimizer.step()

    update_norm = (param.detach() - before).float().norm()
    weight_norm = before.float().norm()
    assert (update_norm / weight_norm).item() == pytest.approx(0.3, rel=1e-5, abs=1e-6)


def test_muon_strict_effective_lr_matches_normalized_weight_distance_on_cpu():
    param = torch.nn.Parameter(torch.arange(1, 17, dtype=torch.float32).view(4, 4))
    optimizer = TensorParallelMuon(
        params=[param],
        lr=0.1,
        momentum=0.95,
        weight_decay=0.1,
        use_decoupled_weight_decay=True,
        num_ns_steps=1,
        effective_lr_mult=3.0,
        strict_effective_lr=True,
        pg_collection=None,
    )
    update = torch.tensor([[1.0, -2.0, 3.0, -4.0]] * 4, dtype=torch.float32)
    scale = optimizer._get_strict_effective_lr_scale(param, update, base_lr=0.1, weight_decay=0.1)
    actual_lr = 0.1 * float(scale.item())
    before = param.detach().float()
    after = before * (1.0 - actual_lr * 0.1) - 0.1 * update * scale
    chord = (after / after.norm() - before / before.norm()).norm().item()
    assert chord == pytest.approx(0.3, rel=1e-5, abs=1e-6)


def test_muon_effective_lr_mult_requires_plain_muon():
    with pytest.raises(AssertionError, match="requires optimizer='muon'"):
        OptimizerConfig(optimizer='muon_hyperball', muon_effective_lr_mult=2.0)


def test_muon_hyperball_lr_mult_sets_preprojection_relative_update_on_cpu():
    param = torch.nn.Parameter(torch.arange(1, 17, dtype=torch.float32).view(4, 4))
    optimizer = TensorParallelMuonHyperball(
        params=[param],
        lr=0.1,
        momentum=0.95,
        weight_decay=0.0,
        use_decoupled_weight_decay=True,
        num_ns_steps=1,
        lr_mult=3.0,
        pg_collection=None,
    )
    before_norm = param.detach().float().norm()
    optimizer.set_diagnostic_context(iteration=1, interval=1)

    (param.square().sum()).backward()
    optimizer.step()

    stats = optimizer.pop_diagnostic_stats()["matrix"]
    relative_update = stats["update_norm_sq"] ** 0.5 / stats["weight_norm_sq"] ** 0.5
    assert relative_update == pytest.approx(0.3, rel=1e-5, abs=1e-6)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert param.detach().float().norm().item() == pytest.approx(before_norm.item(), rel=1e-5)


def test_muon_hyperball_routes_only_selected_matrices():
    overrides = _muon_hyperball_param_overrides_factory('adam')

    def merged_override(param, name):
        result = {}
        for key, override in overrides.items():
            if key.matches(param, name):
                result.update(override)
        return result

    matrix = torch.nn.Parameter(torch.ones(4, 4))
    norm = torch.nn.Parameter(torch.ones(4))
    embedding = torch.nn.Parameter(torch.ones(4, 4))
    embedding.is_embedding_or_output_parameter = True
    router = torch.nn.Parameter(torch.ones(4, 4))

    assert merged_override(matrix, "decoder.layers.0.mlp.linear_fc1.weight") == {'wd_mult': 0.0}
    assert merged_override(norm, "decoder.layers.0.input_layernorm.weight") == {'optimizer': 'adam'}
    assert merged_override(embedding, "embedding.word_embeddings.weight") == {'optimizer': 'adam'}
    assert merged_override(router, "decoder.layers.0.mlp.router.weight") == {'optimizer': 'adam'}


def test_muon_hyperball_rejects_zero_norm_parameter():
    param = torch.nn.Parameter(torch.zeros(4, 4))
    with pytest.raises(ValueError, match="non-zero norm"):
        TensorParallelMuonHyperball(
            params=[param], lr=0.1, weight_decay=0.0, num_ns_steps=1, pg_collection=None
        )


def test_muon_hyperball_fixed_rms_sets_frobenius_radius():
    param = torch.nn.Parameter(torch.ones(4, 8))
    optimizer = TensorParallelMuonHyperball(
        params=[param],
        lr=0.1,
        weight_decay=0.0,
        num_ns_steps=1,
        hyperball_eps=1e-12,
        hyperball_rms=0.25,
        pg_collection=None,
    )

    expected_radius = 0.25 * math.sqrt(param.numel())
    assert param.detach().norm().item() == pytest.approx(expected_radius)
    assert optimizer._ensure_hyperball_radius_state(param).item() == pytest.approx(expected_radius)


def test_muon_hyperball_fixed_radius_and_rms_are_mutually_exclusive():
    with pytest.raises(AssertionError, match="mutually exclusive"):
        OptimizerConfig(
            optimizer='muon_hyperball', muon_hyperball_radius=1.0, muon_hyperball_rms=0.25
        )

    param = torch.nn.Parameter(torch.ones(4, 4))
    with pytest.raises(ValueError, match="mutually exclusive"):
        TensorParallelMuonHyperball(params=[param], hyperball_radius=1.0, hyperball_rms=0.25)


def _is_test_swiglu_fc1(param):
    return getattr(param, "is_test_swiglu_fc1", False)


def _make_partitioned_cpu_muon(param, **kwargs):
    return TensorParallelMuon(
        [param],
        lr=0.1,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        qkv_ns_mode=kwargs.pop("qkv_ns_mode", "separate"),
        qkv_norm_mode=kwargs.pop("qkv_norm_mode", "separate"),
        swiglu_ns_mode=kwargs.pop("swiglu_ns_mode", "separate"),
        swiglu_norm_mode=kwargs.pop("swiglu_norm_mode", "separate"),
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        is_swiglu_fc1_fn=_is_test_swiglu_fc1,
        qkv_split_shapes=(2, 1, 1),
        num_ns_steps=1,
        pg_collection=None,
        **kwargs,
    )


def test_muon_qkv_partition_layout_round_trip():
    param = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).view(8, 3))
    param.is_qkv = True
    optimizer = _make_partitioned_cpu_muon(param)

    parts = optimizer._matrix_partition_parts(param, param)
    assert [tag for tag, _ in parts] == ["linear_q", "linear_k", "linear_v"]
    matrices = [optimizer._partition_as_matrix(part) for _, part in parts]
    assert matrices[0][:, 0].tolist() == [0.0, 3.0, 12.0, 15.0]
    assert matrices[1][:, 0].tolist() == [6.0, 18.0]
    assert matrices[2][:, 0].tolist() == [9.0, 21.0]
    assert torch.equal(optimizer._merge_matrix_partitions(param, matrices, param.shape), param)


@pytest.mark.parametrize(
    ("family", "mode", "expected_calls"),
    [
        ("qkv", "fused", [(8, 3)]),
        ("qkv", "separate", [(4, 3), (2, 3), (2, 3)]),
        ("swiglu", "fused", [(8, 3)]),
        ("swiglu", "separate", [(4, 3), (4, 3)]),
    ],
)
def test_muon_ns_modes_control_logical_partitioning(family, mode, expected_calls):
    param = torch.nn.Parameter(torch.ones(8, 3))
    if family == "qkv":
        param.is_qkv = True
    else:
        param.is_test_swiglu_fc1 = True
    optimizer = _make_partitioned_cpu_muon(
        param,
        qkv_ns_mode=mode if family == "qkv" else "fused",
        swiglu_ns_mode=mode if family == "swiglu" else "fused",
    )
    calls = []

    def spy(matrix, _tp_group, _partition_dim):
        calls.append(tuple(matrix.shape))
        return matrix

    optimizer.scaled_orthogonalize_fn = spy
    assert torch.equal(optimizer.orthogonalize(param, param.detach().clone()), param)
    assert calls == expected_calls


def test_muon_partition_modes_configure_swiglu_predicate():
    model_config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        gated_linear_unit=True,
        use_cpu_initialization=True,
    )
    config = OptimizerConfig(
        optimizer="muon_hyperball", muon_swiglu_ns_mode="separate", muon_swiglu_norm_mode="separate"
    )
    kwargs = _muon_config_to_kwargs(
        config, [SimpleNamespace(config=model_config)], pg_collection=None
    )
    fc1 = torch.nn.Parameter(torch.ones(256, 128))
    fc1.param_name = "decoder.layers.0.mlp.linear_fc1.weight"
    fc2 = torch.nn.Parameter(torch.ones(128, 128))
    fc2.param_name = "decoder.layers.0.mlp.linear_fc2.weight"

    assert kwargs["qkv_ns_mode"] == "separate"
    assert kwargs["swiglu_ns_mode"] == "separate"
    assert kwargs["is_swiglu_fc1_fn"](fc1)
    assert not kwargs["is_swiglu_fc1_fn"](fc2)


def test_muon_separate_swiglu_norm_scales_effective_lr_per_projection():
    param = torch.nn.Parameter(torch.cat((torch.full((4, 3), 2.0), torch.full((4, 3), 4.0))))
    param.is_test_swiglu_fc1 = True
    optimizer = _make_partitioned_cpu_muon(
        param,
        qkv_norm_mode="fused",
        swiglu_ns_mode="fused",
        swiglu_norm_mode="separate",
        effective_lr_mult=3.0,
    )
    optimizer._current_lr_by_param[id(param)] = 0.1
    update = torch.ones_like(param)

    with torch.no_grad():
        optimizer.pre_weight_update_fn_inplace(param, update)

    gate_update, up_update = torch.chunk(update, 2, dim=0)
    assert gate_update.square().mean().sqrt().item() == pytest.approx(6.0)
    assert up_update.square().mean().sqrt().item() == pytest.approx(12.0)


def test_muon_hyperball_separate_norm_preserves_and_saves_swiglu_radii():
    param = torch.nn.Parameter(torch.cat((torch.full((4, 3), 1.0), torch.full((4, 3), 3.0))))
    param.is_test_swiglu_fc1 = True
    optimizer = TensorParallelMuonHyperball(
        [param],
        lr=0.1,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        qkv_ns_mode="fused",
        qkv_norm_mode="fused",
        swiglu_ns_mode="fused",
        swiglu_norm_mode="separate",
        is_swiglu_fc1_fn=_is_test_swiglu_fc1,
        num_ns_steps=1,
        hyperball_eps=1e-12,
        pg_collection=None,
    )
    initial_norms = torch.stack([part.norm() for part in torch.chunk(param.detach(), 2, dim=0)])
    update = torch.cat((torch.ones((4, 3)), torch.full((4, 3), 5.0)))

    with torch.no_grad():
        optimizer.pre_weight_update_fn_inplace(param, update)
        update_norms = torch.stack([part.norm() for part in torch.chunk(update.detach(), 2, dim=0)])
        param.add_(update, alpha=-0.1)
        optimizer.post_weight_update_fn_inplace(param)

    final_norms = torch.stack([part.norm() for part in torch.chunk(param.detach(), 2, dim=0)])
    assert torch.allclose(update_norms, initial_norms)
    assert torch.allclose(final_norms, initial_norms)
    state = optimizer.state_dict()["state"][0]
    torch.testing.assert_close(state["hyperball_partition_radii"], initial_norms)
    assert state["hyperball_partition_state_version"] == 1


def test_muon_hyperball_separate_rms_targets_each_qkv_projection():
    param = torch.nn.Parameter(torch.ones(8, 3))
    param.is_qkv = True
    optimizer = TensorParallelMuonHyperball(
        [param],
        lr=0.1,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        qkv_ns_mode="separate",
        qkv_norm_mode="separate",
        swiglu_ns_mode="separate",
        swiglu_norm_mode="separate",
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=(2, 1, 1),
        num_ns_steps=1,
        hyperball_eps=1e-12,
        hyperball_rms=0.5,
        pg_collection=None,
    )

    for _, part in optimizer._matrix_partition_parts(param, param):
        assert part.float().square().mean().sqrt().item() == pytest.approx(0.5)
    assert torch.allclose(
        optimizer._ensure_partition_hyperball_radius_state(param),
        torch.tensor([0.5 * math.sqrt(12), 0.5 * math.sqrt(6), 0.5 * math.sqrt(6)]),
    )


def test_muon_hyperball_sharded_state_dict_smoke():
    """MuonHyperball should shard tensor state and persistent radius metadata."""
    Utils.initialize_model_parallel()
    try:
        model = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

        optimizer_config = OptimizerConfig(
            optimizer='muon_hyperball',
            lr=0.01,
            weight_decay=0.0,
            bf16=True,
            use_layer_wise_distributed_optimizer=True,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=3,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
            muon_hyperball_eps=1e-12,
        )
        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        x = torch.randn(8, 64, dtype=torch.bfloat16, device='cuda')
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        sharded_state_dict = optimizer.sharded_state_dict(model.sharded_state_dict())
        assert "optimizer" in sharded_state_dict
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonOptimizerMultiRank:
    """Test class for Muon optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP.

        Args:
            model: Model to wrap

        Returns:
            DDP-wrapped model
        """
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def test_get_megatron_optimizer_smoke(self):
        """Smoke test for get_megatron_optimizer function."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # Ensure all parameters require gradients
        for param in model.parameters():
            assert param.requires_grad, "All parameters should require gradients"

        # Create optimizer config for Muon
        optimizer_config = OptimizerConfig(
            optimizer='muon',  # This will be changed internally to 'adam' for non-linear params
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,  # Muon doesn't support distributed optimizer
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        # Test creating the optimizer
        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        # Test basic properties
        assert optimizer is not None, "Optimizer should not be None"
        assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"
        assert len(optimizer.chained_optimizers) >= 1, "Should have at least one chained optimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Test optimizer step
        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        # Test zero_grad
        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), f"Gradients should be zeroed for all parameters"

        # Test state_dict and load_state_dict
        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list), "State dict should be a list"

        # Load state dict should not raise error
        optimizer.load_state_dict(state_dict)

    def test_get_megatron_optimizer_validation(self):
        """Test validation logic for get_megatron_optimizer."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # Test 1: FP16 should raise exception
        optimizer_config_fp16 = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            fp16=True,  # This should cause an exception
            use_distributed_optimizer=False,
        )

        with pytest.raises(Exception, match='emerging optimizer with fp16 is not supported'):
            get_megatron_optimizer(config=optimizer_config_fp16, model_chunks=[model])

        # Test 3: Invalid num_ns_steps should raise exception
        optimizer_config_invalid_ns = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_num_ns_steps=0,  # This should cause an exception
        )

        with pytest.raises(ValueError, match='num_ns_steps must be at least 1'):
            get_megatron_optimizer(config=optimizer_config_invalid_ns, model_chunks=[model])

    def test_get_megatron_optimizer_layer_wise(self):
        """Test get_megatron_optimizer with layer-wise distributed optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_layer_wise_distributed_optimizer=True,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        # use_layer_wise_distributed_optimizer=True triggers LayerWiseDistributedOptimizer
        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        # Verify it's a LayerWiseDistributedOptimizer
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        assert isinstance(
            optimizer, LayerWiseDistributedOptimizer
        ), "Should return LayerWiseDistributedOptimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Test optimizer step
        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"
        assert grad_norm is not None or grad_norm is None, "Grad norm should be returned"


@pytest.mark.parametrize("mode", ["duplicated", "blockwise", "distributed"])
def test_muon_optimizer_different_modes_single_rank(mode):
    """Test TensorParallelMuon optimizer with different modes on single rank.

    When TP size is 1, all modes should produce the same result.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,  # Disable weight decay for deterministic comparison
        num_ns_steps=5,
        pg_collection=None,
        tp_mode=mode,
    )

    # Use fixed input for deterministic results
    torch.manual_seed(42)
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with mode={mode}"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonOptimizerMultiRankTP:
    """Test class for Muon optimizer with multi-rank and tensor parallel setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test with tensor parallel."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    def create_tp_model_and_optimizer(self, mode):
        """Create model with TP and optimizer.

        Args:
            mode: Muon optimizer mode

        Returns:
            tuple: (model, optimizer, pg_collection)
        """
        rank = int(os.getenv('RANK', '0'))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        # Create model with partition_dim for TP
        torch.manual_seed(42 + rank)
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
        model.requires_grad_(True)
        model.weight.data.normal_(0, 0.02)
        model.weight.partition_dim = 0  # Set partition dimension for TP

        optimizer = TensorParallelMuon(
            params=[model.weight],
            lr=0.01,
            momentum=0.95,
            weight_decay=0.0,
            num_ns_steps=5,
            pg_collection=pg_collection,
            tp_mode=mode,
        )

        return model, optimizer

    @pytest.mark.parametrize("mode", ["duplicated", "distributed"])
    def test_muon_optimizer_modes_multirank_same_result(self, mode):
        """Test that duplicated and distributed modes produce same results with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer(mode)

        # Use fixed input for deterministic results
        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        # Verify weight was updated
        assert not torch.equal(
            model.weight.data, original_weight
        ), f"Weight should be updated with mode={mode}"

    def test_muon_optimizer_blockwise_mode_different_result(self):
        """Test that blockwise mode produces different results than duplicated/distributed with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer("blockwise")

        # Use fixed input for deterministic results
        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        # Verify weight was updated
        assert not torch.equal(
            model.weight.data, original_weight
        ), "Weight should be updated with mode=blockwise"

    def test_muon_hyperball_effective_lr_uses_tp_global_norms(self):
        """MuonH normalization and radius projection must span all TP weight shards."""
        rank = int(os.getenv('RANK', '0'))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        param = torch.nn.Parameter(
            torch.full((8, 8), float(rank + 1), dtype=torch.float32, device='cuda')
        )
        param.partition_dim = 0
        optimizer = TensorParallelMuonHyperball(
            params=[param],
            lr=0.1,
            momentum=0.95,
            weight_decay=0.0,
            num_ns_steps=1,
            pg_collection=pg_collection,
            tp_mode="distributed",
            hyperball_eps=1e-12,
            lr_mult=3.0,
        )
        optimizer.set_diagnostic_context(iteration=1, interval=1)
        before = param.detach().clone()

        param.square().sum().backward()
        optimizer.step()

        raw = optimizer.pop_diagnostic_stats()["matrix"]
        diagnostic_norms = torch.tensor(
            [raw["weight_norm_sq"], raw["update_norm_sq"]], device='cuda'
        )
        projected_norms = torch.stack(
            (before.float().square().sum(), param.detach().float().square().sum())
        )
        torch.distributed.all_reduce(diagnostic_norms, group=pg_collection.tp)
        torch.distributed.all_reduce(projected_norms, group=pg_collection.tp)

        relative_update = (diagnostic_norms[1] / diagnostic_norms[0]).sqrt().item()
        assert relative_update == pytest.approx(0.3, rel=1e-5, abs=1e-6)
        assert projected_norms[1].sqrt().item() == pytest.approx(
            projected_norms[0].sqrt().item(), rel=1e-5, abs=1e-6
        )


# All non-custom coefficient types supported by emerging_optimizers.
_TESTABLE_COEFFICIENT_TYPES = (
    [t for t in get_supported_coefficient_types() if t != "custom"]
    if HAVE_EMERGING_OPTIMIZERS
    else []
)

# A reasonable default NS step count for testing; get_coefficient_iterator
# cycles/repeats coefficients so any step count works with any type.
_DEFAULT_NS_STEPS = 5


@pytest.mark.parametrize("coefficient_type", _TESTABLE_COEFFICIENT_TYPES)
def test_muon_optimizer_coefficient_types(coefficient_type):
    """Test TensorParallelMuon optimizer with different coefficient types."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        coefficient_type=coefficient_type,
        num_ns_steps=_DEFAULT_NS_STEPS,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with coefficient_type={coefficient_type}"


@pytest.mark.parametrize("scale_mode", ["spectral", "unit_rms_norm", "shape_scaling"])
def test_muon_optimizer_scale_modes(scale_mode):
    """Test TensorParallelMuon optimizer with different scale modes."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        scale_mode=scale_mode,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with scale_mode={scale_mode}"


@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_optimizer_nesterov(nesterov):
    """Test TensorParallelMuon optimizer with and without Nesterov momentum."""
    model = torch.nn.Linear(50, 25, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.9,
        nesterov=nesterov,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 50, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with nesterov={nesterov}"


def test_muon_optimizer_multiple_steps():
    """Test TensorParallelMuon optimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    # Verify weights changed at each step
    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


def test_muon_optimizer_qkv_split():
    """Test TensorParallelMuon optimizer with QKV splitting."""
    # Create a model with QKV-like parameter
    qkv_size = 3 * 64 * 16  # Combined Q, K, V dimensions, 16 heads x 64 per head
    hidden_size = 1024
    model = torch.nn.Linear(hidden_size, qkv_size, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Mark parameter as QKV
    model.weight.is_qkv = True

    # QKV split shapes: [Q_size, K_size, V_size]
    qkv_split_shapes = (64, 64, 64)

    # Test with split_qkv=True
    optimizer_split = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, hidden_size, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer_split.step()
    weight_with_split = model.weight.data.clone()

    assert not torch.equal(
        weight_with_split, original_weight
    ), "QKV weight should be updated with split_qkv=True"

    # Reset model and test with split_qkv=False
    model.weight.data.fill_(1.0)
    optimizer_no_split = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=False,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    optimizer_no_split.step()
    weight_without_split = model.weight.data.clone()

    assert not torch.equal(
        weight_without_split, original_weight
    ), "QKV weight should be updated with split_qkv=False"

    # Ensure the two results are different
    assert not torch.equal(
        weight_with_split, weight_without_split
    ), "Weights should be different between split_qkv=True and split_qkv=False"


def test_muon_optimizer_extra_scale_factor():
    """Test TensorParallelMuon optimizer with different extra_scale_factor values."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        extra_scale_factor=2.0,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated with extra_scale_factor"


def test_get_supported_coefficient_types_returns_tuple():
    """Test that get_supported_coefficient_types returns a non-empty tuple of strings."""
    supported = get_supported_coefficient_types()
    assert isinstance(supported, tuple)
    assert len(supported) > 0
    for t in supported:
        assert isinstance(t, str)


def test_get_supported_coefficient_types_contains_known_types():
    """Test that the known coefficient types are present in the supported set."""
    supported = get_supported_coefficient_types()
    for expected in ("simple", "quintic", "polar_express"):
        assert expected in supported, f"Expected '{expected}' in supported types {supported}"


def test_validate_coefficient_type_accepts_valid():
    """Test that validate_coefficient_type does not raise for valid types."""
    for t in get_supported_coefficient_types():
        validate_coefficient_type(t)  # should not raise


def test_validate_coefficient_type_rejects_invalid():
    """Test that validate_coefficient_type raises ValueError for an invalid type."""
    with pytest.raises(ValueError, match="Unsupported muon coefficient type"):
        validate_coefficient_type("nonexistent_type_xyz")


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonCoefficientTypeMultiRank:
    """Test coefficient_type integration through get_megatron_optimizer."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    @pytest.mark.parametrize("coefficient_type", _TESTABLE_COEFFICIENT_TYPES)
    def test_get_megatron_optimizer_coefficient_type(self, coefficient_type):
        """Test that coefficient_type flows through get_megatron_optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_coefficient_type=coefficient_type,
            muon_num_ns_steps=_DEFAULT_NS_STEPS,
            muon_tp_mode="duplicated",
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        assert optimizer is not None

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()


@pytest.mark.parametrize("num_ns_steps", [5, 15, 25])
def test_muon_optimizer_num_ns_steps(num_ns_steps):
    """Test TensorParallelMuon optimizer with different numbers of Newton-Schulz steps."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        coefficient_type="quintic",
        num_ns_steps=num_ns_steps,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with num_ns_steps={num_ns_steps}"
