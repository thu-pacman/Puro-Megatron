# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""The migration gate must reject an isolated error, missing step, or NaN."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'feedback_comparison', Path(__file__).with_name('compare_puro_feedback_runs.py')
)
COMPARISON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARISON)


def write_run(directory, records):
    directory.mkdir()
    (directory / 'training.jsonl').write_text(
        ''.join(json.dumps(record) + '\n' for record in records)
    )
    return directory


@pytest.fixture
def records():
    return [
        dict(iteration=i, k1_nll=2.5, k2_nll=2.4, objective_nll=2.45, grad_norm=1.0)
        for i in range(1, 25)
    ]


def test_matching_trajectory_and_resume_suffix(tmp_path, records):
    source = write_run(tmp_path / 'source', records)
    candidate = write_run(tmp_path / 'candidate', records[12:])
    result = COMPARISON.compare(source, candidate, 12, 12, tmp_path / 'result')
    assert result['passed']
    assert all(value == 0 for value in result['max_absolute_error'].values())


@pytest.mark.parametrize('path', ['k1_nll', 'k2_nll'])
def test_single_failed_step_is_not_hidden_by_averaging(tmp_path, records, path):
    source = write_run(tmp_path / 'source', records)
    changed = copy.deepcopy(records)
    changed[13][path] += 0.003
    changed[13]['objective_nll'] += 0.0015
    candidate = write_run(tmp_path / 'candidate', changed)
    assert not COMPARISON.compare(source, candidate, 0, 24, tmp_path / 'result')['passed']


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'nan', 'wrong_objective'])
def test_incomplete_or_invalid_records_fail_closed(tmp_path, records, corruption):
    source = write_run(tmp_path / 'source', records)
    changed = copy.deepcopy(records)
    if corruption == 'missing':
        changed.pop(13)
    elif corruption == 'duplicate':
        changed.append(changed[13])
    elif corruption == 'nan':
        changed[13]['k2_nll'] = float('nan')
    else:
        changed[13]['objective_nll'] = changed[13]['k1_nll']
    candidate = write_run(tmp_path / 'candidate', changed)
    with pytest.raises(ValueError):
        COMPARISON.compare(source, candidate, 0, 24, tmp_path / 'result')


@pytest.mark.parametrize(
    'deterministic,allow_nondeterministic,fp8,accepted',
    [
        (False, '1', 'e4m3', True),
        (True, '0', 'e4m3', True),
        (False, '0', 'e4m3', False),
        (True, '1', 'e4m3', False),
        (True, '0', 'bf16', False),
    ],
)
def test_native_gate_checks_attention_mode_without_relaxing_fp8(
    tmp_path, records, deterministic, allow_nondeterministic, fp8, accepted
):
    native = copy.deepcopy(records)
    for row in native:
        row.update(
            data=[dict(rows=64, sha256='a' * 64)] * 32,
            batch_seconds=30.0,
            input_tokens=8388608,
            peak_allocated_bytes=1 << 30,
        )
    source = write_run(tmp_path / 'source', native)
    candidate = write_run(tmp_path / 'candidate', native)
    manifest = dict(
        world_size=32,
        parameter_count=2040132608,
        packages={},
        environment=dict(NVTE_ALLOW_NONDETERMINISTIC_ALGO=allow_nondeterministic),
        config=dict(
            full_bandwidth='True',
            feedback_passes='2',
            feedback_loss_reduction='mean',
            feedback_fusion='residual_glu',
            micro_batch_size='2',
            global_batch_size='2048',
            seq_length='4096',
            fp8=fp8,
            fp8_recipe='blockwise',
            bf16='True',
            recompute_method='block',
            recompute_num_layers='17',
            output_loss_chunk_size='2048',
            tensor_model_parallel_size='1',
            pipeline_model_parallel_size='1',
            context_parallel_size='1',
            muon_momentum='0.9',
            muon_hyperball_lr_mult='2.0',
        ),
    )
    for directory in (source, candidate):
        (directory / 'manifest.json').write_text(json.dumps(manifest))
    if accepted:
        result = COMPARISON.compare(
            source, candidate, 0, 24, tmp_path / 'result', True, deterministic
        )
        assert result['passed'] and result['tolerance'] == 1e-3
        assert result['native_fp8_configuration_checked']
        assert result['attention_deterministic'] == deterministic
    else:
        with pytest.raises(ValueError):
            COMPARISON.compare(source, candidate, 0, 24, tmp_path / 'result', True, deterministic)
