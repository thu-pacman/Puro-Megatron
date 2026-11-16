# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Fail closed on missing steps, changed inputs/configuration, or NLL error > 1e-4."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

METRICS = ('k1_nll', 'k2_nll', 'objective_nll')
TOLERANCE = 1e-4


def read_steps(path):
    """Read a trajectory, rejecting duplicate steps and non-finite losses."""
    records = {}
    for line in (path / 'training.jsonl').read_text().splitlines():
        value = json.loads(line)
        iteration = value['iteration']
        if iteration in records:
            raise ValueError(f'Duplicate iteration {iteration} in {path}')
        for key in METRICS:
            if not isinstance(value.get(key), (int, float)) or not math.isfinite(value[key]):
                raise ValueError(f'Invalid {key} at iteration {iteration} in {path}')
        if abs(value['objective_nll'] - (value['k1_nll'] + value['k2_nll']) / 2) > 2e-6:
            raise ValueError(f'Objective is not the K1/K2 mean at iteration {iteration}')
        records[iteration] = value
    return records


def compare(reference, candidate, start, steps, output, full_config=False):
    """Compare each selected global step; no averaging can hide a failed step."""
    left, right = read_steps(reference), read_steps(candidate)
    expected = set(range(start + 1, start + steps + 1))
    if set(right) != expected or not expected.issubset(left):
        raise ValueError('Missing or unexpected iterations in the requested comparison interval')
    if full_config:
        manifests = [
            json.loads((directory / 'manifest.json').read_text())
            for directory in (reference, candidate)
        ]
        a, b = manifests
        for key in ('world_size', 'parameter_count', 'config', 'packages', 'environment'):
            if a[key] != b[key]:
                raise ValueError(f'Configuration mismatch for {key}')
        if a['world_size'] != 32 or a['parameter_count'] != 2040132608:
            raise ValueError('Expected the actual 32-GPU Puro-2B feedback model')
        required = dict(
            full_bandwidth='True',
            feedback_passes='2',
            feedback_loss_reduction='mean',
            feedback_fusion='residual_glu',
            micro_batch_size='2',
            global_batch_size='2048',
            seq_length='4096',
            fp8='e4m3',
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
        )
        for key, value in required.items():
            if a['config'][key] != value:
                raise ValueError(f'Unexpected production setting: {key}={a["config"][key]}')
        if a['environment']['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] != '1':
            raise ValueError(
                'The production check must retain the original attention configuration'
            )
    rows = []
    for iteration in sorted(expected):
        a, b = left[iteration], right[iteration]
        if full_config:
            if a.get('data') != b.get('data') or len(a.get('data', [])) != 32:
                raise ValueError(f'Input mismatch at iteration {iteration}')
            if any(item['rows'] != 64 for item in a['data']):
                raise ValueError(f'Unexpected per-rank sample count at iteration {iteration}')
        row = dict(iteration=iteration)
        for key in METRICS:
            row[key + '_reference'] = a[key]
            row[key + '_candidate'] = b[key]
            row[key + '_absolute_error'] = abs(a[key] - b[key])
        row['grad_norm_absolute_error'] = abs(a['grad_norm'] - b['grad_norm'])
        rows.append(row)
    maximum = {key: max(row[key + '_absolute_error'] for row in rows) for key in METRICS}
    result = dict(
        passed=all(value <= TOLERANCE for value in maximum.values()),
        tolerance=TOLERANCE,
        error_kind='absolute_per_global_step',
        start=start,
        steps=steps,
        max_absolute_error=maximum,
        max_gradient_norm_difference=max(row['grad_norm_absolute_error'] for row in rows),
        compared_original_fp8_configuration=full_config,
    )
    if full_config:
        result['performance'] = {}
        for label, records in [('reference', left), ('candidate', right)]:
            stable = [records[i] for i in sorted(expected) if i > start + 4]
            stable = stable or [records[i] for i in sorted(expected)]
            seconds = statistics.median(record['batch_seconds'] for record in stable)
            result['performance'][label] = dict(
                median_batch_seconds=seconds,
                median_input_tokens_per_second=stable[0]['input_tokens'] / seconds,
                peak_allocated_bytes=max(record['peak_allocated_bytes'] for record in stable),
                timing_includes_input_hash_instrumentation=True,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    with output.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-config', action='store_true')
    args = parser.parse_args()
    result = compare(
        args.reference, args.candidate, args.start, args.steps, args.output, args.full_config
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
