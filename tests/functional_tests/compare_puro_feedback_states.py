# Copyright (c) 2026 Puro-Megatron contributors. All rights reserved.

"""Inspect every owned parameter and optimizer state tensor from native probes.

State differences are diagnostic. The independent trajectory gate applies the
required absolute 1e-4 bound to each K1, K2, and mean loss at every global step.
"""

import argparse
import json
import math
from pathlib import Path

import torch


def compare(left, right, left_step, right_step, output, world_size=32, initial=False):
    """Compare all rank snapshots without materializing tensor-sized differences."""
    result = dict(
        reference_step=left_step,
        candidate_step=right_step,
        world_size=world_size,
        initial=initial,
        tensor_count=0,
        bitwise_equal_tensors=0,
        max_absolute_error={},
        changed_scalars=[],
        changed_initial_hashes=[],
        excluded_inactive_scheduler_fields={},
    )

    def visit(a, b, path):
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f'Tensor layout mismatch: {path}')
            result['tensor_count'] += 1
            if torch.equal(a, b):
                result['bitwise_equal_tensors'] += 1
                difference = 0.0
            else:
                difference = 0.0
                for x, y in zip(a.reshape(-1).split(1 << 20), b.reshape(-1).split(1 << 20)):
                    delta = (x.double() - y.double()).abs()
                    value = float(delta.max()) if delta.numel() else 0.0
                    if not math.isfinite(value):
                        raise ValueError(f'Non-finite state: {path}')
                    difference = max(difference, value)
            category = (
                'model'
                if '.model.' in path
                else (
                    'master_weights'
                    if path.endswith('.parameter')
                    else ('gradients' if path.endswith('.gradient') else 'optimizer_state')
                )
            )
            maximum = result['max_absolute_error']
            maximum[category] = max(maximum.get(category, 0.0), difference)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise ValueError(f'State keys differ: {path}')
            for key in a:
                visit(a[key], b[key], f'{path}.{key}')
        elif isinstance(a, (list, tuple)):
            if not isinstance(b, (list, tuple)) or len(a) != len(b):
                raise ValueError(f'State sequence differs: {path}')
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f'{path}.{index}')
        elif a != b:
            if initial and isinstance(a, str) and len(a) == 64:
                result['changed_initial_hashes'].append(path)
            else:
                result['changed_scalars'].append(dict(path=path, reference=a, candidate=b))

    for rank in range(world_size):
        if initial:
            a = json.loads((left / f'initial_{left_step:07d}_rank{rank:02d}.json').read_text())
            b = json.loads((right / f'initial_{right_step:07d}_rank{rank:02d}.json').read_text())
        else:
            a = torch.load(
                left / f'final_{left_step:07d}_rank{rank:02d}.pt',
                map_location='cpu',
                weights_only=False,
            )
            b = torch.load(
                right / f'final_{right_step:07d}_rank{rank:02d}.pt',
                map_location='cpu',
                weights_only=False,
            )
        # The iteration is reported separately, including for cross-step diagnostics.
        a.pop('iteration')
        b.pop('iteration')
        for label, state in [('reference', a), ('candidate', b)]:
            schedule = state['scheduler']
            excluded = {}
            if schedule['wd_incr_style'] == 'constant':
                for key in ('wd_recommend_power', 'wd_recommend_min_ratio'):
                    if key in schedule:
                        excluded[key] = schedule.pop(key)
            if schedule['lr_decay_style'] == 'linear':
                if 'lr_wsd_decay_power' in schedule:
                    excluded['lr_wsd_decay_power'] = schedule.pop('lr_wsd_decay_power')
            if rank == 0:
                result['excluded_inactive_scheduler_fields'][label] = excluded
        visit(a, b, f'rank{rank}')
        del a, b
    result['bitwise_equal'] = (
        result['tensor_count'] == result['bitwise_equal_tensors']
        and not result['changed_scalars']
        and not result['changed_initial_hashes']
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--reference-step', type=int, required=True)
    parser.add_argument('--candidate-step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', action='store_true')
    args = parser.parse_args()
    print(
        json.dumps(
            compare(
                args.reference,
                args.candidate,
                args.reference_step,
                args.candidate_step,
                args.output,
                initial=args.initial,
            ),
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
