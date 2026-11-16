# Source and commit audit

## Fork point

Puro-Megatron forks NVIDIA Megatron-LM at tag `core_v0.16.0`, exact commit:

```text
3bec9aa97dda898d16ff5a89bac0ed2b6682b172
```

The internal branch lineage was `effective_lr` ->
`numeric_efflr_wd_experiment` -> `scaling_ladder`, whose inspected tips were
`57dd9dc23`, `8b9344a67`, and `5df860dc7`. The release does not fork from any
of those tips. It replays only the selected changes onto the clean NVIDIA tag.

## Public patch series

The authoritative list is `git log --reverse core_v0.16.0..main`. The commits
are grouped below by capability. Curated commits preserve the primary original
author. Source SHAs are kept in this audit document rather than repeated in
public commit messages.

| Public commit subject | Source commit(s) | Capability |
| --- | --- | --- |
| `feat(data): add PROM packed NPY pretraining support` | `dd8b4e614`, `ee94062b3`, `aeed231b2`, `5d100adbe`, `4bcd2f63f`, `324b68046`, `af4ccff8f`, `907c4e669`, `e89eb65d8`, `bb90af489`, `d70af3702` | PROM NPY data, packed THD batches, stable resume indexing |
| `fix(data): route GPT batches through the TP data loader` | `fddde3891` | Route GPT batches through the custom TP loader |
| FP8 extra-state compatibility commits | `a071673f1`, `dbbebb670` | Transformer Engine FP8 extra-state compatibility |
| `fix(checkpoint): restore runtime LR bounds after loading arguments` | `0ff1b4fd2` | Restore runtime LR bounds after checkpoint override |
| `feat(training): reset scheduler and dataloader progress on phase resume` | `9239c1e90` | Reset scheduler/data progress for phase resume |
| non-persistent checkpoint retention commit | `12e40b001` | Non-persistent checkpoint retention |
| explicit persistent checkpoint selection commit | `34a35709f` | Honor explicit persistent checkpoint step |
| MuonHyperball and effective-LR commit | `6e4a9a6d4`, `3e76e52e2`, `87df41c57`, `6a9ae0869` | MuonHyperball, AdamW routing, effective-LR diagnostics |
| memory-balanced layer-wise optimizer commit | `1e0a5aab3`, `9d0cc6753`, `edc7f437a`, `75d74a7f4` | Memory-balanced layer-wise distributed optimizer |
| strict effective-LR commit | `57dd9dc23` | Strict effective-LR control |
| power-schedule commit | `2f7d55872` | Open-ended power LR schedule |
| rerun dummy-batch commit | `1aa8d4cc3` | Custom dummy-batch consumption during rerun skip |
| corrupt-data synchronization commits | `636cc5a40`, `1b7d9c26a` | Corrupt-data skip and cross-rank request synchronization |
| theoretical-FLOPs commits | `55666e70d`, `38f66d6e5` | Startup theoretical FLOPs report and TP/SP correction |
| `feat(optimizer): add fixed RMS targets for MuonHyperball` | `02494e215` | Fixed RMS converted to a TP-global Frobenius radius |
| `fix(optimizer): normalize fused matrices by logical projection` | `149a1d1f7` | Separate QKV/SwiGLU normalization and Newton-Schulz modes |
| release-only test alignment commit | release-only | Align stale FLOPs formatting tests; no formula change |

## Full-bandwidth feedback extension

The feedback extension is curated onto `puro-2b` commit
`a7b80e873a0b5e1820ae425b9abd1b9ec578dc5c`. Its training reference is the
frozen Dolma3 fixed-K2 mean-loss implementation from main repository commit
`60ae815114dd49135528c69e34e107867c97f87d` and scripts commit
`5b31e7d701212f62305582a95ec8a9b0b5dc7ad1`.

The selected code includes causal full-layer feedback, mean K1/K2 NLL,
chunked checkpointed output loss, strict HF initialization, persistent
MuonHyperball partition radii, and shared-parameter DDP/checkpoint fixes.
Source files are adapted to the existing public interfaces; unrelated
research modes and cluster launch paths remain excluded. See
[Full-bandwidth feedback](full-bandwidth-feedback.md) for the supported
configuration and migration checks.

## Deliberately excluded from the base release

- `numeric_efflr_wd_experiment`: two-stage and numeric WD sweeps, instant
  checkpoint plans, TE runtime monkey patches, and experimental logging.
- `scaling_ladder`: SFT/curriculum work, evaluation suites, long-context
  experiments, gated attention/norm/activation variants, and data ablations.
- hardware-specific DCU ports, profiler output, cached datasets, checkpoints,
  and run artifacts.

These exclusions keep the public repository reviewable while retaining the
code paths exercised by the release regression runs.
