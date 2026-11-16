# Full-bandwidth feedback continuation

This extension implements the fixed two-pass model used for Puro-2B Dolma3
continuation. Both passes use the same embedding, all decoder layers, final
normalization, and output head. Training minimizes the arithmetic mean of the
two token-level negative log likelihoods:

\[
L = (L_{K=1} + L_{K=2}) / 2.
\]

## Model and causality

Let `e[t]` be the original token embedding and `h1[t]` the normalized decoder
output from the first pass. The second pass starts from

```text
value[t] = hidden_projection(RMSNorm(h1[t - 1]))
gate[t]  = sigmoid(token_projection(RMSNorm(e[t])))
e2[t]    = e[t] + value[t] * gate[t]
h2       = decoder(e2)  # the same 28 layers, with shared parameters
```

The shift stays inside each sample and packed document. The first token in
each document retains its original embedding. Packed THD boundaries are
honored, including padded cumulative sequence lengths. The feedback remains
connected to autograd: the second loss also differentiates through the first
decoder pass. Attention and MLP residual coefficients remain one.

`hidden_projection` is initialized to zero; `token_projection` uses the model
initializer. Consequently, the untrained BF16 feedback model has the same
single-pass output as the baseline. The feedback module adds 8,392,704
parameters at hidden size 2,048. The full model has 2,040,132,608 parameters.
The feedback parameters use AdamW with zero weight decay. MuonHyperball handles
the ordinary matrix parameters; its partition radii are persistent optimizer
state and survive checkpoint reloads.

## Supported interface

Enable `--full-bandwidth --feedback-passes 2 --feedback-fusion residual_glu
--feedback-loss-reduction mean`. These settings select a fixed two-pass mean
objective. Jitter, prefix mixing, sampled passes, middle-layer recurrence, and
other historical research modes are outside this release.

In Python, `GPTModel.forward(..., feedback_passes=1)` provides the ordinary
single-pass path during evaluation. With labels, the default is two passes;
`feedback_passes=2` returns the mean token loss. Without labels, the default is
one pass; explicitly selecting two passes returns the final-pass logits.
Two-pass incremental inference with a KV cache is rejected. Training requires
two passes so that parameters and the objective cannot silently change between
batches.

The training log keeps `lm loss` as the K1 diagnostic for compatibility. Use
`fbt objective` for the optimized mean and `feedback loss/pass1` and
`feedback loss/pass2` for the individual paths.

Training FLOPs include both passes and the feedback projections, following the
existing convention of excluding recomputation and optimizer work. Cumulative
FLOPs restored from older checkpoints retain their historical accounting;
use measured batch time and input tokens when comparing such runs.

This implementation supports data parallel training with TP=PP=CP=1. The
configuration rejects unsupported combinations instead of implying they have
been validated. FP8 additionally requires BF16 parameters, blockwise scaling,
and the explicit `--experimental-feedback-fp8` flag. Disable Transformer
Engine's weight transpose cache when reusing the decoder within a forward.
The implementation sets this automatically for feedback models.

## Continuation recipe

Use [run_puro_2b_feedback.sh](../examples/puro/run_puro_2b_feedback.sh) inside
the same launcher environment as the existing Puro recipe. It takes portable
paths from the environment:

```bash
export TRAIN_DATA_PATH=/datasets/dolma3/npy_4097/train
export VALID_DATA_PATH=/datasets/dolma3/npy_4097/valid
export DATA_CACHE_PATH=/runs/puro-feedback/data_cache
export SAVE_PATH=/runs/puro-feedback/checkpoints
export INIT_FROM_HF=/models/Puro-2B-Base
export PYTHON=python
export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# Run this command on all four nodes, with the corresponding NODE_RANK.
torchrun --nnodes=4 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  --no-python bash examples/puro/run_puro_2b_feedback.sh
```

For InfiniBand, set `NCCL_NET=IB`, `NCCL_IB_DISABLE=0`, and `NCCL_IB_HCA` to
the local adapter names in the launcher environment. Adapter names and Slurm
node lists are deployment details; they are not embedded in the public recipe.

Resume by unsetting `INIT_FROM_HF` and setting `LOAD_PATH` to a native checkpoint
directory. HF initialization strictly validates architecture and weight keys;
the feedback module is the only allowed missing component. Native feedback
checkpoints validate the feedback position, fusion, loss semantics, and
per-rank RNG world size. Loading an older checkpoint with a different
objective raises an error. Do not reinterpret such checkpoints as mean-loss
continuation.

The example preserves the continuation settings:

| Setting | Value |
| --- | --- |
| Parallelism | DP 32, TP 1, PP 1, CP 1 |
| Sequence / micro / global batch | 4,096 / 2 / 2,048 |
| Microbatches per optimizer step | 32 per rank |
| Decoder precision | FP8 E4M3, blockwise |
| Parameters / accumulated gradients | BF16 / FP32 |
| Activation recomputation | First 17 decoder layers, on each pass |
| Output head | 2,048-token chunks, loss checkpointing |
| Optimizer | Memory-balanced layer-wise MuonHyperball + sharded AdamW |
| Muon momentum / epsilon | 0.9 / 1e-12 |
| Muon / AdamW LR ratio | 2 |
| AdamW LR | 0.005 to 0.000005 |
| Schedule | 500-step warmup, linear decay to step 13,892 |
| Checkpoint / validation | Every 212 optimizer steps |
| Validation size | 1,536 samples, subject to the validation data size |
| Seed | 1,234 |

The data loader consumes packed uint32 NPY rows of length 4,097, yielding
4,096 next-token targets. One optimizer step reads 8,388,608 input tokens.
The 13,892-step schedule covers 116,534,542,336 input tokens. These counts do
not multiply by the number of model passes or activation recomputations.

## Migration tests

`tests/unit_tests/models/test_latent_feedback.py` covers zero initialization,
the shift direction, packed boundaries, causality, explicit forward/backward
references, chunked loss, and shared-head fused gradient accumulation with
DDP. `test_feedback_optimizer.py` covers AdamW routing and MuonHyperball radius
and momentum restoration.

`tests/functional_tests/puro_feedback_reference.py` provides a small BF16
trajectory and split-run checkpoint test. The separate native-training probe
records all three losses, exact input hashes on every rank, initialization
hashes, optimizer snapshots, package versions, synchronized batch time, and
peak memory. It leaves the training computation unchanged. The comparison
tool rejects missing steps, changed inputs or configuration, non-finite loss,
or any per-step absolute loss error above 1e-3:

```bash
python tests/functional_tests/compare_puro_feedback_runs.py \
  /runs/reference /runs/candidate --start 0 --steps 24 \
  --full-config --output /runs/acceptance/fresh
```

The native comparison retains the actual FP8 attention settings. Its wall
times include input fingerprint instrumentation and should be reported as
such. Measured migration results are recorded separately from the model
description; a BF16-only result does not establish FP8 trajectory parity.
