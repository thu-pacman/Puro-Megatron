#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${PURO_MEGATRON_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
python_bin=${PYTHON:-python}
train_data=${TRAIN_DATA_PATH:?set TRAIN_DATA_PATH}
valid_data=${VALID_DATA_PATH:?set VALID_DATA_PATH}
data_cache=${DATA_CACHE_PATH:?set DATA_CACHE_PATH}
save_path=${SAVE_PATH:?set SAVE_PATH}

model_args=(
  --use-mcore-models
  --num-layers 28 --hidden-size 2048 --ffn-hidden-size 6144
  --num-attention-heads 16 --group-query-attention --num-query-groups 8 --kv-channels 128
  --untie-embeddings-and-output-weights --padded-vocab-size 151936
  --seq-length 4096 --max-position-embeddings 4096
  --position-embedding-type rope --rotary-base 10000 --rotary-percent 1
  --normalization RMSNorm --norm-epsilon 1e-6 --qk-layernorm
  --swiglu --disable-bias-linear --attention-type thd
  --attention-dropout 0 --hidden-dropout 0 --init-method-std 0.02
  --full-bandwidth --feedback-passes 2
  --feedback-fusion residual_glu --feedback-loss-reduction mean
)

optimizer_args=(
  --optimizer muon_hyperball --muon-hyperball-scalar-optimizer adam
  --muon-momentum 0.9 --muon-hyperball-eps 1e-12 --muon-hyperball-lr-mult 2
  --muon-qkv-norm-mode separate --muon-qkv-ns-mode separate
  --muon-swiglu-norm-mode separate --muon-swiglu-ns-mode separate
  --weight-decay 0.1 --clip-grad 1 --adam-beta1 0.9 --adam-beta2 0.95
  --lr 0.005 --min-lr 0.000005 --lr-warmup-iters 500
  --lr-decay-style linear --lr-decay-iters 13892 --train-iters 13892
  --micro-batch-size 2 --global-batch-size 2048
  --bf16 --accumulate-allreduce-grads-in-fp32 --calculate-per-token-loss
  --use-distributed-optimizer --layerwise-optimizer-memory-balance
  --overlap-grad-reduce --overlap-param-gather
)

data_args=(
  --train-data-path "$train_data" --valid-data-path "$valid_data"
  --data-cache-path "$data_cache" --tokenizer-type NullTokenizer
  --vocab-size 151935 --eos-token-id 151643 --num-workers 0 --no-mmap-bin-files
)

runtime_args=(
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --context-parallel-size 1 --use-tp-pp-dp-mapping
  --fp8-format e4m3 --fp8-recipe blockwise --experimental-feedback-fp8
  --attention-backend flash
  --recompute-granularity full --recompute-method block --recompute-num-layers 17
  --output-loss-chunk-size 2048
  --cross-entropy-loss-fusion --cross-entropy-fusion-impl te
  --manual-gc --seed 1234 --rerun-mode disabled
  --log-interval 1 --log-throughput --eval-iters 1 --eval-interval 212
  --save-interval 212 --save "$save_path" --ckpt-format torch_dist
  --distributed-timeout-minutes 10
)

if [[ -n ${LOAD_PATH:-} ]]; then
  [[ -z ${INIT_FROM_HF:-} ]] || { echo "set only LOAD_PATH or INIT_FROM_HF" >&2; exit 2; }
  runtime_args+=(--load "$LOAD_PATH" --exit-on-missing-checkpoint)
else
  runtime_args+=(--init-from-hf "${INIT_FROM_HF:?set INIT_FROM_HF or LOAD_PATH}")
fi

exec "$python_bin" "$repo_root/pretrain_gpt.py" \
  "${model_args[@]}" "${optimizer_args[@]}" "${data_args[@]}" "${runtime_args[@]}" "$@"
