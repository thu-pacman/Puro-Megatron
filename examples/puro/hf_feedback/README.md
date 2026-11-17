---
license: apache-2.0
base_model: thu-pacman/Puro-2B-Base
library_name: transformers
pipeline_tag: text-generation
datasets:
  - allenai/dolma3_dolmino_mix-100B-1125
tags:
  - puro
  - feedback
  - custom_code
  - base-model
---

# Puro-2B Dolma3 full-layer feedback, step 13,892

This checkpoint continues Puro-2B Base with **two full shared decoder passes**.
At token `t`, a Residual-GLU fuses the normalized first-pass representation at
`t-1` into the original embedding. Every pass uses all 28 decoder layers with
residual coefficient one. Training optimizes `(NLL_K1 + NLL_K2) / 2` with
connected gradients. The default inference output is **K2 logits**.

The final checkpoint contains 2,040,132,608 BF16 parameters, including
8,392,704 feedback parameters. Training used 13,892 optimizer steps and
116,534,542,336 input tokens, starting from `thu-pacman/Puro-2B-Base`.
The local Dolma3/Dolmino mixture is supported as ingredient 1 by discriminating
document-ID and sampled-text comparisons. The original upstream HF revision
and HF-to-Parquet selection manifest were not retained. The last source
Parquet was held out; validation used 1,536 fixed packed rows. Final native
validation NLL was K1 **1.30706394**, K2 **1.29178488**, mean **1.29942441**.

This is a base continuation model. Supply a raw completion prompt; no chat
template is applied by the example script.

## Load and generate

Install the appropriate PyTorch CPU/CUDA build, then:

```bash
pip install -r requirements.txt
```

The tested Transformers version is **4.57.1**. Megatron, Transformer Engine,
FlashAttention, OpenCompass, and the training repository are not required.
The directory is self-contained; copy the entire directory, including both
weight shards, their index, tokenizer/config files, and the two model-code files.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/path/to/puro-2b-dolma3-fbt-k2-13892"
tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda:0").eval()

inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

`trust_remote_code=True` loads the included, inspectable feedback implementation.
Its distinct `puro_feedback` model type prevents an ordinary Qwen3 loader from
silently running K1 and ignoring the feedback weights.

```bash
python /path/to/checkpoint/generate.py --model /path/to/checkpoint \
  --device cuda:0 --feedback-passes both \
  --prompt 'The capital of France is' --max-new-tokens 64

python /path/to/checkpoint/generate.py --model /path/to/checkpoint \
  --device cuda:0 --feedback-passes 2 --interactive

# CPU works too; choose float32 if your CPU does not efficiently support BF16.
python /path/to/checkpoint/generate.py --model /path/to/checkpoint \
  --device cpu --dtype bfloat16 --feedback-passes 2 \
  --prompt 'def fibonacci(n):' --max-new-tokens 32
```

Greedy decoding is the default. Add `--sample --temperature 0.8 --top-p 0.95`
for sampling. Repeat `--prompt` to test a left-padded batch. `--prompt-file`
reads one full UTF-8 prompt. `--no-cache` provides full-prefix reference decoding.
Prompt plus completion must fit the trained 4,096-token context. Both original
EOS IDs, 151643 and 151645, are retained in `generation_config.json`.

## K1, K2, labels, and caches

Use `feedback_passes=1` in `forward` or `generate` to run K1 of this same jointly
trained checkpoint. K1 is not a separately trained single-pass baseline.
K2 is the default; it shares the same decoder weights and adds a second pass.

Generation keeps separate first/second-pass KV caches and the preceding K1
hidden state. Prefill, one-token decoding, and multi-token cached chunks use
the same shift-1 rule. Use one independent document per batch row and start a
fresh cache for each new document. A cache cannot switch between K1 and K2.
Left-padded batches, greedy decoding, sampling, and beam search are covered by
the adapter tests. Static/quantized caches, cache rollback, speculative decoding,
HF tensor/pipeline parallelism, and vLLM integration are outside this adapter.

For full-sequence label scoring, mask padding labels with `-100` and use
`use_cache=False`. With K2, `output.logits` are K2 logits, `output.k1_loss` and
`output.k2_loss` are individual next-token NLLs, and `output.loss` is their
arithmetic mean, matching training. Without labels only the final head is run.

The earlier native-BF16-to-HF conversion check measured K1/K2 logit RMSE
0.07987/0.07274 and KL 0.000668/0.000684 because native TE/FlashAttention and
HF SDPA round differently. This package preserves those exported decoder
weights byte-for-byte and preserves every feedback value. It does not claim
1e-3 native-to-HF logit equality or change the prior benchmark protocol.
`checkpoint_manifest.json` records source and packaged SHA-256 hashes;
`native_export_validation.json`, when included, records the original check.

The local pre-release check uses CPU BF16 with the full 2B weights. Both full
and cached K1/K2 logits match an independent implementation of the prior
evaluation export exactly. Cached and full-prefix greedy outputs match for
the two 8-token smoke completions. BF16 cached versus full-prefix logits
themselves have rounding differences (maximum KL 0.00154 in these prompts);
the same differences occur in the independent reference. Small FP32 tests
check the cache formula to an absolute tolerance of 2e-6. These smoke checks
do not establish equality for every possible prompt or backend.

The base model's recorded license is Apache-2.0, included as `LICENSE-MODEL`.
`LICENSE-CODE` preserves the
source repository's code license. This directory is a local release candidate;
packaging and inference scripts do not upload it to Hugging Face.
