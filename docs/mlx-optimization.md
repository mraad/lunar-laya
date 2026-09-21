# Low-memory native MLX inference

This profile is an additional deployment option for the trained model. It leaves
the original public checkpoint, trained FP16 export and default pilot selection
intact. The default CLI still uses the original model unless `--model` is given.

## Measured result: select Q6

Validated on **2026-09-21, Apple M4 Max, 128 GB unified memory, Python 3.12.12,
MLX 0.32.2**. Q6 is the smallest configuration that passed this comparison.
Compared with the standard FP16 runtime, it reduces the weight file by **59.4%**,
peak active MLX inference memory by **57.6%**, and measured process peak RSS by
**33.6%**. These are different memory measures, not interchangeable claims.

| Runtime | Weights MiB | Peak active MLX MiB | Final cache MiB | Peak process RSS MiB | Correct / 1,928 | Decision P50 ms |
|---|---:|---:|---:|---:|---:|---:|
| Standard FP16 | 614.0 | 872.1 | 747.6 | 1,068.3 | 1,928 | 10.08 |
| Compact FP16 | 614.0¹ | 736.7 | 0 | 1,105.5 | 1,928 | 15.99 |
| Compact Q2 | 96.0 | 215.2 | 0 | 591.5 | 503 | 13.23 |
| Compact Q4 | 172.7 | 293.3 | 0 | 667.5 | 1,713 | 13.46 |
| Compact Q5 | 211.0 | 331.8 | 0 | 668.9 | 1,926 | 13.44 |
| **Compact Q6** | **249.4** | **369.7** | **0** | **708.9** | **1,928** | **13.47** |

¹ Compact FP16 loads the existing full file, discarding unused parameters rather
than writing another FP16 checkpoint. Its active model allocation is 613.6 MiB.
Q6's final active model allocation is 251.4 MiB and its loading peak is also
251.4 MiB. Its two-control decision P95 is 14.15 ms; the original runtime's P95
is 11.15 ms. Saving memory has a latency cost here. System load and allocator
behavior affect these single-process measurements; RSS is not a model-size proxy.

Q2, Q4 and Q5 are **rejected** because they fail exact held-out label agreement.
Q6 subsequently landed **30/30 flights**, seeds 3000–3009 across all three pads,
with **11,147/11,147** proposals matching guidance and zero interventions. Every
recorded post-step state exactly matched the corresponding original trained
FP16 evaluation. This is equality of controls/trajectories, not bit-identical
probabilities. Quantization changes numerical predictions even when choices agree.

The complete reports, rejected rows, checkpoint SHA-256 values and memory byte
counts are in [mlx-memory-results.json](mlx-memory-results.json). The 30-flight
recording is the local artifact `dist/memory-q6.recording.json`; a three-flight
central-pad subset is `dist/compact-demo.json` and appears in the SPA as
**Trained Laya / Q6**. The original FP16 demonstration remains alongside it.

The compact FP16 reference passes all labels, separating code simplification
from quantization error. Twelve Python tests and both JavaScript replay suites
pass; the browser was also checked for Q6 labelling, episode selection and final
touchdown. The GIF's 192 frames and 22,300 ms duration were decoded and checked.
After the final adapter cleanup, a separate CLI retry (`dist/compact-retry.json`)
landed central-pad seeds 3000–3002 again with zero interventions and per-flight
P50 decision latency of 13.14–13.19 ms.

## Implementation

`lunar_laya/compact.py` reuses the pinned Laya-MLX ModernBERT encoder, attention,
decision head, tokenizer, prompt construction and calibration. It changes only
the deployment path:

1. Native `mlx.nn.quantize` packs eligible linear and embedding weights using
   affine quantization with 64 weights per group. Small normalization parameters
   and linear biases remain FP16. Quantization scales and offsets also consume
   memory; the file is larger than the nominal bit count alone would suggest.
2. The compact loader creates the quantized module layout before loading packed
   arrays. Lazy random initializers are replaced before evaluation. It never
   materializes a full FP16 checkpoint during normal quantized startup. Export
   is a separate offline step that does load FP16 weights.
3. The unused escalation classifier and its temperature buffer are removed.
   Flight controls use only the rotation and engine choice logits. The compact
   API returns `answers` with choice, confidence and probabilities; it does not
   return the generic agent's unused escalation/action probability or usage data.
4. Rotation and engine are evaluated sequentially with batch size one. Inputs
   have their actual token lengths, without batch padding. No KV cache is needed:
   this is an encoder decision model, not an autoregressive language model.
5. Input tensors, encoder math, logits, softmax, argmax and entropy confidence run
   in MLX on Metal. Only the small final result transfers to Python for controls
   and JSON. There is no NumPy tensor arithmetic in the compact prediction path.
6. The tokenized question-prefix cache is bounded to the two control questions.
   State tokens and predictions are never cached. The Metal allocator cache limit
   is set to zero to prioritize memory over throughput. **This setting affects
   the entire process**, including other MLX models loaded into it.

Python still runs physics, guidance, orchestration and recording; the Rust
tokenizer runs on CPU. “MLX inference” means all neural computation uses MLX,
not that an application can run without CPU work. No PyTorch or Transformers
module is imported in the measured local runtime. The upstream package retains
NumPy as a dependency and imports it, but the compact path does not use it for
model computations. CUDA/PyTorch are confined to the earlier training stage.

The runtime deliberately does not compile and retain shape-specific graphs or
pad inputs for graph reuse. This prioritizes a small working set. Sequential
questions and an empty allocation cache may be slower than batched FP16 on a
large Mac; performance is measured rather than assumed.

## Export and run

Use the local trained FP16 checkpoint from [training.md](training.md). Export
always requires a **new output directory** and refuses to overwrite one.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m lunar_laya.compact \
  --source models/lunar-laya-supervised-mlx \
  --output models/lunar-laya-q6 --bits 6
HF_HUB_OFFLINE=1 .venv/bin/lunar-laya --pilot laya \
  --model ./models/lunar-laya-q6 --seed 3000 --episodes 3 \
  --out dist/compact-demo.json --replay dist/compact-demo.html
```

The presence of `compact_config.json` selects the compact loader automatically.
That file records precision, group size, quantization mode, original source path
and the choice-only contract. The standard training configuration is copied so
training provenance remains available. Recordings add `pilot.inference`; the
SPA reads its precision rather than displaying FP16 for every model.

Supported export precisions are 2, 3, 4, 5, 6 and 8 bits with group sizes 32, 64
or 128. Availability is not a quality guarantee. Each configuration must be
validated; changing group size or precision invalidates earlier accuracy claims.
Do not deploy the failed low-bit candidates solely because their files are small.
The CLI export default is the validated 6-bit/group-64 combination. This report
does not claim that every supported bit/group combination was evaluated.

## Measurement protocol

`training/benchmark_mlx.py` runs each candidate in a fresh process. It reports
weight file size, peak active MLX bytes during load, peak active MLX bytes during
inference, final active bytes, allocator cache bytes, and peak process RSS.
MLX figures describe its tracked allocations, not total system memory. On macOS,
`resource.getrusage(...).ru_maxrss` is in bytes and includes Python, tokenizer,
Metal/runtime overhead and CPU allocations. RSS and MLX bytes must not be added:
Apple Silicon uses unified memory and the accounting overlaps.

The validation benchmark calls both controls on every one of 1,928 held-out
labelled examples and checks the labelled answer. This intentionally matches the
two-question flight workload, although paired dataset rows repeat a state.
Latency includes token preparation, both model calls and answer formatting.
It excludes model load, physics, replay export and disk writes. Reported P95 uses
the nearest-rank definition. First-call setup is included in the sample set.

The optional flight phase runs seeds 3000–3009 on each of the three pads, records
every transition, and compares proposals with deterministic guidance. Memory
metrics are captured **before** accumulating flight recordings: recording many
episodes and rendering GIFs require additional host memory. A live loop calling
`Pilot.decide` without retaining history has no growing episode buffer.

```bash
# Reference: general-purpose original FP16 runtime.
HF_HUB_OFFLINE=1 .venv/bin/python -m training.benchmark_mlx \
  --model models/lunar-laya-supervised-mlx --output dist/memory-fp16.json
# Isolate sequential/pruned execution from quantization.
HF_HUB_OFFLINE=1 .venv/bin/python -m training.benchmark_mlx --compact \
  --model models/lunar-laya-supervised-mlx --output dist/memory-compact-fp16.json
# Candidate deployment and complete held-out flight matrix.
HF_HUB_OFFLINE=1 .venv/bin/python -m training.benchmark_mlx \
  --model models/lunar-laya-q6 --output dist/memory-q6.json --episodes 10
```

Run these sequentially, without another neural workload, for comparable latency.
By default, the benchmark saves failures as data rather than exiting on imperfect
accuracy. Add `--require-perfect` to exit unsuccessfully after saving any label,
landing or guidance mismatch. Inspect `correct`, `validation_rows`, `landed`,
`episodes`, `guidance_matches` and `decisions` before accepting a candidate. Tests only cover these measured
prompts and seed ranges; they do not establish correctness for arbitrary states.

## Review and simplification

Recording schema/provenance construction is shared by the CLI and evaluation
script instead of duplicated. Large checkpoint checksums stream through
`hashlib.file_digest` rather than reading a second full copy into host memory.
Trained SPA labels now use dataset provenance, not a filename substring. The
episode runner rejects empty step budgets and uses a standard P95 calculation.
These changes preserve the simulator and assistance semantics.

Source review and runnable checks were used. CodeRabbit could not perform a diff
review because this workspace has no initialized Git repository; no CodeRabbit
clean-review claim is made. The optimization keeps upstream neural layers rather
than maintaining a second transformer implementation.

This is a measured low-memory profile, not proof of a global memory minimum.
Mixed precision, different group sizes, distillation or retraining could improve
the tradeoff further and would need new evaluations. The goal here is to reduce
memory while preserving the existing trained task and its original artifacts.
