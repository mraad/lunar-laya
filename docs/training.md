# Additional CUDA training and local MLX deployment

The trained checkpoint also has an additive [low-memory MLX deployment](mlx-optimization.md).
The [animated flight and reproduction guide](media.md) shows a fresh local FP16
run after review. Both preserve the original training artifacts described here.

## What changed, and what remains available

Training is an **additional workflow**. The original public checkpoint, CLI
defaults, guidance baseline, assistance behavior, and earlier recordings remain
available. The new model lives in its own directories and is selected explicitly
with `--model`. No weights were written into the cached upstream checkpoint.

This is supervised fine-tuning of Laya's existing typed-decision architecture.
It teaches the model to follow the guidance requests already present in the
flight prompt. It does not train a policy from pixels, discover the guidance
law, use reinforcement learning, or eliminate the deterministic teacher from
input construction. Raw execution means there is no *post-inference override*;
the prompt still contains guidance's requested rotation and throttle labels.

## Training files

| File | Role |
|---|---|
| `training/data.py` | Generate deterministic teacher-labelled JSONL data and checksummed manifests |
| `training/train.py` | CUDA distributed training, validation, best-checkpoint export and parity fixtures |
| `training/verify_mlx.py` | Verify transfer checksum, convert to MLX, check backend parity and run held-out raw flights |
| `training/requirements.txt` | Additional CUDA-side dependencies, separate from local inference |
| `tests/test_training_data.py` | Label mapping, train/validation seed separation, and non-overwriting data generation |

Run these modules from the source checkout. They are not imported by baseline
or ordinary local inference, and CUDA/PyTorch dependencies are not added to the
default package requirements. The source distribution includes the training
scripts; the runtime wheel retains the smaller inference/simulation package.

## Measured remote environment

The run used a user-authorized remote training machine and only
**CUDA devices 0, 1 and 2**. Each was an NVIDIA RTX PRO 6000 Blackwell Server
Edition with approximately 96 GB of memory. Device 3 had another workload and
was not used. After completion, devices 0–2 reported zero allocated memory in
the GPU inventory check.

Remote experiment directory (home-relative, with identifying account details omitted):

```text
~/lunar-laya-training/2026-09-21/
  lunar_laya/             simulator and guidance used to produce labels
  training/              training code
  reference/laya/         pinned upstream PyTorch implementation
  reference/LICENSE      upstream license
  data/                  train.jsonl, validation.jsonl, manifest.json
  checkpoint-v1/         separately trained model and evaluation fixtures
  train-v1.log            training progress and completion record
```

Python was 3.12.3, PyTorch 2.12.0+cu130, Transformers 5.8.1 and safetensors
0.7.0. Upstream Laya source was copied from the adjacent checkout at revision
`42626c348753fbb17572a813127df2278a1ec527`, with its license. The local MLX
runtime remains the implementation pinned in `pyproject.toml`.

## Dataset construction

Each example contains a full prompt, question ID, categorical label, split,
seed, target and origin (`teacher_rollout` or `synthetic_recovery`). Each flight
state creates two examples: one for rotation and one for engine power. Labels
come directly from `guidance()` and the same label dictionaries used at inference.

Teacher rollouts record every eighth control decision. Training uses 32 seeds
across all three pads; validation uses four different seeds across all pads.
Synthetic examples broaden the velocity, altitude and angle combinations beyond
the nearly stable final descent of teacher rollouts.

| Split | Rollout seeds | Synthetic state IDs | Total choice examples |
|---|---|---|---:|
| Training | 1000–1031, all pads | 10000–11999 | 13,022 |
| Validation | 2000–2003, all pads | 20000–20399 | 1,928 |
| Final local flight evaluation | 3000–3009, all pads | None | 30 complete flight attempts |

Training synthetic randomness uses a separate local RNG seed of 73019;
validation uses 81019. Synthetic states use a target-relative x offset uniform
within ±100 m, y in [150, 600] m, vx in [−18, 18] m/s, vy in [−20, 8] m/s,
angle in [−40, 40] degrees and fuel in [5, 100]. They are independent control
examples, not simulated recovery trajectories or guarantees that all such states
are recoverable.

| Label | Training examples | Validation examples |
|---|---:|---:|
| Rotation left | 1,576 | 284 |
| Rotation hold | 3,419 | 415 |
| Rotation right | 1,516 | 265 |
| Engine off | 2,787 | 432 |
| Engine half | 3,316 | 444 |
| Engine full | 408 | 88 |

No class weights or oversampling are used. Synthetic validation supplies full
engine requests that are absent from the sampled nominal validation rollouts.
Splits separate initial state seeds and synthetic RNG draws, but share the
prompt template, guidance law, terrain and option meanings. This is a test of
generalization to new states in that task, not unrelated language or terrain.

The manifest records JSONL SHA-256 hashes. Dataset generation refuses an
existing destination to prevent silently replacing a previously measured split:

```bash
python3 -m training.data --output dist/training-data-new
```

The actual run's data is retained locally in `dist/training-data-v2/`.

## Optimization and checkpoint selection

The starting model was `aac6fef/laya-multilingual-mlx` at revision
`f2b4faf51023039425946074e2cf1361d2db11d5`. The trainer maps MLX-export names
back to upstream PyTorch parameter names and loads them strictly. A shape/name
mismatch fails rather than silently creating randomly initialized parameters.

The last **two encoder layers**, encoder final normalization, decision head,
type embedding and scorer are trainable: **24,801,793 parameters**. Earlier
encoder layers are frozen. The escalation/action head is also frozen because
flight controls use only the two choice questions. Its action probability is
not a trained flight-safety estimate.

| Setting | Value |
|---|---|
| Objective | Cross-entropy against teacher choice index |
| Epochs | 3 |
| Distributed workers | 3, one per authorized GPU |
| Per-GPU batch | 16 choice examples |
| Nominal global batch | 48 choice examples |
| Encoder learning rate | 1e−5 |
| Other trainable parameter learning rate | 1e−4 |
| Optimizer | AdamW, weight decay 0.01 |
| Schedule | Cosine decay to 1e−6 over all updates |
| Gradient clipping | Global norm 1.0 |
| Mixed precision | CUDA BF16 autocast; FP32 master parameters |
| Seed | 718 for model RNG and distributed sampler |
| Updates | 272 per epoch, 816 total |
| Saved weights | FP16 safetensors |

The distributed sampler shuffles each epoch and can pad its shards with repeated
examples so ranks have equal length. The final batch can be smaller than 16.
The log's training loss is the average of rank-zero batch means, not a global
sample-weighted loss. Validation runs on rank zero with the entire validation
split and no distributed padding; the other ranks wait at a barrier.

Best checkpoint selection uses validation accuracy. A strict improvement is
required to replace the best checkpoint, so ties preserve the earlier epoch.
All three epochs achieved 100% validation accuracy; **epoch one** was selected.
The recorded elapsed time through all three train/validation epochs was about
25.1 seconds, excluding model download, model construction, dataset tokenization,
final fixtures and network transfer. This is not an end-to-end training-service
latency claim.

| Epoch | Rank-zero mean training loss | Validation accuracy |
|---|---:|---:|
| 1 | 0.0749715 | 1.0000 |
| 2 | 0.0001597 | 1.0000 |
| 3 | 0.0000372 | 1.0000 |

The small loss is expected for explicit-label following. No temperature fitting
is performed; temperatures are reset to one. Good categorical accuracy does not
establish probability calibration. Optimizer state is not exported, so this
script does not provide an exact training-resume checkpoint.

## Reproduce on a CUDA host

Copy the source and generated data to a new experiment directory. Use an
isolated environment rather than changing a shared machine's dependencies:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r training/requirements.txt
CUDA_VISIBLE_DEVICES=0,1,2 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=3 -m training.train \
  --data data --output checkpoint-new --epochs 3 --batch-size 16
```

Install a CUDA-compatible PyTorch build for the host. The measured host already
had its working CUDA stack installed, and the actual invocation used
`PYTHONPATH=reference:.` with the copied upstream source rather than reinstalling
its environment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=reference:. \
  python3 -m torch.distributed.run --standalone --nproc_per_node=3 \
  -m training.train --data data --output checkpoint-v1 --epochs 3 --batch-size 16
```

Use a new `--output` directory for each experiment. The trainer refuses an
existing output. Do not rerun the example with the measured directory unless
you choose a new name. To change the number of visible GPUs, keep
`--nproc_per_node` consistent with the allowed visible device count.

## Return to Apple MLX

The additional files in this workspace are:

```text
models/lunar-laya-supervised-torch/    transferred CUDA export and fixtures
models/lunar-laya-supervised-mlx/      converted local inference export
```

The CUDA export's SHA-256 is:

```text
2fd0ebcbc4436c293fff517fb7302dbae715be0d7a142f85ea47f2645d795c7a
```

The verification script checks that checksum before loading the transferred
weights. It converts into a **new** destination, verifies token IDs and selected
answers against 32 saved CUDA fixtures, and permits at most 0.03 absolute
probability error for BF16 CUDA versus FP16 MLX. Conversion is a parameter-name
and storage-layout export, not a second training phase or quantization.

```bash
# Replace training-host with your configured SSH alias.
scp -r training-host:~/lunar-laya-training/2026-09-21/checkpoint-v1 \
  models/lunar-laya-supervised-torch

HF_HUB_OFFLINE=1 .venv/bin/python -m training.verify_mlx \
  --source models/lunar-laya-supervised-torch \
  --output models/lunar-laya-supervised-mlx \
  --episodes 10
```

The copy/conversion examples assume the destinations do not yet exist. This
workspace already has them after the measured run. Run normal inference below
to reuse the existing model, or choose new destination names for another export.

The verification script then runs **raw Laya mode** on seeds 3000–3009 for all
three pads. It records executed actions, compares each proposal with guidance
on that same state, and writes:

- `dist/trained-evaluation.json`: all 30 full trajectories.
- `dist/trained-evaluation.metrics.json`: parity and flight summary metrics.
- `dist/trained-demo.json`: the first three central-pad test flights.
- `dist/trained-demo.html`: a portable replay of that subset.

## Run the additional model locally

### Measured local verification

The exported model passed **32/32** CUDA-to-MLX selected-answer comparisons,
with identical token IDs and maximum absolute probability difference
**0.00005798**. All **30/30** test flights (seeds 3000–3009, all three pads)
landed in raw mode with **zero guidance overrides**. The model's **11,147/11,147**
proposed commands matched guidance on their observed states.

Pooled local decision latency was **10.66 ms P50 / 11.82 ms P95**. Evaluation
took approximately **120.7 seconds** excluding transfer, conversion and the
preceding parity checks. That inference timing follows parity calls, so it is
not a cold-start benchmark. Browser work also occurred on the Mac during this
run; these measurements are not an isolated hardware-performance comparison.

The compact report, training metadata and every flight summary are retained in
[training-results.json](training-results.json). Full trajectories remain under
ignored `dist/`. These results demonstrate reliable guidance following for the
tested distribution; explicit requested labels are still part of every prompt.

The SPA uses a paired subset: seeds 3000–3002 on pad index 1. All four modes
were run on those exact starts, with results saved in
[spa-comparison.json](spa-comparison.json):

| Pilot | Landings | Decisions | Guidance overrides |
|---|---:|---:|---:|
| Baseline | 3/3 | 1,132 | 0 |
| Original pretrained, raw | 0/3 | 367 | 0 |
| Original pretrained, assisted | 3/3 | 1,132 | 576 |
| Additional trained model, raw | 3/3 | 1,132 | 0 |

All three original raw flights went out of bounds. Every before/after state in
the trained three-flight subset matched the corresponding baseline trajectory.
The trained model's ordinary `lunar-laya --model ...` CLI path also passed a
five-decision offline smoke run; its `truncated` result is expected for that
short budget, rather than an attempted landing.

### Inference command

```bash
HF_HUB_OFFLINE=1 .venv/bin/lunar-laya --pilot laya \
  --model ./models/lunar-laya-supervised-mlx \
  --seed 3000 --episodes 3 \
  --out dist/trained-local.json --replay dist/trained-local.html
open dist/trained-local.html
```

Use `--pilot assisted` with the same model only when explicitly studying
assistance. To return to the original model, omit `--model`; its default remains
the public checkpoint. Nothing in training replaces that default.

For a fair SPA comparison, generate original-mode recordings on the same three
test seeds without overwriting the earlier seed-100 experiments:

```bash
.venv/bin/lunar-laya --pilot baseline --seed 3000 --episodes 3 \
  --out dist/baseline-demo.json --replay dist/baseline-demo.html
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 \
  .venv/bin/lunar-laya --pilot laya --seed 3000 --episodes 3 \
  --out dist/pretrained-demo.json --replay dist/pretrained-demo.html
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 \
  .venv/bin/lunar-laya --pilot assisted --seed 3000 --episodes 3 \
  --out dist/assisted-demo.json --replay dist/assisted-demo.html
python3 web/build.py dist/baseline-demo.json dist/pretrained-demo.json \
  dist/assisted-demo.json dist/trained-demo.json
```

## Interpretation and next experiments

Validation action accuracy and flight success answer different questions. A few
incorrect actions can destabilize a trajectory even if classification accuracy
is high. Conversely, exact guidance following should reproduce the teacher's
trajectory. The local evaluation measures both; report raw outcomes separately
from assisted outcomes.

This experiment deliberately preserves the existing prompt, including target
labels. Removing those labels, training directly from telemetry, learning from
pixels or optimizing flight reward would create different tasks requiring new
datasets and evaluation. Do not present the present result as evidence for those
capabilities. The frozen escalation head and uncalibrated probabilities should
not be used as autonomous flight-safety guarantees.
