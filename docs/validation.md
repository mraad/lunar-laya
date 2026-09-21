# Validation and measured results

For the subsequent native MLX memory review, quantization accuracy comparison,
30-flight Q6 retest and final CLI retry, see [MLX optimization](mlx-optimization.md).
The [GIF guide](media.md) records the fresh FP16 retry used for documentation.

This page preserves the **original public-checkpoint** evaluation. A separate
checkpoint has since been trained on `training-host` and validated locally: see
[training and MLX verification](training.md) and
[training-results.json](training-results.json). Its 30/30 raw-flight result does
not replace the original model's failures reported here.

Measured on **2026-09-21** in this workspace. These are local results, not
upstream Laya benchmark numbers. The compact machine-readable report is
[validation.json](validation.json); full trajectories and standalone replays
were generated under ignored `dist/`.

## Environment and checkpoint

- Apple Silicon, macOS 26.6.2, Python 3.12.12.
- MLX 0.32.2; tokenizers 0.23.2; Laya-MLX 0.1.0.
- Laya-MLX sibling source checkout at
  `fc1df62828a3fedf4d8229fdac1cbd85f1cdf337` (also pinned in the MLX extra).
- Model `aac6fef/laya-multilingual-mlx`, resolved Hub revision
  `f2b4faf51023039425946074e2cf1361d2db11d5`.
- GPU inference, default FP16, CPU prompt cache enabled, no compilation option.

No hardware model-specific performance claim is made. Other Apple chips,
thermal conditions, dependencies and checkpoints can change these timings.

## End-to-end model experiment

All modes used seeds **100, 101, 102**, target **1** (central pad), up to 900
decisions each, and identical simulation parameters.

| Pilot | Landings | Other outcomes | Decisions | Interventions | P50 / P95 decision latency |
|---|---:|---|---:|---:|---|
| Baseline | 3/3 | None | 1,114 | 0 | 0.0030 / 0.0040 ms |
| Assisted Laya | 3/3 | None | 1,114 | 522 (46.9%) | 10.14 / 10.59 ms |
| Raw Laya execution | 0/3 | 3 out of bounds | 549 | 0 | 10.10 / 10.53 ms |

Latency percentiles pool individual decision times within each mode, include
first-call overhead and exclude loading, physics and rendering. These are
decision-loop measurements, not a promise of browser FPS or hard real-time
behavior. The initial five-decision smoke test had a first-call latency around
936 ms; later full-run processes reused lower-level caches. No warmup calls
were discarded from the reported full runs.

| Seed | Assisted duration | Assisted overrides | Raw outcome / duration |
|---|---:|---:|---|
| 100 | 73.80 s | 162 / 369 | out of bounds / 16.82 s |
| 101 | 74.48 s | 167 / 373 | out of bounds / 66.24 s |
| 102 | 74.38 s | 193 / 372 | out of bounds / 26.26 s |

Every one of the 1,114 assisted before/after state pairs was checked against
its baseline counterpart and matched exactly, as the assistance policy implies.
Therefore the successful assisted flights validate the integration and guidance
controller, **not independent model flight competence**. The raw model failed
even though its prompt contained requested guidance labels. Three episodes
are a smoke comparison, not a statistically strong model evaluation.

Reproduce after installing the MLX extra:

```bash
.venv/bin/lunar-laya --pilot baseline --episodes 3 --seed 100 --out dist/baseline.json --replay dist/baseline.html
.venv/bin/lunar-laya --pilot assisted --episodes 3 --seed 100 \
  --revision f2b4faf51023039425946074e2cf1361d2db11d5 \
  --out dist/assisted.json --replay dist/assisted.html
.venv/bin/lunar-laya --pilot laya --episodes 3 --seed 100 \
  --revision f2b4faf51023039425946074e2cf1361d2db11d5 \
  --out dist/raw.json --replay dist/raw.html
```

This workspace's downloaded weights are in `.cache/huggingface`, so to reuse
them here without another download, prefix model commands with:

```bash
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv/bin/lunar-laya --pilot assisted
```

For another checkout, omit those environment variables on the first run so the
runtime can download to the usual Hugging Face cache.

## Physics and guidance checks

The baseline landed **300/300** episodes on seeds 100–199 across all three pads
(100 episodes per pad). This covers the defined seeded spawn distribution with
the fixed terrain. It does not cover arbitrary initial velocities, empty tanks,
terrain changes or propulsion faults.

```bash
python3 - <<'PY'
from collections import Counter
from lunar_laya.game import Game
from lunar_laya.pilot import guidance

outcomes = Counter()
for target in range(3):
    for seed in range(100, 200):
        game = Game(seed, target)
        while game.state.status == "flying":
            game.step(guidance(game)[0])
        outcomes[game.state.status] += 1
print(outcomes)  # Counter({'landed': 300})
PY
```

Eight standard-library unittest methods pass, including a separate 30-flight
baseline matrix (seeds 0–9, all pads), deterministic seeding, gravity/thrust
signs, dry and partially empty tanks, invalid commands, landing/crash envelopes,
pad-edge collision, score multipliers, bounds/timeouts, terminal absorption,
model mapping, explicit interventions, recording continuity, truncation, and
safe JSON-to-HTML embedding.

The dependency-free Node check executes the replay script with a minimal fake
DOM/canvas and checks initial state, scrubbing to terminal state, restart, and
timed completion. It validates playback logic, not pixel appearance, browser
compatibility or screen-reader quality.

```bash
python3 -m unittest discover -s tests -v
node tests/test_replay.cjs
python3 -m compileall -q lunar_laya tests
uv build
```

Both source distribution and wheel built successfully; archive inspection
confirmed the replay template is included and model weights/cache directories
are excluded. The installed `lunar-laya` entry point was exercised for baseline,
assisted and raw runs.

## Verification limits

Automated browser inspection of the local HTML was blocked by the browser
tool's URL policy. No visual browser pass is claimed; the generated replay is
available for opening directly, and the JavaScript syntax and playback logic
checks passed. The workspace initially had no Git repository, so CodeRabbit's
Git-diff review was not run; validation used local source inspection and the
checks above.

The original public Laya checkpoint is not a reliable autonomous flight policy
for this prompt. Those runs used no lunar-specific fine-tuning. There is no
confidence-based safety guarantee, physical spacecraft validation, or claim of
Atari ROM fidelity. Improvements
to raw policy performance should be evaluated on held-out seeds and reported
separately from guidance-assisted outcomes.

The additional supervised model is reported separately. The new HTTP-served
VanillaJS SPA was subsequently opened and visually inspected in Chrome; its
navigation and playback were exercised. The earlier local-file browser block
remains a historical verification limit for the original single-file replay.
