# Developer and experiment guide

For the additional compact native MLX profile, precision selection and measured
memory results, see [MLX optimization](mlx-optimization.md). For the animated
trained-model flight and export commands, see [media reproduction](media.md).

## 1. Work from the source boundaries

Start with `game.py`, then `pilot.py`, then `cli.py`. The replay consumes the
runner's output; it is not an alternative implementation of the simulation.
Keeping these boundaries lets a physics regression run without model weights
and lets a replay be rebuilt without repeating inference.

| Function or class | Inputs | Result / side effects |
|---|---|---|
| `Game(seed=0, target=1)` | Seed and valid target index | Creates a fresh mutable state |
| `Game.snapshot()` | None | New dictionary of current state values |
| `Game.step(command)` | Valid `Command` | Advances up to ten substeps and returns a snapshot |
| `Game.surface_height()` | Current craft x | Maximum terrain height under the horizontal hull footprint |
| `ground(x)` | Horizontal coordinate | Interpolated terrain height, with x clamped to the world interval |
| `Command(turn=0, throttle=0)` | Turn −1/0/1 and finite power in [0, 1] | Immutable validated control value |
| `guidance(game)` | Current game | Guidance command and numeric navigation metrics |
| `observation(game, requested, metrics)` | Game and guidance output | Exact text supplied to Laya |
| `Pilot(mode, model, revision, agent)` | Controller configuration | Loads model once unless baseline or an agent is supplied |
| `Pilot.decide(game)` | Current game | Executed command and full decision record |
| `run_episode(pilot, seed, target, steps)` | Pilot and positive decision budget | Summary and transition frames |
| `export_replay(record, path)` | Complete recording dictionary and destination | Writes standalone HTML, creating parents |

The Python API is intentionally small and currently version 0.1.0. The CLI
checks positive episode/step counts and conflicting output paths. If calling
helpers directly, preserve their preconditions yourself: in particular, pass a
positive `steps` value to `run_episode`, and export a nonempty recording using
schema version 1. There is no general recording-schema validator or legacy
schema migration layer.

## 2. Simulate a flight directly

This example requires no package installation when run from the repository root:

```python
from lunar_laya.game import Game
from lunar_laya.pilot import guidance

game = Game(seed=42, target=1)
while game.state.status == "flying":
    command, metrics = guidance(game)
    game.step(command)

print(game.snapshot())
assert game.state.status == "landed"
```

To inspect a single actuator response without guidance:

```python
from lunar_laya.game import Command, Game, State

game = Game()
game.state = State(x=500, y=400, vx=0, vy=0, angle=0)
before = game.snapshot()
after = game.step(Command(turn=0, throttle=1))
print("vertical velocity change:", after["vy"] - before["vy"])
# Approximately +0.676 m/s: (5.0 − 1.62) × 0.2.
```

Overwriting `game.state` is useful in tests, but bypasses the seeded spawn
distribution. Results from custom states should be reported as custom-state
experiments, not as the documented seeded baseline.

## 3. Inspect guidance and the exact model input

```python
from lunar_laya.game import Game
from lunar_laya.pilot import QUESTIONS, guidance, observation

game = Game(seed=0)
request, metrics = guidance(game)
print(request)
print(metrics)
print(observation(game, request, metrics))
print(QUESTIONS)
```

For seed zero and target one, the first request is turn left and engine off.
The lander starts to the right of the pad, moving right, and descending more
slowly than the high-altitude descent target. Guidance initially requests a
leftward tilt while allowing gravity to increase the descent rate. A request
to rotate with the main engine off is valid: attitude jets consume fuel
independently from main-engine power.

The first five-decision MLX smoke run recorded the following first-frame
proposal, rounded here exactly as the runtime returned its probabilities:

| Question | Option probabilities | Selected option |
|---|---|---|
| Rotation | left 0.2618; hold 0.6942; right 0.0440 | hold |
| Engine | off 0.3139; half 0.3233; full 0.3628 | full |

The proposed `{turn: 0, throttle: 1}` differed from the guidance request
`{turn: -1, throttle: 0}`. Assisted mode executed the latter and set
`intervened=true`. This is an actual integration example, not a fabricated
ideal answer. Full-run aggregate results are in [validation](validation.md).

## 4. Call the real model through the project adapter

After installing the MLX extra and downloading/caching the checkpoint:

```python
from lunar_laya.game import Game
from lunar_laya.pilot import Pilot

pilot = Pilot(
    mode="laya",
    model="aac6fef/laya-multilingual-mlx",
    revision="f2b4faf51023039425946074e2cf1361d2db11d5",
)
game = Game(seed=100)
command, decision = pilot.decide(game)
print(decision["answers"])
print("proposed:", decision["proposed"])
print("executed:", decision["executed"])
game.step(command)
```

Set `HF_HOME` and `HF_HUB_OFFLINE` in the environment before starting Python if
you need the workspace cache or offline behavior. Construct a pilot once and
reuse it; constructing one for every frame would repeatedly load model weights.
Do not mutate a loaded agent's weights or module structure while assuming its
inference caches still represent the original model.

`agent=` supports contract tests with an object exposing
`predict(state, questions)`. Such tests verify answer mapping, error behavior
and assistance independently of GPU setup. They do not establish model accuracy,
timing or actual MLX execution. Injected-agent provenance does not automatically
identify a real checkpoint, so use ordinary model loading for reportable runs.

## 5. Regenerate HTML from saved JSON

Use this after changing the replay template, or when you have received a JSON
recording but not its matching HTML:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from lunar_laya.cli import export_replay

record = json.loads(Path("dist/assisted.json").read_text())
assert record["schema_version"] == 1
export_replay(record, "dist/assisted-regenerated.html")
PY
open dist/assisted-regenerated.html
```

This operation does not load Laya, advance physics or change measurements.
It renders the same recorded data with the current template. Always retain the
JSON when you want to distinguish changes in visualization from changes in
controller behavior. A saved HTML file does not automatically update when its
source template changes.

## 6. Analyze outcomes and latency

The following standard-library script summarizes a recording, counts all final
outcomes, reports intervention rate per decision and computes pooled latency
percentiles:

```bash
python3 - <<'PY'
from collections import Counter
import json
import math
from pathlib import Path
import statistics

record = json.loads(Path("dist/assisted.json").read_text())
episodes = record["episodes"]
decisions = [f["decision"] for e in episodes for f in e["frames"]]
times = sorted(d["latency_ms"] for d in decisions)
outcomes = Counter(e["summary"]["status"] for e in episodes)
overrides = sum(d["intervened"] for d in decisions)
print("pilot:", record["pilot"])
print("outcomes:", dict(outcomes))
print("landing rate:", outcomes["landed"] / len(episodes))
print("interventions:", overrides, "/", len(decisions))
print("intervention rate:", overrides / len(decisions))
print("P50 ms:", statistics.median(times))
print("P95 ms:", times[math.ceil(0.95 * len(times)) - 1])
PY
```

This script assumes a valid, nonempty recording from the runner. An average of
per-episode P95 values is not the pooled P95 of all decisions. If you publish
latency, state which aggregation was used, whether cold calls were retained,
and whether model loading was included. This implementation keeps first calls
in decision percentiles and records loading separately.

Do not average only successful episodes unless explicitly reporting that
conditional population. Keep `crashed`, `out_of_bounds`, `timeout` and
`truncated` distinct: they indicate different failure or evaluation conditions.

## 7. Check paired trajectory equality

For baseline and assisted recordings generated with matching seeds, targets and
budgets, the exact-command replacement policy implies identical trajectories:

```bash
python3 - <<'PY'
import json
from pathlib import Path

baseline = json.loads(Path("dist/baseline.json").read_text())
assisted = json.loads(Path("dist/assisted.json").read_text())
assert baseline["world"] == assisted["world"]
assert len(baseline["episodes"]) == len(assisted["episodes"])
count = 0
for a, b in zip(baseline["episodes"], assisted["episodes"]):
    for key in ("seed", "target", "status", "score"):
        assert a["summary"][key] == b["summary"][key]
    assert len(a["frames"]) == len(b["frames"])
    for x, y in zip(a["frames"], b["frames"]):
        assert x["before"] == y["before"]
        assert x["after"] == y["after"]
        assert x["decision"]["executed"] == y["decision"]["executed"]
        count += 1
print("Identical transitions:", count)
PY
```

Decision records themselves are not identical: baseline has no model answers,
different timing, and no interventions. Match on seed and target if your
recordings use different episode ordering; the example assumes the same order.

## 8. Design a meaningful new experiment

Before changing code, identify the question and the measurement it requires:

| Question | Change | Useful evaluation |
|---|---|---|
| Can Laya execute explicit guidance reliably? | Prompt or checkpoint | Raw landing rate and agreement with guidance |
| Can it reason from telemetry without requested labels? | Remove label requests, revise instructions | Raw held-out outcomes; compare against both original prompt and baseline |
| Does fine-tuning improve control? | Train outside this repository, load compatible exported model | Held-out seeds/states, raw outcomes, action agreement, calibration |
| Can assistance preserve more model freedom? | Replace exact-command shield | Raw/assisted outcomes, intervention rate, counterexamples |
| Is guidance robust to different physics? | Change thrust, fuel or terrain | Baseline outcome matrix before model comparison |
| Does an optimization reduce inference time? | Runtime path or model precision | Paired decisions, output agreement and separately measured cold/steady timing |

Current seeds vary initial state but not terrain. A new seed is not a new lunar
landscape. Broader generalization needs explicit terrain and state families,
which are not provided by the current CLI. If using guidance commands as
training targets, describe the model as imitating that controller and separate
training, tuning and final evaluation states.

Three raw failures show a limitation for these cases. They do not establish a
precise population failure probability. Similarly, 300 successful baseline
episodes establish a tested sample, not a mathematical safety proof.

## 9. Reproducibility checklist

For a published experiment, retain:

1. This project's exact source state, including any uncommitted changes.
2. Laya-MLX source revision, and whether it was a modified editable checkout.
3. Model repository and resolved commit, or checksums for a local export.
4. Python, MLX, tokenizer and other installed dependency versions.
5. OS and hardware description, with concurrent GPU workload noted.
6. Pilot mode, seed interval, target, step budget and physics changes.
7. Full JSON trajectories, including failed and truncated episodes.
8. Timing boundaries, aggregation method and warmup/cold-call treatment.

The current JSON records many of these but not all: it does not embed the
complete source tree, a source commit for this project, GPU hardware model,
weight hashes, or every dependency version. Capture a dependency manifest when
needed:

```bash
uv pip freeze --python .venv/bin/python > dist/environment.txt
```

There is no committed full dependency lockfile. The MLX source dependency is
pinned, but transitive packages can resolve differently at installation time.
The measured report names the runtime versions actually exercised.

## 10. Tests, builds and review

```bash
python3 -m unittest discover -s tests -v
node tests/test_replay.cjs
python3 -m compileall -q lunar_laya tests
uv build
```

Use the smallest relevant check after a change, then run the suite before
sharing it. Physics changes need contact, fuel and seeded-flight coverage.
Prompt changes require real-model evaluation; fake-agent tests alone cannot
validate them. Replay changes need playback-logic tests and a visual inspection
when browser access is available. Documentation-only changes need link and
example verification, not a new neural benchmark.

The wheel contains the Python package and replay template. The source
distribution additionally includes documentation and tests. Neither should
contain model weights, caches, a virtual environment, or generated runs. Model
downloads are deployment artifacts, not source files.

The existing test suite covers:

| Test area | Regression guarded against |
|---|---|
| Seeds and gravity | Nondeterministic starts or incorrect acceleration sign |
| Thrust direction | Confusion between world y and canvas y, or degrees and radians |
| Dry/partial tanks | Thrust or rotation after exhaustion, or free full-power last substeps |
| Commands | Invalid turn values and nonfinite/out-of-range throttle |
| Contact | Unsafe velocity/tilt accepted, pad-edge overlap, wrong score multiplier |
| Terminal state | Simulation continuing after a terminal outcome |
| Baseline matrix | Guidance regressions on the fixed seeded distribution |
| Model adapter | Proposal/execution confusion or hidden interventions in raw mode |
| Recording/export | Broken frame continuity, incorrect truncation, script-closing strings |
| Replay script | Incorrect initial state, scrubbing, restart, or timed completion |

The [additional training workflow](training.md) now provides CUDA supervised
fine-tuning and MLX verification. The [VanillaJS SPA](../web/README.md) adds a
multi-recording flight deck and comparison views. Both preserve these original
simulation and recording APIs.

Known omissions remain explicit: no manual-play interface, sound, spacecraft
rigid-body physics, adaptive engine identification, reinforcement-learning loop,
multi-agent inference server, streaming recording, or formal controller safety proof.
