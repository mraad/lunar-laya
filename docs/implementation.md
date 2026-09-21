# Implementation guide

## The experiment

Lunar landing couples four quantities: position, velocity, attitude and fuel.
Thrust accelerates along the craft's current attitude, so tilting to correct
horizontal motion also reduces upward thrust. Turning the engine off does not
stop motion. Landing requires cancelling velocity before contact, aligning the
craft, and staying inside a flat pad.

This project makes each decision inspectable. The deterministic simulator owns
the truth; a feedback controller computes guidance; Laya answers typed control
questions; the runner records the before/decision/after transition. The browser
only renders that recording. None of the replay's JavaScript simulates physics
or calls a model.

```text
seed → Game.state → guidance → telemetry and requested corrections
                                    ↓
                        Laya-MLX predict(two choice questions)
                                    ↓
                        rotation and engine probabilities
                                    ↓
                proposed command → optional guidance replacement
                                    ↓
                         Game.step(executed command)
                                    ↓
                     JSON transitions → standalone HTML replay
```

The baseline bypasses inference. Raw Laya and assisted Laya receive the same
prompt. Assistance is applied after inference, making paired comparisons
possible. There is no online weight update, MPC rollout, autoregressive
decoder, cloud endpoint or hidden fallback when a checkpoint fails to load.
The optional [training workflow](training.md) runs separately on CUDA and exports
a new model for the same local MLX inference interface. It does not alter the
default checkpoint or simulator.

## Files and boundaries

`game.py` has only standard-library dependencies and exposes `Game`, `State`,
`Command`, the terrain and pad constants. `Game.snapshot()` returns a new dict;
recorded frames cannot mutate when later steps update the live state.

`pilot.py` imports `laya_mlx` only when a model-backed pilot is constructed.
`Pilot.decide(game)` returns an executed `Command` and a serializable decision
record. Tests inject a fake agent with the same `predict(state, questions)` API.

`cli.py` constructs one pilot per invocation and reuses it across episodes.
Each episode creates a new game with seed `--seed + episode_index`. The model
does not receive prior turns; prompt tokenization caching does not add recurrent
memory. Inference failures stop the command with a nonzero exit status.

`replay.html` is packaged alongside the Python source and included in wheels.
Export embeds the recording as JSON, escaping `<` to prevent a string such as a
model path from closing the script element. Dynamic UI labels use `textContent`,
not HTML interpolation. The resulting file is portable and works offline.

## Coordinates, terrain and initial state

The world is 1,000 metres wide; positive x points right and positive y points up.
Angle is in degrees from upright; positive angle tilts thrust toward the right.
Terrain is a fixed piecewise-linear polyline, interpolated by `ground(x)`.

| Target index | Pad x interval | Surface y | Multiplier | Successful score |
|---|---|---|---|---|
| 0 | 150–240 m | 30 m | 2× | 100 |
| 1 (default) | 440–560 m | 20 m | 1× | 50 |
| 2 | 775–825 m | 40 m | 4× | 200 |

`random.Random(seed)` controls the initial state without modifying Python's
global random generator. Relative to the chosen pad centre, x is uniform within
±100 m; horizontal velocity is uniform within ±8 m/s; attitude is uniform within
±12°. Initial centre altitude is y=450 m, vertical velocity is −8 m/s, fuel is
100 units and score is zero. The target tells guidance where to land. Touchdown
on *any* pad can score if all contact requirements pass.

## Integration and fuel

The public control interval is 0.2 simulated seconds, subdivided into ten
0.02-second steps. This separates model decision frequency (5 Hz simulated)
from physics integration (50 Hz simulated). A command contains rotation
`turn ∈ {-1, 0, 1}` and finite `throttle ∈ [0, 1]`. Laya chooses among 0%, 50%,
and 100% throttle; the simulator itself also accepts intermediate values.

For each substep, with dt=0.02:

```text
requested_fuel = (0.65 × throttle + 0.04 × |turn|) × dt
fraction = min(1, remaining_fuel / requested_fuel)   # 1 when cost is zero
remaining_fuel = max(0, remaining_fuel − requested_fuel)
angle = wrap_to_[-180,180)(angle + 30 × turn × fraction × dt)
a_x = 5 × throttle × fraction × sin(angle)
a_y = 5 × throttle × fraction × cos(angle) − 1.62
v_x += a_x × dt; v_y += a_y × dt
x += v_x × dt; y += v_y × dt
```

This is semi-implicit Euler: velocity is updated before position. Trigonometric
functions convert the stored degree angle to radians. A partially empty tank
scales both actuators for its last burn, so the model cannot receive a full
substep of thrust from almost no fuel. At zero fuel, both engine and rotation
stop, but gravity and inertia continue. Fuel exhaustion alone does not terminate
flight: an unpowered craft can still make a safe landing.

Mass is constant; there is no drag, wind, torque or angular momentum. Rotation
directly changes angle at 30°/s while commanded. Those are arcade simplifications,
not spacecraft engineering assumptions.

## Contact and terminal states

The collision hull is an **axis-aligned square of half-width/height 8 m**,
independent of visual attitude. The terrain height under the hull is the maximum
of both horizontal endpoints and every terrain vertex between them. Contact
occurs when the hull bottom reaches that height. This catches pad-edge and
terrain-peak overlap without implementing rigid-body legs.

A landing requires, at the actual contact substep:

- The entire horizontal hull footprint lies inside one pad.
- `|vx| ≤ 2 m/s` and `−3 ≤ vy ≤ 0 m/s`.
- `|angle| ≤ 8°`.

Passing awards `50 × pad_multiplier`, otherwise the outcome is `crashed`.
The hull crossing either side boundary, or its centre rising above y=750 m,
ends as `out_of_bounds`. An episode still flying at 180 simulated seconds ends
as `timeout`. The runner's separate `--steps` budget produces `truncated` in
the episode summary if no physical terminal state was reached. A truncated
frame's `after.status` remains `flying`; truncation is an evaluator outcome.

Once physically terminal, subsequent `Game.step()` calls leave the state
unchanged. A terminal control interval may contain fewer than ten substeps.
The recorded times preserve that shorter duration. The centre can penetrate
the terrain slightly during the final substep; contact velocity is retained
for evaluation rather than reset to zero.

## Guidance controller

Guidance is a small position/velocity feedback law, not the adjacent project's
adaptive model predictive controller. Let `dx = pad_centre − x` and
`h = max(0, y − pad_y − 8)`:

```text
desired_vx = clip(0.15 × dx, −10, 10)
desired_ax = clip(0.65 × (desired_vx − vx), −1.8, 1.8)
desired_vy = −min(12, max(0.7, 0.12 × h))
```

When |dx| > 35 m and h < 120 m, guidance raises the desired vertical velocity
to at least `0.15 × (120 − h)` to recover horizontal alignment above obstacles.
It then computes:

```text
required_ay = 1.62 + 0.8 × (desired_vy − vy)
desired_angle = clip(degrees(atan2(desired_ax, max(0.8, required_ay))), −30, 30)
```

Rotation follows the sign of `desired_angle − angle` outside a ±3° deadband.
Required engine power is `required_ay / (5 × max(0.5, cos(angle)))`, clipped
to [0, 1]. It is rounded to zero below 0.25, half below 0.75, and full otherwise.
These gains and caps live together in `guidance()` so experiments can change
them without a configuration framework. They are tested on the project's start
distribution, not proven safe for arbitrary states or other terrain.

## What Laya does

The default checkpoint is `aac6fef/laya-multilingual-mlx`, with FP16 inference
on the MLX GPU device. Two `choice` questions are submitted in one `predict`
call: rotation (`left`, `hold`, `right`) and engine (`off`, `half`, `full`).
The standard runtime batches question rows. The additional compact MLX runtime
processes them sequentially using packed quantized weights and omits the unused
escalation head; see [memory optimization](mlx-optimization.md). Each row still encodes its own state and
question; this project does not claim shared hidden-state encoding.

The state is a compact text description containing:

- Requested rotation and power labels computed by deterministic guidance.
- Pad-relative altitude and horizontal offset.
- Current horizontal/vertical velocity, angle and fuel.
- Desired angle and desired vertical velocity.
- Coordinate conventions and landing tolerances.

`result['answers'][question]['choice']` maps to actuator values. Probability
maps are retained unmodified in the recording. There is no generated output to
parse. An unexpected choice fails rather than silently substituting a control.

The prompt deliberately supplies guidance requests: numeric control from an
untrained language encoder is not assumed to be reliable. To study independent
model reasoning, remove those requests in `observation()`, change `QUESTIONS`,
and evaluate raw execution across held-out seeds. That is a different experiment
and should not reuse the current success claims. The additional trained model
retains the requested labels and learns to follow them; it is guidance imitation,
not an independently learned telemetry-to-control policy.

With `--pilot assisted`, **every disagreement** with guidance is replaced by the
entire guidance command. Consequently, assisted trajectories are exactly those
of the baseline for the same initial state. The shield is an integration aid,
not a learned safety system or formal safety certificate. `intervened` means
the proposed command differed; it is not a measure of an imminent crash.

## Recording schema (version 1)

```text
schema_version: 1
pilot: {mode, model, revision, resolved_path?}
runtime: {python, platform, packages, load_seconds}
world: {terrain, pads, dt, control_steps}
episodes[]:
  summary: {seed, target, x, y, vx, vy, angle, fuel, time, status, score,
            decisions, interventions, latency_p50_ms, latency_p95_ms}
  frames[]:
    before: {x, y, vx, vy, angle, fuel, time, status, score}
    decision:
      proposed: {turn, throttle}
      executed: {turn, throttle}
      intervened: boolean
      answers: Laya answers dictionary, or null for baseline
      guidance: {target_dx, altitude, desired_angle, desired_vy}
      prompt: exact input text
      latency_ms: end-to-end decision time
    after: state after executing this command
```

Frame n's `after` equals frame n+1's `before`. The renderer displays the state
*before* the decision beside the associated probabilities and commands, then
advances after the frame's actual simulated duration. At the final timeline
position, it shows the terminal summary and last decision. The path is a trace
of observed positions, not a model forecast.

Latency spans guidance, prompt construction, tokenization, synchronized model
inference, output decoding and assistance. It excludes model loading, physics,
file writing and rendering. There is no warmup exclusion, so the first decision
can be slower. P50 is the median; P95 is nearest rank. Model loading has its
own timer. Simulation waits for inference: 5 decisions/s in simulated time
does **not** establish real-time deadline compliance. Replay speed uses
simulation time rather than original inference wall time.

## Reproducibility and extensions

Pin the runtime commit in `pyproject.toml`, the model commit with `--revision`,
and retain the JSON runtime versions. For a Hub cache snapshot the resolved
commit is recorded automatically. Local paths have no inferred revision or
weight hash; archive/checksum them separately if publishing results. Floating
dependency versions can change performance; the validation report records the
versions actually tested.

For a new policy, keep `Game.step()` and recording boundaries unchanged, modify
`Pilot.decide()`, and compare identical seed/target pairs. For physics changes,
update contact and fuel regressions first. Never silently mark a guidance or
fake-model run as a Laya run. Do not derive model accuracy from assisted landing
rate. Keep failed and truncated episodes when reporting outcomes.

## Privacy of published provenance

Recorded checkpoint locations are made portable before publication: paths inside
the current project become relative, home-directory paths use `~/`, and other
absolute paths retain only a `<local>/` prefix and checkpoint basename. This
applies recursively to model and training metadata, including the source path
written by the compact exporter. Actual model loading still uses the original
local path. Model identifiers, revisions, checksums and measured results remain
intact. Private training-machine aliases and account names are replaced with
placeholders in the documentation; configure your own `training-host` SSH alias
when following the remote commands.

The checked documentation images contain no EXIF or personal metadata. Public
upstream project attribution is retained. Local ignored model files and historical
generated recordings are not publication artifacts; regenerate recordings before
sharing them so they receive the current path sanitization.
