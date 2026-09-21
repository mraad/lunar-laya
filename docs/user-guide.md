# User guide

For the additional compact native MLX profile, precision selection and measured
memory results, see [MLX optimization](mlx-optimization.md). For the animated
trained-model flight and export commands, see [media reproduction](media.md).

This guide describes the original self-contained replay and shared Python CLI.
The additional [VanillaJS SPA](../web/README.md) provides multi-recording
navigation and comparison at `http://127.0.0.1:8766/` when its local server runs.
Its playback starts paused, unlike the original replay's autoplay. See
[training](training.md) for the separate trained checkpoint and local deployment.

## 1. What you are opening

Lunar Laya has two execution stages:

1. Python runs a complete simulation, using the selected controller and, for
   model modes, a locally loaded Laya checkpoint.
2. The generated HTML application displays the completed recording.

Opening an HTML file does not download a model, run MLX, or start a new flight.
The recording, drawing code and styles are embedded in that file. You can view
it on a machine that has neither Python nor Apple Silicon. The replay has no
keyboard piloting, live inference, upload button, or editable physics controls.
Change a simulation option in the CLI, generate another recording, then open it.

There are two different HTML files to distinguish:

| File | Purpose |
|---|---|
| `lunar_laya/replay.html` | Source template with a null recording placeholder; do not open this as the application |
| `dist/assisted.html`, `dist/raw.html`, or your `--replay` path | Exported application with embedded flight data; open this file |

## 2. Open the demonstration already generated in this workspace

From the repository directory on macOS:

```bash
open dist/assisted.html
```

This uses your default HTML application. You can also use Finder to open the
file, or the browser's File → Open File command. For comparison:

```bash
open dist/raw.html
open dist/baseline.html
```

The demonstration files contain seeds 100–102 targeting the central pad.
The assisted and baseline recordings land; the raw model recording shows the
actual out-of-bounds failures. See [validation](validation.md) for exact counts.
Generated files are local artifacts, so these paths will not exist in a fresh
checkout until you run the commands that create them.

## 3. Installation choices

### Baseline with no installation

From the repository root, with Python 3.11 or newer:

```bash
python3 -m lunar_laya --pilot baseline
open dist/replay.html
```

This path runs on platforms supported by Python's standard library. The
baseline never imports the MLX runtime and does not require model weights.

### Installed command without model support

```bash
uv venv --python 3.12
uv pip install -e .
.venv/bin/lunar-laya --pilot baseline
```

The `-e` flag creates an editable installation: edits to Python source are used
by subsequent commands without reinstalling. Changes to package metadata or
dependencies may require another installation. Without `-e`, the installed
package is a copy rather than a live reference to the working directory.

### Installed command with MLX

Use an Apple Silicon Mac and a compatible macOS/Python environment; the runtime
requirements and tested versions are listed in the README and validation report.

```bash
uv venv --python 3.12
uv pip install -e '.[mlx]'
.venv/bin/lunar-laya --pilot assisted \
  --out dist/assisted.json --replay dist/assisted.html
```

The extra fetches the pinned Laya-MLX Git source and installs its dependencies.
It does not include model weights. Weights are resolved on the first model load.
No account credentials are required by this project for the public default
checkpoint. An installed baseline alone does not make `--pilot assisted` work.

### Sibling reference checkout

When `../laya-mlx` exists, this alternative uses that source tree directly:

```bash
uv pip install -e ../laya-mlx -e .
```

The tested sibling revision is recorded in [validation](validation.md). Editing
the sibling changes your runtime, so the pinned extra's revision no longer
describes arbitrary modified sibling code. Keep that distinction in experiment
notes.

## 4. Complete command-line reference

Both `python3 -m lunar_laya` and the installed `lunar-laya` entry point use the
same argument parser. Run them with the interpreter/environment that contains
the dependencies you intend to use.

| Option | Default | Accepted values and effect |
|---|---|---|
| `--pilot` | `assisted` | `baseline`, `laya`, `assisted`; selects the control path |
| `--model` | `aac6fef/laya-multilingual-mlx` | Hub repository ID or local checkpoint directory; ignored for baseline inference |
| `--revision` | Unspecified | Model Hub commit or tag; pin a commit for repeatable weights; does not pin Python packages |
| `--episodes` | `1` | Positive integer; runs sequential episodes with a shared loaded model |
| `--seed` | `0` | Integer; episode n uses `seed + n`, beginning at n=0 |
| `--target` | `1` | `0`, `1`, `2`; chosen landing pad and initial spawn centre |
| `--steps` | `900` | Positive integer; maximum decisions in each episode, not physics substeps |
| `--out` | `dist/run.json` | Destination for the full JSON recording |
| `--replay` | `dist/replay.html` | Destination for the standalone HTML application |
| `-h`, `--help` | — | Print help and exit without loading a model |

Output parent directories are created automatically. JSON and HTML paths must
be different. Existing files at those destinations are overwritten. Paths are
relative to the shell's current working directory, not necessarily the package
directory. Put paths containing spaces in quotes.

The CLI prints one JSON summary line per completed episode to stdout and the
output paths to stderr. Model download progress may also appear on stderr.
A crash or out-of-bounds flight is a valid simulation outcome and normally
returns exit code zero. Installation, loading or execution errors return a
nonzero exit code; an invalid CLI argument normally exits with code two.
Do not use process exit code alone as a landing-success metric.

Recordings are written after all requested episodes finish. There is no live
checkpoint or resume feature. Interrupting a long run can lose the in-memory
recording even if some episode summaries have already printed. For long
experiments, run smaller batches with separate output names. If JSON writing
succeeds but HTML export fails, the JSON can be used to regenerate the replay.

## 5. Replay controls

| Control | Behavior |
|---|---|
| Pause / Play | Pause or resume the current flight; playback starts automatically on page load |
| Restart | Return to frame zero and resume playback |
| Timeline slider | Select a decision boundary and pause there |
| Speed | Choose 0.5×, 1×, 4×, or 16× simulated-time playback |
| Flight | Select a recorded episode by seed and final outcome |
| View state sent to Laya | Expand the exact recorded prompt |

Changing the selected flight resets its timeline. It preserves the current
play/pause state. At the final timeline position playback stops; Play starts
again from the beginning. There are no custom keyboard shortcuts, but the
buttons, selects and range input use native browser focus and keyboard behavior.

Playback advances between recorded states; it does not interpolate new physics
states between decisions. The final decision can be shorter than 0.2 seconds
because contact or another terminal event occurs within a physics substep.
The player uses the actual recorded before/after timestamps for this interval.
Wall-clock catch-up is capped per animation frame, so returning to a backgrounded
tab does not jump through an arbitrarily long flight segment.

## 6. Read the display accurately

### Top telemetry

| Display | Definition |
|---|---|
| Altitude | `max(0, centre_y − selected_pad_y − 8)` in metres; hull-bottom height relative to the selected pad |
| Vertical | Current `vy` in m/s; negative means descending, positive means climbing |
| Horizontal | Current `vx` in m/s; negative means leftward motion |
| Fuel | Remaining units shown as a percentage because the initial tank has 100 units |

Altitude is **not local terrain clearance**. A mountain under the craft can be
higher than the selected pad. Positive displayed altitude does not rule out a
terrain collision. Ground collision uses the full horizontal hull footprint.

### Flight telemetry panel

Status is `FLYING`, `LANDED`, `CRASHED`, `OUT OF BOUNDS`, `TIMEOUT`, or
`TRUNCATED`. Mission time is simulated time, not inference wall time. The
selected target is displayed as 01, 02 or 03, while the CLI indexes it as 0, 1
or 2. Positive tilt is toward the right. Score is awarded on a safe landing
and is zero during flight or after a failed landing.

The selected pad is highlighted. Other pads remain valid landing surfaces;
the simulator awards their own multiplier if the craft safely reaches them.
The faint path is the history of recorded positions, not a predicted trajectory.
The flame indicates commanded thrust when fuel remains, rather than a
measurement of independently simulated engine combustion.

### Decision panel

Rotation and engine probability bars come directly from Laya's recorded
answers. Rotation `L`, `—`, and `R` mean turn left, hold angle, and turn right;
power is shown as a percentage. These controls affect acceleration and angle,
not an instantaneous position or velocity change.

`Proposed` is the model command, or the guidance command in baseline mode.
`Executed` is what the simulator actually received. In assisted mode any
disagreement replaces the entire command with guidance. The message marks that
intervention. At the end, the panel shows the final flight's total interventions
and the last decision. Baseline explicitly displays no model probabilities.

Probabilities are option preferences, not certified chances of landing. Two
separate choice distributions are not a jointly evaluated distribution over all
nine rotation/engine combinations. A high model preference for one choice does
not make that choice physically safe.

## 7. Local weights and offline operation

The default model loader uses the Hugging Face cache unless you pass a local
path. This workspace's validation used a project-specific cache:

```bash
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 \
  .venv/bin/lunar-laya --pilot assisted \
  --out dist/offline.json --replay dist/offline.html
```

`HF_HUB_OFFLINE=1` prevents a Hub lookup from downloading missing files. It does
not create a checkpoint. Omit it for an initial online download. If you use a
different `HF_HOME`, the same model may need downloading into that other cache.

A local checkpoint directory must have the layout expected by Laya-MLX,
including `model.safetensors`, `rl_agent_config.json`, `encoder/config.json`,
and the tokenizer files. Pass a directory, not the safetensors file itself:

```bash
HF_HUB_OFFLINE=1 .venv/bin/lunar-laya \
  --model ./models/laya-multilingual-mlx
```

Use a path with `./` or an absolute path to make local intent unambiguous.
For a Hub model, use `--revision` to select the tested model commit. For a local
directory, retain your own source revision and checksum record; the application
does not infer or verify a weight hash.

## 8. Troubleshooting

| Symptom | Check or remedy |
|---|---|
| `No module named lunar_laya` | Run from the repository root or install into the interpreter being invoked |
| MLX pilot requires the extra | Install `.[mlx]` into `.venv`; use `.venv/bin/lunar-laya`, not another environment's command |
| MLX or Metal cannot run | Verify Apple Silicon and compatible macOS/runtime; use baseline to isolate simulation from GPU setup |
| Model not found in offline mode | Reuse the cache that contains the model or allow the first download |
| Local checkpoint incomplete | Check directory layout and tokenizer files; do not pass a weights file alone |
| Blank page after opening HTML | Open an exported `dist/*.html`, not the source template; ensure export completed |
| Browser shows an older run | Verify the exact file path and reload; an already open page retains its embedded old recording |
| Raw Laya flies away or crashes | This is a measured model limitation; inspect proposals and compare the same seed in assisted mode |
| Assisted model looks successful despite many overrides | Expected: the guidance command determines every executed action in this mode |
| Outcome is truncated | Increase `--steps`; distinguish the evaluator budget from a physical crash or timeout |
| Model seems fast but simulation takes longer | Loading/download time is separate; every decision waits for synchronous inference |
| Export is large | Each frame includes states, prompt and model answers, then embeds them again in HTML; reduce episode count for smaller artifacts |

HTML exports are self-contained and can be shared without JSON or model files.
They include recorded prompts, model provenance and runtime metadata, including
local checkpoint paths when recorded. JSON is the better artifact for analysis;
HTML is the convenient artifact for viewing. Neither includes the neural weights.

## 9. Where to go next

- For equations and source boundaries, read [implementation](implementation.md).
- To write a policy, regenerate a replay or analyze JSON, read
  [development](development.md).
- To interpret actual success and latency numbers, read
  [validation](validation.md).
