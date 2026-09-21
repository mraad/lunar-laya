# VanillaJS mission-control SPA

The application is plain HTML, CSS and JavaScript. No framework, npm
dependencies, bundler, CDN, remote fonts or cloud API is required. Python's
standard library builds the static files and can serve them locally.

## Build and open

Generate recordings with `lunar-laya` first, or use the existing workspace
artifacts. From the repository root:

```bash
python3 web/build.py dist/baseline.json dist/raw.json dist/assisted.json
python3 -m http.server 8766 --bind 127.0.0.1 --directory dist/web
```

Open **http://127.0.0.1:8766/**. Keep that server running while using the
application; Ctrl+C stops it. If the port is already occupied, use another port
and open the corresponding address. Serve only `dist/web`, rather than the
repository containing checkpoints and caches.

To include the separately trained model, after its evaluation creates a demo:

```bash
python3 web/build.py dist/baseline-demo.json dist/pretrained-demo.json \
  dist/assisted-demo.json dist/trained-demo.json
```

The `*-demo.json` files are the matched-seed recordings described in
[training documentation](../docs/training.md). The trained run is selected
initially when present; otherwise the app prefers assisted mode. This default
affects the demo selection, not the Python CLI's checkpoint or pilot defaults.

Unlike the original self-contained HTML export, this SPA fetches a manifest
and separate recordings. Open it through HTTP, not `file://`. It needs no
Internet connection once the files are built. Static hosting can serve the same
directory; hash navigation avoids a server-side route fallback requirement.

## Screens and controls

**Flight deck:** choose a pilot and recorded episode, play/pause, restart, scrub
the decision timeline, and select 0.5×, 1×, 4× or 16× playback. Space toggles
playback when focus is outside form and navigation controls. Selecting a new
recording or episode starts paused so you can inspect its initial conditions.

The canvas shows terrain, the chosen pad, the lander and its past trajectory.
The side panel shows actual per-option probabilities, proposed/executed
commands, guidance override status, decision latency and the exact model prompt.
The mission summary deliberately shows the known *final* outcome even while
viewing an earlier frame; the status under the canvas describes the selected
frame. Altitude is relative to the selected pad, not local terrain clearance.

**Compare pilots:** aggregate landing counts, decision/override totals and a
per-episode table. Each card opens its recording in the flight deck. Seed and
pad are shown so unmatched datasets cannot be mistaken for paired trials.
P50 values in this table are per episode, not pooled across the full run.

**How it works:** explains simulation, typed model decisions, assistance,
training and recorded playback. The browser never loads neural weights or
runs a training job. It does not provide manual spacecraft piloting.

**Download current recording:** downloads the currently selected JSON dataset.
It includes all of that run's episodes, not just the displayed episode.

## Data flow

```text
Python simulation → dist/*.json
                         ↓
                   web/build.py
                         ↓
  dist/web/index.html + styles.css + app.js
           manifest.json + run-0.json + run-1.json + …
                         ↓
        local HTTP server → browser fetch → canvas / DOM
```

The builder validates a nonempty schema-1 recording, copies the static assets,
and writes a small manifest containing mode labels and episode summaries.
Large transition arrays stay in separate files and are fetched when selected.
The builder recognizes trained checkpoints through the recorded
`pilot.training.dataset.teacher == "lunar_laya.pilot.guidance"` metadata.
Filenames do not establish training provenance. The precision badge uses
`pilot.inference.bits`, with FP16 as the legacy recording default.
The optimized demonstration can be added as a fifth recording by appending
`dist/compact-demo.json` to the build command. It appears as `Trained Laya / Q6`;
the original trained FP16 recording remains a separate selection. See
[measured memory and accuracy](../docs/mlx-optimization.md) for its validation.

The client cancels a previous recording fetch when the selection changes and
checks request identity before applying a response. Failed loads show an error
message and disable playback. DOM text uses `textContent`, and probability bars
clamp their displayed width. Export files are treated as application-generated
schema-1 data, not as arbitrary validated uploads.

## Rendering and timing

`requestAnimationFrame` handles pacing, but DOM/canvas redraws occur when the
recorded frame changes or a control changes the display. Playback uses the
difference between each frame's `after.time` and `before.time`. It caps elapsed
wall time per browser frame to avoid large jumps after backgrounding a tab.
There is no interpolated physics state, collision calculation or policy call
in the browser. The terrain helper is only for drawing the vertical reference
line under the lander.

The responsive layout switches from a persistent sidebar and telemetry column
to a horizontal navigation bar and stacked panels on smaller screens. Native
buttons, selects, range inputs and details elements provide keyboard interaction.
Canvas imagery has a textual label; current flight state is also visible as DOM
text. The visualization does not expose every point of the drawn trajectory to
a screen reader.

## Development and checks

```bash
node --check web/app.js
node web/app.test.cjs
python3 web/build.py dist/baseline.json dist/raw.json dist/assisted.json
```

The dependency-free Node test checks manifest/recording loading, comparison
population, timeline scrubbing, restart, timed completion, hash routing and
failed-fetch behavior with a minimal fake DOM/canvas. Browser inspection checks
the rendered result separately. After editing source files, rebuild and reload
the browser; the server serves the copied files in `dist/web`.

The output directory is reusable. A rebuild overwrites generated asset and
recording filenames but does not delete unrelated/stale files already present
there. The manifest controls which runs appear in the interface. Use a new
`--output` directory for a clean, separately archived application build.
