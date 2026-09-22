# Animated trained-model documentation

![Trained Laya landing](assets/trained-laya.gif)

The animation is rendered from a fresh local FP16 MLX inference recording,
`dist/trained-retry.json`, produced after the code review changes. It shows seed
3000 on the central pad (target index 1), a 76.22-second simulated descent,
382 decisions, successful touchdown, score 50 and zero post-inference guidance
overrides. The fresh retry also landed seeds 3001 and 3002. Requested rotation
and power labels are present in the input: this is supervised guidance following.

## Reproduce

```bash
uv pip install -e '.[media]'
HF_HUB_OFFLINE=1 .venv/bin/lunar-laya --pilot laya \
  --model ./models/lunar-laya-supervised-mlx --seed 3000 --episodes 3 \
  --out dist/trained-retry.json --replay dist/trained-retry.html
.venv/bin/python -m lunar_laya.gif dist/trained-retry.json \
  --output docs/assets/trained-laya.gif --episode 0 --speed 4
```

The renderer needs Pillow (validated with 12.3.0), but no MLX installation or
model once the JSON recording exists. The inference command needs Apple Silicon
and the trained checkpoint described in [training.md](training.md). The original
FP16 and additional quantized models are separate artifacts. To animate another
validated compact recording, pass its path and a different output filename.

## Rendering and provenance

`lunar_laya/gif.py` draws the actual recorded positions, terrain, trajectory,
attitude, engine command and model probabilities. It does not invent or
interpolate intermediate physics states; the in-page flight deck does
interpolate between control stages so playback is not 5 fps, but this exporter
deliberately does not, so every frame here is a state the simulator produced.
The lander is the chamfered-box silhouette shared with the SPA, lunar-mpc and
lunar-mpc-laya, drawn as a legible symbol rather than to scale: at 0.43 pixels
per world unit a to-scale one would be five pixels across, so its vertical
offset is measured on the rotated silhouette and no part lands inside the
terrain at any tilt. The terminal frame displays the final
summary; its probabilities and latency belong to the last executed decision.
The screenshot is a generated illustration of measured data, not browser capture.

The GIF is 1000 × 620 pixels with a 64-color palette per frame, 10 frames per
second and infinite looping. At 4× speed it samples a recorded state every
0.4 simulation seconds, holds the first frame for 800 ms and the final frame
for 2,500 ms. This export has 192 sampled frames, a 22.3-second playback cycle,
and is 2,137,489 bytes. GIF timing is quantized to its native 10 ms clock.

The exporter also writes a final-frame PNG and a JSON sidecar containing the
source SHA-256, model path, seed, outcome, timing and episode summary. The checked
artifact's exact provenance is [assets/trained-laya.json](assets/trained-laya.json).
Fresh inference timings vary, so regenerated JSON hashes and rendered latency
labels need not be byte-identical even when trajectories match.

Input checks require a nonempty schema-1 trained raw-model recording, increasing
timestamps, no guidance interventions, a valid episode and a positive finite
playback speed. Other pilot modes are rejected to prevent misleading labels.
Pillow holds sampled images in host memory while exporting; this offline media
step is separate from the low-memory MLX inference measurements.

Embed using `![Trained Laya landing](docs/assets/trained-laya.gif)` from the root
README, or copy the GIF into another documentation site. Keep the accompanying
caption explaining the playback speed and explicit guidance input.
