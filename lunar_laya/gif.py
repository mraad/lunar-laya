"""Render an actual trained-model recording as a portable documentation GIF."""

import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG, PANEL, GRID = "#0b111b", "#101d29", "#203140"
TEXT, MUTED, MINT, AMBER = "#e6f4f4", "#94aabd", "#90edd0", "#f3c583"
FONTS = {size: ImageFont.load_default(size=size) for size in (13, 15, 18, 23, 30)}


def render(record, episode, index, speed):
    """Draw only recorded states; the final index denotes the terminal summary."""
    frames, summary = episode["frames"], episode["summary"]
    terminal = index == len(frames)
    frame = frames[min(index, len(frames) - 1)]
    state = summary if terminal else frame["before"]
    decision = frame["decision"]
    image = Image.new("RGB", (1000, 620), BG)
    draw = ImageDraw.Draw(image)

    def text(x, y, value, size=15, color=TEXT):
        draw.text((x, y), str(value), font=FONTS[size], fill=color)

    text(26, 20, "LUNAR LAYA", 30, MINT)
    text(26, 58, "CUDA fine-tuning / local MLX inference / actual flight", 15, MUTED)
    text(740, 24, "TRAINED MODEL", 18, MINT)
    text(740, 52, "RAW EXECUTION", 15)
    draw.line((24, 89, 976, 89), fill=GRID)
    pad = record["world"]["pads"][summary["target"]]
    metrics = [("PAD ALTITUDE", f"{max(0, state['y'] - pad[2] - 8):.0f} m"),
               ("VERTICAL", f"{state['vy']:+.1f} m/s"),
               ("HORIZONTAL", f"{state['vx']:+.1f} m/s"),
               ("FUEL", f"{state['fuel']:.1f}%")]
    for x, (label, value) in zip((26, 208, 390, 570), metrics):
        text(x, 102, label, 13, MUTED)
        text(x, 124, value, 23)
    draw.rounded_rectangle((24, 165, 704, 542), radius=10, fill=PANEL, outline=GRID)
    draw.rounded_rectangle((724, 102, 976, 542), radius=10, fill=PANEL, outline=GRID)
    # Fixed world bounds keep terrain and trajectories comparable across flights.
    def point(x, y):
        return (38 + x * 0.65, 509 - y * 0.43)

    for i in range(55):
        x, y = 40 + i * 173 % 650, 182 + i * 79 % 240
        draw.point((x, y), fill=MUTED if i % 7 == 0 else GRID)
    for y in (200, 400, 600):
        draw.line([point(0, y), point(1000, y)], fill=GRID)
    terrain = [point(x, y) for x, y in record["world"]["terrain"]]
    draw.polygon([(38, 529), *terrain, (688, 529)], fill="#182b37")
    draw.line(terrain, fill=MUTED, width=2)
    path = [point(f["before"]["x"], f["before"]["y"]) for f in frames[:index + 1]]
    if terminal:
        path.append(point(state["x"], state["y"]))
    if len(path) > 1:
        draw.line(path, fill="#467b77", width=2)
    for i, p in enumerate(record["world"]["pads"]):
        color = MINT if i == summary["target"] else MUTED
        draw.line([point(p[0], p[2]), point(p[1], p[2])], fill=color, width=4)
        x, y = point(p[0], p[2])
        text(x, y + 8, f"PAD {i + 1} / {p[3]}x", 13, color)
    x, y = point(state["x"], state["y"])
    angle = math.radians(state["angle"])

    def ship(points):
        return [(x + a * math.cos(angle) - b * math.sin(angle),
                 y + a * math.sin(angle) + b * math.cos(angle)) for a, b in points]

    draw.line(ship([(0,-10),(7,-4),(6,4),(-6,4),(-7,-4),(0,-10)]), fill=MINT, width=2)
    for sign in (-1, 1):
        draw.line(ship([(sign*5,4),(sign*7,8),(sign*11,8)]), fill=MINT, width=2)
    command = decision["executed"]
    if not terminal and state["fuel"] > 0 and command["throttle"] > 0:
        draw.line(ship([(-3,6),(0,10+14*command["throttle"]),(3,6)]), fill=AMBER, width=2)
    text(42, 180, f"SEED {summary['seed']} / PAD {summary['target'] + 1}", 13, MUTED)
    text(742, 120, "MODEL DECISION", 13, MUTED)
    for name, row_y in (("rotation", 155), ("engine", 250)):
        answer = decision["answers"][name]
        text(742, row_y, f"{name.upper()} / {answer['choice'].upper()}", 18, MINT)
        for j, (label, probability) in enumerate(answer["probabilities"].items()):
            row = row_y + 27 + j * 20
            text(742, row, label, 13, MUTED)
            draw.rectangle((790, row+4, 908, row+10), fill=GRID)
            if probability > 0:
                draw.rectangle((790, row+4, 790+118*probability, row+10), fill=MINT)
            text(916, row-1, f"{probability*100:.0f}%", 13)
    draw.line((740, 352, 960, 352), fill=GRID)
    text(742, 368, f"T+ {state['time']:.2f} s", 23)
    text(742, 405, f"Decision: {decision['latency_ms']:.1f} ms", 15, MUTED)
    text(742, 431, "No guidance overrides", 15, MINT)
    status = state["status"].replace("_", " ").upper()
    text(742, 477, status, 23, MINT if status in ("FLYING", "LANDED") else AMBER)
    text(742, 509, f"Score {state['score']} / {index} decisions", 13, MUTED)
    draw.rounded_rectangle((24, 555, 976, 559), radius=2, fill=GRID)
    draw.line((24, 557, 24+952*index/len(frames), 557), fill=MINT, width=3)
    text(24, 573, f"RENDERED RECORDING / {speed:g}x SIMULATION TIME / GUIDANCE-FOLLOWING MODEL", 15, MINT)
    text(24, 597, "Requested control labels remain in the model input. No browser capture or invented trajectory.", 13, MUTED)
    return image


def export_gif(source, output, episode_index=0, speed=4):
    source, output = Path(source), Path(output)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    if output.suffix.lower() != ".gif" or output.resolve() == source.resolve():
        raise ValueError("output must be a separate .gif file")
    record = json.loads(source.read_text())
    teacher = record.get("pilot", {}).get("training", {}).get("dataset", {}).get("teacher")
    if record.get("schema_version") != 1 or teacher != "lunar_laya.pilot.guidance" or record["pilot"]["mode"] != "laya":
        raise ValueError("expected a schema-1 trained Laya recording in raw mode")
    if not 0 <= episode_index < len(record["episodes"]):
        raise ValueError("episode index is out of range")
    episode = record["episodes"][episode_index]
    frames = episode["frames"]
    if not frames or any(f["decision"]["intervened"] for f in frames):
        raise ValueError("expected a nonempty flight without guidance overrides")
    times = [f["before"]["time"] for f in frames]
    end = episode["summary"]["time"]
    if any(b <= a for a,b in zip(times, times[1:])) or end <= times[-1]:
        raise ValueError("recorded timestamps must increase")
    # 10 FPS is exactly representable in GIF's 10 ms clock. Sample actual states.
    indices = [bisect_right(times, times[0] + i*speed/10)-1
               for i in range(math.ceil((end-times[0])*10/speed))] + [len(frames)]
    images = [render(record, episode, index, speed).quantize(colors=64, dither=Image.Dither.NONE)
              for index in indices]
    durations = [100] * len(images)
    durations[0], durations[-1] = 800, 2500
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=durations,
                   loop=0, optimize=True, disposal=1)
    render(record, episode, len(frames), speed).save(output.with_suffix(".png"))
    metadata = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "seed": episode["summary"]["seed"], "target": episode["summary"]["target"],
                "model": record["pilot"]["model"], "status": episode["summary"]["status"],
                "simulation_seconds": end-times[0], "speed": speed, "fps": 10,
                "duration_ms": sum(durations), "sampled_frames": len(images),
                "size_bytes": output.stat().st_size, "summary": episode["summary"]}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/trained-laya.gif"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--speed", type=float, default=4)
    args = parser.parse_args()
    try:
        print(json.dumps(export_gif(args.recording, args.output, args.episode, args.speed), indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"lunar-laya GIF: {exc}\n")


if __name__ == "__main__":
    main()
