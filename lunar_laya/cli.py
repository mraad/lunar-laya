"""Finite, reproducible runs and portable browser replays."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

from .game import DT, CONTROL_STEPS, PADS, TERRAIN, Game
from .pilot import Pilot


def run_episode(pilot, seed, target, steps):
    if steps < 1:
        raise ValueError("steps must be at least 1")
    game = Game(seed, target)
    frames = []
    while game.state.status == "flying" and len(frames) < steps:
        before = game.snapshot()
        command, decision = pilot.decide(game)
        after = game.step(command)
        frames.append({"before": before, "decision": decision, "after": after})
    terminal = game.snapshot()
    if terminal["status"] == "flying":
        terminal["status"] = "truncated"
    times = sorted(f["decision"]["latency_ms"] for f in frames)
    summary = {"seed": seed, "target": target, **terminal,
               "decisions": len(frames),
               "interventions": sum(f["decision"]["intervened"] for f in frames),
               "latency_p50_ms": statistics.median(times),
               "latency_p95_ms": times[math.ceil(len(times) * 0.95) - 1]}
    return {"summary": summary, "frames": frames}


def new_recording(pilot):
    """Common schema and runtime provenance for CLI and training evaluation."""
    packages = {}
    for name in ("laya-mlx", "mlx", "tokenizers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"schema_version": 1, "pilot": dict(pilot.provenance),
            "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": packages},
            "world": {"terrain": TERRAIN, "pads": PADS, "dt": DT,
                      "control_steps": CONTROL_STEPS}, "episodes": []}


def export_replay(record, path):
    # Escape '<' so even a model path containing </script> remains inert JSON.
    payload = json.dumps(record, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("replay.html").read_text()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("/* RECORDING */null", payload))


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", choices=("assisted", "laya", "baseline"), default="assisted")
    parser.add_argument("--model", default="aac6fef/laya-multilingual-mlx")
    parser.add_argument("--revision", help="Hugging Face model commit or tag")
    parser.add_argument("--episodes", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=int, choices=range(3), default=1)
    parser.add_argument("--steps", type=positive_int, default=900, help="maximum decisions per episode")
    parser.add_argument("--out", type=Path, default=Path("dist/run.json"))
    parser.add_argument("--replay", type=Path, default=Path("dist/replay.html"))
    args = parser.parse_args(argv)
    if args.out.resolve() == args.replay.resolve():
        parser.error("--out and --replay must be different files")
    try:
        start = perf_counter()
        pilot = Pilot(args.pilot, args.model, args.revision)
        load_seconds = perf_counter() - start
        record = new_recording(pilot)
        record["runtime"]["load_seconds"] = load_seconds
        for seed in range(args.seed, args.seed + args.episodes):
            episode = run_episode(pilot, seed, args.target, args.steps)
            record["episodes"].append(episode)
            print(json.dumps(episode["summary"], allow_nan=False), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        export_replay(record, args.replay)
        print(f"Recording: {args.out}\nReplay: {args.replay}", file=sys.stderr)
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"lunar-laya: {exc}\n")


if __name__ == "__main__":
    main()
