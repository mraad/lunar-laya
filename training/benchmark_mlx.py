"""Fresh-process memory/accuracy measurement; optionally record held-out flights."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time

import mlx.core as mx

from lunar_laya.cli import new_recording, run_episode
from lunar_laya.game import Game, State
from lunar_laya.pilot import Pilot, QUESTIONS, guidance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--validation", type=Path, default=Path("dist/training-data-v2/validation.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=0, help="fresh flight seeds per pad")
    parser.add_argument("--require-perfect", action="store_true", help="fail after saving if labels or flights disagree")
    args = parser.parse_args()
    if args.episodes < 0:
        parser.error("episodes must be nonnegative")
    started = time.perf_counter()
    if args.compact:
        from lunar_laya.compact import CompactAgent
        pilot = Pilot("laya", agent=CompactAgent(args.model))
    else:
        pilot = Pilot("laya", str(args.model))
    load_seconds = time.perf_counter() - started
    load_peak = mx.get_peak_memory()
    mx.reset_peak_memory()
    rows = [json.loads(line) for line in args.validation.read_text().splitlines()]
    matches, latency, errors = 0, [], []
    # Both questions per call, matching flight workload; check the labelled one.
    for i, row in enumerate(rows):
        started = time.perf_counter()
        answer = pilot.agent.predict(row["state"], QUESTIONS)["answers"][row["question"]]
        latency.append((time.perf_counter() - started) * 1000)
        correct = answer["choice"] == row["label"]
        matches += correct
        if not correct:
            errors.append({"row": i, "expected": row["label"], "actual": answer["choice"]})
    latency.sort()
    report = {"pilot": pilot.provenance, "load_seconds": load_seconds,
              "weights_bytes": (args.model / "model.safetensors").stat().st_size,
              "load_peak_bytes": load_peak, "inference_peak_bytes": mx.get_peak_memory(),
              "active_bytes": mx.get_active_memory(), "cache_bytes": mx.get_cache_memory(),
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "validation_rows": len(rows), "correct": matches, "errors": errors,
              "decision_p50_ms": statistics.median(latency),
              "decision_p95_ms": latency[math.ceil(len(latency)*.95)-1],
              "torch_imported": "torch" in sys.modules, "transformers_imported": "transformers" in sys.modules}
    print(json.dumps({"validation_correct": matches, "validation_rows": len(rows)}), flush=True)
    record = new_recording(pilot)
    guidance_matches = decisions = 0
    for target in range(3):
        for seed in range(3000, 3000 + args.episodes):
            episode = run_episode(pilot, seed, target, 900)
            record["episodes"].append(episode)
            game = Game(seed, target)
            for frame in episode["frames"]:
                game.state = State(**frame["before"])
                guidance_matches += asdict(guidance(game)[0]) == frame["decision"]["proposed"]
                decisions += 1
            print(json.dumps(episode["summary"]), flush=True)
    report.update(episodes=len(record["episodes"]),
                  landed=sum(e["summary"]["status"] == "landed" for e in record["episodes"]),
                  decisions=decisions, guidance_matches=guidance_matches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if record["episodes"]:
        args.output.with_suffix(".recording.json").write_text(json.dumps(record, separators=(",", ":")) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("errors", "pilot")}), flush=True)
    if args.require_perfect and (matches != len(rows) or report["landed"] != report["episodes"]
                                or guidance_matches != decisions):
        raise SystemExit("Validation failed; saved report contains the mismatches")


if __name__ == "__main__":
    main()
