"""Verify exported CUDA answers against local MLX, then run held-out raw flights."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from laya_mlx import Agent
from laya_mlx.convert import convert

from lunar_laya.cli import export_replay, new_recording, run_episode
from lunar_laya.game import Game, State
from lunar_laya.pilot import Pilot, QUESTIONS, guidance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, default=Path("dist/trained-evaluation.json"))
    parser.add_argument("--episodes", type=int, default=10, help="episodes per pad")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    expected = (args.source / "weights.sha256").read_text().split()[0]
    with (args.source / "model.safetensors").open("rb") as weights:
        actual = hashlib.file_digest(weights, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("Transferred checkpoint checksum mismatch")
    convert(args.source, args.output, dtype="float16")
    agent = Agent(args.output, device="gpu", cache_prompts=True)
    parity = []
    for fixture in json.loads((args.source / "parity.json").read_text()):
        row = fixture["row"]
        question = {row["question"]: QUESTIONS[row["question"]]}
        items, _ = agent.prepare(row["state"], question)
        if items[0]["ids"] != fixture["ids"]:
            raise AssertionError("CUDA/MLX tokenizer mismatch")
        answer = agent.predict(row["state"], question)["answers"][row["question"]]
        probabilities = list(answer["probabilities"].values())
        error = float(np.max(np.abs(np.array(probabilities) - fixture["probabilities"])))
        agreement = int(np.argmax(probabilities)) == int(np.argmax(fixture["probabilities"]))
        if not agreement or error > 0.03:
            raise AssertionError(f"CUDA/MLX probability mismatch: {error}")
        parity.append({"agreement": agreement, "max_probability_error": error})
    pilot = Pilot("laya", agent=agent)
    pilot.provenance["source_sha256"] = actual
    record = new_recording(pilot)
    matches, total = 0, 0
    start = time.perf_counter()
    for target in range(3):
        for seed in range(3000, 3000 + args.episodes):
            episode = run_episode(pilot, seed, target, 900)
            record["episodes"].append(episode)
            game = Game(seed, target)
            for frame in episode["frames"]:
                game.state = State(**frame["before"])
                matches += asdict(guidance(game)[0]) == frame["decision"]["proposed"]
                total += 1
            print(json.dumps(episode["summary"]), flush=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    report = {"source_sha256": actual, "parity": parity, "episodes": len(record["episodes"]),
              "landed": sum(e["summary"]["status"] == "landed" for e in record["episodes"]),
              "guidance_matches": matches, "decisions": total,
              "wall_seconds": time.perf_counter()-start,
              "summaries": [e["summary"] for e in record["episodes"]]}
    args.record.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    # A compact, three-flight central-pad demo uses fresh held-out states.
    record["episodes"] = [e for e in record["episodes"] if e["summary"]["target"] == 1][:3]
    (args.record.parent / "trained-demo.json").write_text(json.dumps(record, separators=(",", ":"))+"\n")
    export_replay(record, args.record.parent / "trained-demo.html")
    print(json.dumps({k:v for k,v in report.items() if k not in ("parity", "summaries")}), flush=True)


if __name__ == "__main__":
    main()
