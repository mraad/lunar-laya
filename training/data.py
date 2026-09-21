"""Generate guidance-labelled decisions with disjoint seed ranges."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from lunar_laya.game import Game, PADS, State
from lunar_laya.pilot import QUESTIONS, THROTTLES, TURNS, guidance, observation


def examples(game, split, source):
    command, metrics = guidance(game)
    state = observation(game, command, metrics)
    labels = {"rotation": next(k for k, v in TURNS.items() if v == command.turn),
              "engine": next(k for k, v in THROTTLES.items() if v == command.throttle)}
    return [{"state": state, "question": q, "label": labels[q], "split": split,
             "seed": game.seed, "target": game.target, "source": source}
            for q in QUESTIONS]


def generate(output, train_seeds=32, synthetic=2000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"teacher": "lunar_laya.pilot.guidance", "questions": QUESTIONS,
                "prompt_contains_requested_labels": True, "splits": {}}
    for split, start, count in (("train", 1000, train_seeds), ("validation", 2000, 4)):
        rows = []
        for seed in range(start, start + count):
            for target in range(3):
                game = Game(seed, target)
                step = 0
                while game.state.status == "flying":
                    if step % 8 == 0:
                        rows.extend(examples(game, split, "teacher_rollout"))
                    game.step(guidance(game)[0])
                    step += 1
        if synthetic:
            rng = random.Random(73019 if split == "train" else 81019)
            for i in range(synthetic if split == "train" else 400):
                target = i % 3
                pad = PADS[target]
                game = Game((10000 if split == "train" else 20000) + i, target)
                game.state = State((pad[0] + pad[1])/2 + rng.uniform(-100, 100),
                                   rng.uniform(150, 600), rng.uniform(-18, 18),
                                   rng.uniform(-20, 8), rng.uniform(-40, 40), rng.uniform(5, 100))
                rows.extend(examples(game, split, "synthetic_recovery"))
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["splits"][split] = {
            "rows": len(rows), "rollout_seeds": [start, start + count - 1],
            "labels": dict(Counter(f"{r['question']}:{r['label']}" for r in rows)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output), indent=2))
