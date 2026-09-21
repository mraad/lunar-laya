import json
from pathlib import Path
import tempfile
import unittest

from lunar_laya.game import Game
from lunar_laya.pilot import QUESTIONS
from training.data import examples, generate


class TrainingDataTests(unittest.TestCase):
    def test_labels_and_split_isolation(self):
        rows = examples(Game(0), "train", "test")
        self.assertEqual([r["label"] for r in rows], ["left", "off"])
        for row in rows:
            self.assertIn(row["label"], QUESTIONS[row["question"]]["criteria"])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "data"
            manifest = generate(output, train_seeds=1, synthetic=2)
            splits = {name:[json.loads(s) for s in (output/f"{name}.jsonl").read_text().splitlines()]
                      for name in ("train", "validation")}
            self.assertFalse({r["seed"] for r in splits["train"]} & {r["seed"] for r in splits["validation"]})
            self.assertEqual(manifest["splits"]["train"]["rows"], len(splits["train"]))
            with self.assertRaises(FileExistsError):
                generate(output)


if __name__ == "__main__":
    unittest.main()
