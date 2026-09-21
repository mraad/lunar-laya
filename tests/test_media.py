import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from lunar_laya.cli import new_recording, run_episode
from lunar_laya.pilot import Pilot
from test_lunar import FakeAgent
from web.build import build


class ArtifactTests(unittest.TestCase):
    def record(self):
        agent = FakeAgent()
        agent.cfg = {"training": {"dataset": {"teacher": "lunar_laya.pilot.guidance"}}}
        agent.model_id = "unrelated-name"
        agent.inference = {"bits": 6}
        pilot = Pilot("laya", agent=agent)
        record = new_recording(pilot)
        record["episodes"] = [run_episode(pilot, 1, 1, 3)]
        return record

    def test_builder_uses_provenance_not_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.json"
            record = self.record()
            source.write_text(json.dumps(record))
            out = build(Path(tmp)/"web", [source])
            run = json.loads((out/"manifest.json").read_text())["runs"][0]
            self.assertTrue(run["trained"])
            self.assertEqual(run["inference"]["bits"], 6)
            record["pilot"]["training"] = {}
            record["pilot"]["model"] = "lunar-laya-untrained"
            source.write_text(json.dumps(record))
            build(out, [source])
            self.assertFalse(json.loads((out/"manifest.json").read_text())["runs"][0]["trained"])

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "install media extra for GIF test")
    def test_gif_timing_and_reject_assistance(self):
        from PIL import Image
        from lunar_laya.gif import export_gif
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/"input.json", Path(tmp)/"flight.gif"
            record = self.record()
            source.write_text(json.dumps(record))
            metadata = export_gif(source, output)
            duration = 0
            with Image.open(output) as gif:
                self.assertEqual(gif.size, (1000, 620))
                for i in range(gif.n_frames):
                    gif.seek(i)
                    duration += gif.info["duration"]
            self.assertEqual(duration, metadata["duration_ms"])
            record["episodes"][0]["frames"][0]["decision"]["intervened"] = True
            source.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                export_gif(source, output)
