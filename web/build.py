"""Build the dependency-free SPA from actual recorded runs."""

import argparse
import json
from pathlib import Path
import shutil


def build(output, recordings):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    for name in ("index.html", "app.js", "styles.css"):
        shutil.copyfile(source / name, output / name)
    runs = []
    for i, path in enumerate(recordings):
        record = json.loads(Path(path).read_text())
        if record.get("schema_version") != 1 or not record.get("episodes"):
            raise ValueError(f"Not a nonempty schema-1 recording: {path}")
        filename = f"run-{i}.json"
        model = record["pilot"].get("model") or "No model"
        mode = record["pilot"]["mode"]
        trained = mode != "baseline" and record["pilot"].get("training", {}).get("dataset", {}).get("teacher") == "lunar_laya.pilot.guidance"
        label = {"baseline": "Guidance baseline", "assisted": "Assisted Laya", "laya": "Pretrained Laya"}[mode]
        if trained:
            label = "Trained Laya" + (" + assistance" if mode == "assisted" else "")
        bits = record["pilot"].get("inference", {}).get("bits", 16)
        if mode != "baseline" and bits != 16:
            label += f" / Q{bits}"
        summaries = [e["summary"] for e in record["episodes"]]
        runs.append({"id": str(i), "label": label, "file": filename,
                     "mode": mode, "trained": trained,
                     "model": model, "inference": record["pilot"].get("inference", {"bits": 16}),
                     "summaries": summaries})
        # Data files remain separate so switching runs does not download all trajectories.
        (output / filename).write_text(json.dumps(record, separators=(",", ":"), allow_nan=False))
    (output / "manifest.json").write_text(json.dumps({"version": 1, "runs": runs}, indent=2)+"\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dist/web"))
    args = parser.parse_args()
    print(build(args.output, args.recordings))
