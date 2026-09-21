"""Local, choice-only MLX runtime and additive quantized checkpoint export."""

import argparse
import json
import math
from pathlib import Path
import shutil

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from laya_mlx.agent import Agent
from laya_mlx.common import temp_bucket
from laya_mlx.model import DecisionModel, EncoderConfig, sanitize_weights
from laya_mlx.prepared import PrefixCache
from laya_mlx.tokenizer import Tokenizer

from .pilot import public_provenance


class ChoiceModel(DecisionModel):
    """Reuse the upstream encoder; omit the unused escalation classifier."""

    def __init__(self, encoder_config, agent_config):
        super().__init__(encoder_config, agent_config)
        del self["act_head"]
        del self["temperature"]

    def __call__(self, ids, markers):
        mask = mx.ones(ids.shape, dtype=mx.bool_)
        h = self.encoder(ids, mask)
        h = h + self.type_emb(mx.array([0]))[:, None, :]
        h = self.head(h, mask[:, None, None, :])
        return self.scorer(h[:, markers]).squeeze(-1).astype(mx.float32)[0]


def quantize(model, bits, group_size):
    nn.quantize(model, bits=bits, group_size=group_size,
                class_predicate=lambda _, m: hasattr(m, "to_quantized")
                and m.weight.shape[-1] % group_size == 0)


class CompactAgent:
    """One question at a time, no NumPy tensor math or unused action head.

    Reuses upstream token preparation only. Cache limit is process-wide; this
    explicit low-memory profile disables retained Metal allocation caches.
    """

    _to_internal = staticmethod(Agent._to_internal)
    prepare = Agent.prepare

    def __init__(self, model):
        mx.set_cache_limit(0)
        self.model_id = str(model)
        self.model_dir = Path(model).expanduser().resolve()
        self.cfg = json.loads((self.model_dir / "rl_agent_config.json").read_text())
        enc = EncoderConfig.from_dict(json.loads((self.model_dir / "encoder/config.json").read_text()))
        if not 4 < self.cfg.get("head_max_len", 192) < self.cfg.get("max_len", 512) <= enc.max_position_embeddings:
            raise ValueError("Invalid token budgets")
        self.temperature = self.cfg.get("temperature", [1., 1., 1.])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        if len(self.temperature) != 3 or any(not math.isfinite(float(t)) or float(t) <= 0
                for t in [*self.temperature, *self.temperature_by_options.values()]):
            raise ValueError("Invalid calibration temperatures")
        profile = self.model_dir / "compact_config.json"
        self.inference = json.loads(profile.read_text()) if profile.exists() else {"bits": 16}
        if profile.exists() and (self.inference.get("mode") != "affine"
                                 or self.inference.get("choice_only") is not True):
            raise ValueError("Unsupported compact checkpoint format")
        self.inference = {**self.inference, "runtime": "mlx-choice-sequential", "cache_limit_bytes": 0}
        self.tok = Tokenizer(self.model_dir / "tokenizer")
        self._prefix_cache = PrefixCache(capacity=2)
        with mx.stream(mx.gpu):
            self.model = ChoiceModel(enc, self.cfg)
            bits = self.inference["bits"]
            if bits != 16:
                quantize(self.model, bits, self.inference["group_size"])
            weights = sanitize_weights(mx.load(str(self.model_dir / "model.safetensors")))
            weights = {k: v for k, v in weights.items() if not k.startswith("act_head.") and k != "temperature"}
            if bits == 16:
                weights = {k: v.astype(mx.float16) for k, v in weights.items()}
            self.model.load_weights(list(weights.items()), strict=True)
            self.model.eval()
            mx.eval(self.model.parameters())

    def predict(self, state, questions):
        if any(q.get("type") != "choice" for q in questions.values()):
            raise ValueError("Compact lander runtime supports choice questions only")
        items, internal = self.prepare(state, questions)
        answers = {}
        for (qid, _), item, question in zip(questions.items(), items, internal):
            labels = list(question["crit"])
            with mx.stream(mx.gpu):
                logits = self.model(mx.array([item["ids"]]), mx.array(item["markers"]))
                scale = self.temperature_by_options.get(temp_bucket(0, len(labels)), self.temperature[0])
                p = mx.softmax(logits / max(1e-3, float(scale)))
                confidence = (mx.clip(1 + (p * mx.log(mx.maximum(p, 1e-12))).sum()
                                      / math.log(len(labels)), 0, 1) if len(labels) > 1 else mx.array(1.))
                choice = mx.argmax(p)
                mx.eval(p, confidence, choice)
                values = p.tolist()
                if not all(math.isfinite(v) for v in values):
                    raise FloatingPointError("Non-finite choice probabilities")
                answers[qid] = {"type": "choice", "choice": labels[choice.item()],
                                "confidence": round(confidence.item(), 4),
                                "probabilities": {label: round(v, 4) for label, v in zip(labels, values)}}
        return {"answers": answers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(2, 3, 4, 5, 6, 8), default=6)
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory; existing checkpoints are preserved")
    agent = CompactAgent(args.source)
    if agent.inference["bits"] != 16:
        parser.error("Export from FP16, not an already quantized checkpoint")
    quantize(agent.model, args.bits, args.group_size)
    mx.eval(agent.model.parameters())
    args.output.mkdir(parents=True)
    for name in ("encoder", "tokenizer"):
        shutil.copytree(args.source / name, args.output / name)
    shutil.copy2(args.source / "rl_agent_config.json", args.output)
    mx.save_safetensors(str(args.output / "model.safetensors"), dict(tree_flatten(agent.model.parameters())))
    profile = {"bits": args.bits, "group_size": args.group_size, "mode": "affine",
               "source": public_provenance(str(args.source.resolve())), "choice_only": True}
    (args.output / "compact_config.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(profile))


if __name__ == "__main__":
    main()
