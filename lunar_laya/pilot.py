"""Laya controls rotation and throttle; assistance is explicit and recorded."""

import math
from pathlib import Path
from time import perf_counter

from .game import Command, GRAVITY, PADS, RADIUS, THRUST

QUESTIONS = {
    "rotation": {
        "type": "choice",
        "instructions": "Control lunar lander tilt. Follow the requested tilt correction.",
        "criteria": {"left": "Decrease tilt angle", "hold": "Keep current tilt", "right": "Increase tilt angle"},
    },
    "engine": {
        "type": "choice",
        "instructions": "Control lunar lander engine. Follow the requested engine power.",
        "criteria": {"off": "Zero thrust", "half": "Half thrust", "full": "Full thrust"},
    },
}
TURNS = {"left": -1, "hold": 0, "right": 1}
THROTTLES = {"off": 0.0, "half": 0.5, "full": 1.0}


def public_provenance(value):
    """Keep checkpoint provenance without publishing personal absolute paths."""
    if isinstance(value, dict):
        return {key: public_provenance(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_provenance(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        path = Path(value)
        for root, prefix in ((Path.cwd(), ""), (Path.home(), "~/")):
            if path.is_relative_to(root):
                return prefix + str(path.relative_to(root))
        return "<local>/" + path.name
    return value


def guidance(game):
    """PD navigation produces a desired tilt and vertical acceleration, not a search."""
    s = game.state
    pad = PADS[game.target]
    dx = (pad[0] + pad[1]) / 2 - s.x
    altitude = max(0, s.y - pad[2] - RADIUS)
    desired_vx = max(-10, min(10, dx * 0.15))
    ax = max(-1.8, min(1.8, (desired_vx - s.vx) * 0.65))
    desired_vy = -min(12, max(0.7, altitude * 0.12))
    # Hold above obstacles until horizontal alignment is recovered.
    if abs(dx) > 35 and altitude < 120:
        desired_vy = max(desired_vy, (120 - altitude) * 0.15)
    ay = GRAVITY + (desired_vy - s.vy) * 0.8
    desired_angle = max(-30, min(30, math.degrees(math.atan2(ax, max(0.8, ay)))))
    error = desired_angle - s.angle
    turn = 1 if error > 3 else -1 if error < -3 else 0
    power = max(0, min(1, ay / (THRUST * max(0.5, math.cos(math.radians(s.angle))))))
    throttle = 0 if power < 0.25 else 0.5 if power < 0.75 else 1.0
    return Command(turn, throttle), {"target_dx": dx, "altitude": altitude,
                                      "desired_angle": desired_angle, "desired_vy": desired_vy}


def observation(game, requested, metrics):
    s = game.state
    rotation = next(k for k, v in TURNS.items() if v == requested.turn)
    engine = next(k for k, v in THROTTLES.items() if v == requested.throttle)
    return (f"Lunar landing. Requested tilt correction: {rotation}. Requested engine power: {engine}. "
            f"Altitude {metrics['altitude']:.1f} m. Pad offset {metrics['target_dx']:.1f} m. "
            f"Horizontal velocity {s.vx:.1f} m/s. Vertical velocity {s.vy:.1f} m/s. "
            f"Tilt {s.angle:.1f} degrees; desired {metrics['desired_angle']:.1f}. "
            f"Desired vertical velocity {metrics['desired_vy']:.1f} m/s. Fuel {s.fuel:.1f}. "
            "Positive x is right, positive y is up, positive tilt is right. "
            "Land upright with horizontal speed <=2 and downward speed <=3 m/s.")


class Pilot:
    def __init__(self, mode="assisted", model="aac6fef/laya-multilingual-mlx", revision=None, agent=None):
        if mode not in ("baseline", "laya", "assisted"):
            raise ValueError("unknown pilot mode")
        self.mode = mode
        self.agent = agent
        self.provenance = {"mode": mode, "model": None, "revision": revision}
        if mode != "baseline" and agent is None:
            try:
                import laya_mlx
            except ImportError as exc:
                raise RuntimeError("MLX pilot requires an Apple Silicon Mac and the mlx extra; see README.md") from exc
            if (Path(model).expanduser() / "compact_config.json").is_file():
                from .compact import CompactAgent
                self.agent = CompactAgent(model)
            else:
                self.agent = laya_mlx.load(model, revision=revision, device="gpu", cache_prompts=True)
        if mode != "baseline":
            self.provenance.update(model=getattr(self.agent, "model_id", None),
                                   training=getattr(self.agent, "cfg", {}).get("training", {}))
            self.provenance["inference"] = getattr(self.agent, "inference", {"bits": 16, "runtime": "laya-mlx"})
            # Hub snapshot directory names identify the resolved commit even without --revision.
            if getattr(self.agent, "model_dir", None) is not None:
                model_dir = Path(self.agent.model_dir)
                self.provenance["resolved_path"] = str(model_dir)
                if model_dir.parent.name == "snapshots":
                    self.provenance["revision"] = model_dir.name
            self.provenance = public_provenance(self.provenance)

    def decide(self, game):
        start = perf_counter()
        reference, metrics = guidance(game)
        prompt = observation(game, reference, metrics)
        answers = None
        proposed = reference
        if self.mode != "baseline":
            result = self.agent.predict(prompt, QUESTIONS)
            answers = result["answers"]
            proposed = Command(TURNS[answers["rotation"]["choice"]],
                               THROTTLES[answers["engine"]["choice"]])
        # ponytail: exact guidance shield; replace with a validated reachable-set shield
        # if preserving more model freedom becomes a research requirement.
        intervened = self.mode == "assisted" and proposed != reference
        executed = reference if intervened else proposed
        return executed, {"proposed": {"turn": proposed.turn, "throttle": proposed.throttle},
                          "executed": {"turn": executed.turn, "throttle": executed.throttle},
                          "intervened": intervened, "answers": answers,
                          "guidance": metrics, "prompt": prompt,
                          "latency_ms": (perf_counter() - start) * 1000}
