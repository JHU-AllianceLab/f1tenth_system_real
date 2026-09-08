"""Export paired SB3 checkpoints to a Python-version-neutral TorchScript bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "f1tenth_safety_rl_gym").is_dir():
        return candidate
    raise FileNotFoundError("Pass --workspace-root pointing to the f1tenth_safe checkout")


def _install_training_imports(root: Path) -> None:
    for path in (root / "f1tenth_safety_rl_gym" / "src", root / "safety-stable-baselines"):
        if not path.is_dir():
            raise FileNotFoundError(path)
        sys.path.insert(0, str(path))


def _resolve_waypoints(config_path: Path, config: dict) -> Path:
    value = Path(str(config["environment"]["waypoint_path"])).expanduser()
    candidates = [
        value,
        Path.cwd() / value,
        config_path.parent / value,
        config_path.parent.parent / value,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve waypoint_path {value}")


def _validate_p1_pair(p1_path: Path, p2_path: Path) -> None:
    candidates = (
        p2_path.parent / "warm_start.json",
        p2_path.parent.parent / "warm_start.json",
    )
    provenance_path = next((path for path in candidates if path.is_file()), None)
    if provenance_path is None:
        raise FileNotFoundError(
            "P2 warm_start.json is required to verify its frozen P1 checkpoint"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = provenance.get("frozen_p1_sha256")
    if not expected:
        raise ValueError("P2 provenance does not record frozen_p1_sha256")
    actual = _sha256(p1_path)
    if actual != expected:
        raise ValueError(
            "P1 does not match the frozen P1 used to train P2: "
            f"expected {expected}, got {actual}"
        )


def _export_model(model, output: Path, name: str) -> dict:
    import torch
    from torch import nn

    class PhysicalActor(nn.Module):
        def __init__(self, actor, low, high):
            super().__init__()
            self.actor = actor
            self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32))
            self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32))

        def forward(self, observation):
            normalized = self.actor(observation, deterministic=True)
            return self.low + 0.5 * (normalized + 1.0) * (self.high - self.low)

    class TwinCritic(nn.Module):
        def __init__(self, critic):
            super().__init__()
            self.critic = critic

        def forward(self, observation, normalized_action):
            first, second = self.critic(observation, normalized_action)
            return torch.cat((first, second), dim=1)

    observation_size = int(np.prod(model.observation_space.shape))
    low = np.asarray(model.action_space.low, dtype=np.float32)
    high = np.asarray(model.action_space.high, dtype=np.float32)
    if low.shape != (2,) or high.shape != (2,):
        raise ValueError("Only two-dimensional control checkpoints are supported")
    example_observation = torch.zeros((2, observation_size), dtype=torch.float32)
    example_action = torch.zeros((2, 2), dtype=torch.float32)
    actor = PhysicalActor(model.actor.cpu().eval(), low, high).eval()
    critic = TwinCritic(model.critic.cpu().eval()).eval()
    actor_file = f"{name}_actor.pt"
    critic_file = f"{name}_critic.pt"
    torch.jit.trace(actor, example_observation, check_trace=True).save(
        str(output / actor_file)
    )
    torch.jit.trace(
        critic, (example_observation, example_action), check_trace=True
    ).save(str(output / critic_file))
    return {
        "observation_size": observation_size,
        "actor": actor_file,
        "critic": critic_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-model", required=True)
    parser.add_argument("--p2-model", required=True)
    parser.add_argument("--config", required=True, help="Resolved training YAML")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace-root")
    args = parser.parse_args()

    root = _workspace_root(args.workspace_root)
    _install_training_imports(root)
    from f1tenth_safety_rl_gym.models import (
        MixedBehaviorReachAvoidSAC,
        MixedBehaviorSafetySAC,
    )

    p1_path = Path(args.p1_model).expanduser().resolve()
    p2_path = Path(args.p2_model).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for path in (p1_path, p2_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing bundle: {output}")
    output.mkdir(parents=True)
    try:
        _validate_p1_pair(p1_path, p2_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        waypoints = _resolve_waypoints(config_path, config)
        p1 = MixedBehaviorSafetySAC.load(p1_path, device="cpu")
        p2 = MixedBehaviorReachAvoidSAC.load(p2_path, device="cpu")
        p1_entry = _export_model(p1, output, "p1")
        p2_entry = _export_model(p2, output, "p2")
        if p1_entry["observation_size"] + 1 != p2_entry["observation_size"]:
            raise ValueError("P1/P2 observation dimensions are not a composed pair")
        if not (
            np.array_equal(p1.action_space.low, p2.action_space.low)
            and np.array_equal(p1.action_space.high, p2.action_space.high)
        ):
            raise ValueError("P1/P2 action bounds do not match")
        shutil.copy2(config_path, output / "resolved_config.yaml")
        shutil.copy2(waypoints, output / "centerline.csv")
        manifest = {
            "format_version": 1,
            "action_low": np.asarray(p1.action_space.low, dtype=float).tolist(),
            "action_high": np.asarray(p1.action_space.high, dtype=float).tolist(),
            "p1": p1_entry,
            "p2": p2_entry,
            "training_config": "resolved_config.yaml",
            "waypoints": "centerline.csv",
            "source": {
                "p1_path": str(p1_path),
                "p1_sha256": _sha256(p1_path),
                "p2_path": str(p2_path),
                "p2_sha256": _sha256(p2_path),
                "config_path": str(config_path),
                "config_sha256": _sha256(config_path),
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output)
        raise
    print(f"Exported deployment bundle: {output}")


if __name__ == "__main__":
    main()
