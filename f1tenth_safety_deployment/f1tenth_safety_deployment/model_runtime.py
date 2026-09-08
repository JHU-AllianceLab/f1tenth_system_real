"""Portable TorchScript inference without Stable-Baselines3 at runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TorchScriptSafetyModel:
    def __init__(self, bundle_path: str | Path, name: str, device: str = "cpu"):
        import torch

        bundle = Path(bundle_path).expanduser().resolve()
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("format_version", -1)) != 1:
            raise ValueError("Unsupported deployment bundle format")
        if name not in ("p1", "p2"):
            raise ValueError("Model name must be p1 or p2")
        entry = manifest[name]
        self.observation_size = int(entry["observation_size"])
        self.action_low = np.asarray(manifest["action_low"], dtype=np.float32)
        self.action_high = np.asarray(manifest["action_high"], dtype=np.float32)
        self._torch = torch
        self._device = torch.device(device)
        self._actor = torch.jit.load(str(bundle / entry["actor"]), map_location=self._device)
        self._critic = torch.jit.load(str(bundle / entry["critic"]), map_location=self._device)
        self._actor.eval()
        self._critic.eval()

    def _observations(self, observation: np.ndarray, count: int = 1):
        values = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        if values.shape[1] != self.observation_size:
            raise ValueError(
                f"Expected {self.observation_size} observations, got {values.shape[1]}"
            )
        if count > 1:
            values = np.repeat(values, count, axis=0)
        return self._torch.as_tensor(values, device=self._device)

    def safe_action(self, observation: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            action = self._actor(self._observations(observation))
        return action[0].cpu().numpy().astype(np.float32)

    def twin_q_values(self, observation: np.ndarray, actions: np.ndarray) -> np.ndarray:
        physical = np.asarray(actions, dtype=np.float32).reshape(-1, 2)
        normalized = 2.0 * (physical - self.action_low) / (
            self.action_high - self.action_low
        ) - 1.0
        with self._torch.no_grad():
            values = self._critic(
                self._observations(observation, physical.shape[0]),
                self._torch.as_tensor(normalized, device=self._device),
            )
        return values.cpu().numpy().astype(np.float32)

    def value(self, observation: np.ndarray) -> float:
        action = self.safe_action(observation)
        return float(np.min(self.twin_q_values(observation, action.reshape(1, 2))))

