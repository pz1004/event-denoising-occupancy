"""Tiny quantization-friendly classifier for native state features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from .core import StateConfig


class TinyEventClassifier(nn.Module):
    """Two-layer ReLU classifier with frozen affine feature normalization."""

    def __init__(
        self, feature_dim: int, hidden_dim: int = 32, context_mode: str = "causal"
    ):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        if context_mode not in {"causal", "bidirectional"}:
            raise ValueError("context_mode must be causal or bidirectional")
        self.context_mode = context_mode
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_scale", torch.ones(feature_dim))
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, 2),
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def set_normalizer(self, mean: np.ndarray, scale: np.ndarray) -> None:
        mean = np.asarray(mean, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)
        if mean.shape != (self.feature_dim,) or scale.shape != (self.feature_dim,):
            raise ValueError("Normalizer shape does not match feature_dim")
        scale = np.maximum(scale, 1e-4)
        self.feature_mean.copy_(torch.from_numpy(mean))
        self.feature_scale.copy_(torch.from_numpy(scale))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        return self.classifier(normalized)

    @torch.inference_mode()
    def noise_probability(
        self, features: np.ndarray, device: torch.device, batch_size: int = 262_144
    ) -> np.ndarray:
        self.eval()
        probabilities = []
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(features[start : start + batch_size], dtype=np.float32)
            ).to(device)
            logits = self(batch)
            probabilities.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        if not probabilities:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(probabilities)


def save_checkpoint(
    path: Path,
    model: TinyEventClassifier,
    state_config: StateConfig,
    metadata: Dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": {
            "feature_dim": model.feature_dim,
            "hidden_dim": model.hidden_dim,
            "context_mode": model.context_mode,
            "state_dict": model.state_dict(),
        },
        "state_config": state_config.to_dict(),
        "metadata": dict(metadata),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path, map_location: str = "cpu"
) -> Tuple[TinyEventClassifier, StateConfig, Dict]:
    payload = torch.load(Path(path), map_location=map_location)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported Native3D-SSM checkpoint format")
    config = StateConfig.from_dict(payload["state_config"])
    model_spec = payload["model"]
    context_mode = model_spec.get("context_mode", "causal")
    expected_dim = config.feature_dim * (2 if context_mode == "bidirectional" else 1)
    if model_spec["feature_dim"] != expected_dim:
        raise ValueError("Checkpoint feature dimension and state config disagree")
    model = TinyEventClassifier(
        feature_dim=model_spec["feature_dim"],
        hidden_dim=model_spec["hidden_dim"],
        context_mode=context_mode,
    )
    model.load_state_dict(model_spec["state_dict"], strict=True)
    return model, config, dict(payload.get("metadata", {}))
