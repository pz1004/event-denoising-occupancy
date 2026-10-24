"""Deeper FPGA-oriented classifier with explicit fake-quantization support."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .advanced import AdvancedConfig


def _ste_fake_quantize(value: torch.Tensor, clip: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-tensor fake quantization with a straight-through gradient."""

    qmax = float(2 ** (bits - 1) - 1)
    safe_clip = torch.clamp(clip, min=1e-7)
    scale = safe_clip / qmax
    quantized = torch.clamp(torch.round(value / scale), -qmax, qmax) * scale
    return value + (quantized - value).detach()


class AdvancedEventClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
    ):
        super().__init__()
        hidden_dims = tuple(int(value) for value in hidden_dims)
        if feature_dim <= 0 or not hidden_dims or any(value <= 0 for value in hidden_dims):
            raise ValueError("Feature and hidden dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dims = hidden_dims
        dimensions = (self.feature_dim,) + hidden_dims + (2,)
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(len(dimensions) - 1)
        )
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_scale", torch.ones(self.feature_dim))
        self.register_buffer("qat_enabled", torch.zeros(1, dtype=torch.uint8))
        self.register_buffer("quant_bits", torch.tensor([12, 8, 8], dtype=torch.int32))
        self.register_buffer("input_clip", torch.ones(1))
        self.register_buffer("activation_clips", torch.ones(len(hidden_dims)))
        self.register_buffer("weight_clips", torch.ones(len(self.layers)))

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def is_qat_enabled(self) -> bool:
        return bool(self.qat_enabled.item())

    def set_normalizer(self, mean: np.ndarray, scale: np.ndarray) -> None:
        mean = np.asarray(mean, dtype=np.float32)
        scale = np.maximum(np.asarray(scale, dtype=np.float32), 1e-4)
        if mean.shape != (self.feature_dim,) or scale.shape != (self.feature_dim,):
            raise ValueError("Normalizer shape does not match feature_dim")
        self.feature_mean.copy_(torch.from_numpy(mean))
        self.feature_scale.copy_(torch.from_numpy(scale))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = (features - self.feature_mean) / self.feature_scale
        if self.is_qat_enabled:
            value = _ste_fake_quantize(value, self.input_clip[0], int(self.quant_bits[0]))
        for index, layer in enumerate(self.layers):
            weight = layer.weight
            if self.is_qat_enabled:
                weight = _ste_fake_quantize(
                    weight, self.weight_clips[index], int(self.quant_bits[1])
                )
            value = F.linear(value, weight, layer.bias)
            if index < len(self.layers) - 1:
                value = F.relu(value)
                if self.is_qat_enabled:
                    value = _ste_fake_quantize(
                        value,
                        self.activation_clips[index],
                        int(self.quant_bits[2]),
                    )
        return value

    @torch.inference_mode()
    def calibrate_qat(
        self,
        features: np.ndarray,
        device: torch.device,
        percentile: float = 99.9,
        maximum_samples: int = 131_072,
    ) -> None:
        if not 90.0 <= percentile <= 100.0:
            raise ValueError("percentile must be in [90, 100]")
        if len(features) > maximum_samples:
            indices = np.linspace(0, len(features) - 1, maximum_samples, dtype=np.int64)
            sample = features[indices]
        else:
            sample = features
        # torch 1.12 quantile cannot index arbitrarily large flattened tensors.
        # Bound calibration by element count as well as event count while keeping
        # deterministic coverage across the full sampled feature matrix.
        maximum_input_elements = 8_000_000
        element_limited_samples = max(1, maximum_input_elements // self.feature_dim)
        if len(sample) > element_limited_samples:
            indices = np.linspace(
                0, len(sample) - 1, element_limited_samples, dtype=np.int64
            )
            sample = sample[indices]
        value = torch.from_numpy(
            np.array(sample, dtype=np.float32, order="C", copy=True)
        ).to(device)
        value = (value - self.feature_mean) / self.feature_scale
        self.input_clip[0] = torch.quantile(value.abs().flatten(), percentile / 100.0)
        for index, layer in enumerate(self.layers):
            self.weight_clips[index] = torch.max(layer.weight.abs())
            value = F.linear(value, layer.weight, layer.bias)
            if index < len(self.layers) - 1:
                value = F.relu(value)
                self.activation_clips[index] = torch.quantile(
                    value.abs().flatten(), percentile / 100.0
                )

    def enable_qat(
        self, input_bits: int = 12, weight_bits: int = 8, activation_bits: int = 8
    ) -> None:
        if min(input_bits, weight_bits, activation_bits) < 2:
            raise ValueError("Quantized tensors require at least two bits")
        self.quant_bits.copy_(
            torch.tensor(
                [input_bits, weight_bits, activation_bits],
                dtype=torch.int32,
                device=self.quant_bits.device,
            )
        )
        self.qat_enabled.fill_(1)

    def disable_qat(self) -> None:
        self.qat_enabled.zero_()

    @torch.inference_mode()
    def noise_probability(
        self,
        features: np.ndarray,
        device: torch.device,
        batch_size: int = 65_536,
    ) -> np.ndarray:
        self.eval()
        outputs = []
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(
                np.array(
                    features[start : start + batch_size],
                    dtype=np.float32,
                    order="C",
                    copy=True,
                )
            ).to(device)
            outputs.append(torch.softmax(self(batch), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def save_advanced_checkpoint(
    path: Path,
    model: AdvancedEventClassifier,
    config: AdvancedConfig,
    metadata: Dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model": {
            "feature_dim": model.feature_dim,
            "hidden_dims": list(model.hidden_dims),
            "state_dict": model.state_dict(),
        },
        "advanced_config": config.to_dict(),
        "metadata": dict(metadata),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_advanced_checkpoint(
    path: Path, map_location: str = "cpu"
) -> Tuple[AdvancedEventClassifier, AdvancedConfig, Dict]:
    payload = torch.load(Path(path), map_location=map_location)
    if payload.get("format_version") != 2:
        raise ValueError("Unsupported advanced checkpoint format")
    config = AdvancedConfig.from_dict(payload["advanced_config"])
    spec = payload["model"]
    if int(spec["feature_dim"]) != config.feature_dim:
        raise ValueError("Checkpoint feature dimension and advanced config disagree")
    model = AdvancedEventClassifier(spec["feature_dim"], spec["hidden_dims"])
    model.load_state_dict(spec["state_dict"], strict=True)
    return model, config, dict(payload.get("metadata", {}))
