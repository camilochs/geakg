"""Neural Architecture representation for NAS.

A NeuralArchitecture is a DAG of ArchitectureLayers with:
- Layer types (conv, linear, lstm, etc.)
- Activation functions
- Skip/residual connections
- Training hyperparameters (optimizer, lr, schedule)
- Regularization settings (dropout, normalization, weight decay)
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field


@dataclass
class ArchitectureLayer:
    """A single layer in the neural architecture."""

    layer_id: int
    layer_type: str = "linear"  # linear, conv2d, lstm, gru, attention
    units: int = 128
    activation: str = "relu"
    dropout: float = 0.0
    normalization: str = "none"  # none, batch, layer, group
    kernel_size: int = 3  # For conv layers

    def param_count(self) -> int:
        """Estimated parameter count for this layer."""
        if self.layer_type == "linear":
            return self.units * self.units  # rough estimate
        elif self.layer_type == "conv2d":
            return self.units * self.kernel_size * self.kernel_size
        elif self.layer_type in ("lstm", "gru"):
            mult = 4 if self.layer_type == "lstm" else 3
            return mult * self.units * self.units
        return self.units


@dataclass
class NeuralArchitecture:
    """A complete neural architecture specification.

    This is the "solution" in the NAS search — analogous to a tour in TSP.
    """

    layers: list[ArchitectureLayer] = field(default_factory=list)
    skip_connections: list[tuple[int, int]] = field(default_factory=list)

    # Training hyperparameters
    optimizer: str = "adam"  # sgd, adam, adamw, lamb
    learning_rate: float = 1e-3
    lr_schedule: str = "cosine"  # cosine, step, warmup_cosine, cyclical
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 100

    # Data augmentation
    augmentation: str = "standard"  # none, standard, cutout, mixup, autoaugment

    # Loss
    loss_fn: str = "cross_entropy"  # cross_entropy, label_smoothing, focal

    def total_params(self) -> int:
        """Estimated total parameter count."""
        return sum(layer.param_count() for layer in self.layers)

    def depth(self) -> int:
        """Number of layers."""
        return len(self.layers)

    def copy(self) -> NeuralArchitecture:
        """Deep copy of the architecture."""
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "layers": [
                {
                    "layer_id": l.layer_id,
                    "layer_type": l.layer_type,
                    "units": l.units,
                    "activation": l.activation,
                    "dropout": l.dropout,
                    "normalization": l.normalization,
                    "kernel_size": l.kernel_size,
                }
                for l in self.layers
            ],
            "skip_connections": self.skip_connections,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "lr_schedule": self.lr_schedule,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "augmentation": self.augmentation,
            "loss_fn": self.loss_fn,
        }

    @staticmethod
    def random(search_space: "NASSearchSpace | None" = None) -> NeuralArchitecture:
        """Generate a random valid architecture."""
        if search_space is not None:
            return search_space.random_architecture()

        # Default random architecture
        n_layers = random.randint(2, 6)
        layers = []
        for i in range(n_layers):
            layers.append(ArchitectureLayer(
                layer_id=i,
                layer_type=random.choice(["linear", "conv2d"]),
                units=random.choice([32, 64, 128, 256]),
                activation=random.choice(["relu", "gelu", "silu"]),
                dropout=random.choice([0.0, 0.1, 0.2, 0.3]),
                normalization=random.choice(["none", "batch", "layer"]),
            ))

        skips = []
        if n_layers > 2 and random.random() > 0.5:
            src = random.randint(0, n_layers - 3)
            tgt = random.randint(src + 2, n_layers - 1)
            skips.append((src, tgt))

        return NeuralArchitecture(
            layers=layers,
            skip_connections=skips,
            optimizer=random.choice(["adam", "adamw", "sgd"]),
            learning_rate=random.choice([1e-2, 3e-3, 1e-3, 3e-4]),
            lr_schedule=random.choice(["cosine", "step", "warmup_cosine"]),
            weight_decay=random.choice([0.0, 1e-4, 1e-3]),
            augmentation=random.choice(["none", "standard", "cutout"]),
            loss_fn=random.choice(["cross_entropy", "label_smoothing"]),
        )
