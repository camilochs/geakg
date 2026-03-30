"""NAS Search Space: Defines valid architectures and constraints.

Constrains the architecture search to valid and resource-feasible
designs. Inspired by NASNet, DARTS, and Once-for-All search spaces.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from src.domains.nas.architecture import NeuralArchitecture, ArchitectureLayer


@dataclass
class NASSearchSpace:
    """Search space definition for NAS."""

    # Layer types
    layer_types: list[str] = field(default_factory=lambda: [
        "linear", "conv2d",
    ])
    activations: list[str] = field(default_factory=lambda: [
        "relu", "gelu", "silu", "tanh", "sigmoid", "mish",
    ])
    optimizers: list[str] = field(default_factory=lambda: [
        "sgd", "adam", "adamw",
    ])
    lr_schedules: list[str] = field(default_factory=lambda: [
        "cosine", "step", "warmup_cosine", "cyclical",
    ])
    normalizations: list[str] = field(default_factory=lambda: [
        "none", "batch", "layer", "group",
    ])
    augmentations: list[str] = field(default_factory=lambda: [
        "none", "standard", "cutout", "mixup",
    ])
    loss_functions: list[str] = field(default_factory=lambda: [
        "cross_entropy", "label_smoothing", "focal",
    ])

    # Architecture constraints
    min_layers: int = 2
    max_layers: int = 8
    min_units: int = 16
    max_units: int = 512
    unit_choices: list[int] = field(default_factory=lambda: [
        16, 32, 64, 128, 256, 512,
    ])
    dropout_choices: list[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.5,
    ])
    lr_choices: list[float] = field(default_factory=lambda: [
        1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4,
    ])
    weight_decay_choices: list[float] = field(default_factory=lambda: [
        0.0, 1e-5, 1e-4, 1e-3, 1e-2,
    ])
    batch_size_choices: list[int] = field(default_factory=lambda: [
        32, 64, 128, 256,
    ])

    # Resource constraints
    max_params: int = 10_000_000  # 10M parameters
    max_depth: int = 20

    def random_architecture(self) -> NeuralArchitecture:
        """Generate a random valid architecture within the search space."""
        n_layers = random.randint(self.min_layers, self.max_layers)
        layers = []

        for i in range(n_layers):
            layers.append(ArchitectureLayer(
                layer_id=i,
                layer_type=random.choice(self.layer_types),
                units=random.choice(self.unit_choices),
                activation=random.choice(self.activations),
                dropout=random.choice(self.dropout_choices),
                normalization=random.choice(self.normalizations),
                kernel_size=random.choice([1, 3, 5]) if "conv" in self.layer_types else 3,
            ))

        # Random skip connections
        skips = []
        if n_layers > 2 and random.random() > 0.4:
            n_skips = random.randint(1, min(3, n_layers - 2))
            for _ in range(n_skips):
                src = random.randint(0, n_layers - 3)
                tgt = random.randint(src + 2, n_layers - 1)
                if (src, tgt) not in skips:
                    skips.append((src, tgt))

        arch = NeuralArchitecture(
            layers=layers,
            skip_connections=skips,
            optimizer=random.choice(self.optimizers),
            learning_rate=random.choice(self.lr_choices),
            lr_schedule=random.choice(self.lr_schedules),
            weight_decay=random.choice(self.weight_decay_choices),
            batch_size=random.choice(self.batch_size_choices),
            augmentation=random.choice(self.augmentations),
            loss_fn=random.choice(self.loss_functions),
        )

        # Ensure resource constraints
        while arch.total_params() > self.max_params and len(arch.layers) > self.min_layers:
            arch.layers.pop()

        return arch

    def is_valid(self, arch: NeuralArchitecture) -> bool:
        """Check if architecture is within search space constraints."""
        if len(arch.layers) < self.min_layers:
            return False
        if len(arch.layers) > self.max_layers:
            return False
        if arch.total_params() > self.max_params:
            return False

        for layer in arch.layers:
            if layer.layer_type not in self.layer_types:
                return False
            if layer.activation not in self.activations:
                return False
            if layer.units < self.min_units or layer.units > self.max_units:
                return False

        # Validate skip connections
        for src, tgt in arch.skip_connections:
            if src >= tgt:
                return False
            if tgt >= len(arch.layers):
                return False

        return True
