"""NASRoleSchema: 18 roles for Neural Architecture Search.

Defines 5 categories (TOPOLOGY, ACTIVATION, TRAINING, REGULARIZATION, EVALUATION)
with roles inspired by NAS literature (DARTS, ENAS, NASNet, Once-for-All).
"""

from src.geakg.core.role_schema import RoleSchema


# ---- NAS Role Catalog ----

NAS_CATEGORIES = [
    "topology",
    "activation",
    "training",
    "regularization",
    "evaluation",
]

NAS_ROLE_CATALOG: dict[str, dict] = {
    # === TOPOLOGY (entry point) ===
    "topo_feedforward": {
        "description": "Feedforward network: sequential MLP or CNN without skip connections",
        "category": "topology",
        "expected_cost": "O(L)",
        "exploration_bias": 0.2,
    },
    "topo_residual": {
        "description": "Residual/skip connections between layers (ResNet, DenseNet style)",
        "category": "topology",
        "expected_cost": "O(L²)",
        "exploration_bias": 0.3,
    },
    "topo_recursive": {
        "description": "Recurrent or partially recursive layers (LSTM, GRU)",
        "category": "topology",
        "expected_cost": "O(L·T)",
        "exploration_bias": 0.4,
    },
    "topo_cell_based": {
        "description": "Cell-based search: discover normal+reduction cells to stack",
        "category": "topology",
        "expected_cost": "O(C²)",
        "exploration_bias": 0.5,
    },

    # === ACTIVATION ===
    "act_standard": {
        "description": "Standard activations: ReLU, Sigmoid, Tanh",
        "category": "activation",
        "expected_cost": "O(1)",
        "exploration_bias": 0.1,
    },
    "act_modern": {
        "description": "Modern activations: GELU, SiLU/Swish, Mish",
        "category": "activation",
        "expected_cost": "O(1)",
        "exploration_bias": 0.2,
    },
    "act_parametric": {
        "description": "Parametric activations: PReLU, APL (learnable parameters)",
        "category": "activation",
        "expected_cost": "O(P)",
        "exploration_bias": 0.3,
    },
    "act_mixed": {
        "description": "Mixed activations: different activation per layer",
        "category": "activation",
        "expected_cost": "O(L)",
        "exploration_bias": 0.4,
    },

    # === TRAINING ===
    "train_optimizer": {
        "description": "Optimizer selection: SGD+momentum, Adam, AdamW, LAMB",
        "category": "training",
        "expected_cost": "O(1)",
        "exploration_bias": 0.2,
    },
    "train_schedule": {
        "description": "LR schedule: cosine annealing, warmup+decay, cyclical",
        "category": "training",
        "expected_cost": "O(1)",
        "exploration_bias": 0.2,
    },
    "train_augmentation": {
        "description": "Data augmentation: cutout, mixup, AutoAugment",
        "category": "training",
        "expected_cost": "O(N)",
        "exploration_bias": 0.3,
    },
    "train_loss": {
        "description": "Loss function: cross-entropy, label smoothing, focal loss",
        "category": "training",
        "expected_cost": "O(1)",
        "exploration_bias": 0.2,
    },

    # === REGULARIZATION ===
    "reg_dropout": {
        "description": "Dropout regularization: Dropout, DropPath, DropBlock",
        "category": "regularization",
        "expected_cost": "O(1)",
        "exploration_bias": 0.2,
    },
    "reg_normalization": {
        "description": "Normalization: BatchNorm, LayerNorm, GroupNorm",
        "category": "regularization",
        "expected_cost": "O(1)",
        "exploration_bias": 0.2,
    },
    "reg_weight_decay": {
        "description": "Weight decay: L2 regularization, decoupled weight decay",
        "category": "regularization",
        "expected_cost": "O(1)",
        "exploration_bias": 0.1,
    },
    "reg_structural": {
        "description": "Structural constraints: max params, max FLOPs, max latency",
        "category": "regularization",
        "expected_cost": "O(L)",
        "exploration_bias": 0.3,
    },

    # === EVALUATION ===
    "eval_proxy": {
        "description": "Proxy evaluation: few epochs, subset of data, early stopping",
        "category": "evaluation",
        "expected_cost": "O(E·N)",
        "exploration_bias": 0.1,
    },
    "eval_full": {
        "description": "Full evaluation: train to convergence on full dataset",
        "category": "evaluation",
        "expected_cost": "O(E·N·L)",
        "exploration_bias": 0.1,
    },
}

# Valid transitions between NAS categories
NAS_CATEGORY_TRANSITIONS: dict[str, list[str]] = {
    "topology": ["activation", "topology"],
    "activation": ["activation", "training"],
    "training": ["training", "regularization"],
    "regularization": ["regularization", "evaluation"],
    "evaluation": ["topology", "training", "activation"],
}


class NASRoleSchema(RoleSchema):
    """RoleSchema for Neural Architecture Search (18 roles, 5 categories).

    Categories: TOPOLOGY (entry), ACTIVATION, TRAINING, REGULARIZATION, EVALUATION
    Revisitable: TOPOLOGY, ACTIVATION, TRAINING, REGULARIZATION
    """

    def get_all_roles(self) -> list[str]:
        return list(NAS_ROLE_CATALOG.keys())

    def get_role_category(self, role_id: str) -> str:
        if role_id not in NAS_ROLE_CATALOG:
            raise KeyError(f"Unknown NAS role: {role_id}")
        return NAS_ROLE_CATALOG[role_id]["category"]

    def get_categories(self) -> list[str]:
        return list(NAS_CATEGORIES)

    def get_roles_by_category(self, category: str) -> list[str]:
        return [
            role_id for role_id, info in NAS_ROLE_CATALOG.items()
            if info["category"] == category
        ]

    def get_entry_categories(self) -> list[str]:
        return ["topology"]

    def get_category_transitions(self) -> dict[str, list[str]]:
        return dict(NAS_CATEGORY_TRANSITIONS)

    def get_role_metadata(self, role_id: str) -> dict:
        if role_id not in NAS_ROLE_CATALOG:
            raise KeyError(f"Unknown NAS role: {role_id}")
        return dict(NAS_ROLE_CATALOG[role_id])

    def is_revisitable_category(self, category: str) -> bool:
        # Everything except evaluation is revisitable
        return category != "evaluation"

    def get_role_description_for_llm(self) -> str:
        lines = [
            "# Abstract Roles for Neural Architecture Search",
            "",
            "These roles represent NAS design decisions.",
            "Design a meta-search by specifying which roles connect to which.",
            "",
        ]
        for category in NAS_CATEGORIES:
            lines.append(f"## {category.upper()}")
            roles = self.get_roles_by_category(category)
            for role_id in roles:
                info = NAS_ROLE_CATALOG[role_id]
                lines.append(f"- **{role_id}**: {info['description']}")
                lines.append(f"  - Cost: {info['expected_cost']}, Exploration: {info['exploration_bias']:.1f}")
            lines.append("")
        return "\n".join(lines)
