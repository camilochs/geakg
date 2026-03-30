"""OptimizationRoleSchema: Wraps the existing 11-role vocabulary.

This schema reuses the data from roles.py (ROLE_CATALOG, VALID_CATEGORY_TRANSITIONS)
without duplicating it. The AbstractRole enum values are strings, so they
work directly as role IDs.
"""

from src.geakg.core.role_schema import RoleSchema
from src.geakg.layers.l0.roles import (
    AbstractRole,
    RoleCategory,
    ROLE_CATALOG,
    VALID_CATEGORY_TRANSITIONS,
)


class OptimizationRoleSchema(RoleSchema):
    """RoleSchema for combinatorial optimization (11 roles, 3 categories).

    Categories: CONSTRUCTION (entry), LOCAL_SEARCH, PERTURBATION
    Revisitable: LOCAL_SEARCH, PERTURBATION
    """

    def get_all_roles(self) -> list[str]:
        return [r.value for r in AbstractRole]

    def get_role_category(self, role_id: str) -> str:
        try:
            role = AbstractRole(role_id)
        except ValueError:
            raise KeyError(f"Unknown optimization role: {role_id}")
        return ROLE_CATALOG[role]["category"].value

    def get_categories(self) -> list[str]:
        return [c.value for c in RoleCategory]

    def get_roles_by_category(self, category: str) -> list[str]:
        cat_enum = RoleCategory(category)
        return [
            role.value for role, info in ROLE_CATALOG.items()
            if info["category"] == cat_enum
        ]

    def get_entry_categories(self) -> list[str]:
        return [RoleCategory.CONSTRUCTION.value]

    def get_category_transitions(self) -> dict[str, list[str]]:
        return {
            src.value: [tgt.value for tgt in targets]
            for src, targets in VALID_CATEGORY_TRANSITIONS.items()
        }

    def get_role_metadata(self, role_id: str) -> dict:
        try:
            role = AbstractRole(role_id)
        except ValueError:
            raise KeyError(f"Unknown optimization role: {role_id}")
        info = ROLE_CATALOG[role]
        return {
            "description": info["description"],
            "category": info["category"].value,
            "expected_cost": info["expected_cost"],
            "exploration_bias": info["exploration_bias"],
            "typical_quality": info.get("typical_quality", ""),
            "when_to_use": info.get("when_to_use", ""),
        }

    def is_revisitable_category(self, category: str) -> bool:
        return category in (
            RoleCategory.LOCAL_SEARCH.value,
            RoleCategory.PERTURBATION.value,
        )

    def get_role_description_for_llm(self) -> str:
        from src.geakg.layers.l0.roles import get_role_description_for_llm
        return get_role_description_for_llm()
