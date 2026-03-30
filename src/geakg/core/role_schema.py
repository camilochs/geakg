"""RoleSchema: Abstract protocol for role vocabularies.

The RoleSchema decouples the GEAKG framework from any fixed set of roles.
Each case study provides its own schema:
  - OptimizationRoleSchema: 11 roles (const_*, ls_*, pert_*)
  - NASRoleSchema: 18 roles (topo_*, act_*, train_*, reg_*, eval_*)

The ACO, MetaGraph, and Bindings all delegate role-related decisions
to the schema instead of hardcoding them.
"""

from abc import ABC, abstractmethod


class RoleSchema(ABC):
    """Abstract protocol for defining a vocabulary of roles.

    A RoleSchema answers all questions about roles that the framework
    needs: what roles exist, how they're categorized, which categories
    are entry points, which are revisitable, and what transitions
    are valid between categories.
    """

    @abstractmethod
    def get_all_roles(self) -> list[str]:
        """Get all role IDs in this schema."""

    @abstractmethod
    def get_role_category(self, role_id: str) -> str:
        """Get the category for a role ID.

        Raises:
            KeyError: If role_id is not in the schema.
        """

    @abstractmethod
    def get_categories(self) -> list[str]:
        """Get all category names."""

    @abstractmethod
    def get_roles_by_category(self, category: str) -> list[str]:
        """Get all role IDs belonging to a category."""

    @abstractmethod
    def get_entry_categories(self) -> list[str]:
        """Get categories that serve as starting points for ACO paths."""

    @abstractmethod
    def get_category_transitions(self) -> dict[str, list[str]]:
        """Get valid transitions between categories.

        Returns:
            Dict mapping source category to list of valid target categories.
        """

    @abstractmethod
    def get_role_metadata(self, role_id: str) -> dict:
        """Get metadata for a role (description, cost, exploration_bias, etc.).

        Raises:
            KeyError: If role_id is not in the schema.
        """

    @abstractmethod
    def is_revisitable_category(self, category: str) -> bool:
        """Whether roles in this category can be visited multiple times in a path."""

    @abstractmethod
    def get_role_description_for_llm(self) -> str:
        """Generate a text description of all roles for LLM prompts."""

    # --- Convenience methods with default implementations ---

    def is_valid_role(self, role_id: str) -> bool:
        """Check if a role ID is in this schema."""
        return role_id in self.get_all_roles()

    def is_entry_role(self, role_id: str) -> bool:
        """Check if a role belongs to an entry category."""
        if not self.is_valid_role(role_id):
            return False
        category = self.get_role_category(role_id)
        return category in self.get_entry_categories()

    def is_valid_transition(self, source: str, target: str) -> bool:
        """Check if a transition between two roles is valid.

        Same role is always valid (repetition). Otherwise checks
        category-level transition rules.
        """
        if source == target:
            return True
        src_cat = self.get_role_category(source)
        tgt_cat = self.get_role_category(target)
        transitions = self.get_category_transitions()
        return tgt_cat in transitions.get(src_cat, [])

    def get_entry_roles(self) -> list[str]:
        """Get all roles that belong to entry categories."""
        roles = []
        for cat in self.get_entry_categories():
            roles.extend(self.get_roles_by_category(cat))
        return roles
