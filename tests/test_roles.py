"""Tests for Abstract Roles module."""

import pytest

from src.geakg.roles import (
    AbstractRole,
    RoleCategory,
    RoleNode,
    ROLE_CATALOG,
    get_role_node,
    get_all_role_nodes,
    get_roles_by_category,
    get_construction_roles,
    get_local_search_roles,
    get_perturbation_roles,
    get_role_description_for_llm,
    is_valid_role_transition,
    get_valid_next_roles,
    VALID_CATEGORY_TRANSITIONS,
)


class TestAbstractRoleEnum:
    """Tests for AbstractRole enumeration."""

    def test_has_11_roles(self):
        """Should have 11 abstract roles (3 categories)."""
        assert len(AbstractRole) == 11

    def test_role_values_are_lowercase(self):
        """All role values should be lowercase strings."""
        for role in AbstractRole:
            assert role.value == role.value.lower()
            assert isinstance(role.value, str)

    def test_construction_roles_exist(self):
        """Should have 4 construction roles."""
        construction = [r for r in AbstractRole if r.value.startswith("const_")]
        assert len(construction) == 4

    def test_local_search_roles_exist(self):
        """Should have 4 local search roles."""
        ls = [r for r in AbstractRole if r.value.startswith("ls_")]
        assert len(ls) == 4

    def test_perturbation_roles_exist(self):
        """Should have 3 perturbation roles."""
        pert = [r for r in AbstractRole if r.value.startswith("pert_")]
        assert len(pert) == 3


class TestRoleCatalog:
    """Tests for ROLE_CATALOG completeness."""

    def test_catalog_has_all_roles(self):
        """Catalog should have entry for every role."""
        for role in AbstractRole:
            assert role in ROLE_CATALOG, f"Missing catalog entry for {role}"

    def test_catalog_entries_have_required_fields(self):
        """Each catalog entry should have all required fields."""
        required_fields = {
            "description", "category", "expected_cost", "exploration_bias",
            "typical_quality", "when_to_use", "examples_tsp", "examples_jssp"
        }
        for role, info in ROLE_CATALOG.items():
            for field in required_fields:
                assert field in info, f"Missing {field} in {role}"

    def test_exploration_bias_in_range(self):
        """Exploration bias should be between 0 and 1."""
        for role, info in ROLE_CATALOG.items():
            bias = info["exploration_bias"]
            assert 0.0 <= bias <= 1.0, f"Invalid bias for {role}: {bias}"

    def test_categories_are_valid(self):
        """All categories should be valid RoleCategory values."""
        for role, info in ROLE_CATALOG.items():
            assert isinstance(info["category"], RoleCategory)

    def test_examples_are_lists(self):
        """TSP and JSSP examples should be lists."""
        for role, info in ROLE_CATALOG.items():
            assert isinstance(info["examples_tsp"], list)
            assert isinstance(info["examples_jssp"], list)


class TestRoleNode:
    """Tests for RoleNode model."""

    def test_create_role_node(self):
        """Should create RoleNode with valid data."""
        node = RoleNode(
            role=AbstractRole.CONST_GREEDY,
            description="Test description",
            category=RoleCategory.CONSTRUCTION,
            expected_cost="O(n²)",
            exploration_bias=0.1,
        )
        assert node.role == AbstractRole.CONST_GREEDY
        assert node.exploration_bias == 0.1

    def test_is_construction(self):
        """is_construction should return True for construction roles."""
        node = get_role_node(AbstractRole.CONST_GREEDY)
        assert node.is_construction() is True
        assert node.is_local_search() is False

    def test_is_local_search(self):
        """is_local_search should return True for LS roles."""
        node = get_role_node(AbstractRole.LS_INTENSIFY_SMALL)
        assert node.is_local_search() is True
        assert node.is_construction() is False

    def test_is_perturbation(self):
        """is_perturbation should return True for perturbation roles."""
        node = get_role_node(AbstractRole.PERT_ESCAPE_SMALL)
        assert node.is_perturbation() is True


class TestRoleHelpers:
    """Tests for role helper functions."""

    def test_get_role_node(self):
        """get_role_node should return valid RoleNode."""
        node = get_role_node(AbstractRole.LS_INTENSIFY_MEDIUM)
        assert isinstance(node, RoleNode)
        assert node.role == AbstractRole.LS_INTENSIFY_MEDIUM
        assert node.category == RoleCategory.LOCAL_SEARCH

    def test_get_all_role_nodes(self):
        """get_all_role_nodes should return 11 nodes."""
        nodes = get_all_role_nodes()
        assert len(nodes) == 11
        assert all(isinstance(n, RoleNode) for n in nodes)

    def test_get_construction_roles(self):
        """Should return 4 construction roles."""
        roles = get_construction_roles()
        assert len(roles) == 4
        assert AbstractRole.CONST_GREEDY in roles
        assert AbstractRole.CONST_INSERTION in roles

    def test_get_local_search_roles(self):
        """Should return 4 local search roles."""
        roles = get_local_search_roles()
        assert len(roles) == 4
        assert AbstractRole.LS_INTENSIFY_SMALL in roles
        assert AbstractRole.LS_CHAIN in roles

    def test_get_perturbation_roles(self):
        """Should return 3 perturbation roles."""
        roles = get_perturbation_roles()
        assert len(roles) == 3
        assert AbstractRole.PERT_ESCAPE_SMALL in roles


class TestRoleDescriptionForLLM:
    """Tests for LLM prompt generation."""

    def test_description_contains_all_roles(self):
        """Description should mention all 11 roles."""
        desc = get_role_description_for_llm()
        for role in AbstractRole:
            assert role.value in desc, f"Missing {role.value} in description"

    def test_description_has_sections(self):
        """Description should have category sections."""
        desc = get_role_description_for_llm()
        assert "CONSTRUCTION" in desc
        assert "LOCAL_SEARCH" in desc
        assert "PERTURBATION" in desc

    def test_description_not_empty(self):
        """Description should not be empty."""
        desc = get_role_description_for_llm()
        assert len(desc) > 500  # Should be substantial


class TestRoleTransitions:
    """Tests for role transition rules."""

    def test_valid_category_transitions_defined(self):
        """All categories should have valid transitions defined."""
        for cat in RoleCategory:
            assert cat in VALID_CATEGORY_TRANSITIONS
            assert isinstance(VALID_CATEGORY_TRANSITIONS[cat], list)

    def test_construction_can_go_to_local_search(self):
        """Construction should be able to transition to local search."""
        assert is_valid_role_transition(
            AbstractRole.CONST_GREEDY,
            AbstractRole.LS_INTENSIFY_SMALL
        )

    def test_local_search_can_go_to_perturbation(self):
        """Local search should be able to transition to perturbation."""
        assert is_valid_role_transition(
            AbstractRole.LS_INTENSIFY_SMALL,
            AbstractRole.PERT_ESCAPE_SMALL
        )

    def test_perturbation_can_go_to_local_search(self):
        """Perturbation should be able to transition to local search."""
        assert is_valid_role_transition(
            AbstractRole.PERT_ESCAPE_SMALL,
            AbstractRole.LS_INTENSIFY_SMALL
        )

    def test_same_role_always_valid(self):
        """Transition to same role should always be valid."""
        for role in AbstractRole:
            assert is_valid_role_transition(role, role)

    def test_get_valid_next_roles_includes_same(self):
        """Valid next roles should include same role."""
        for role in AbstractRole:
            valid = get_valid_next_roles(role)
            assert role in valid

    def test_get_valid_next_roles_for_construction(self):
        """Construction should lead to LS or perturbation."""
        valid = get_valid_next_roles(AbstractRole.CONST_GREEDY)
        # Should include LS roles
        assert AbstractRole.LS_INTENSIFY_SMALL in valid
        # Should include perturbation (for restart)
        assert AbstractRole.PERT_ESCAPE_SMALL in valid

    def test_get_valid_next_roles_for_local_search(self):
        """Local search should lead to LS or perturbation."""
        valid = get_valid_next_roles(AbstractRole.LS_INTENSIFY_SMALL)
        # Can chain to another LS
        assert AbstractRole.LS_INTENSIFY_MEDIUM in valid
        # Can escape via perturbation
        assert AbstractRole.PERT_ESCAPE_SMALL in valid
