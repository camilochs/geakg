"""Tests for Domain Bindings module."""

import pytest

from src.geakg.roles import AbstractRole, RoleCategory
from src.geakg.bindings import (
    OperatorBinding,
    DomainBindings,
    BindingRegistry,
    create_tsp_bindings,
    create_jssp_bindings,
    get_binding_stats,
)


class TestOperatorBinding:
    """Tests for OperatorBinding dataclass."""

    def test_create_binding(self):
        """Should create binding with required fields."""
        binding = OperatorBinding(
            operator_id="two_opt",
            role=AbstractRole.LS_INTENSIFY_SMALL,
            domain="tsp",
        )
        assert binding.operator_id == "two_opt"
        assert binding.role == AbstractRole.LS_INTENSIFY_SMALL
        assert binding.domain == "tsp"
        assert binding.priority == 1  # default
        assert binding.weight == 1.0  # default

    def test_binding_with_priority(self):
        """Should respect priority setting."""
        binding = OperatorBinding(
            operator_id="two_opt",
            role=AbstractRole.LS_INTENSIFY_SMALL,
            domain="tsp",
            priority=3,
            weight=2.0,
        )
        assert binding.priority == 3
        assert binding.weight == 2.0

    def test_auto_description(self):
        """Should generate description if not provided."""
        binding = OperatorBinding(
            operator_id="two_opt",
            role=AbstractRole.LS_INTENSIFY_SMALL,
            domain="tsp",
        )
        assert "two_opt" in binding.description
        assert "ls_intensify_small" in binding.description


class TestDomainBindings:
    """Tests for DomainBindings class."""

    @pytest.fixture
    def empty_bindings(self):
        """Create empty domain bindings."""
        return DomainBindings(domain="test")

    @pytest.fixture
    def sample_bindings(self):
        """Create sample domain bindings."""
        bindings = DomainBindings(domain="test")
        bindings.add_binding(OperatorBinding(
            operator_id="op1",
            role=AbstractRole.LS_INTENSIFY_SMALL,
            domain="test",
            priority=2,
            weight=1.5,
        ))
        bindings.add_binding(OperatorBinding(
            operator_id="op2",
            role=AbstractRole.LS_INTENSIFY_SMALL,
            domain="test",
            priority=1,
            weight=1.0,
        ))
        bindings.add_binding(OperatorBinding(
            operator_id="op3",
            role=AbstractRole.CONST_GREEDY,
            domain="test",
            priority=1,
            weight=1.0,
        ))
        return bindings

    def test_add_binding(self, empty_bindings):
        """Should add binding to domain."""
        empty_bindings.add_binding(OperatorBinding(
            operator_id="test_op",
            role=AbstractRole.CONST_GREEDY,
            domain="test",
        ))
        assert empty_bindings.has_role(AbstractRole.CONST_GREEDY)

    def test_get_operators_for_role(self, sample_bindings):
        """Should return all operators for a role."""
        ops = sample_bindings.get_operators_for_role(AbstractRole.LS_INTENSIFY_SMALL)
        assert len(ops) == 2
        assert "op1" in ops
        assert "op2" in ops

    def test_get_operators_for_unbound_role(self, sample_bindings):
        """Should return empty list for unbound role."""
        ops = sample_bindings.get_operators_for_role(AbstractRole.LS_CHAIN)
        assert ops == []

    def test_get_primary_operator_by_priority(self, sample_bindings):
        """Should return highest priority operator."""
        primary = sample_bindings.get_primary_operator(AbstractRole.LS_INTENSIFY_SMALL)
        assert primary == "op1"  # priority 2 > priority 1

    def test_get_primary_operator_unbound(self, sample_bindings):
        """Should return None for unbound role."""
        primary = sample_bindings.get_primary_operator(AbstractRole.LS_CHAIN)
        assert primary is None

    def test_select_operator_deterministic(self, sample_bindings):
        """Deterministic selection should return primary."""
        selected = sample_bindings.select_operator(
            AbstractRole.LS_INTENSIFY_SMALL,
            mode="primary"
        )
        assert selected == "op1"

    def test_select_operator_weighted(self, sample_bindings):
        """Weighted selection should work without error."""
        # Run multiple times to test stochastic selection
        selections = set()
        for _ in range(100):
            selected = sample_bindings.select_operator(
                AbstractRole.LS_INTENSIFY_SMALL,
                mode="weighted"
            )
            selections.add(selected)

        # With weights 1.5 and 1.0, both should be selected sometimes
        assert "op1" in selections or "op2" in selections

    def test_has_role(self, sample_bindings):
        """has_role should return correct boolean."""
        assert sample_bindings.has_role(AbstractRole.LS_INTENSIFY_SMALL)
        assert sample_bindings.has_role(AbstractRole.CONST_GREEDY)
        assert not sample_bindings.has_role(AbstractRole.LS_CHAIN)

    def test_get_bound_roles(self, sample_bindings):
        """Should return list of bound roles."""
        roles = sample_bindings.get_bound_roles()
        assert AbstractRole.LS_INTENSIFY_SMALL in roles
        assert AbstractRole.CONST_GREEDY in roles
        assert len(roles) == 2

    def test_repr(self, sample_bindings):
        """repr should show domain and counts."""
        r = repr(sample_bindings)
        assert "test" in r
        assert "roles=2" in r
        assert "operators=3" in r


class TestBindingRegistry:
    """Tests for BindingRegistry singleton."""

    def setup_method(self):
        """Reset singleton before each test."""
        BindingRegistry.reset()

    def test_singleton_pattern(self):
        """Should return same instance."""
        r1 = BindingRegistry()
        r2 = BindingRegistry()
        assert r1 is r2

    def test_has_default_domains(self):
        """Should have TSP and JSSP by default."""
        registry = BindingRegistry()
        assert registry.has_domain("tsp")
        assert registry.has_domain("jssp")

    def test_list_domains(self):
        """Should list registered domains."""
        registry = BindingRegistry()
        domains = registry.list_domains()
        assert "tsp" in domains
        assert "jssp" in domains

    def test_get_domain(self):
        """Should return domain bindings."""
        registry = BindingRegistry()
        tsp = registry.get_domain("tsp")
        assert isinstance(tsp, DomainBindings)
        assert tsp.domain == "tsp"

    def test_get_nonexistent_domain(self):
        """Should return None for unknown domain."""
        registry = BindingRegistry()
        unknown = registry.get_domain("unknown")
        assert unknown is None

    def test_register_custom_domain(self):
        """Should allow registering new domains."""
        registry = BindingRegistry()
        custom = DomainBindings(domain="custom")
        custom.add_binding(OperatorBinding(
            operator_id="custom_op",
            role=AbstractRole.CONST_GREEDY,
            domain="custom",
        ))
        registry.register_domain(custom)

        assert registry.has_domain("custom")
        retrieved = registry.get_domain("custom")
        assert retrieved.get_primary_operator(AbstractRole.CONST_GREEDY) == "custom_op"


class TestTSPBindings:
    """Tests for TSP domain bindings."""

    @pytest.fixture
    def tsp_bindings(self):
        """Get TSP bindings."""
        return create_tsp_bindings()

    def test_domain_is_tsp(self, tsp_bindings):
        """Domain should be 'tsp'."""
        assert tsp_bindings.domain == "tsp"

    def test_has_construction_roles(self, tsp_bindings):
        """Should have all construction roles bound."""
        assert tsp_bindings.has_role(AbstractRole.CONST_GREEDY)
        assert tsp_bindings.has_role(AbstractRole.CONST_INSERTION)
        assert tsp_bindings.has_role(AbstractRole.CONST_SAVINGS)
        assert tsp_bindings.has_role(AbstractRole.CONST_RANDOM)

    def test_has_local_search_roles(self, tsp_bindings):
        """Should have all local search roles bound."""
        assert tsp_bindings.has_role(AbstractRole.LS_INTENSIFY_SMALL)
        assert tsp_bindings.has_role(AbstractRole.LS_INTENSIFY_MEDIUM)
        assert tsp_bindings.has_role(AbstractRole.LS_INTENSIFY_LARGE)
        assert tsp_bindings.has_role(AbstractRole.LS_CHAIN)

    def test_has_perturbation_roles(self, tsp_bindings):
        """Should have all perturbation roles bound."""
        assert tsp_bindings.has_role(AbstractRole.PERT_ESCAPE_SMALL)
        assert tsp_bindings.has_role(AbstractRole.PERT_ESCAPE_LARGE)
        assert tsp_bindings.has_role(AbstractRole.PERT_ADAPTIVE)

    def test_all_11_roles_bound(self, tsp_bindings):
        """All 11 roles should have at least one binding."""
        bound = tsp_bindings.get_bound_roles()
        assert len(bound) == 11

    def test_two_opt_is_primary_for_small_ls(self, tsp_bindings):
        """two_opt should be primary for LS_INTENSIFY_SMALL."""
        primary = tsp_bindings.get_primary_operator(AbstractRole.LS_INTENSIFY_SMALL)
        assert primary == "two_opt"

    def test_greedy_nn_is_primary_for_greedy(self, tsp_bindings):
        """greedy_nearest_neighbor should be primary for CONST_GREEDY."""
        primary = tsp_bindings.get_primary_operator(AbstractRole.CONST_GREEDY)
        assert primary == "greedy_nearest_neighbor"

    def test_double_bridge_is_primary_for_escape(self, tsp_bindings):
        """double_bridge should be primary for PERT_ESCAPE_SMALL."""
        primary = tsp_bindings.get_primary_operator(AbstractRole.PERT_ESCAPE_SMALL)
        assert primary == "double_bridge"

    def test_multiple_operators_for_small_ls(self, tsp_bindings):
        """LS_INTENSIFY_SMALL should have multiple operators."""
        ops = tsp_bindings.get_operators_for_role(AbstractRole.LS_INTENSIFY_SMALL)
        assert len(ops) >= 3
        assert "two_opt" in ops
        assert "swap" in ops


class TestJSSPBindings:
    """Tests for JSSP domain bindings."""

    @pytest.fixture
    def jssp_bindings(self):
        """Get JSSP bindings."""
        return create_jssp_bindings()

    def test_domain_is_jssp(self, jssp_bindings):
        """Domain should be 'jssp'."""
        assert jssp_bindings.domain == "jssp"

    def test_all_11_roles_bound(self, jssp_bindings):
        """All 11 roles should have at least one binding."""
        bound = jssp_bindings.get_bound_roles()
        assert len(bound) == 11

    def test_spt_dispatch_is_primary_for_greedy(self, jssp_bindings):
        """spt_dispatch should be primary for CONST_GREEDY."""
        primary = jssp_bindings.get_primary_operator(AbstractRole.CONST_GREEDY)
        assert primary == "spt_dispatch"

    def test_adjacent_swap_in_small_ls(self, jssp_bindings):
        """adjacent_swap should be in LS_INTENSIFY_SMALL."""
        ops = jssp_bindings.get_operators_for_role(AbstractRole.LS_INTENSIFY_SMALL)
        assert "adjacent_swap" in ops or "critical_swap" in ops


class TestBindingStats:
    """Tests for binding statistics."""

    def setup_method(self):
        """Reset singleton before each test."""
        BindingRegistry.reset()

    def test_get_binding_stats_unknown_domain(self):
        """Should return error for unknown domain."""
        stats = get_binding_stats("unknown")
        assert "error" in stats


class TestCrossdomainConsistency:
    """Tests verifying TSP and JSSP use same role vocabulary."""

    def setup_method(self):
        """Reset singleton before each test."""
        BindingRegistry.reset()

    def test_same_roles_both_domains(self):
        """Both domains should bind all 11 roles."""
        tsp = create_tsp_bindings()
        jssp = create_jssp_bindings()

        tsp_roles = set(tsp.get_bound_roles())
        jssp_roles = set(jssp.get_bound_roles())

        assert tsp_roles == jssp_roles
        assert len(tsp_roles) == 11

    def test_different_operators_same_roles(self):
        """Same role should have different operators per domain."""
        tsp = create_tsp_bindings()
        jssp = create_jssp_bindings()

        # CONST_GREEDY should differ
        tsp_greedy = tsp.get_primary_operator(AbstractRole.CONST_GREEDY)
        jssp_greedy = jssp.get_primary_operator(AbstractRole.CONST_GREEDY)
        assert tsp_greedy != jssp_greedy
        assert tsp_greedy == "greedy_nearest_neighbor"
        assert jssp_greedy == "spt_dispatch"

        # LS_INTENSIFY_SMALL should differ
        tsp_ls = tsp.get_primary_operator(AbstractRole.LS_INTENSIFY_SMALL)
        jssp_ls = jssp.get_primary_operator(AbstractRole.LS_INTENSIFY_SMALL)
        assert tsp_ls != jssp_ls
        assert tsp_ls == "two_opt"
        assert jssp_ls in ["adjacent_swap", "critical_swap"]
