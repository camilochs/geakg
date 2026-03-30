"""Test that the same GEAKG framework works with both case studies.

This is the key generality test: demonstrates that MetaGraph, Bindings,
and the ACO-related infrastructure work identically with both
OptimizationRoleSchema and NASRoleSchema, without any code changes.
"""

import pytest

from src.geakg.core.role_schema import RoleSchema
from src.geakg.core.schemas.optimization import OptimizationRoleSchema
from src.geakg.core.schemas.nas import NASRoleSchema
from src.geakg.core.case_study import CaseStudy
from src.geakg.layers.l0.metagraph import MetaGraph, MetaEdge
from src.geakg.layers.l0.patterns import (
    create_hybrid_meta_graph,
    create_nas_meta_graph,
)


@pytest.fixture
def opt_schema():
    return OptimizationRoleSchema()


@pytest.fixture
def nas_schema():
    return NASRoleSchema()


class TestMetaGraphGenerality:
    """Same MetaGraph class works with both schemas."""

    def test_optimization_meta_graph(self, opt_schema):
        mg = create_hybrid_meta_graph()
        # Should have nodes and edges
        assert len(mg.nodes) > 0
        assert len(mg.edges) > 0

    def test_nas_meta_graph(self, nas_schema):
        mg = create_nas_meta_graph(nas_schema)
        assert mg.role_schema is nas_schema
        assert len(mg.nodes) > 0
        assert len(mg.edges) > 0

    def test_both_have_entry_roles(self, opt_schema, nas_schema):
        """Both schemas should produce MetaGraphs with entry roles."""
        opt_mg = create_hybrid_meta_graph()
        nas_mg = create_nas_meta_graph(nas_schema)

        # Optimization: construction roles are entry
        opt_entry = opt_mg.get_construction_roles()
        assert len(opt_entry) > 0

        # NAS: topology roles are entry
        nas_entry = nas_mg.get_entry_roles()
        assert len(nas_entry) > 0

    def test_get_successors_works_for_both(self, nas_schema):
        """get_successors() returns string roles for both."""
        opt_mg = create_hybrid_meta_graph()
        nas_mg = create_nas_meta_graph(nas_schema)

        # Optimization: successors of construction roles
        opt_construction = opt_mg.get_construction_roles()
        if opt_construction:
            successors = opt_mg.get_successors(opt_construction[0])
            assert all(isinstance(s, str) for s in successors)

        # NAS: successors of topology roles
        nas_entry = nas_mg.get_entry_roles()
        if nas_entry:
            successors = nas_mg.get_successors(nas_entry[0])
            assert all(isinstance(s, str) for s in successors)

    def test_validate_transitions_both(self, nas_schema):
        """validate_transitions passes for both schemas."""
        opt_mg = create_hybrid_meta_graph()
        nas_mg = create_nas_meta_graph(nas_schema)

        # Both should have valid transitions (no exception raised)
        opt_mg.validate_transitions()
        nas_mg.validate_transitions()


class TestCaseStudyFactory:
    """CaseStudy factory methods work correctly."""

    def test_optimization_case_study(self):
        cs = CaseStudy.optimization(domain="tsp", pattern="hybrid")
        assert cs.name == "optimization_tsp"
        assert isinstance(cs.role_schema, OptimizationRoleSchema)
        assert len(cs.base_operators) == 11

    def test_nas_case_study(self):
        cs = CaseStudy.nas(dataset="cifar10")
        assert cs.name == "nas_cifar10"
        assert isinstance(cs.role_schema, NASRoleSchema)
        assert len(cs.base_operators) == 18

    def test_optimization_creates_meta_graph(self):
        cs = CaseStudy.optimization()
        mg = cs.create_meta_graph()
        assert isinstance(mg, MetaGraph)
        assert len(mg.edges) > 0

    def test_nas_creates_meta_graph(self):
        cs = CaseStudy.nas()
        mg = cs.create_meta_graph()
        assert isinstance(mg, MetaGraph)
        assert len(mg.edges) > 0
        assert mg.role_schema is not None

    def test_optimization_base_operators_cover_all_roles(self):
        cs = CaseStudy.optimization()
        warnings = cs.validate()
        assert len(warnings) == 0

    def test_nas_base_operators_cover_all_roles(self):
        cs = CaseStudy.nas()
        warnings = cs.validate()
        assert len(warnings) == 0

    def test_get_base_operator(self):
        cs = CaseStudy.optimization()
        code = cs.get_base_operator("const_greedy")
        assert "def const_greedy" in code

    def test_nas_get_base_operator(self):
        cs = CaseStudy.nas()
        code = cs.get_base_operator("topo_feedforward")
        assert "def topo_feedforward" in code

    def test_invalid_role_raises(self):
        cs = CaseStudy.optimization()
        with pytest.raises(KeyError):
            cs.get_base_operator("topo_feedforward")  # NAS role in optimization

    def test_different_patterns(self):
        for pattern in ["ils", "vns", "hybrid"]:
            cs = CaseStudy.optimization(pattern=pattern)
            mg = cs.create_meta_graph()
            assert len(mg.edges) > 0

    def test_different_datasets(self):
        for dataset in ["cifar10", "cifar100"]:
            cs = CaseStudy.nas(dataset=dataset)
            assert dataset in cs.name


class TestSameCodeDifferentSchemas:
    """The crucial generality test: identical code path, different schemas."""

    @pytest.mark.parametrize("cs_factory,expected_roles", [
        (lambda: CaseStudy.optimization(), 11),
        (lambda: CaseStudy.nas(), 18),
    ])
    def test_get_all_roles(self, cs_factory, expected_roles):
        cs = cs_factory()
        assert len(cs.get_all_roles()) == expected_roles

    @pytest.mark.parametrize("cs_factory", [
        lambda: CaseStudy.optimization(),
        lambda: CaseStudy.nas(),
    ])
    def test_meta_graph_has_entry_roles(self, cs_factory):
        cs = cs_factory()
        mg = cs.create_meta_graph()
        entry = mg.get_entry_roles()
        assert len(entry) > 0

    @pytest.mark.parametrize("cs_factory", [
        lambda: CaseStudy.optimization(),
        lambda: CaseStudy.nas(),
    ])
    def test_schema_transitions_are_consistent(self, cs_factory):
        """All transitions in MetaGraph respect the schema."""
        cs = cs_factory()
        mg = cs.create_meta_graph()
        schema = cs.role_schema

        for edge in mg.edges.values():
            src = edge.source_str
            tgt = edge.target_str

            src_cat = schema.get_role_category(src)
            tgt_cat = schema.get_role_category(tgt)

            # Transition must be valid per schema
            transitions = schema.get_category_transitions()
            if src_cat != tgt_cat:
                assert tgt_cat in transitions.get(src_cat, []), (
                    f"Invalid transition: {src} ({src_cat}) -> {tgt} ({tgt_cat})"
                )
