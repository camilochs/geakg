"""Tests for RoleSchema protocol and both implementations.

Verifies that OptimizationRoleSchema and NASRoleSchema both satisfy
the RoleSchema abstract interface with consistent behavior.
"""

import pytest

from src.geakg.core.role_schema import RoleSchema
from src.geakg.core.schemas.optimization import OptimizationRoleSchema
from src.geakg.core.schemas.nas import NASRoleSchema


@pytest.fixture
def opt_schema():
    return OptimizationRoleSchema()


@pytest.fixture
def nas_schema():
    return NASRoleSchema()


class TestRoleSchemaProtocol:
    """Both schemas must satisfy the RoleSchema protocol."""

    @pytest.mark.parametrize("schema_cls", [OptimizationRoleSchema, NASRoleSchema])
    def test_is_role_schema(self, schema_cls):
        schema = schema_cls()
        assert isinstance(schema, RoleSchema)

    @pytest.mark.parametrize("schema_cls", [OptimizationRoleSchema, NASRoleSchema])
    def test_get_all_roles_not_empty(self, schema_cls):
        schema = schema_cls()
        roles = schema.get_all_roles()
        assert len(roles) > 0
        assert all(isinstance(r, str) for r in roles)

    @pytest.mark.parametrize("schema_cls", [OptimizationRoleSchema, NASRoleSchema])
    def test_get_categories_not_empty(self, schema_cls):
        schema = schema_cls()
        categories = schema.get_categories()
        assert len(categories) > 0
        assert all(isinstance(c, str) for c in categories)

    @pytest.mark.parametrize("schema_cls", [OptimizationRoleSchema, NASRoleSchema])
    def test_every_role_has_category(self, schema_cls):
        schema = schema_cls()
        for role in schema.get_all_roles():
            cat = schema.get_role_category(role)
            assert cat in schema.get_categories()

    @pytest.mark.parametrize("schema_cls", [OptimizationRoleSchema, NASRoleSchema])
    def test_entry_categories_subset_of_categories(self, schema_cls):
        schema = schema_cls()
        entry = schema.get_entry_categories()
        assert len(entry) > 0
        for cat in entry:
            assert cat in schema.get_categories()

    @pytest.mark.parametrize("schema_cls", [OptimizationRoleSchema, NASRoleSchema])
    def test_roles_by_category_covers_all(self, schema_cls):
        schema = schema_cls()
        all_from_categories = []
        for cat in schema.get_categories():
            all_from_categories.extend(schema.get_roles_by_category(cat))
        assert set(all_from_categories) == set(schema.get_all_roles())

    @pytest.mark.parametrize("schema_cls", [OptimizationRoleSchema, NASRoleSchema])
    def test_transitions_reference_valid_categories(self, schema_cls):
        schema = schema_cls()
        transitions = schema.get_category_transitions()
        valid_cats = set(schema.get_categories())
        for src, targets in transitions.items():
            assert src in valid_cats, f"Source '{src}' not in categories"
            for tgt in targets:
                assert tgt in valid_cats, f"Target '{tgt}' not in categories"

    @pytest.mark.parametrize("schema_cls", [OptimizationRoleSchema, NASRoleSchema])
    def test_role_metadata_has_description(self, schema_cls):
        schema = schema_cls()
        for role in schema.get_all_roles():
            meta = schema.get_role_metadata(role)
            assert "description" in meta
            assert len(meta["description"]) > 0

    @pytest.mark.parametrize("schema_cls", [OptimizationRoleSchema, NASRoleSchema])
    def test_llm_description_not_empty(self, schema_cls):
        schema = schema_cls()
        desc = schema.get_role_description_for_llm()
        assert len(desc) > 100

    @pytest.mark.parametrize("schema_cls", [OptimizationRoleSchema, NASRoleSchema])
    def test_is_valid_role(self, schema_cls):
        schema = schema_cls()
        for role in schema.get_all_roles():
            assert schema.is_valid_role(role)
        assert not schema.is_valid_role("nonexistent_role_xyz")

    @pytest.mark.parametrize("schema_cls", [OptimizationRoleSchema, NASRoleSchema])
    def test_same_role_transition_always_valid(self, schema_cls):
        schema = schema_cls()
        for role in schema.get_all_roles():
            assert schema.is_valid_transition(role, role)

    @pytest.mark.parametrize("schema_cls", [OptimizationRoleSchema, NASRoleSchema])
    def test_entry_roles_belong_to_entry_categories(self, schema_cls):
        schema = schema_cls()
        entry_roles = schema.get_entry_roles()
        entry_cats = set(schema.get_entry_categories())
        for role in entry_roles:
            assert schema.get_role_category(role) in entry_cats

    @pytest.mark.parametrize("schema_cls", [OptimizationRoleSchema, NASRoleSchema])
    def test_invalid_role_raises_key_error(self, schema_cls):
        schema = schema_cls()
        with pytest.raises(KeyError):
            schema.get_role_category("completely_fake_role")
        with pytest.raises(KeyError):
            schema.get_role_metadata("completely_fake_role")


class TestOptimizationSchema:
    """Optimization-specific schema tests."""

    def test_has_11_roles(self, opt_schema):
        assert len(opt_schema.get_all_roles()) == 11

    def test_has_3_categories(self, opt_schema):
        cats = opt_schema.get_categories()
        assert set(cats) == {"construction", "local_search", "perturbation"}

    def test_entry_is_construction(self, opt_schema):
        assert opt_schema.get_entry_categories() == ["construction"]

    def test_construction_has_4_roles(self, opt_schema):
        assert len(opt_schema.get_roles_by_category("construction")) == 4

    def test_local_search_has_4_roles(self, opt_schema):
        assert len(opt_schema.get_roles_by_category("local_search")) == 4

    def test_perturbation_has_3_roles(self, opt_schema):
        assert len(opt_schema.get_roles_by_category("perturbation")) == 3

    def test_construction_not_revisitable(self, opt_schema):
        assert not opt_schema.is_revisitable_category("construction")

    def test_local_search_revisitable(self, opt_schema):
        assert opt_schema.is_revisitable_category("local_search")

    def test_perturbation_revisitable(self, opt_schema):
        assert opt_schema.is_revisitable_category("perturbation")


class TestNASSchema:
    """NAS-specific schema tests."""

    def test_has_18_roles(self, nas_schema):
        assert len(nas_schema.get_all_roles()) == 18

    def test_has_5_categories(self, nas_schema):
        cats = nas_schema.get_categories()
        assert set(cats) == {"topology", "activation", "training", "regularization", "evaluation"}

    def test_entry_is_topology(self, nas_schema):
        assert nas_schema.get_entry_categories() == ["topology"]

    def test_topology_has_4_roles(self, nas_schema):
        roles = nas_schema.get_roles_by_category("topology")
        assert len(roles) == 4
        assert "topo_feedforward" in roles
        assert "topo_residual" in roles

    def test_activation_has_4_roles(self, nas_schema):
        assert len(nas_schema.get_roles_by_category("activation")) == 4

    def test_training_has_4_roles(self, nas_schema):
        assert len(nas_schema.get_roles_by_category("training")) == 4

    def test_regularization_has_4_roles(self, nas_schema):
        assert len(nas_schema.get_roles_by_category("regularization")) == 4

    def test_evaluation_has_2_roles(self, nas_schema):
        assert len(nas_schema.get_roles_by_category("evaluation")) == 2

    def test_evaluation_not_revisitable(self, nas_schema):
        assert not nas_schema.is_revisitable_category("evaluation")

    def test_topology_revisitable(self, nas_schema):
        assert nas_schema.is_revisitable_category("topology")

    def test_topology_to_activation_valid(self, nas_schema):
        assert nas_schema.is_valid_transition("topo_feedforward", "act_standard")

    def test_activation_to_training_valid(self, nas_schema):
        assert nas_schema.is_valid_transition("act_modern", "train_optimizer")

    def test_training_to_regularization_valid(self, nas_schema):
        assert nas_schema.is_valid_transition("train_schedule", "reg_dropout")

    def test_regularization_to_evaluation_valid(self, nas_schema):
        assert nas_schema.is_valid_transition("reg_structural", "eval_proxy")

    def test_evaluation_to_topology_valid(self, nas_schema):
        assert nas_schema.is_valid_transition("eval_proxy", "topo_residual")

    def test_topology_to_training_invalid(self, nas_schema):
        """Cannot skip activation and go directly to training."""
        assert not nas_schema.is_valid_transition("topo_feedforward", "train_optimizer")
