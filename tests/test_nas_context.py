"""Tests for NAS domain: architecture, context, search space, operators."""

import pytest

from src.domains.nas.architecture import NeuralArchitecture, ArchitectureLayer
from src.domains.nas.search_space import NASSearchSpace
from src.domains.nas.context import NASContext
from src.domains.nas.config import NASDomainConfig


@pytest.fixture
def search_space():
    return NASSearchSpace()


@pytest.fixture
def context(search_space):
    return NASContext(search_space=search_space, dataset="cifar10")


@pytest.fixture
def sample_arch():
    """A simple 3-layer architecture."""
    return NeuralArchitecture(
        layers=[
            ArchitectureLayer(layer_id=0, layer_type="linear", units=128, activation="relu"),
            ArchitectureLayer(layer_id=1, layer_type="linear", units=64, activation="gelu"),
            ArchitectureLayer(layer_id=2, layer_type="linear", units=32, activation="silu"),
        ],
        skip_connections=[(0, 2)],
        optimizer="adam",
        learning_rate=1e-3,
    )


class TestArchitectureLayer:
    def test_param_count_linear(self):
        layer = ArchitectureLayer(layer_id=0, layer_type="linear", units=128)
        assert layer.param_count() == 128 * 128

    def test_param_count_conv2d(self):
        layer = ArchitectureLayer(layer_id=0, layer_type="conv2d", units=64, kernel_size=3)
        assert layer.param_count() == 64 * 3 * 3


class TestNeuralArchitecture:
    def test_depth(self, sample_arch):
        assert sample_arch.depth() == 3

    def test_total_params(self, sample_arch):
        expected = 128 * 128 + 64 * 64 + 32 * 32
        assert sample_arch.total_params() == expected

    def test_copy_is_deep(self, sample_arch):
        copied = sample_arch.copy()
        assert copied is not sample_arch
        assert copied.layers is not sample_arch.layers
        copied.layers[0].units = 999
        assert sample_arch.layers[0].units == 128

    def test_to_dict(self, sample_arch):
        d = sample_arch.to_dict()
        assert len(d["layers"]) == 3
        assert d["optimizer"] == "adam"
        assert d["skip_connections"] == [(0, 2)]

    def test_random_creates_valid(self):
        arch = NeuralArchitecture.random()
        assert 2 <= arch.depth() <= 6
        assert arch.optimizer in ("adam", "adamw", "sgd")


class TestNASSearchSpace:
    def test_random_architecture_valid(self, search_space):
        for _ in range(10):
            arch = search_space.random_architecture()
            assert search_space.is_valid(arch)

    def test_too_many_layers_invalid(self, search_space):
        layers = [
            ArchitectureLayer(layer_id=i, layer_type="linear", units=32, activation="relu")
            for i in range(20)
        ]
        arch = NeuralArchitecture(layers=layers)
        assert not search_space.is_valid(arch)

    def test_too_few_layers_invalid(self, search_space):
        arch = NeuralArchitecture(layers=[
            ArchitectureLayer(layer_id=0, layer_type="linear", units=32, activation="relu"),
        ])
        assert not search_space.is_valid(arch)

    def test_invalid_skip_connection(self, search_space):
        arch = NeuralArchitecture(
            layers=[
                ArchitectureLayer(layer_id=0, layer_type="linear", units=64, activation="relu"),
                ArchitectureLayer(layer_id=1, layer_type="linear", units=64, activation="relu"),
                ArchitectureLayer(layer_id=2, layer_type="linear", units=64, activation="relu"),
            ],
            skip_connections=[(2, 0)],  # Invalid: source >= target
        )
        assert not search_space.is_valid(arch)


class TestNASContext:
    def test_family_is_architecture(self, context):
        from src.geakg.contexts.base import OptimizationFamily
        assert context.family == OptimizationFamily.ARCHITECTURE

    def test_domain_is_nas(self, context):
        assert context.domain == "nas"

    def test_evaluate_returns_negative(self, context, sample_arch):
        """Fitness = negative accuracy, so should be <= 0."""
        fitness = context.evaluate(sample_arch)
        assert fitness <= 0

    def test_valid_architecture(self, context, sample_arch):
        assert context.valid(sample_arch)

    def test_random_solution_is_valid(self, context):
        for _ in range(5):
            arch = context.random_solution()
            assert context.valid(arch)

    def test_copy(self, context, sample_arch):
        copied = context.copy(sample_arch)
        assert copied is not sample_arch
        assert copied.depth() == sample_arch.depth()

    def test_instance_data(self, context):
        data = context.instance_data
        assert "dimension" in data
        assert "dataset" in data
        assert data["dataset"] == "cifar10"


class TestNASDomainConfig:
    def test_create_config(self):
        config = NASDomainConfig(dataset="cifar100")
        assert config.dataset == "cifar100"
        assert config.search_space is not None

    def test_create_context(self):
        config = NASDomainConfig()
        ctx = config.create_context()
        assert ctx.domain == "nas"

    def test_validate_solution(self, sample_arch):
        config = NASDomainConfig()
        assert config.validate_solution(sample_arch)
        assert not config.validate_solution("not an architecture")


class TestNASOperators:
    """Test the generic architecture operators."""

    def test_add_layer(self, sample_arch, context):
        from src.geakg.generic_operators.architecture import add_layer
        result = add_layer(sample_arch, context)
        assert result.depth() == sample_arch.depth() + 1

    def test_remove_layer(self, sample_arch, context):
        from src.geakg.generic_operators.architecture import remove_layer
        result = remove_layer(sample_arch, context)
        assert result.depth() == sample_arch.depth() - 1

    def test_remove_layer_minimum(self, context):
        """Cannot remove below 2 layers."""
        from src.geakg.generic_operators.architecture import remove_layer
        arch = NeuralArchitecture(layers=[
            ArchitectureLayer(layer_id=0, layer_type="linear", units=64, activation="relu"),
            ArchitectureLayer(layer_id=1, layer_type="linear", units=32, activation="relu"),
        ])
        result = remove_layer(arch, context)
        assert result.depth() == 2

    def test_add_skip_connection(self, context):
        from src.geakg.generic_operators.architecture import add_skip_connection
        arch = NeuralArchitecture(layers=[
            ArchitectureLayer(layer_id=i, layer_type="linear", units=64, activation="relu")
            for i in range(4)
        ])
        result = add_skip_connection(arch, context)
        assert len(result.skip_connections) >= len(arch.skip_connections)

    def test_change_optimizer(self, sample_arch, context):
        from src.geakg.generic_operators.architecture import change_optimizer
        result = change_optimizer(sample_arch, context)
        # Should change (or at least return valid)
        assert result.optimizer in ("sgd", "adam", "adamw")

    def test_adjust_dropout(self, sample_arch, context):
        from src.geakg.generic_operators.architecture import adjust_dropout
        result = adjust_dropout(sample_arch, context)
        assert any(l.dropout >= 0 for l in result.layers)

    def test_change_normalization(self, sample_arch, context):
        from src.geakg.generic_operators.architecture import change_normalization
        result = change_normalization(sample_arch, context)
        valid_norms = {"none", "batch", "layer", "group"}
        assert all(l.normalization in valid_norms for l in result.layers)

    def test_validate_architecture_fixes_skips(self, context):
        from src.geakg.generic_operators.architecture import validate_architecture
        arch = NeuralArchitecture(
            layers=[
                ArchitectureLayer(layer_id=0, layer_type="linear", units=64, activation="relu"),
                ArchitectureLayer(layer_id=1, layer_type="linear", units=64, activation="relu"),
            ],
            skip_connections=[(0, 5), (3, 1)],  # Invalid skips
        )
        result = validate_architecture(arch, context)
        # Invalid skip connections should be removed
        for s, t in result.skip_connections:
            assert 0 <= s < len(result.layers)
            assert 0 <= t < len(result.layers)
            assert s < t


class TestNASOperatorDispatcher:
    """Test the operator dispatch in src/domains/nas/operators.py."""

    def test_known_operator(self, sample_arch, context):
        from src.domains.nas.operators import apply_nas_operator
        result = apply_nas_operator("change_optimizer", sample_arch, context)
        assert result is not None
        assert isinstance(result, NeuralArchitecture)

    def test_unknown_operator_returns_none(self, sample_arch, context):
        from src.domains.nas.operators import apply_nas_operator
        result = apply_nas_operator("nonexistent_op", sample_arch, context)
        assert result is None
