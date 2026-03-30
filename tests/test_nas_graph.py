"""Tests for NAS-Bench-Graph integration.

Tests:
- GraphArchitecture: construction, validation, copy, serialization, Hamming
- NASBenchGraphEvaluator: proxy evaluation, accuracy range, determinism
- NASBenchGraphContext: interface compliance
- 18 graph operators: all produce valid architectures
- Base operators (code strings): all compile and run correctly
"""

import random

import pytest

from src.domains.nas.graph_architecture import (
    GraphArchitecture,
    GRAPH_NUM_NODES,
    GRAPH_NUM_OPS,
    GRAPH_OPERATIONS,
    GRAPH_MAX_CONN,
)
from src.domains.nas.graph_evaluator import NASBenchGraphEvaluator
from src.domains.nas.graph_context import NASBenchGraphContext


# =============================================================================
# GraphArchitecture Tests
# =============================================================================


class TestGraphArchitecture:
    """Tests for GraphArchitecture dataclass."""

    def test_default_construction(self):
        arch = GraphArchitecture()
        assert len(arch.connectivity) == GRAPH_NUM_NODES
        assert len(arch.operations) == GRAPH_NUM_NODES
        assert all(c == 0 for c in arch.connectivity)
        assert all(o == 0 for o in arch.operations)

    def test_custom_construction(self):
        arch = GraphArchitecture(
            connectivity=[0, 1, 2, 3],
            operations=[0, 1, 2, 3],
        )
        assert arch.connectivity == [0, 1, 2, 3]
        assert arch.operations == [0, 1, 2, 3]

    def test_invalid_connectivity_length(self):
        with pytest.raises(ValueError, match="Expected 4 connectivity"):
            GraphArchitecture(connectivity=[0, 1], operations=[0, 0, 0, 0])

    def test_invalid_operations_length(self):
        with pytest.raises(ValueError, match="Expected 4 operations"):
            GraphArchitecture(connectivity=[0, 0, 0, 0], operations=[0, 1])

    def test_invalid_connectivity_value(self):
        with pytest.raises(ValueError, match="must be in"):
            GraphArchitecture(connectivity=[0, 1, 4, 0], operations=[0, 0, 0, 0])

    def test_negative_connectivity_value(self):
        with pytest.raises(ValueError, match="must be in"):
            GraphArchitecture(connectivity=[0, -1, 0, 0], operations=[0, 0, 0, 0])

    def test_invalid_operation_value(self):
        with pytest.raises(ValueError, match="must be in"):
            GraphArchitecture(connectivity=[0, 0, 0, 0], operations=[0, 1, 9, 0])

    def test_negative_operation_value(self):
        with pytest.raises(ValueError, match="must be in"):
            GraphArchitecture(connectivity=[0, 0, 0, 0], operations=[0, -1, 0, 0])

    def test_random(self):
        rng = random.Random(42)
        arch = GraphArchitecture.random(rng)
        assert len(arch.connectivity) == GRAPH_NUM_NODES
        assert len(arch.operations) == GRAPH_NUM_NODES
        assert all(0 <= c < GRAPH_MAX_CONN for c in arch.connectivity)
        assert all(0 <= o < GRAPH_NUM_OPS for o in arch.operations)

    def test_random_reproducible(self):
        a1 = GraphArchitecture.random(random.Random(42))
        a2 = GraphArchitecture.random(random.Random(42))
        assert a1.connectivity == a2.connectivity
        assert a1.operations == a2.operations

    def test_random_different_seeds(self):
        a1 = GraphArchitecture.random(random.Random(42))
        a2 = GraphArchitecture.random(random.Random(99))
        # Very unlikely to be equal with different seeds
        assert (a1.connectivity != a2.connectivity or
                a1.operations != a2.operations)

    def test_copy_independent(self):
        arch = GraphArchitecture(
            connectivity=[0, 1, 2, 3],
            operations=[0, 1, 2, 3],
        )
        copy = arch.copy()
        assert copy.connectivity == arch.connectivity
        assert copy.operations == arch.operations
        copy.connectivity[0] = 3
        copy.operations[0] = 8
        assert arch.connectivity[0] == 0
        assert arch.operations[0] == 0

    def test_to_dict(self):
        arch = GraphArchitecture(
            connectivity=[0, 1, 2, 3],
            operations=[0, 1, 2, 3],
        )
        d = arch.to_dict()
        assert d["connectivity"] == [0, 1, 2, 3]
        assert d["operations"] == [0, 1, 2, 3]
        assert "ops_names" in d
        assert d["ops_names"][0] == "gcn"
        assert d["ops_names"][3] == "gin"

    def test_from_dict(self):
        d = {"connectivity": [3, 2, 1, 0], "operations": [8, 7, 6, 5]}
        arch = GraphArchitecture.from_dict(d)
        assert arch.connectivity == [3, 2, 1, 0]
        assert arch.operations == [8, 7, 6, 5]

    def test_roundtrip_dict(self):
        original = GraphArchitecture.random(random.Random(42))
        restored = GraphArchitecture.from_dict(original.to_dict())
        assert original.connectivity == restored.connectivity
        assert original.operations == restored.operations

    def test_to_index(self):
        arch = GraphArchitecture(
            connectivity=[0, 0, 0, 0],
            operations=[0, 0, 0, 0],
        )
        assert arch.to_index() == 0

    def test_from_index_roundtrip(self):
        for seed in [0, 1, 42, 100]:
            rng = random.Random(seed)
            arch = GraphArchitecture.random(rng)
            idx = arch.to_index()
            restored = GraphArchitecture.from_index(idx)
            assert arch.connectivity == restored.connectivity, f"Failed for seed={seed}"
            assert arch.operations == restored.operations, f"Failed for seed={seed}"

    def test_hamming_distance_identical(self):
        arch = GraphArchitecture(
            connectivity=[0, 1, 2, 3],
            operations=[0, 1, 2, 3],
        )
        assert arch.hamming_distance(arch.copy()) == 0

    def test_hamming_distance_all_different(self):
        a = GraphArchitecture(
            connectivity=[0, 0, 0, 0],
            operations=[0, 0, 0, 0],
        )
        b = GraphArchitecture(
            connectivity=[1, 1, 1, 1],
            operations=[1, 1, 1, 1],
        )
        assert a.hamming_distance(b) == 8  # All 8 positions differ

    def test_hamming_distance_partial(self):
        a = GraphArchitecture(
            connectivity=[0, 1, 2, 3],
            operations=[0, 1, 2, 3],
        )
        b = GraphArchitecture(
            connectivity=[0, 1, 0, 0],
            operations=[0, 1, 0, 0],
        )
        assert a.hamming_distance(b) == 4

    def test_get_op_name(self):
        arch = GraphArchitecture(
            connectivity=[0, 0, 0, 0],
            operations=[0, 1, 7, 8],
        )
        assert arch.get_op_name(0) == "gcn"
        assert arch.get_op_name(1) == "gat"
        assert arch.get_op_name(2) == "identity"
        assert arch.get_op_name(3) == "fc"

    def test_num_gnn_ops(self):
        arch = GraphArchitecture(
            connectivity=[0, 0, 0, 0],
            operations=[0, 1, 7, 8],  # gcn, gat, identity, fc
        )
        assert arch.num_gnn_ops() == 2  # gcn and gat

    def test_num_identity(self):
        arch = GraphArchitecture(
            connectivity=[0, 0, 0, 0],
            operations=[7, 7, 0, 1],
        )
        assert arch.num_identity() == 2

    def test_num_fc(self):
        arch = GraphArchitecture(
            connectivity=[0, 0, 0, 0],
            operations=[8, 8, 8, 0],
        )
        assert arch.num_fc() == 3

    def test_hash_and_eq(self):
        a = GraphArchitecture(
            connectivity=[0, 1, 2, 3],
            operations=[0, 1, 2, 3],
        )
        b = GraphArchitecture(
            connectivity=[0, 1, 2, 3],
            operations=[0, 1, 2, 3],
        )
        assert a == b
        assert hash(a) == hash(b)
        s = {a, b}
        assert len(s) == 1

    def test_repr(self):
        arch = GraphArchitecture(
            connectivity=[0, 1, 2, 3],
            operations=[0, 1, 2, 3],
        )
        r = repr(arch)
        assert "GraphArchitecture" in r

    def test_operations_list(self):
        assert len(GRAPH_OPERATIONS) == 9
        assert GRAPH_OPERATIONS[0] == "gcn"
        assert GRAPH_OPERATIONS[8] == "fc"

    def test_constants(self):
        assert GRAPH_NUM_NODES == 4
        assert GRAPH_NUM_OPS == 9
        assert GRAPH_MAX_CONN == 4


# =============================================================================
# NASBenchGraphEvaluator Tests
# =============================================================================


class TestNASBenchGraphEvaluator:
    """Tests for NASBenchGraphEvaluator (proxy mode)."""

    @pytest.fixture
    def evaluator(self):
        return NASBenchGraphEvaluator(dataset="cora", use_proxy=True, seed=42)

    def test_proxy_mode(self, evaluator):
        assert evaluator.is_proxy is True

    def test_evaluate_returns_float(self, evaluator):
        arch = GraphArchitecture.random(random.Random(42))
        acc = evaluator.evaluate(arch)
        assert isinstance(acc, float)

    def test_accuracy_in_range(self, evaluator):
        """Accuracy should be in realistic range [40, 99]."""
        rng = random.Random(42)
        for _ in range(100):
            arch = GraphArchitecture.random(rng)
            acc = evaluator.evaluate(arch)
            assert 40.0 <= acc <= 99.0, f"Accuracy {acc} out of range"

    def test_deterministic(self, evaluator):
        """Same architecture should give same accuracy."""
        arch = GraphArchitecture(
            connectivity=[0, 1, 2, 0],
            operations=[0, 1, 3, 7],
        )
        acc1 = evaluator.evaluate(arch)
        acc2 = evaluator.evaluate(arch)
        assert acc1 == acc2

    def test_different_archs_different_acc(self, evaluator):
        """Different architectures should generally give different accuracies."""
        a = GraphArchitecture(
            connectivity=[0, 0, 0, 0],
            operations=[0, 0, 0, 0],
        )
        b = GraphArchitecture(
            connectivity=[0, 1, 2, 3],
            operations=[1, 3, 7, 8],
        )
        acc_a = evaluator.evaluate(a)
        acc_b = evaluator.evaluate(b)
        assert acc_a != acc_b

    def test_eval_count(self, evaluator):
        arch = GraphArchitecture.random(random.Random(42))
        assert evaluator.eval_count == 0
        evaluator.evaluate(arch)
        assert evaluator.eval_count == 1
        evaluator.evaluate(arch)  # cached, but still counts
        assert evaluator.eval_count == 2

    def test_reset_eval_count(self, evaluator):
        arch = GraphArchitecture.random(random.Random(42))
        evaluator.evaluate(arch)
        evaluator.reset_eval_count()
        assert evaluator.eval_count == 0

    def test_get_stats(self, evaluator):
        stats = evaluator.get_stats()
        assert stats["dataset"] == "cora"
        assert stats["is_proxy"] is True
        assert "eval_count" in stats

    def test_good_arch_higher_acc(self, evaluator):
        """A GNN-rich arch should have higher accuracy than all-identity."""
        good = GraphArchitecture(
            connectivity=[0, 1, 2, 0],
            operations=[0, 1, 3, 0],  # gcn, gat, gin, gcn
        )
        bad = GraphArchitecture(
            connectivity=[0, 0, 0, 0],
            operations=[7, 7, 7, 7],  # all identity
        )
        acc_good = evaluator.evaluate(good)
        acc_bad = evaluator.evaluate(bad)
        assert acc_good > acc_bad, (
            f"Good arch ({acc_good}) should have higher acc than bad ({acc_bad})"
        )

    def test_invalid_dataset(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            NASBenchGraphEvaluator(dataset="invalid_dataset", use_proxy=True)


# =============================================================================
# NASBenchGraphContext Tests
# =============================================================================


class TestNASBenchGraphContext:
    """Tests for NASBenchGraphContext."""

    @pytest.fixture
    def ctx(self):
        evaluator = NASBenchGraphEvaluator(dataset="cora", use_proxy=True, seed=42)
        return NASBenchGraphContext(evaluator=evaluator)

    def test_family(self, ctx):
        from src.geakg.contexts.base import OptimizationFamily
        assert ctx.family == OptimizationFamily.ARCHITECTURE

    def test_domain(self, ctx):
        assert ctx.domain == "nas_bench_graph"

    def test_dimension(self, ctx):
        assert ctx.dimension == 8

    def test_evaluate_negative(self, ctx):
        """Context returns -accuracy for minimization."""
        arch = GraphArchitecture.random(random.Random(42))
        fitness = ctx.evaluate(arch)
        assert isinstance(fitness, float)
        assert fitness < 0  # Negated accuracy

    def test_valid(self, ctx):
        arch = GraphArchitecture(
            connectivity=[0, 1, 2, 3],
            operations=[0, 1, 2, 3],
        )
        assert ctx.valid(arch) is True

    def test_random_solution(self, ctx):
        sol = ctx.random_solution()
        assert isinstance(sol, GraphArchitecture)
        assert ctx.valid(sol)

    def test_copy(self, ctx):
        arch = GraphArchitecture(
            connectivity=[0, 1, 2, 3],
            operations=[0, 1, 2, 3],
        )
        copied = ctx.copy(arch)
        assert copied.connectivity == arch.connectivity
        assert copied.operations == arch.operations
        copied.connectivity[0] = 3
        assert arch.connectivity[0] == 0

    def test_instance_data(self, ctx):
        data = ctx.instance_data
        assert data["dimension"] == 8
        assert data["dataset"] == "cora"
        assert data["metric"] == "accuracy"
        assert data["direction"] == "maximize"


# =============================================================================
# Graph Architecture Operators Tests
# =============================================================================


class TestGraphArchitectureOperators:
    """Tests for all 18 graph architecture operators."""

    @pytest.fixture
    def ctx(self):
        evaluator = NASBenchGraphEvaluator(dataset="cora", use_proxy=True, seed=42)
        return NASBenchGraphContext(evaluator=evaluator)

    def _validate_result(self, result):
        """Check that operator output is a valid GraphArchitecture."""
        assert hasattr(result, "connectivity"), "Result must have .connectivity"
        assert hasattr(result, "operations"), "Result must have .operations"
        assert len(result.connectivity) == GRAPH_NUM_NODES, (
            f"Expected {GRAPH_NUM_NODES} connectivity, got {len(result.connectivity)}"
        )
        assert len(result.operations) == GRAPH_NUM_NODES, (
            f"Expected {GRAPH_NUM_NODES} operations, got {len(result.operations)}"
        )
        for i, c in enumerate(result.connectivity):
            assert 0 <= c < GRAPH_MAX_CONN, (
                f"Connectivity {i} has value {c}, must be in [0, {GRAPH_MAX_CONN - 1}]"
            )
        for i, o in enumerate(result.operations):
            assert 0 <= o < GRAPH_NUM_OPS, (
                f"Operation {i} has value {o}, must be in [0, {GRAPH_NUM_OPS - 1}]"
            )

    def test_all_operators_produce_valid_output(self, ctx):
        """Every operator must produce a valid GraphArchitecture."""
        from src.geakg.generic_operators.graph_architecture import (
            GRAPH_ARCHITECTURE_OPERATORS,
        )

        rng = random.Random(42)
        all_ops = GRAPH_ARCHITECTURE_OPERATORS.get_all_operators()
        assert len(all_ops) == 18, f"Expected 18 operators, got {len(all_ops)}"

        for op in all_ops:
            for _ in range(5):
                arch = GraphArchitecture.random(rng)
                result = op.function(arch, ctx)
                self._validate_result(result)

    def test_operators_dont_mutate_input(self, ctx):
        """Operators must not modify the input architecture."""
        from src.geakg.generic_operators.graph_architecture import (
            GRAPH_ARCHITECTURE_OPERATORS,
        )

        for op in GRAPH_ARCHITECTURE_OPERATORS.get_all_operators():
            arch = GraphArchitecture(
                connectivity=[0, 1, 2, 3],
                operations=[0, 1, 2, 3],
            )
            original_conn = list(arch.connectivity)
            original_ops = list(arch.operations)
            _ = op.function(arch, ctx)
            assert arch.connectivity == original_conn, (
                f"Operator {op.operator_id} mutated input connectivity"
            )
            assert arch.operations == original_ops, (
                f"Operator {op.operator_id} mutated input operations"
            )

    def test_topology_operators(self, ctx):
        """Test topology operators: A₀ with different connectivity biases."""
        from src.geakg.generic_operators.graph_architecture import (
            topo_feedforward, topo_residual, topo_recursive, topo_cell_based,
        )

        arch = GraphArchitecture(connectivity=[1, 2, 3, 1], operations=[0, 1, 2, 3])

        # All topology operators leave operations unchanged
        for topo_fn in [topo_feedforward, topo_residual, topo_recursive, topo_cell_based]:
            result = topo_fn(arch, ctx)
            self._validate_result(result)
            assert result.operations == [0, 1, 2, 3]

        # feedforward: sets 1 connectivity to 0 (input)
        result = topo_feedforward(arch, ctx)
        assert 0 in result.connectivity

        # residual: sets connectivity[i] = i (chain)
        result = topo_residual(arch, ctx)
        assert any(c == i for i, c in enumerate(result.connectivity))

        # recursive: swaps 2 connectivity values (same multiset)
        result = topo_recursive(arch, ctx)
        assert sorted(result.connectivity) == sorted(arch.connectivity)

        # cell_based: randomizes all connectivity
        result = topo_cell_based(arch, ctx)
        self._validate_result(result)

    def test_activation_operators(self, ctx):
        """Test activation operators: A₀ with different operation biases."""
        from src.geakg.generic_operators.graph_architecture import (
            act_standard, act_modern, act_parametric, act_mixed,
        )

        arch = GraphArchitecture(connectivity=[0, 0, 0, 0], operations=[0, 1, 2, 3])

        # All activation operators leave connectivity unchanged
        for act_fn in [act_standard, act_modern, act_parametric, act_mixed]:
            result = act_fn(arch, ctx)
            self._validate_result(result)
            assert result.connectivity == [0, 0, 0, 0]

        # standard: flip 1 op to random
        result = act_standard(arch, ctx)
        diffs = sum(1 for a, b in zip(arch.operations, result.operations) if a != b)
        assert diffs <= 1

        # modern: nudge 1 op by ±1
        result = act_modern(arch, ctx)
        diffs = sum(1 for a, b in zip(arch.operations, result.operations) if a != b)
        assert diffs <= 1

        # parametric: swap 2 ops (same multiset)
        result = act_parametric(arch, ctx)
        assert sorted(result.operations) == sorted(arch.operations)

        # mixed: randomize 2 ops
        result = act_mixed(arch, ctx)
        diffs = sum(1 for a, b in zip(arch.operations, result.operations) if a != b)
        assert diffs <= 2

    def test_regularization_structural(self, ctx):
        """Test structural regularization: A₀ shuffles all connectivity."""
        from src.geakg.generic_operators.graph_architecture import reg_structural

        arch = GraphArchitecture(
            connectivity=[0, 1, 2, 3],
            operations=[7, 7, 8, 8],
        )
        result = reg_structural(arch, ctx)
        self._validate_result(result)
        # Operations unchanged — reg_structural only mutates connectivity
        assert result.operations == [7, 7, 8, 8]
        # Shuffle preserves multiset
        assert sorted(result.connectivity) == sorted(arch.connectivity)

    def test_eval_proxy_clamps(self, ctx):
        """Test eval_proxy clamps to valid range."""
        from src.geakg.generic_operators.graph_architecture import eval_proxy

        arch = GraphArchitecture(
            connectivity=[0, 1, 2, 3],
            operations=[0, 1, 2, 3],
        )
        result = eval_proxy(arch, ctx)
        assert all(0 <= c <= 3 for c in result.connectivity)
        assert all(0 <= o <= 8 for o in result.operations)

    def test_operator_role_coverage(self):
        """All 18 NAS roles must have at least one graph operator."""
        from src.geakg.generic_operators.graph_architecture import (
            GRAPH_ARCHITECTURE_OPERATORS,
        )

        expected_roles = {
            "topo_feedforward", "topo_residual", "topo_recursive", "topo_cell_based",
            "act_standard", "act_modern", "act_parametric", "act_mixed",
            "train_optimizer", "train_schedule", "train_augmentation", "train_loss",
            "reg_dropout", "reg_normalization", "reg_weight_decay", "reg_structural",
            "eval_proxy", "eval_full",
        }
        covered_roles = set(GRAPH_ARCHITECTURE_OPERATORS.operators_by_role.keys())
        assert covered_roles == expected_roles


# =============================================================================
# Base Operators (Code Strings) Tests
# =============================================================================


class TestGraphBaseOperators:
    """Tests for graph base operator code strings."""

    def test_all_operators_compile(self):
        """All 18 base operators must compile without errors."""
        from src.geakg.layers.l1.base_operators_nas_graph import (
            NAS_GRAPH_BASE_OPERATORS,
            ALL_NAS_GRAPH_ROLES,
        )

        assert len(NAS_GRAPH_BASE_OPERATORS) == 18

        for role in ALL_NAS_GRAPH_ROLES:
            code = NAS_GRAPH_BASE_OPERATORS[role]
            namespace: dict = {}
            exec(compile(code, f"<{role}>", "exec"), namespace)

            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    func = obj
                    break
            assert func is not None, f"No function found for role {role}"

    def test_all_operators_produce_valid_output(self):
        """All base operators produce valid GraphArchitecture."""
        from src.geakg.layers.l1.base_operators_nas_graph import (
            NAS_GRAPH_BASE_OPERATORS,
            ALL_NAS_GRAPH_ROLES,
        )

        rng = random.Random(42)

        for role in ALL_NAS_GRAPH_ROLES:
            code = NAS_GRAPH_BASE_OPERATORS[role]
            namespace: dict = {}
            exec(compile(code, f"<{role}>", "exec"), namespace)
            func = next(
                obj for name, obj in namespace.items()
                if callable(obj) and not name.startswith("_")
            )

            for _ in range(3):
                arch = GraphArchitecture.random(rng)

                class _Ctx:
                    def evaluate(self, sol):
                        return 80.0

                result = func(arch, _Ctx())
                assert hasattr(result, "connectivity"), (
                    f"Role {role}: result has no .connectivity"
                )
                assert hasattr(result, "operations"), (
                    f"Role {role}: result has no .operations"
                )
                assert len(result.connectivity) == GRAPH_NUM_NODES, (
                    f"Role {role}: expected {GRAPH_NUM_NODES} connectivity"
                )
                assert len(result.operations) == GRAPH_NUM_NODES, (
                    f"Role {role}: expected {GRAPH_NUM_NODES} operations"
                )
                assert all(0 <= c < GRAPH_MAX_CONN for c in result.connectivity), (
                    f"Role {role}: invalid connectivity values {result.connectivity}"
                )
                assert all(0 <= o < GRAPH_NUM_OPS for o in result.operations), (
                    f"Role {role}: invalid operation values {result.operations}"
                )

    def test_get_operator_function(self):
        """Test the get function."""
        from src.geakg.layers.l1.base_operators_nas_graph import (
            get_nas_graph_base_operator,
        )

        code = get_nas_graph_base_operator("topo_feedforward")
        assert "def topo_feedforward" in code

        with pytest.raises(KeyError):
            get_nas_graph_base_operator("nonexistent_role")


# =============================================================================
# Integration Tests
# =============================================================================


class TestGraphIntegration:
    """Integration tests for the Graph NAS pipeline."""

    def test_evaluator_context_consistency(self):
        """Evaluator accuracy should match context's -accuracy."""
        evaluator = NASBenchGraphEvaluator(dataset="cora", use_proxy=True, seed=42)
        ctx = NASBenchGraphContext(evaluator=evaluator)

        arch = GraphArchitecture.random(random.Random(42))
        acc_eval = evaluator.evaluate(arch)
        fitness_ctx = ctx.evaluate(arch)
        assert fitness_ctx == -acc_eval

    def test_operator_explores_search_space(self):
        """Running multiple operators should explore the search space."""
        from src.geakg.generic_operators.graph_architecture import (
            GRAPH_ARCHITECTURE_OPERATORS,
        )

        evaluator = NASBenchGraphEvaluator(dataset="cora", use_proxy=True, seed=42)
        ctx = NASBenchGraphContext(evaluator=evaluator)

        rng = random.Random(42)
        best_acc = 0.0
        all_ops = GRAPH_ARCHITECTURE_OPERATORS.get_all_operators()

        for _ in range(50):
            arch = GraphArchitecture.random(rng)
            # Apply 3 random operators
            for _ in range(3):
                op = rng.choice(all_ops)
                arch = op.function(arch, ctx)
            acc = evaluator.evaluate(arch)
            if acc > best_acc:
                best_acc = acc

        # After 50 iterations, should find something reasonable
        assert best_acc > 70.0, f"Best accuracy {best_acc} is too low"

    def test_exports(self):
        """Test that __init__.py exports work."""
        from src.domains.nas import (
            GraphArchitecture,
            NASBenchGraphEvaluator,
            NASBenchGraphContext,
        )

        arch = GraphArchitecture.random(random.Random(42))
        evaluator = NASBenchGraphEvaluator(dataset="cora", use_proxy=True, seed=42)
        ctx = NASBenchGraphContext(evaluator=evaluator)
        assert ctx.valid(arch)
