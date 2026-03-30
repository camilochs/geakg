"""Factory Functions for Creating Standard MetaGraph Patterns.

Part of L1 (Unified Knowledge Layer): Pre-defined meta-algorithm templates
that can be instantiated for any domain.

Available patterns:
- ILS (Iterated Local Search): construct -> improve -> perturb -> repeat
- VNS (Variable Neighborhood Search): systematic neighborhood exploration
- Hybrid: Multiple paths for MMAS exploration
- NAS: Neural Architecture Search pattern (topology -> activation -> training -> ...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .roles import AbstractRole
from .conditions import EdgeCondition, ConditionType
from .metagraph import MetaGraph, MetaEdge

if TYPE_CHECKING:
    from src.geakg.core.role_schema import RoleSchema


def create_ils_meta_graph() -> MetaGraph:
    """Create a standard ILS-style meta-graph.

    Structure:
        CONST_GREEDY -> LS_INTENSIFY_SMALL -> LS_INTENSIFY_MEDIUM
                                           -> PERT_ESCAPE_SMALL (on stagnation)
        PERT_ESCAPE_SMALL -> LS_INTENSIFY_SMALL (re-optimize)
    """
    mg = MetaGraph(
        name="ils_standard",
        description="Iterated Local Search: construct, improve, perturb, repeat",
    )

    # Construction to local search
    mg.add_edge(MetaEdge(
        source=AbstractRole.CONST_GREEDY,
        target=AbstractRole.LS_INTENSIFY_SMALL,
        weight=0.90,
        reasoning="Start with fast local search after construction",
    ))

    # Local search chain
    mg.add_edge(MetaEdge(
        source=AbstractRole.LS_INTENSIFY_SMALL,
        target=AbstractRole.LS_INTENSIFY_MEDIUM,
        weight=0.75,
        reasoning="Progress to more complex moves after basic improvement",
    ))
    mg.add_edge(MetaEdge(
        source=AbstractRole.LS_INTENSIFY_MEDIUM,
        target=AbstractRole.LS_INTENSIFY_SMALL,
        weight=0.60,
        reasoning="Return to fast moves after complex moves",
    ))

    # Escape on stagnation
    mg.add_edge(MetaEdge(
        source=AbstractRole.LS_INTENSIFY_SMALL,
        target=AbstractRole.PERT_ESCAPE_SMALL,
        weight=0.35,
        conditions=[EdgeCondition(
            condition_type=ConditionType.STAGNATION,
            threshold=3.0,
            reason="Escape local optimum after stagnation",
        )],
        condition_boost=3.0,
        reasoning="Perturb when stuck to escape local optimum",
    ))
    mg.add_edge(MetaEdge(
        source=AbstractRole.LS_INTENSIFY_MEDIUM,
        target=AbstractRole.PERT_ESCAPE_SMALL,
        weight=0.30,
        conditions=[EdgeCondition(
            condition_type=ConditionType.DIVERSITY_LOW,
            threshold=0.3,
            reason="Diversify when population converges",
        )],
        condition_boost=2.5,
        reasoning="Perturb when diversity drops",
    ))

    # Re-optimize after perturbation
    mg.add_edge(MetaEdge(
        source=AbstractRole.PERT_ESCAPE_SMALL,
        target=AbstractRole.LS_INTENSIFY_SMALL,
        weight=0.90,
        reasoning="Always re-optimize after perturbation",
    ))

    return mg


def create_vns_meta_graph() -> MetaGraph:
    """Create a Variable Neighborhood Search style meta-graph.

    Structure:
        CONST_GREEDY -> LS_CHAIN
        LS_CHAIN -> PERT_ESCAPE_SMALL (on stagnation) -> LS_CHAIN
                 -> PERT_ESCAPE_LARGE (on deep stagnation) -> LS_CHAIN
    """
    mg = MetaGraph(
        name="vns_standard",
        description="Variable Neighborhood Search with escalating perturbation",
    )

    # Construction to VND
    mg.add_edge(MetaEdge(
        source=AbstractRole.CONST_GREEDY,
        target=AbstractRole.LS_CHAIN,
        weight=0.90,
        reasoning="Start with systematic neighborhood exploration",
    ))

    # VND to perturbations (escalating)
    mg.add_edge(MetaEdge(
        source=AbstractRole.LS_CHAIN,
        target=AbstractRole.PERT_ESCAPE_SMALL,
        weight=0.40,
        conditions=[EdgeCondition(
            condition_type=ConditionType.STAGNATION,
            threshold=2.0,
        )],
        condition_boost=2.5,
        reasoning="Mild perturbation on early stagnation",
    ))
    mg.add_edge(MetaEdge(
        source=AbstractRole.LS_CHAIN,
        target=AbstractRole.PERT_ESCAPE_LARGE,
        weight=0.25,
        conditions=[EdgeCondition(
            condition_type=ConditionType.STAGNATION,
            threshold=5.0,
        )],
        condition_boost=3.0,
        reasoning="Strong perturbation on prolonged stagnation",
    ))

    # Perturbations back to VND
    mg.add_edge(MetaEdge(
        source=AbstractRole.PERT_ESCAPE_SMALL,
        target=AbstractRole.LS_CHAIN,
        weight=0.90,
        reasoning="Re-explore neighborhoods after mild perturbation",
    ))
    mg.add_edge(MetaEdge(
        source=AbstractRole.PERT_ESCAPE_LARGE,
        target=AbstractRole.LS_CHAIN,
        weight=0.85,
        reasoning="Re-explore neighborhoods after strong perturbation",
    ))

    return mg


def create_hybrid_meta_graph() -> MetaGraph:
    """Create a hybrid meta-graph with multiple construction and LS options.

    More complex structure allowing the MMAS to discover good paths.
    """
    mg = MetaGraph(
        name="hybrid_exploration",
        description="Hybrid with multiple paths for MMAS exploration",
    )

    # Multiple construction options
    mg.add_edge(MetaEdge(
        source=AbstractRole.CONST_GREEDY,
        target=AbstractRole.LS_INTENSIFY_SMALL,
        weight=0.80,
    ))
    mg.add_edge(MetaEdge(
        source=AbstractRole.CONST_INSERTION,
        target=AbstractRole.LS_INTENSIFY_SMALL,
        weight=0.75,
    ))

    # Full LS ladder
    mg.add_edge(MetaEdge(
        source=AbstractRole.LS_INTENSIFY_SMALL,
        target=AbstractRole.LS_INTENSIFY_MEDIUM,
        weight=0.70,
    ))
    mg.add_edge(MetaEdge(
        source=AbstractRole.LS_INTENSIFY_MEDIUM,
        target=AbstractRole.LS_INTENSIFY_LARGE,
        weight=0.50,
    ))
    mg.add_edge(MetaEdge(
        source=AbstractRole.LS_INTENSIFY_LARGE,
        target=AbstractRole.LS_INTENSIFY_SMALL,
        weight=0.40,
    ))

    # Shortcut back
    mg.add_edge(MetaEdge(
        source=AbstractRole.LS_INTENSIFY_MEDIUM,
        target=AbstractRole.LS_INTENSIFY_SMALL,
        weight=0.55,
    ))

    # Escape routes with conditions
    mg.add_edge(MetaEdge(
        source=AbstractRole.LS_INTENSIFY_SMALL,
        target=AbstractRole.PERT_ESCAPE_SMALL,
        weight=0.30,
        conditions=[EdgeCondition(
            condition_type=ConditionType.STAGNATION,
            threshold=3.0,
        )],
        condition_boost=2.5,
    ))
    mg.add_edge(MetaEdge(
        source=AbstractRole.LS_INTENSIFY_MEDIUM,
        target=AbstractRole.PERT_ESCAPE_LARGE,
        weight=0.25,
        conditions=[EdgeCondition(
            condition_type=ConditionType.STAGNATION,
            threshold=5.0,
        )],
        condition_boost=3.0,
    ))

    # Adaptive perturbation on low diversity
    mg.add_edge(MetaEdge(
        source=AbstractRole.LS_INTENSIFY_LARGE,
        target=AbstractRole.PERT_ADAPTIVE,
        weight=0.35,
        conditions=[EdgeCondition(
            condition_type=ConditionType.DIVERSITY_LOW,
            threshold=0.2,
        )],
        condition_boost=2.0,
    ))

    # Return from perturbations
    mg.add_edge(MetaEdge(
        source=AbstractRole.PERT_ESCAPE_SMALL,
        target=AbstractRole.LS_INTENSIFY_SMALL,
        weight=0.90,
    ))
    mg.add_edge(MetaEdge(
        source=AbstractRole.PERT_ESCAPE_LARGE,
        target=AbstractRole.LS_INTENSIFY_SMALL,
        weight=0.85,
    ))
    mg.add_edge(MetaEdge(
        source=AbstractRole.PERT_ADAPTIVE,
        target=AbstractRole.LS_INTENSIFY_MEDIUM,
        weight=0.80,
    ))

    return mg


def create_nas_meta_graph(schema: "RoleSchema") -> MetaGraph:
    """Create a Neural Architecture Search meta-graph.

    Structure follows the NAS design pipeline:
        TOPOLOGY -> ACTIVATION -> TRAINING -> REGULARIZATION -> EVALUATION
        EVALUATION -> TOPOLOGY (redesign), TRAINING (refine), ACTIVATION (adjust)

    Args:
        schema: NASRoleSchema instance.

    Returns:
        MetaGraph configured for NAS.
    """
    mg = MetaGraph(
        name="nas_standard",
        description="Neural Architecture Search: topology -> activation -> training -> reg -> eval",
    )
    mg.role_schema = schema

    # Add all roles from schema
    for role_id in schema.get_all_roles():
        meta = schema.get_role_metadata(role_id)
        mg.add_role_generic(
            role_id=role_id,
            description=meta.get("description", role_id),
            category=meta.get("category", ""),
            expected_cost=meta.get("expected_cost", "O(n)"),
            exploration_bias=meta.get("exploration_bias", 0.5),
        )

    # TOPOLOGY -> ACTIVATION (structure defined -> choose activations)
    mg.add_edge(MetaEdge(source="topo_feedforward", target="act_standard", weight=0.80))
    mg.add_edge(MetaEdge(source="topo_residual", target="act_modern", weight=0.85))
    mg.add_edge(MetaEdge(source="topo_recursive", target="act_standard", weight=0.75))
    mg.add_edge(MetaEdge(source="topo_cell_based", target="act_mixed", weight=0.80))

    # TOPOLOGY -> TOPOLOGY (refine topology)
    mg.add_edge(MetaEdge(source="topo_feedforward", target="topo_residual", weight=0.50))
    mg.add_edge(MetaEdge(source="topo_residual", target="topo_cell_based", weight=0.40))

    # ACTIVATION -> ACTIVATION (experiment with combinations)
    mg.add_edge(MetaEdge(source="act_standard", target="act_modern", weight=0.60))
    mg.add_edge(MetaEdge(source="act_modern", target="act_parametric", weight=0.45))
    mg.add_edge(MetaEdge(source="act_parametric", target="act_mixed", weight=0.40))

    # ACTIVATION -> TRAINING
    mg.add_edge(MetaEdge(source="act_standard", target="train_optimizer", weight=0.75))
    mg.add_edge(MetaEdge(source="act_modern", target="train_optimizer", weight=0.80))
    mg.add_edge(MetaEdge(source="act_mixed", target="train_optimizer", weight=0.70))
    mg.add_edge(MetaEdge(source="act_parametric", target="train_optimizer", weight=0.70))

    # TRAINING -> TRAINING (combine optimizer + schedule + augmentation)
    mg.add_edge(MetaEdge(source="train_optimizer", target="train_schedule", weight=0.85))
    mg.add_edge(MetaEdge(source="train_schedule", target="train_augmentation", weight=0.65))
    mg.add_edge(MetaEdge(source="train_augmentation", target="train_loss", weight=0.60))
    mg.add_edge(MetaEdge(source="train_optimizer", target="train_loss", weight=0.50))

    # TRAINING -> REGULARIZATION
    mg.add_edge(MetaEdge(source="train_schedule", target="reg_dropout", weight=0.70))
    mg.add_edge(MetaEdge(source="train_loss", target="reg_dropout", weight=0.70))
    mg.add_edge(MetaEdge(source="train_augmentation", target="reg_normalization", weight=0.60))

    # REGULARIZATION -> REGULARIZATION (combine)
    mg.add_edge(MetaEdge(source="reg_dropout", target="reg_normalization", weight=0.75))
    mg.add_edge(MetaEdge(source="reg_normalization", target="reg_weight_decay", weight=0.65))
    mg.add_edge(MetaEdge(source="reg_weight_decay", target="reg_structural", weight=0.55))
    mg.add_edge(MetaEdge(source="reg_dropout", target="reg_structural", weight=0.50))

    # REGULARIZATION -> EVALUATION
    mg.add_edge(MetaEdge(source="reg_structural", target="eval_proxy", weight=0.85))
    mg.add_edge(MetaEdge(source="reg_weight_decay", target="eval_proxy", weight=0.75))
    mg.add_edge(MetaEdge(source="reg_normalization", target="eval_proxy", weight=0.70))
    mg.add_edge(MetaEdge(source="reg_dropout", target="eval_proxy", weight=0.65))

    # EVALUATION -> feedback loops
    mg.add_edge(MetaEdge(source="eval_proxy", target="topo_residual", weight=0.40,
                         reasoning="Redesign topology if result unsatisfactory"))
    mg.add_edge(MetaEdge(source="eval_proxy", target="train_optimizer", weight=0.50,
                         reasoning="Refine training if result acceptable"))
    mg.add_edge(MetaEdge(source="eval_proxy", target="act_modern", weight=0.35,
                         reasoning="Change activations if suboptimal"))
    mg.add_edge(MetaEdge(source="eval_proxy", target="eval_full", weight=0.60,
                         reasoning="Full evaluation for promising architectures"))

    return mg
