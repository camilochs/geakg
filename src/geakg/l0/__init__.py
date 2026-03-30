"""L0: BACKWARD COMPATIBILITY STUB.

DEPRECATED: This module re-exports from src.geakg.layers.l1 and src.geakg.offline.
New code should import directly:

    # For operators
    from src.geakg.layers.l1 import L1Generator, L1Config, OperatorPool

    # For iterative refinement
    from src.geakg.offline import run_iterative_refinement

This stub exists for backward compatibility with existing imports.
"""

# Core classes from L1 (operator generation)
from src.geakg.layers.l1.pool import Operator, OperatorPool
from src.geakg.layers.l1.base_operators import ALL_ROLES, BASE_OPERATORS, get_role_category
from src.geakg.layers.l1.design_space import (
    DESIGN_AXES,
    TOTAL_COMBINATIONS,
    sample_design_point,
    format_design_point,
    sample_design_point_for_category,
)
from src.geakg.layers.l1.prompts import DESIGN_SPACE_PROMPT, build_design_space_prompt
from src.geakg.layers.l1.racing import mini_frace, rank_operators
from src.geakg.layers.l1.generator import L1Generator, L1Config, L0Generator, L0Config
from src.geakg.layers.l0.topology_generator import (
    L0MetaGraphGenerator,
    create_default_metagraph_for_pool,
)
from src.geakg.layers.l1.hook import L1SynthesisHook, L0SynthesisHook

# Iterative refinement from offline
from src.geakg.offline.iterative_refinement import (
    WeakSpot,
    SnapshotAnalysis,
    IterativeRefinementConfig,
    analyze_snapshot,
    generate_contextual_operator,
    run_iterative_refinement,
    save_snapshot_for_transfer,
    load_snapshot_for_transfer,
)

__all__ = [
    # Core classes
    "Operator",
    "OperatorPool",
    "L1Generator",
    "L1Config",
    "L0Generator",  # Alias for backward compat
    "L0Config",  # Alias for backward compat
    # Base operators
    "ALL_ROLES",
    "BASE_OPERATORS",
    "get_role_category",
    # Design space
    "DESIGN_AXES",
    "TOTAL_COMBINATIONS",
    "sample_design_point",
    "format_design_point",
    "sample_design_point_for_category",
    # Prompts
    "DESIGN_SPACE_PROMPT",
    "build_design_space_prompt",
    # Racing
    "mini_frace",
    "rank_operators",
    # MetaGraph generation
    "L0MetaGraphGenerator",
    "create_default_metagraph_for_pool",
    # Hook
    "L1SynthesisHook",
    "L0SynthesisHook",  # Alias for backward compat
    # Iterative refinement
    "WeakSpot",
    "SnapshotAnalysis",
    "IterativeRefinementConfig",
    "analyze_snapshot",
    "generate_contextual_operator",
    "run_iterative_refinement",
    "save_snapshot_for_transfer",
    "load_snapshot_for_transfer",
]
