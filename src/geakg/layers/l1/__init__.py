"""L1: Operator Generation Layer for NS-SE.

This layer generates EXECUTABLE CODE for each role:
- AFO (Always-From-Original): Generate variants from base operators
- Design-Space Prompting: Explore orthogonal design axes
- Mini F-Race: Statistical selection with Wilcoxon test (experimental)

Architecture:
    L0: MetaGraph Topology (offline, LLM) - src/geakg/layers/l0/
    L1: Operator Generation (offline, LLM) - src/geakg/layers/l1/  <-- YOU ARE HERE
    L2: ACO Training (offline, ACO) - src/geakg/layers/l2/
    Online: Symbolic Executor - src/geakg/online/

Usage:
    from src.geakg.layers.l1 import L1Generator, L1Config, OperatorPool

    # Generate pool
    generator = L1Generator(llm_client, domain="tsp")
    pool = generator.generate(instances, evaluate_fn)

    # Save/load
    pool.save("pools/tsp.json")
    pool = OperatorPool.load("pools/tsp.json")
"""

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
from src.geakg.layers.l1.generator import L1Generator, L1Config
from src.geakg.layers.l1.hook import L1SynthesisHook

# Aliases for backward compatibility (L0 -> L1)
L0Generator = L1Generator
L0Config = L1Config
L0SynthesisHook = L1SynthesisHook

__all__ = [
    # Core classes
    "Operator",
    "OperatorPool",
    "L1Generator",
    "L1Config",
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
    # Hook
    "L1SynthesisHook",
    # Backward compatibility aliases
    "L0Generator",
    "L0Config",
    "L0SynthesisHook",
]
