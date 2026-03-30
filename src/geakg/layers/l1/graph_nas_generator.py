"""NAS-Bench-Graph L1 Generator: Operator synthesis for GraphArchitecture.

Adapts the L1Generator pipeline for NAS-Bench-Graph GNN architectures.
Key differences from NAS-Bench-201 and NAS-Bench-NLP:
- Solutions are GraphArchitecture (4 conn [0,3] + 4 ops [0,8])
- Operations are GNN-specific (GCN, GAT, GIN, GraphSAGE, etc.)
- Design axes reference GNN design concepts (message passing, attention, etc.)
- Validation checks connectivity [0,3] and operations [0,8]
- Metric is accuracy (higher is better, same direction as NAS-Bench-201)

Usage:
    generator = GraphNASGeneratorL1(llm_client, config=GraphNASL1Config(...))
    pool = generator.generate(evaluator)
    pool.save("pools/graph_nas_pool.json")
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.geakg.layers.l1.base_operators_nas_graph import (
    ALL_NAS_GRAPH_ROLES,
    NAS_GRAPH_BASE_OPERATORS,
)
from src.geakg.layers.l1.pool import Operator, OperatorPool

if TYPE_CHECKING:
    from src.domains.nas.graph_evaluator import NASBenchGraphEvaluator
    from src.llm.client import OllamaClient, OpenAIClient


# Estimated tokens per Graph NAS variant (prompt + response)
GRAPH_NAS_TOKENS_PER_VARIANT = 1200

# ---- Graph NAS Design Space ----

GRAPH_NAS_DESIGN_AXES: dict[str, list[str]] = {
    "selection": [
        "targeted (identify weakest component, fix it specifically)",
        "random (change a random node)",
        "pattern-based (apply a known good architecture pattern)",
        "analytical (count op types, fix imbalance)",
    ],
    "scope": [
        "single-node (modify 1 node: op or conn)",
        "two-node (modify 2 nodes coordinately)",
        "all-ops (set all 4 operations to a specific pattern)",
        "full-arch (modify both connectivity and operations together)",
    ],
    "knowledge": [
        "op-counting (count GNN vs identity vs FC, ensure >= 2 GNN ops)",
        "connectivity-aware (analyze chain vs star vs mixed, adjust depth/breadth)",
        "combo-aware (use proven combos: GAT+GCN, GIN+identity, diverse ops)",
        "position-aware (strong GNN at node 0, aggregation at node 3)",
    ],
    "acceptance": [
        "always (always apply the modification)",
        "conditional (only modify if current arch has a detectable weakness)",
        "multi-trial (try 2-3 variants, pick by heuristic score)",
        "fallback (try primary strategy, fallback to simpler if N/A)",
    ],
}

GRAPH_NAS_TOTAL_COMBINATIONS = 1
for _options in GRAPH_NAS_DESIGN_AXES.values():
    GRAPH_NAS_TOTAL_COMBINATIONS *= len(_options)  # 4^4 = 256


def _sample_graph_nas_design_point(rng: random.Random, category: str) -> dict[str, str]:
    """Sample a Graph NAS design point, optionally respecting category constraints."""
    constraints: dict[str, list[str]] = {}
    if category == "topology":
        constraints["knowledge"] = [
            "connectivity-aware (analyze chain vs star vs mixed, adjust depth/breadth)",
            "position-aware (strong GNN at node 0, aggregation at node 3)",
        ]
    elif category == "activation":
        constraints["knowledge"] = [
            "op-counting (count GNN vs identity vs FC, ensure >= 2 GNN ops)",
            "combo-aware (use proven combos: GAT+GCN, GIN+identity, diverse ops)",
        ]
    elif category == "regularization":
        constraints["selection"] = [
            "analytical (count op types, fix imbalance)",
            "targeted (identify weakest component, fix it specifically)",
        ]
    elif category == "training":
        constraints["scope"] = [
            "single-node (modify 1 node: op or conn)",
            "two-node (modify 2 nodes coordinately)",
        ]

    point = {}
    for axis, options in GRAPH_NAS_DESIGN_AXES.items():
        available = constraints.get(axis, options)
        point[axis] = rng.choice(available)
    return point


def _format_design_point(point: dict[str, str]) -> str:
    return "\n".join(f"- {axis}: {choice}" for axis, choice in point.items())


def _design_point_key(point: dict[str, str]) -> tuple:
    return tuple(sorted(point.items()))


# ---- Graph NAS Role Category Mapping ----

_GRAPH_NAS_ROLE_CATEGORY = {
    "topo_feedforward": "topology",
    "topo_residual": "topology",
    "topo_recursive": "topology",
    "topo_cell_based": "topology",
    "act_standard": "activation",
    "act_modern": "activation",
    "act_parametric": "activation",
    "act_mixed": "activation",
    "train_optimizer": "training",
    "train_schedule": "training",
    "train_augmentation": "training",
    "train_loss": "training",
    "reg_dropout": "regularization",
    "reg_normalization": "regularization",
    "reg_weight_decay": "regularization",
    "reg_structural": "regularization",
    "eval_proxy": "evaluation",
    "eval_full": "evaluation",
}

# ---- Graph NAS Category Guidance ----

_GRAPH_NAS_CATEGORY_GUIDANCE = {
    "topology": """**Topology operators** modify the DAG connectivity (solution.connectivity).
The BEST connectivity for GNNs balances depth and breadth.
- Chain [0,1,2,3] gives multi-hop aggregation (each node sees k more hops).
- Star [0,0,0,0] gives parallel extraction from input.
- Balanced [0,0,1,2] is often optimal — 2 nodes read input, then depth builds.
- Analyze the current connectivity pattern and move towards a known good one.
- Do NOT randomize blindly — count how many nodes read from input (from_input) and adjust.
- If from_input >= 3: too flat, add depth by chaining nodes 2→1, 3→2.
- If from_input <= 1: too deep, add breadth by connecting a node to 0 (input).""",

    "activation": """**Activation operators** change GNN layer types (solution.operations).
GCN (0) + GAT (1) combo is the strongest. At least 2 GNN ops (indices 0-6) are needed.
- 1 identity (7) as skip connection is OK; 2+ identity hurts performance.
- FC (8) is almost always bad for graphs — replace it immediately.
- Best combos: GAT+GCN (attention+message-passing), GIN+identity (expressive+residual), 3+ diverse GNN ops.
- Worst patterns: multiple FC (8), all-identity (7), single op repeated 4 times.
- Count ops first: `n_gnn = sum(1 for o in result.operations if o <= 6)`. If n_gnn < 2, upgrade identity/FC to GCN/GAT.
- Op reference: GCN=0, GAT=1, SAGE=2, GIN=3, Cheb=4, ARMA=5, k-GNN=6, identity=7, FC=8.""",

    "training": """**Training operators** make small targeted changes (1-2 elements) to preserve quality.
Best moves:
- Upgrade identity (7) or FC (8) to GCN (0) or GAT (1) — adds graph awareness.
- Swap GCN (0) <-> GAT (1) for attention vs message-passing diversity.
- Change 1 op to its neighbor (+-1 index) — conservative but safe.
- Never change more than 2 elements at once — small perturbations preserve good structure.
- Before modifying, analyze: count GNN ops, check for FC/identity excess, then fix the weakest.""",

    "regularization": """**Regularization operators** simplify without destroying.
- Replace complex ops (ARMA=5, k-GNN=6, Cheb=4) with simpler GCN=0 or SAGE=2.
- Replace FC=8 with identity=7 (at least it passes features through).
- After pruning, ensure at least 2 GNN ops (indices 0-6) remain — otherwise you destroy graph awareness.
- Identity=7 is better than FC=8 but worse than any GNN op.
- Good simplification target: 2 GCN/GAT + 1 GIN/SAGE + 1 identity (balanced and simple).""",

    "evaluation": """**Evaluation operators** validate or score the architecture.
- Clamp connectivity to [0,3] and operations to [0,8].
- These operators typically don't modify the architecture substantively.""",
}

# ---- Graph NAS Prompt Template ----

GRAPH_NAS_DESIGN_SPACE_PROMPT = """You are designing a {role_category} operator for **Neural Architecture Search** on NAS-Bench-Graph.

**DOMAIN CONTEXT - NAS-Bench-Graph (GNN Architecture Search):**
- A GNN architecture has 4 computing nodes in a DAG (input -> 4 nodes -> output)
- Each node has:
  - `connectivity[i]` in [0,3]: which prior node feeds this node (0=input, 1-3=prior computing node)
  - `operations[i]` in [0,8]: GNN layer type
- Operations: {{gcn=0, gat=1, sage=2, gin=3, cheb=4, arma=5, k-gnn=6, identity=7, fc=8}}
- The solution object has `solution.connectivity` (4 ints [0,3]) and `solution.operations` (4 ints [0,8])
- Use `solution.copy()` to clone before modification
- Dataset: node classification (Cora, CiteSeer, PubMed, etc.)
- Metric: accuracy (HIGHER is better)

**Known good patterns (from NAS-Bench-Graph analysis):**
- Best ops: GAT+GCN combo, GIN+identity (residual), diverse GNN mix (3+ different GNN ops)
- Worst ops: multiple FC (8), all-identity (7), single op repeated 4 times
- Best connectivity: chain [0,1,2,3] or balanced [0,0,1,2]; star [0,0,0,0] is OK
- Position matters: strong GNN (GCN/GAT/GIN) at node 0, varied ops at nodes 1-3
- At least 2 GNN ops (indices 0-6) are essential; 3 is better

**Important: ctx has NO useful methods for operators.**
- Do NOT call ctx.evaluate(), ctx.cost(), ctx.delta(), or ctx.neighbors() — they don't exist
- Your operator ONLY modifies solution.connectivity and solution.operations
- Use architectural heuristics (op counting, pattern matching) instead of evaluation

{category_guidance}

**GOAL: Create a POWERFUL {role_category} operator using your knowledge of GNN architecture design.**

The original operator (A₀) is intentionally simple. You must create something MUCH BETTER by:
- Analyzing the current architecture BEFORE modifying — don't change blindly
- Making informed decisions based on connectivity patterns and operation distribution
- Using GNN design insights (message passing, attention, spectral methods, skip connections)

**Original operator (A₀) - this is WEAK, you must do BETTER:**
```python
{original_code}
```

**Example of a good operator (for reference):**
```python
def topo_balanced_chain(solution, ctx):
    import random
    result = solution.copy()
    from_input = sum(1 for c in result.connectivity if c == 0)
    if from_input >= 3:  # Too flat (star-like) -> add depth
        result.connectivity[2] = 1
        result.connectivity[3] = 2
    elif from_input <= 1:  # Too deep (chain-like) -> add breadth
        idx = random.randint(1, 3)
        result.connectivity[idx] = 0
    # Ensure at least 2 GNN ops
    n_gnn = sum(1 for o in result.operations if o <= 6)
    if n_gnn < 2:
        for i in range(4):
            if result.operations[i] >= 7:  # identity or FC
                result.operations[i] = random.choice([0, 1, 3])  # GCN, GAT, GIN
                break
    return result
```

**Design Space - guide your approach:**
{design_point}

**Rules:**
1. Copy first: `result = solution.copy()`
2. Modify `result.connectivity[i]` (int 0-3) and/or `result.operations[i]` (int 0-8)
3. Always return `result` (a valid GraphArchitecture)
4. `import random` at function start if needed
5. 4 computing nodes indexed 0-3
6. Analyze the architecture BEFORE modifying — don't change blindly
7. Use op counting: `n_gnn = sum(1 for o in result.operations if o <= 6)`

Return JSON:
{{
  "name": "{category}_<descriptive_name>",
  "design_choices": {{"selection": "...", "scope": "...", "knowledge": "...", "acceptance": "..."}},
  "structural_changes": "<explain your GNN-specific algorithmic approach>",
  "code": "def {category}_<name>(solution, ctx):\\n    ..."
}}
"""

GRAPH_NAS_REFINEMENT_PROMPT = """Fix the errors in this GNN NAS operator while keeping the design choices.

**Original code with errors:**
```python
{code}
```

**Errors to fix:**
{errors}

**Design choices (preserve these):**
{design_point}

**Rules:**
- solution.connectivity is a list of 4 integers in [0, 3]
- solution.operations is a list of 4 integers in [0, 8]
- Use solution.copy() to clone
- Return a GraphArchitecture with valid connectivity and operations

Return JSON:
{{
  "code": "def <fixed_function>(solution, ctx):\\n    ...",
  "changes_made": "<describe what was fixed>"
}}
"""


# ---- Graph NAS L1 Config ----

@dataclass
class GraphNASL1Config:
    """Configuration for Graph NAS L1 operator generation."""

    max_tokens: int = 15_000
    pool_size_per_role: int = 3
    temperature: float = 0.7
    max_refine_attempts: int = 2
    seed: int = 42
    roles: list[str] = field(default_factory=lambda: ALL_NAS_GRAPH_ROLES.copy())
    skip_eval_roles: bool = True


# ---- Graph NAS L1 Generator ----

class GraphNASGeneratorL1:
    """L1 Generator specialized for NAS-Bench-Graph GraphArchitecture operators.

    Generates operator variants for 18 NAS roles using Design-Space Prompting
    with GNN-specific axes (selection, scope, knowledge, acceptance).

    Usage:
        llm = OpenAIClient(model="gpt-4o-mini")
        gen = GraphNASGeneratorL1(llm, config=GraphNASL1Config(max_tokens=15000))
        pool = gen.generate(evaluator)
    """

    def __init__(
        self,
        llm_client: "OllamaClient | OpenAIClient",
        config: GraphNASL1Config | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or GraphNASL1Config()
        self.rng = random.Random(self.config.seed)
        self.tokens_used = 0
        self.used_combinations: set[tuple] = set()
        self.generation_stats: dict[str, Any] = {
            "total_attempts": 0,
            "valid_count": 0,
            "refined_count": 0,
            "failed_count": 0,
        }

    def generate(
        self,
        evaluator: "NASBenchGraphEvaluator | None" = None,
    ) -> OperatorPool:
        """Generate Graph NAS operator pool using Design-Space Prompting.

        Args:
            evaluator: Optional evaluator for fitness-based selection.

        Returns:
            OperatorPool with operators per NAS role.
        """
        roles = [
            r for r in self.config.roles
            if not (self.config.skip_eval_roles and r.startswith("eval_"))
        ]

        n_roles = len(roles)
        tokens_per_role = self.config.max_tokens // max(n_roles, 1)
        variants_per_role = max(1, tokens_per_role // GRAPH_NAS_TOKENS_PER_VARIANT)

        logger.info(
            f"[Graph-NAS-L1] Starting generation: {n_roles} roles, "
            f"~{variants_per_role} variants/role, {self.config.max_tokens} token budget"
        )

        pool = OperatorPool(
            metadata={
                "domain": "nas_bench_graph",
                "tokens_used": 0,
                "llm_model": getattr(self.llm_client, "model", "unknown"),
            }
        )

        for role in roles:
            if self.tokens_used >= self.config.max_tokens:
                logger.info("[Graph-NAS-L1] Token budget exhausted")
                break

            logger.info(f"[Graph-NAS-L1] Generating variants for {role}")
            candidates = self._generate_role_variants(
                role=role,
                max_variants=variants_per_role,
                evaluator=evaluator,
            )

            for op in candidates[:self.config.pool_size_per_role]:
                pool.add_operator(op)

            logger.info(f"[Graph-NAS-L1]   {role}: {len(candidates)} valid operators")

        # Add A₀ for eval roles
        if self.config.skip_eval_roles:
            for role in self.config.roles:
                if role.startswith("eval_") and role in NAS_GRAPH_BASE_OPERATORS:
                    base_op = Operator(
                        name=f"{role}_base",
                        code=NAS_GRAPH_BASE_OPERATORS[role].strip(),
                        role=role,
                        design_choices={},
                        interaction_effects="Base operator (A₀) — evaluation role",
                    )
                    pool.add_operator(base_op)

        pool.metadata["tokens_used"] = self.tokens_used
        pool.metadata["generation_stats"] = dict(self.generation_stats)

        logger.info(
            f"[Graph-NAS-L1] Complete: {pool.total_operators} operators across "
            f"{len(pool.roles)} roles, {self.tokens_used} tokens used"
        )

        return pool

    def _generate_role_variants(
        self,
        role: str,
        max_variants: int,
        evaluator: "NASBenchGraphEvaluator | None" = None,
    ) -> list[Operator]:
        """Generate variants for a single Graph NAS role."""
        category = _GRAPH_NAS_ROLE_CATEGORY.get(role, "topology")
        base_code = NAS_GRAPH_BASE_OPERATORS.get(role, "")

        base_op = Operator(
            name=f"{role}_base",
            code=base_code.strip(),
            role=role,
            design_choices={},
            interaction_effects="Base operator (A₀)",
        )
        candidates = [base_op]

        for _ in range(max_variants):
            if self.tokens_used >= self.config.max_tokens:
                break

            design_point = self._sample_unique_point(category)
            if design_point is None:
                break

            self.generation_stats["total_attempts"] += 1

            variant = self._generate_variant(role, category, base_code, design_point)
            if variant is None:
                self.generation_stats["failed_count"] += 1
                continue

            if self._validate_graph_operator(variant):
                candidates.append(variant)
                self.generation_stats["valid_count"] += 1
                logger.info(f"[Graph-NAS-L1]   + {variant.name}")
            else:
                refined = self._refine_variant(variant)
                if refined and self._validate_graph_operator(refined):
                    candidates.append(refined)
                    self.generation_stats["refined_count"] += 1
                    logger.info(f"[Graph-NAS-L1]   + {refined.name} (refined)")
                else:
                    self.generation_stats["failed_count"] += 1

        if evaluator and len(candidates) > self.config.pool_size_per_role:
            candidates = self._rank_by_fitness(candidates, evaluator)

        return candidates

    def _sample_unique_point(self, category: str) -> dict[str, str] | None:
        """Sample a Graph NAS design point not yet used."""
        for _ in range(50):
            point = _sample_graph_nas_design_point(self.rng, category)
            key = _design_point_key(point)
            if key not in self.used_combinations:
                self.used_combinations.add(key)
                return point
        return None

    def _generate_variant(
        self,
        role: str,
        category: str,
        base_code: str,
        design_point: dict[str, str],
    ) -> Operator | None:
        """Generate a single Graph NAS operator variant via LLM."""
        category_guidance = _GRAPH_NAS_CATEGORY_GUIDANCE.get(category, "")

        prompt = GRAPH_NAS_DESIGN_SPACE_PROMPT.format(
            role_category=category,
            category=category,
            category_guidance=category_guidance,
            original_code=base_code.strip(),
            design_point=_format_design_point(design_point),
        )

        try:
            response = self.llm_client.query(
                prompt=prompt,
                temperature=self.config.temperature,
            )
            self.tokens_used += GRAPH_NAS_TOKENS_PER_VARIANT

            parsed = self._parse_response(response.content)
            if parsed is None:
                return None

            code = self._clean_code(parsed.get("code", ""))
            if not code:
                return None

            return Operator(
                name=parsed.get("name", f"{category}_variant"),
                code=code,
                role=role,
                design_choices=parsed.get("design_choices", design_point),
                interaction_effects=parsed.get(
                    "structural_changes",
                    parsed.get("interaction_effects", ""),
                ),
            )

        except Exception as e:
            logger.warning(f"[Graph-NAS-L1] Generation failed for {role}: {e}")
            self.tokens_used += GRAPH_NAS_TOKENS_PER_VARIANT // 2
            return None

    def _validate_graph_operator(self, operator: Operator) -> bool:
        """Validate that a Graph NAS operator compiles and produces valid architectures."""
        from src.domains.nas.graph_architecture import (
            GraphArchitecture,
            GRAPH_NUM_NODES,
            GRAPH_NUM_OPS,
            GRAPH_MAX_CONN,
        )

        try:
            compiled = compile(operator.code, "<string>", "exec")
            namespace: dict = {}
            exec(compiled, namespace)

            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    func = obj
                    break

            if func is None:
                return False

            test_rng = random.Random(42)
            for _ in range(3):
                test_arch = GraphArchitecture.random(test_rng)

                class _DummyCtx:
                    pass

                result = func(test_arch, _DummyCtx())

                if not hasattr(result, "connectivity") or not hasattr(result, "operations"):
                    return False
                if len(result.connectivity) != GRAPH_NUM_NODES:
                    return False
                if len(result.operations) != GRAPH_NUM_NODES:
                    return False
                if not all(0 <= c < GRAPH_MAX_CONN for c in result.connectivity):
                    return False
                if not all(0 <= o < GRAPH_NUM_OPS for o in result.operations):
                    return False

            return True

        except Exception:
            return False

    def _refine_variant(self, operator: Operator) -> Operator | None:
        """Try to fix a broken operator."""
        errors = self._collect_errors(operator)
        if not errors:
            return operator

        for _ in range(self.config.max_refine_attempts):
            prompt = GRAPH_NAS_REFINEMENT_PROMPT.format(
                code=operator.code.strip(),
                errors="\n".join(f"- {e}" for e in errors),
                design_point=_format_design_point(operator.design_choices),
            )

            try:
                response = self.llm_client.query(prompt=prompt, temperature=0.3)
                self.tokens_used += GRAPH_NAS_TOKENS_PER_VARIANT // 2

                parsed = self._parse_response(response.content)
                if parsed and "code" in parsed:
                    operator.code = self._clean_code(parsed["code"])
                    if self._validate_graph_operator(operator):
                        return operator

            except Exception:
                pass

        return None

    def _collect_errors(self, operator: Operator) -> list[str]:
        """Collect error messages from Graph NAS operator validation."""
        from src.domains.nas.graph_architecture import (
            GraphArchitecture,
            GRAPH_NUM_NODES,
            GRAPH_NUM_OPS,
            GRAPH_MAX_CONN,
        )

        errors = []

        try:
            compiled = compile(operator.code, "<string>", "exec")
            namespace: dict = {}
            exec(compiled, namespace)
        except SyntaxError as e:
            errors.append(f"SyntaxError: {e.msg} at line {e.lineno}")
            return errors
        except Exception as e:
            errors.append(f"Compilation error: {e}")
            return errors

        func = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("_"):
                func = obj
                break

        if func is None:
            errors.append("No function definition found in the code")
            return errors

        try:
            test_arch = GraphArchitecture.random(random.Random(42))

            class _DummyCtx:
                pass

            result = func(test_arch, _DummyCtx())

            if not hasattr(result, "connectivity") or not hasattr(result, "operations"):
                errors.append(
                    f"Return type {type(result).__name__} missing .connectivity or .operations. "
                    "Must return a GraphArchitecture (use solution.copy())."
                )
            else:
                if len(result.connectivity) != GRAPH_NUM_NODES:
                    errors.append(
                        f"result.connectivity has length {len(result.connectivity)}, "
                        f"expected {GRAPH_NUM_NODES}"
                    )
                if len(result.operations) != GRAPH_NUM_NODES:
                    errors.append(
                        f"result.operations has length {len(result.operations)}, "
                        f"expected {GRAPH_NUM_NODES}"
                    )
                bad_conn = [c for c in result.connectivity if c < 0 or c >= GRAPH_MAX_CONN]
                if bad_conn:
                    errors.append(
                        f"Invalid connectivity values {bad_conn}. "
                        f"Each must be in [0, {GRAPH_MAX_CONN - 1}]."
                    )
                bad_ops = [o for o in result.operations if o < 0 or o >= GRAPH_NUM_OPS]
                if bad_ops:
                    errors.append(
                        f"Invalid operation values {bad_ops}. "
                        f"Each must be in [0, {GRAPH_NUM_OPS - 1}]."
                    )
        except Exception as e:
            errors.append(f"Runtime error: {type(e).__name__}: {e}")

        return errors

    def _rank_by_fitness(
        self,
        candidates: list[Operator],
        evaluator: "NASBenchGraphEvaluator",
    ) -> list[Operator]:
        """Rank operators by average fitness on random architectures.

        Higher accuracy is better for Graph NAS.
        """
        from src.domains.nas.graph_architecture import GraphArchitecture

        rng = random.Random(self.config.seed)
        n_test = 20

        test_archs = [GraphArchitecture.random(rng) for _ in range(n_test)]

        scores: list[tuple[Operator, float]] = []
        for op in candidates:
            try:
                compiled = compile(op.code, "<string>", "exec")
                namespace: dict = {}
                exec(compiled, namespace)
                func = next(
                    obj for name, obj in namespace.items()
                    if callable(obj) and not name.startswith("_")
                )

                total_acc = 0.0
                valid = 0
                for arch in test_archs:
                    class _Ctx:
                        pass

                    try:
                        result = func(arch.copy(), _Ctx())
                        if (hasattr(result, "connectivity") and
                                hasattr(result, "operations") and
                                len(result.connectivity) == 4 and
                                len(result.operations) == 4):
                            if (all(0 <= c < 4 for c in result.connectivity) and
                                    all(0 <= o < 9 for o in result.operations)):
                                acc = evaluator.evaluate(result)
                                total_acc += acc
                                valid += 1
                    except Exception:
                        pass

                avg_acc = total_acc / valid if valid > 0 else 0.0
                op.fitness_scores = [avg_acc]
                scores.append((op, avg_acc))

            except Exception:
                scores.append((op, 0.0))

        # Sort by accuracy descending (higher is better for Graph NAS)
        scores.sort(key=lambda x: x[1], reverse=True)
        return [op for op, _ in scores]

    def _parse_response(self, content: str) -> dict | None:
        """Parse LLM response, handling various formats."""
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        start = content.find("{")
        if start != -1:
            depth = 0
            in_string = False
            escape = False
            for i, char in enumerate(content[start:], start):
                if escape:
                    escape = False
                    continue
                if char == "\\" and in_string:
                    escape = True
                    continue
                if char == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(content[start: i + 1])
                        except json.JSONDecodeError:
                            break

        code_match = re.search(r"```python\s*([\s\S]*?)\s*```", content)
        if code_match:
            return {
                "name": "extracted_operator",
                "code": code_match.group(1),
                "design_choices": {},
                "structural_changes": "Extracted from non-JSON response",
            }

        return None

    def _clean_code(self, code: str) -> str:
        """Clean up generated code."""
        code = re.sub(r"```python\s*", "", code)
        code = re.sub(r"```\s*", "", code)
        return code.strip()
