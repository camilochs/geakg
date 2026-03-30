"""NAS-specific L1 Generator: Operator synthesis for CellArchitecture.

Adapts the L1Generator pipeline for NAS-Bench-201 cell architectures.
Key differences from combinatorial optimization:
- Solutions are CellArchitecture (6 edges, 5 ops) not permutations
- Design axes are NAS-specific (connectivity, operation_bias, etc.)
- Validation checks edge bounds [0,4] not permutation validity
- Prompt template references cell operations, not tour/distance

Usage:
    generator = NASGeneratorL1(llm_client, config=NASL1Config(...))
    pool = generator.generate(evaluator, roles=["topo_feedforward", ...])
    pool.save("pools/nas_pool.json")
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.geakg.layers.l1.base_operators_nas_bench import (
    ALL_NAS_BENCH_ROLES,
    NAS_BENCH_BASE_OPERATORS,
)
from src.geakg.layers.l1.pool import Operator, OperatorPool

if TYPE_CHECKING:
    from src.domains.nas.nasbench_evaluator import NASBench201Evaluator
    from src.llm.client import OllamaClient, OpenAIClient


# Estimated tokens per NAS variant (prompt + response)
NAS_TOKENS_PER_VARIANT = 1200

# ---- NAS Design Space ----

NAS_DESIGN_AXES: dict[str, list[str]] = {
    "connectivity": [
        "sparse (prefer fewer non-none edges, 3-4 active)",
        "dense (prefer more non-none edges, 5-6 active)",
        "balanced (mix of active and inactive, ~4-5 active)",
        "adaptive (choose connectivity based on current cell state)",
    ],
    "operation_bias": [
        "conv-heavy (prefer nor_conv_3x3 for capacity)",
        "lightweight (prefer nor_conv_1x1 and skip_connect)",
        "diverse (mix all operation types for complementarity)",
        "pooling-aware (use avg_pool_3x3 strategically for downsampling)",
    ],
    "edge_priority": [
        "early-edges (prioritize edges 0-2, closer to input node)",
        "late-edges (prioritize edges 3-5, closer to output node)",
        "bottleneck (modify the edge with worst estimated contribution)",
        "uniform (treat all 6 edges equally)",
    ],
    "strategy": [
        "targeted (change exactly 1 edge based on analysis)",
        "batch (change 2-3 edges in a coordinated pattern)",
        "constructive (build good patterns: skip+conv combinations)",
        "destructive (remove weak edges by setting to none, then rebuild)",
    ],
}

NAS_TOTAL_COMBINATIONS = 1
for _options in NAS_DESIGN_AXES.values():
    NAS_TOTAL_COMBINATIONS *= len(_options)  # 4^4 = 256


def _sample_nas_design_point(rng: random.Random, category: str) -> dict[str, str]:
    """Sample a NAS design point, optionally respecting category constraints."""
    # Category-specific biases
    constraints: dict[str, list[str]] = {}
    if category == "topology":
        constraints["strategy"] = NAS_DESIGN_AXES["strategy"]  # all valid
    elif category == "activation":
        constraints["operation_bias"] = [
            "lightweight (prefer nor_conv_1x1 and skip_connect)",
            "diverse (mix all operation types for complementarity)",
            "pooling-aware (use avg_pool_3x3 strategically for downsampling)",
        ]
    elif category == "regularization":
        constraints["connectivity"] = [
            "sparse (prefer fewer non-none edges, 3-4 active)",
            "balanced (mix of active and inactive, ~4-5 active)",
            "adaptive (choose connectivity based on current cell state)",
        ]

    point = {}
    for axis, options in NAS_DESIGN_AXES.items():
        available = constraints.get(axis, options)
        point[axis] = rng.choice(available)
    return point


def _format_nas_design_point(point: dict[str, str]) -> str:
    return "\n".join(f"- {axis}: {choice}" for axis, choice in point.items())


def _design_point_key(point: dict[str, str]) -> tuple:
    return tuple(sorted(point.items()))


# ---- NAS Role Category Mapping ----

_NAS_ROLE_CATEGORY = {
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

# ---- NAS Category Guidance ----

_NAS_CATEGORY_GUIDANCE = {
    "topology": """**Topology operators** modify the cell's edge structure.
- Good topology operators create well-connected cells with diverse operations
- Skip connections + convolutions are complementary (ResNet principle)
- Edges to node 3 (edges 3,4,5) have higher impact on accuracy
- All-none or all-same architectures perform poorly""",

    "activation": """**Activation operators** change operation types with semantic intent.
- Replace weak operations (none) with useful ones (conv, skip, pool)
- nor_conv_1x1 is lightweight but effective for channel mixing
- avg_pool_3x3 provides spatial invariance without learnable parameters
- Balance between parametric (conv) and non-parametric (pool, skip) ops""",

    "training": """**Training operators** make incremental modifications.
- Small targeted changes (1-2 edges) preserve good existing structure
- Changing conv_3x3 <-> conv_1x1 adjusts model capacity
- Adding trainable ops (conv) to none edges increases learning capacity
- Neighbor operation changes (±1 in op index) are conservative but safe""",

    "regularization": """**Regularization operators** simplify or prune the cell.
- Setting edges to none acts as drop-path regularization
- Replacing heavy ops (conv_3x3) with light ops (skip, conv_1x1) reduces overfitting
- Ensure minimum connectivity (at least 3-4 non-none edges)
- Over-pruning destroys model capacity; under-pruning causes overfitting""",

    "evaluation": """**Evaluation operators** validate or score the architecture.
- Clamp edges to valid range [0, 4]
- Count operation types for heuristic quality estimate
- These operators typically don't modify the architecture""",
}

# ---- NAS Prompt Template ----

NAS_DESIGN_SPACE_PROMPT = """You are designing a {role_category} operator for **Neural Architecture Search** on NAS-Bench-201.

**DOMAIN CONTEXT - NAS-Bench-201:**
- A cell architecture is a DAG with 4 nodes and 6 directed edges
- Each edge chooses from 5 operations: {{none=0, skip_connect=1, nor_conv_1x1=2, nor_conv_3x3=3, avg_pool_3x3=4}}
- Total search space: 5^6 = 15,625 unique architectures
- Goal: maximize classification accuracy (higher is better)
- The solution object has `solution.edges` (list of 6 ints in [0,4]) and `solution.copy()`
- Edge layout: edges[0]=0→1, edges[1]=0→2, edges[2]=1→2, edges[3]=0→3, edges[4]=1→3, edges[5]=2→3
- Known good patterns: diverse operations, skip+conv combinations, 4-5 active (non-none) edges

{category_guidance}

**GOAL: Create a POWERFUL {role_category} operator using your knowledge of effective NAS techniques.**

The original operator (A₀) is intentionally simple. You must create something MUCH BETTER by:
- Analyzing the current cell state before modifying
- Making informed decisions based on operation distribution
- Using NAS-Bench-201 insights (e.g., edge position importance)

**Original operator (A₀) - this is WEAK, you must do BETTER:**
```python
{original_code}
```

**Design Space - guide your approach:**
{design_point}

**Rules:**
1. Copy first: `result = solution.copy()`
2. Modify result.edges[i] (integers 0-4, for 6 edges indexed 0-5)
3. Return `result` (always return a valid CellArchitecture)
4. Import random at function start if needed
5. Access edges via `result.edges[i]` or `len(result.edges)` (always 6)

Return JSON:
{{
  "name": "{category}_<descriptive_name>",
  "design_choices": {{"connectivity": "...", "operation_bias": "...", "edge_priority": "...", "strategy": "..."}},
  "structural_changes": "<explain your NAS-specific algorithmic approach>",
  "code": "def {category}_<name>(solution, ctx):\\n    ..."
}}
"""

NAS_REFINEMENT_PROMPT = """Fix the errors in this NAS operator while keeping the design choices.

**Original code with errors:**
```python
{code}
```

**Errors to fix:**
{errors}

**Design choices (preserve these):**
{design_point}

**Rules:**
- solution.edges is a list of 6 integers in [0, 4]
- Use solution.copy() to clone
- Return a CellArchitecture with valid edges

Return JSON:
{{
  "code": "def <fixed_function>(solution, ctx):\\n    ...",
  "changes_made": "<describe what was fixed>"
}}
"""


# ---- NAS L1 Config ----

@dataclass
class NASL1Config:
    """Configuration for NAS L1 operator generation."""

    max_tokens: int = 15_000
    """Total token budget for generation."""

    pool_size_per_role: int = 3
    """How many operators to keep per role after selection."""

    temperature: float = 0.7
    """LLM temperature for generation."""

    max_refine_attempts: int = 2
    """Maximum refinement attempts per variant."""

    seed: int = 42
    """Random seed."""

    roles: list[str] = field(default_factory=lambda: ALL_NAS_BENCH_ROLES.copy())
    """Roles to generate operators for."""

    skip_eval_roles: bool = True
    """Skip generation for evaluation roles (eval_proxy, eval_full)."""


# ---- NAS L1 Generator ----

class NASGeneratorL1:
    """L1 Generator specialized for NAS-Bench-201 CellArchitecture operators.

    Generates operator variants for 18 NAS roles using Design-Space Prompting
    with NAS-specific axes (connectivity, operation_bias, edge_priority, strategy).

    Usage:
        llm = OpenAIClient(model="gpt-4o-mini")
        gen = NASGeneratorL1(llm, config=NASL1Config(max_tokens=15000))
        pool = gen.generate(evaluator)
    """

    def __init__(
        self,
        llm_client: "OllamaClient | OpenAIClient",
        config: NASL1Config | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or NASL1Config()
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
        evaluator: "NASBench201Evaluator | None" = None,
    ) -> OperatorPool:
        """Generate NAS operator pool using AFO + Design-Space Prompting.

        Args:
            evaluator: Optional evaluator for fitness-based selection.
                      If None, selection is based only on validation.

        Returns:
            OperatorPool with operators per NAS role.
        """
        roles = [
            r for r in self.config.roles
            if not (self.config.skip_eval_roles and r.startswith("eval_"))
        ]

        n_roles = len(roles)
        tokens_per_role = self.config.max_tokens // max(n_roles, 1)
        variants_per_role = max(1, tokens_per_role // NAS_TOKENS_PER_VARIANT)

        logger.info(
            f"[NAS-L1] Starting generation: {n_roles} roles, "
            f"~{variants_per_role} variants/role, {self.config.max_tokens} token budget"
        )

        pool = OperatorPool(
            metadata={
                "domain": "nas_bench",
                "tokens_used": 0,
                "llm_model": getattr(self.llm_client, "model", "unknown"),
            }
        )

        for role in roles:
            if self.tokens_used >= self.config.max_tokens:
                logger.info("[NAS-L1] Token budget exhausted")
                break

            logger.info(f"[NAS-L1] Generating variants for {role}")
            candidates = self._generate_role_variants(
                role=role,
                max_variants=variants_per_role,
                evaluator=evaluator,
            )

            # Add A₀ first, then variants
            for op in candidates[:self.config.pool_size_per_role]:
                pool.add_operator(op)

            logger.info(f"[NAS-L1]   {role}: {len(candidates)} valid operators")

        # Also add A₀ for eval roles (no LLM needed)
        if self.config.skip_eval_roles:
            for role in self.config.roles:
                if role.startswith("eval_") and role in NAS_BENCH_BASE_OPERATORS:
                    base_op = Operator(
                        name=f"{role}_base",
                        code=NAS_BENCH_BASE_OPERATORS[role].strip(),
                        role=role,
                        design_choices={},
                        interaction_effects="Base operator (A₀) — evaluation role",
                    )
                    pool.add_operator(base_op)

        pool.metadata["tokens_used"] = self.tokens_used
        pool.metadata["generation_stats"] = dict(self.generation_stats)

        logger.info(
            f"[NAS-L1] Complete: {pool.total_operators} operators across "
            f"{len(pool.roles)} roles, {self.tokens_used} tokens used"
        )
        logger.info(f"[NAS-L1] Stats: {self.generation_stats}")

        return pool

    def _generate_role_variants(
        self,
        role: str,
        max_variants: int,
        evaluator: "NASBench201Evaluator | None" = None,
    ) -> list[Operator]:
        """Generate variants for a single NAS role."""
        category = _NAS_ROLE_CATEGORY.get(role, "topology")
        base_code = NAS_BENCH_BASE_OPERATORS.get(role, "")

        # Always include A₀
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

            # Sample unique design point
            design_point = self._sample_unique_point(category)
            if design_point is None:
                break

            self.generation_stats["total_attempts"] += 1

            # Generate variant via LLM
            variant = self._generate_variant(role, category, base_code, design_point)
            if variant is None:
                self.generation_stats["failed_count"] += 1
                continue

            # Validate
            if self._validate_nas_operator(variant):
                candidates.append(variant)
                self.generation_stats["valid_count"] += 1
                logger.info(f"[NAS-L1]   + {variant.name}")
            else:
                # Try refinement
                refined = self._refine_variant(variant)
                if refined and self._validate_nas_operator(refined):
                    candidates.append(refined)
                    self.generation_stats["refined_count"] += 1
                    logger.info(f"[NAS-L1]   + {refined.name} (refined)")
                else:
                    self.generation_stats["failed_count"] += 1

        # If evaluator available, rank by fitness
        if evaluator and len(candidates) > self.config.pool_size_per_role:
            candidates = self._rank_by_fitness(candidates, evaluator)

        return candidates

    def _sample_unique_point(self, category: str) -> dict[str, str] | None:
        """Sample a NAS design point not yet used."""
        for _ in range(50):
            point = _sample_nas_design_point(self.rng, category)
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
        """Generate a single NAS operator variant via LLM."""
        category_guidance = _NAS_CATEGORY_GUIDANCE.get(category, "")

        prompt = NAS_DESIGN_SPACE_PROMPT.format(
            role_category=category,
            category=category,
            category_guidance=category_guidance,
            original_code=base_code.strip(),
            design_point=_format_nas_design_point(design_point),
        )

        try:
            response = self.llm_client.query(
                prompt=prompt,
                temperature=self.config.temperature,
            )
            self.tokens_used += NAS_TOKENS_PER_VARIANT

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
            logger.warning(f"[NAS-L1] Generation failed for {role}: {e}")
            self.tokens_used += NAS_TOKENS_PER_VARIANT // 2
            return None

    def _validate_nas_operator(self, operator: Operator) -> bool:
        """Validate that a NAS operator compiles and produces valid CellArchitectures."""
        from src.domains.nas.cell_architecture import CellArchitecture, NUM_EDGES, NUM_OPS

        try:
            compiled = compile(operator.code, "<string>", "exec")
            namespace: dict = {}
            exec(compiled, namespace)

            # Find the function
            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    func = obj
                    break

            if func is None:
                return False

            # Test on 3 random architectures
            test_rng = random.Random(42)
            for _ in range(3):
                test_arch = CellArchitecture.random(test_rng)

                class _DummyCtx:
                    pass

                result = func(test_arch, _DummyCtx())

                # Must return something with .edges
                if not hasattr(result, "edges"):
                    return False
                if len(result.edges) != NUM_EDGES:
                    return False
                if not all(0 <= e < NUM_OPS for e in result.edges):
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
            prompt = NAS_REFINEMENT_PROMPT.format(
                code=operator.code.strip(),
                errors="\n".join(f"- {e}" for e in errors),
                design_point=_format_nas_design_point(operator.design_choices),
            )

            try:
                response = self.llm_client.query(prompt=prompt, temperature=0.3)
                self.tokens_used += NAS_TOKENS_PER_VARIANT // 2

                parsed = self._parse_response(response.content)
                if parsed and "code" in parsed:
                    operator.code = self._clean_code(parsed["code"])
                    if self._validate_nas_operator(operator):
                        return operator

            except Exception:
                pass

        return None

    def _collect_errors(self, operator: Operator) -> list[str]:
        """Collect error messages from NAS operator validation."""
        from src.domains.nas.cell_architecture import CellArchitecture, NUM_EDGES, NUM_OPS

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
            test_arch = CellArchitecture.random(random.Random(42))

            class _DummyCtx:
                pass

            result = func(test_arch, _DummyCtx())

            if not hasattr(result, "edges"):
                errors.append(
                    f"Return type {type(result).__name__} has no .edges attribute. "
                    "Must return a CellArchitecture (use solution.copy())."
                )
            elif len(result.edges) != NUM_EDGES:
                errors.append(
                    f"result.edges has length {len(result.edges)}, expected {NUM_EDGES}"
                )
            elif not all(0 <= e < NUM_OPS for e in result.edges):
                bad = [e for e in result.edges if e < 0 or e >= NUM_OPS]
                errors.append(
                    f"Invalid edge values {bad}. Each edge must be in [0, {NUM_OPS - 1}]."
                )
        except Exception as e:
            errors.append(f"Runtime error: {type(e).__name__}: {e}")

        return errors

    def _rank_by_fitness(
        self,
        candidates: list[Operator],
        evaluator: "NASBench201Evaluator",
    ) -> list[Operator]:
        """Rank operators by average fitness on random architectures."""
        from src.domains.nas.cell_architecture import CellArchitecture

        rng = random.Random(self.config.seed)
        n_test = 20  # Test each operator on 20 random architectures

        # Generate test architectures
        test_archs = [CellArchitecture.random(rng) for _ in range(n_test)]

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
                        if hasattr(result, "edges") and len(result.edges) == 6:
                            if all(0 <= e < 5 for e in result.edges):
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

        # Sort by accuracy descending (higher is better in NAS)
        scores.sort(key=lambda x: x[1], reverse=True)
        return [op for op, _ in scores]

    def _parse_response(self, content: str) -> dict | None:
        """Parse LLM response, handling various formats."""
        # Try direct JSON
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try markdown JSON block
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object
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

        # Extract code from python block
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
