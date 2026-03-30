"""Iterative Pool Refinement: ACO-guided operator generation.

This module implements the iterative refinement loop:
1. Start with generic operators (base pool)
2. Run ACO to discover structural patterns
3. Analyze snapshot to find "weak spots" (roles with low diversity, ineffective transitions)
4. Generate operators specifically for those weak contexts
5. Repeat until pool is good enough

The key insight: ACO discovers WHAT contexts need better operators,
then we generate operators specifically for those contexts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from src.geakg.layers.l1.program_generator import generate_best_program

if TYPE_CHECKING:
    from src.geakg.aco import MetaACOSelector
    from src.geakg.instance_pool import InstancePool
    from src.geakg.layers.l1.pool import Operator, OperatorPool
    from src.llm.client import OllamaClient, OpenAIClient


@dataclass
class WeakSpot:
    """A weak spot identified in the ACO snapshot.

    Represents a context where the current pool is underperforming
    and could benefit from a new specialized operator.
    """

    role: str
    """The role that needs improvement."""

    category: str
    """Role category (construction, local_search, perturbation)."""

    reason: str
    """Why this is a weak spot (e.g., 'low diversity', 'high failure rate')."""

    context: dict[str, Any] = field(default_factory=dict)
    """Additional context for generation (transitions, stats, etc.)."""

    priority: float = 1.0
    """Priority score (higher = more important to fix)."""


@dataclass
class ValidationStats:
    """Statistics for operator validation errors during generation."""

    total_generated: int = 0
    """Total operators attempted to generate."""

    successful: int = 0
    """Operators that passed validation."""

    timeout_errors: int = 0
    """Operators that timed out (possible infinite loops)."""

    syntax_errors: int = 0
    """Operators with syntax errors."""

    runtime_errors: int = 0
    """Operators that crashed during execution."""

    invalid_result_errors: int = 0
    """Operators that returned invalid permutations."""

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for snapshot."""
        return {
            "total_generated": self.total_generated,
            "successful": self.successful,
            "timeout_errors": self.timeout_errors,
            "syntax_errors": self.syntax_errors,
            "runtime_errors": self.runtime_errors,
            "invalid_result_errors": self.invalid_result_errors,
        }


@dataclass
class SnapshotAnalysis:
    """Analysis results from an ACO snapshot."""

    weak_spots: list[WeakSpot]
    """Identified weak spots that need new operators."""

    role_diversity: dict[str, int]
    """Number of effective operators per role."""

    transition_patterns: dict[tuple[str, str], float]
    """Learned transition strengths between roles."""

    operator_effectiveness: dict[str, float]
    """Operator -> effectiveness score (0-1)."""

    overall_gap: float
    """Best gap achieved by ACO."""

    def has_weak_spots(self) -> bool:
        """Check if there are weak spots to address."""
        return len(self.weak_spots) > 0

    def get_top_weak_spots(self, n: int = 5, balance_categories: bool = True) -> list[WeakSpot]:
        """Get the top N weak spots, balanced across categories.

        Args:
            n: Maximum number of weak spots to return
            balance_categories: If True, distribute spots across categories
                               (construction, local_search, perturbation)

        Returns:
            List of weak spots to address
        """
        if not balance_categories:
            return sorted(self.weak_spots, key=lambda x: -x.priority)[:n]

        # Group by category
        by_category: dict[str, list[WeakSpot]] = {
            "construction": [],
            "local_search": [],
            "perturbation": [],
        }
        # Sort by priority, with random tiebreaker to avoid always picking the same role
        import random
        shuffled = self.weak_spots.copy()
        random.shuffle(shuffled)  # Randomize first
        for spot in sorted(shuffled, key=lambda x: -x.priority):  # Then sort by priority (stable)
            if spot.category in by_category:
                by_category[spot.category].append(spot)

        # Round-robin selection across categories
        result = []
        categories = ["construction", "local_search", "perturbation"]
        indices = {cat: 0 for cat in categories}

        while len(result) < n:
            added_this_round = False
            for cat in categories:
                if len(result) >= n:
                    break
                if indices[cat] < len(by_category[cat]):
                    result.append(by_category[cat][indices[cat]])
                    indices[cat] += 1
                    added_this_round = True

            if not added_this_round:
                break  # No more spots in any category

        return result


def analyze_snapshot(
    selector: "MetaACOSelector",
    pool: "OperatorPool",
    min_effective_pheromone: float = 0.5,
    min_operators_per_role: int = 2,
) -> SnapshotAnalysis:
    """Analyze ACO snapshot to identify weak spots.

    Args:
        selector: The ACO selector after training
        pool: Current operator pool
        min_effective_pheromone: Minimum pheromone to consider an operator "effective"
        min_operators_per_role: Minimum operators needed per role

    Returns:
        SnapshotAnalysis with identified weak spots
    """
    weak_spots = []

    # Get pheromone data
    operator_pheromones = selector.get_operator_pheromones()
    role_pheromones = selector.pheromones

    # Get operator stats
    operator_stats = selector.get_operator_stats()

    # Analyze role diversity
    role_diversity: dict[str, int] = {}
    operator_effectiveness: dict[str, float] = {}

    # Get max pheromone for normalization
    max_tau = max(operator_pheromones.values()) if operator_pheromones else 1.0

    for role in pool.roles:
        # Count effective operators (pheromone above threshold)
        effective_count = 0
        role_operators = []

        for (r, op), tau in operator_pheromones.items():
            if r == role:
                normalized_tau = tau / max_tau if max_tau > 0 else 0
                operator_effectiveness[op] = normalized_tau
                role_operators.append((op, normalized_tau))

                if normalized_tau >= min_effective_pheromone:
                    effective_count += 1

        role_diversity[role] = effective_count

        # Check for weak spots
        if effective_count < min_operators_per_role:
            # Low diversity - need more operators
            category = _get_role_category(role)

            # Get transition context
            incoming = [
                (src, tau)
                for (src, tgt), tau in role_pheromones.items()
                if tgt == role
            ]
            outgoing = [
                (tgt, tau)
                for (src, tgt), tau in role_pheromones.items()
                if src == role
            ]

            weak_spots.append(
                WeakSpot(
                    role=role,
                    category=category,
                    reason=f"Low diversity: only {effective_count} effective operators",
                    context={
                        "current_operators": role_operators,
                        "incoming_transitions": incoming,
                        "outgoing_transitions": outgoing,
                    },
                    priority=2.0 - effective_count / min_operators_per_role,
                )
            )

    # Check for roles with high failure rates
    for op_id, stats in operator_stats.items():
        if stats.get("total_uses", 0) >= 10:  # Enough data
            success_rate = stats.get("total_successes", 0) / stats["total_uses"]
            if success_rate < 0.3:  # Less than 30% success
                # Find the role for this operator
                role = None
                for (r, o), _ in operator_pheromones.items():
                    if o == op_id:
                        role = r
                        break

                if role:
                    weak_spots.append(
                        WeakSpot(
                            role=role,
                            category=_get_role_category(role),
                            reason=f"High failure rate: {op_id} has {success_rate:.0%} success",
                            context={
                                "failing_operator": op_id,
                                "stats": stats,
                            },
                            priority=1.5,
                        )
                    )

    # Note: We skip checking for missing transition coverage because:
    # 1. Not all transitions should exist (e.g., const→const is forbidden)
    # 2. The metagraph already defines valid transitions
    # 3. This would generate too many low-priority weak spots

    # Build transition patterns
    transition_patterns = {
        (src, tgt): tau for (src, tgt), tau in role_pheromones.items()
    }

    # Get overall gap from selector
    overall_gap = selector.best_fitness if hasattr(selector, "best_fitness") else float("inf")

    return SnapshotAnalysis(
        weak_spots=weak_spots,
        role_diversity=role_diversity,
        transition_patterns=transition_patterns,
        operator_effectiveness=operator_effectiveness,
        overall_gap=overall_gap,
    )


def _get_role_category(role: str) -> str:
    """Get category from role name."""
    if role.startswith("const"):
        return "construction"
    elif role.startswith("ls"):
        return "local_search"
    elif role.startswith("pert"):
        return "perturbation"
    return "local_search"


# Prompt for contextual operator generation with AFO + Design-Space + Evolutionary Feedback
# Working examples by category - these compile and run correctly
CATEGORY_EXAMPLES = {
    "construction": {
        "name": "const_greedy_swap",
        "code": """def const_greedy_swap(s, ctx):
    import random
    n = len(s)
    # Start with a complete random tour
    tour = list(range(n))
    random.shuffle(tour)
    best_cost = ctx.evaluate(tour)
    # Greedy improvement: try swaps
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                tour[i], tour[j] = tour[j], tour[i]
                cost = ctx.evaluate(tour)
                if cost < best_cost:
                    best_cost = cost
                    improved = True
                else:
                    tour[i], tour[j] = tour[j], tour[i]
    return tour""",
        "description": "Start with random complete tour, then greedy swap improvement. Always returns n elements."
    },
    "local_search": {
        "name": "ls_2opt_first",
        "code": """def ls_2opt_first(s, ctx):
    n = len(s)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                d = ctx.delta(s, '2opt', i, j)
                if d < -1e-9:
                    s[i+1:j+1] = s[i+1:j+1][::-1]
                    improved = True
                    break
            if improved:
                break
    return s""",
        "description": "First-improvement 2-opt using ctx.delta() for O(1) evaluation"
    },
    "perturbation": {
        "name": "pert_double_bridge",
        "code": """def pert_double_bridge(s, ctx):
    import random
    n = len(s)
    if n < 8:
        i, j = random.sample(range(n), 2)
        s[i], s[j] = s[j], s[i]
        return s
    pos = sorted(random.sample(range(1, n), 3))
    p1, p2, p3 = pos
    return s[:p1] + s[p2:p3] + s[p1:p2] + s[p3:]""",
        "description": "Double-bridge perturbation for escaping local optima"
    },
}

CONTEXTUAL_OPERATOR_PROMPT = """Generate a state-of-the-art {category} operator for {domain} optimization.

## Task
Role: {role}
{reason}
Use well-known, proven techniques from the combinatorial optimization literature.

## Existing Operators (learn from these)
{existing_operators_with_fitness}

## Design Direction
{design_point}

## API Reference
- `s`: permutation list [0..n-1] (input, use len(s) to get n)
- `ctx.evaluate(tour)` -> float: total cost of a COMPLETE tour
- `ctx.delta(s, move, i, j)` -> float: cost change (O(1), negative=better) [local_search only]
- `ctx.neighbors(s, i, k)` -> list[int]: k nearest positions [local_search only]

## Category-Specific Rules
{category_rules}

## Working Example ({category})
```python
{example_code}
```
{example_description}

## Requirements
1. Use 4-space indentation consistently
2. random.choice/sample need list not set: random.choice(list(myset))
3. Always return a valid permutation of [0..n-1]
4. EFFICIENCY: Must complete in <1 second for n=100. Avoid O(n³) loops. Use early termination.

## Output
Return JSON with "name" and "code".

```json
{{
  "name": "{role}_<suffix>",
  "code": "def {role}_<suffix>(s, ctx):\\n    ..."
}}
```"""

# Category-specific rules to include in prompt
CATEGORY_RULES = {
    "construction": """- BUILD a new tour from scratch (use n = len(s) to get size)
- MUST return a COMPLETE tour: list with exactly n elements containing [0..n-1] each once
- SAFE PATTERN: Start with `tour = list(range(n))`, shuffle, then improve
- Use ctx.evaluate(tour) only on COMPLETE tours
- Limit iterations: max 1-2 passes over the tour""",

    "local_search": """- IMPROVE the existing tour s
- Use ctx.delta(s, move, i, j) for O(1) move evaluation (NOT ctx.evaluate in loops)
- Use first-improvement: return as soon as you find an improving move
- Limit iterations with max_iter or early termination""",

    "perturbation": """- PERTURB the tour to escape local optima (DO NOT optimize)
- Make random structural changes: 2-4 random swaps, segment shuffle, or double-bridge
- Must be FAST: O(n) or O(1), no nested loops over full tour
- Return immediately after perturbation""",
}


def _format_operators_with_fitness(pool: "OperatorPool | None", role: str, max_ops: int = 3) -> str:
    """Format existing operators with their code and fitness for evolutionary feedback.

    Shows the best operators first so the LLM can learn from successful patterns.

    Args:
        pool: Operator pool
        role: Role to get operators for
        max_ops: Maximum operators to show (to limit prompt size)

    Returns:
        Formatted string with operators ranked by fitness
    """
    if not pool:
        return "None yet (this is the first operator for this role)."

    operators = pool.get_operators_for_role(role)
    if not operators:
        return "None yet (this is the first operator for this role)."

    # Sort by avg_delta (lower/more negative = better, inf goes last)
    sorted_ops = sorted(
        operators,
        key=lambda o: o.avg_fitness if o.avg_fitness != float("inf") else 1e10
    )

    lines = []
    for i, op in enumerate(sorted_ops[:max_ops], 1):
        # Format fitness (avg_delta: negative = reduces cost = good)
        if op.avg_fitness == float("inf") or not op.fitness_scores:
            fitness_str = "not evaluated"
        else:
            delta = op.avg_fitness
            if delta < 0:
                fitness_str = f"reduces cost by {-delta:.1f} per use"
            elif delta > 0:
                fitness_str = f"increases cost by {delta:.1f} per use"
            else:
                fitness_str = "no effect"

        lines.append(f"### {i}. {op.name} ({fitness_str})")
        lines.append(f"```python\n{op.code}\n```")

        # Brief analysis hint
        if op.avg_fitness != float("inf") and op.fitness_scores:
            if "delta" in op.code:
                lines.append("✓ Uses delta() for efficiency")
            elif "evaluate" in op.code and "for" in op.code:
                lines.append("⚠ Uses evaluate() in loop (slow)")
            if "neighbors" in op.code:
                lines.append("✓ Uses candidate lists")
        lines.append("")

    # Add summary if there are more operators
    if len(operators) > max_ops:
        lines.append(f"({len(operators) - max_ops} more operators not shown)")

    return "\n".join(lines)


def generate_contextual_operator(
    weak_spot: WeakSpot,
    llm_client: "OllamaClient | OpenAIClient",
    domain: str,
    temperature: float = 0.7,
    max_refine_attempts: int = 2,
    instances: list[Any] | None = None,
    ctx_factory: Callable[[Any], Any] | None = None,
    pool: "OperatorPool | None" = None,
) -> tuple["Operator | None", str]:
    """Generate an operator using AFO + Design-Space Prompting.

    Uses the AFO (Always-From-Original) principle: variants are generated
    from the base operator A₀, not iteratively from other variants.

    Also uses Design-Space Prompting with 4 orthogonal axes to ensure
    structural diversity rather than superficial variation.

    Args:
        weak_spot: The weak spot to address
        llm_client: LLM client for generation
        domain: Target domain (e.g., "tsp", "vrp")
        temperature: LLM temperature
        max_refine_attempts: Max attempts to fix validation errors
        instances: Test instances for validation
        ctx_factory: Factory to create domain context
        pool: Operator pool to get base operator A₀ from

    Returns:
        Tuple of (operator, error_type) where error_type is one of:
        - "ok": success
        - "parse_error": failed to parse LLM response
        - "syntax": syntax/compilation error
        - "timeout": execution timeout (possible infinite loop)
        - "runtime": runtime exception
        - "invalid_result": returned invalid permutation
    """
    import re
    from src.geakg.layers.l1.pool import Operator
    from src.geakg.layers.l1.base_operators import BASE_OPERATORS
    from src.geakg.layers.l1.design_space import sample_design_point_for_category, format_design_point

    # Sample design point for this category
    rng = random.Random()
    design_point = sample_design_point_for_category(rng, weak_spot.category)
    design_point_str = format_design_point(design_point)

    # Get existing operators with fitness for evolutionary feedback
    existing_ops_with_fitness = _format_operators_with_fitness(pool, weak_spot.role, max_ops=3)
    logger.debug(f"[ITERATIVE] Feedback for {weak_spot.role}:\n{existing_ops_with_fitness}")

    # Get working example and rules for this category
    example = CATEGORY_EXAMPLES.get(weak_spot.category, CATEGORY_EXAMPLES["local_search"])
    category_rules = CATEGORY_RULES.get(weak_spot.category, CATEGORY_RULES["local_search"])

    prompt = CONTEXTUAL_OPERATOR_PROMPT.format(
        domain=domain.upper(),
        role=weak_spot.role,
        category=weak_spot.category,
        reason=weak_spot.reason,
        design_point=design_point_str,
        existing_operators_with_fitness=existing_ops_with_fitness,
        example_code=example["code"],
        example_description=example["description"],
        category_rules=category_rules,
    )

    def parse_response(content: str) -> dict | None:
        """Parse LLM response to extract operator data."""
        # Try to extract JSON
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try direct parse
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        return None

    def create_operator(data: dict) -> Operator:
        """Create operator from parsed data."""
        code = data.get("code", "")
        code = re.sub(r"```python\s*", "", code)
        code = re.sub(r"```\s*", "", code)

        # Generate unique suffix (6-char hex) to match is_synth_operator pattern in execution.py
        # Pattern expected: _[a-f0-9]{6}$
        unique_suffix = "".join(random.choices("0123456789abcdef", k=6))

        # Get base name from LLM response
        base_name = data.get("name", f"{weak_spot.role}_variant")

        # Append unique suffix
        unique_name = f"{base_name}_{unique_suffix}"

        # Update the function name in the code to match
        # Find the function definition and replace the name
        code = re.sub(
            rf"def\s+{re.escape(base_name)}\s*\(",
            f"def {unique_name}(",
            code,
            count=1
        )

        # Merge design choices from response with sampled design point
        response_choices = data.get("design_choices", {})
        merged_choices = {
            **design_point,  # Sampled design axes
            **response_choices,  # LLM's implementation details
            "context": weak_spot.reason,
            "afo_base": weak_spot.role,
        }

        return Operator(
            name=unique_name,
            code=code.strip(),
            role=weak_spot.role,
            design_choices=merged_choices,
            interaction_effects=data.get("structural_changes", ""),
        )

    def collect_errors(operator: Operator, timeout_seconds: float = 5.0) -> tuple[list[str], str]:
        """Collect validation errors for refinement prompt.

        Returns:
            Tuple of (errors_list, error_type) where error_type is one of:
            - "ok": no errors
            - "syntax": syntax/compilation error
            - "timeout": execution timeout
            - "runtime": runtime exception
            - "invalid_result": invalid permutation
        """
        import concurrent.futures

        errors = []

        # Check compilation
        try:
            namespace: dict = {
                "random": __import__("random"),
                "math": __import__("math"),
            }
            exec(operator.code, namespace)

            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    func = obj
                    break

            if func is None:
                errors.append("No function definition found in code")
                return errors, "syntax"

        except SyntaxError as e:
            errors.append(f"SyntaxError: {e.msg} at line {e.lineno}")
            return errors, "syntax"
        except Exception as e:
            errors.append(f"Compilation error: {e}")
            return errors, "syntax"

        # Test execution with timeout using signal (Unix) for true interruption
        if instances and ctx_factory:
            import signal

            class TimeoutError(Exception):
                pass

            def timeout_handler(signum, frame):
                raise TimeoutError("Operator execution timed out")

            try:
                instance = instances[0]
                n = instance.get("dimension", 10) if isinstance(instance, dict) else 10
                ctx = ctx_factory(instance)
                solution = list(range(n))

                # Set up timeout using signal (works on Unix/macOS)
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout_seconds))

                try:
                    result = func(solution, ctx)
                finally:
                    signal.alarm(0)  # Cancel the alarm
                    signal.signal(signal.SIGALRM, old_handler)

                if not isinstance(result, list):
                    errors.append(f"Return type is {type(result).__name__}, expected list")
                    return errors, "invalid_result"
                elif len(result) != n:
                    errors.append(f"Returned list length {len(result)}, expected {n}")
                    return errors, "invalid_result"
                elif set(result) != set(range(n)):
                    missing = set(range(n)) - set(result)
                    extra = set(result) - set(range(n))
                    errors.append(f"Result is not a valid permutation. Missing: {missing}, Extra: {extra}")
                    return errors, "invalid_result"

            except TimeoutError:
                errors.append(f"Timeout: operator took longer than {timeout_seconds}s (possible infinite loop)")
                return errors, "timeout"
            except Exception as e:
                errors.append(f"Runtime error: {type(e).__name__}: {e}")
                return errors, "runtime"

        return errors, "ok"

    # Initial generation with retries for parse failures
    max_parse_retries = 2
    operator = None
    last_error_type = "parse_error"

    for parse_attempt in range(max_parse_retries + 1):
        try:
            logger.debug(f"[ITERATIVE] Calling LLM for {weak_spot.role}...")
            response = llm_client.query(prompt=prompt, temperature=temperature)
            logger.debug(f"[ITERATIVE] LLM responded for {weak_spot.role}")
            data = parse_response(response.content)

            if data is None:
                if parse_attempt < max_parse_retries:
                    logger.debug(f"[ITERATIVE] Parse failed, retrying ({parse_attempt + 1}/{max_parse_retries})")
                    continue
                logger.warning("[ITERATIVE] Failed to parse LLM response after retries")
                return None, "parse_error"

            operator = create_operator(data)
            break  # Success

        except Exception as e:
            if parse_attempt < max_parse_retries:
                logger.debug(f"[ITERATIVE] Generation error, retrying: {e}")
                continue
            logger.warning(f"[ITERATIVE] Generation failed after retries: {e}")
            return None, "parse_error"

    if operator is None:
        return None, "parse_error"

    # Validate and refine if needed
    if instances and ctx_factory:
        for attempt in range(max_refine_attempts + 1):
            errors, last_error_type = collect_errors(operator)

            if not errors:
                return operator, "ok"  # Valid!

            if attempt >= max_refine_attempts:
                logger.warning(f"[ITERATIVE] Max refinement attempts reached for {operator.name}. Last errors: {errors}")
                return None, last_error_type

            # Build refinement prompt - ask for complete fixed code
            errors_str = "; ".join(errors)

            # Category-specific fix hints
            category = weak_spot.category
            if category == "construction":
                fix_hint = """CRITICAL for construction operators:
- MUST return a list with exactly n elements (n = len(s))
- The list must contain each number from 0 to n-1 exactly once
- SAFE PATTERN: Start with `tour = list(range(n))`, then shuffle/modify it
- DO NOT build incrementally - risk of incomplete tour"""
            else:
                fix_hint = """Requirements:
- Return valid permutation with same elements as input s
- Use 4-space indentation consistently"""

            refinement_prompt = f"""Fix this Python function. Error: {errors_str}

Original code:
```python
{operator.code}
```

{fix_hint}

Return JSON with fixed code:
```json
{{"name": "{operator.name}", "code": "def {operator.name}(s, ctx):\\n    ..."}}
```"""

            try:
                refine_response = llm_client.query(prompt=refinement_prompt, temperature=0.2)
                refine_data = parse_response(refine_response.content)

                if refine_data and "code" in refine_data:
                    operator = create_operator(refine_data)
                    logger.debug(f"[ITERATIVE] Refinement applied for {operator.name}")
                else:
                    logger.debug("[ITERATIVE] Failed to parse refinement response")
                    continue

            except Exception as e:
                logger.debug(f"[ITERATIVE] Refinement failed: {e}")
                continue

    return operator, "ok"


def validate_operator(
    operator: "Operator",
    instances: list[Any],
    ctx_factory: Callable[[Any], Any],
    timeout_seconds: float = 5.0,
) -> tuple[bool, str]:
    """Validate that an operator compiles and produces valid solutions.

    Args:
        operator: Operator to validate
        instances: Test instances
        ctx_factory: Factory to create domain context
        timeout_seconds: Max seconds per operator execution (default 5s)

    Returns:
        Tuple of (is_valid, error_type) where error_type is one of:
        - "ok": no error
        - "syntax": syntax/compilation error
        - "timeout": execution timeout (possible infinite loop)
        - "runtime": runtime exception
        - "invalid_result": returned invalid permutation
    """
    # Check compilation first (fast, no timeout needed)
    try:
        namespace: dict = {
            "random": __import__("random"),
            "math": __import__("math"),
        }
        exec(operator.code, namespace)

        # Find function
        func = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("_"):
                func = obj
                break

        if func is None:
            return False, "syntax"

    except Exception as e:
        logger.debug(f"[ITERATIVE] Compilation failed: {e}")
        return False, "syntax"

    # Test on instances with timeout using signal (Unix) for true interruption
    import signal

    class TimeoutError(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutError("Operator execution timed out")

    for instance in instances[:3]:
        try:
            n = instance.get("dimension", 10) if isinstance(instance, dict) else 10
            ctx = ctx_factory(instance)
            solution = list(range(n))
            random.shuffle(solution)

            # Set up timeout using signal (works on Unix/macOS)
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))

            try:
                result = func(solution, ctx)
            finally:
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)

            # Validate result
            if not isinstance(result, list):
                return False, "invalid_result"
            if len(result) != n:
                return False, "invalid_result"
            if set(result) != set(range(n)):
                return False, "invalid_result"

        except TimeoutError:
            logger.debug(f"[ITERATIVE] Operator timed out after {timeout_seconds}s")
            return False, "timeout"
        except Exception as e:
            logger.debug(f"[ITERATIVE] Execution failed: {e}")
            return False, "runtime"

    return True, "ok"


@dataclass
class IterativeRefinementConfig:
    """Configuration for iterative refinement."""

    max_rounds: int = 3
    """Maximum refinement rounds."""

    aco_timeout: int = 60
    """Seconds to run ACO per round."""

    weak_spots_per_round: int = 3
    """Max weak spots to address per round."""

    n_ants: int = 10
    """Number of ants per ACO iteration."""

    min_effective_pheromone: float = 0.5
    """Threshold for effective operator pheromone."""

    initial_operators_per_category: int = 1
    """Number of operators to generate per CATEGORY (construction, local_search, perturbation)
    BEFORE first ACO round. Set to 0 to disable initial generation (original behavior).
    With default=1, generates 3 operators total (1 per category)."""


def run_iterative_refinement(
    pool: "OperatorPool",
    instance_pool: "InstancePool",
    domain_config: Any,
    llm_client: "OllamaClient | OpenAIClient",
    config: IterativeRefinementConfig | None = None,
    ctx_factory: Callable[[Any], Any] | None = None,
    use_llm_metagraph: bool = True,
    output_dir: str | Path | None = None,
    max_tokens: int | None = None,
    llamea_wrapper: Any | None = None,
) -> tuple["OperatorPool", dict]:
    """Run the iterative refinement loop.

    Uses InstancePool with Instance Hardness Sampling for multi-instance training.

    Args:
        pool: Initial operator pool (base/generic operators)
        instance_pool: InstancePool with loaded instances (supports hardness sampling)
        domain_config: Domain configuration
        llm_client: LLM client for generation
        config: Refinement configuration
        ctx_factory: Factory to create domain context
        use_llm_metagraph: If True, LLM generates metagraph topology (L2/L3).
                          If False, uses predefined structure.
        output_dir: Optional directory to save visualizations (best path per round).
        max_tokens: Optional maximum total tokens. Training stops when reached.
        llamea_wrapper: Optional LLaMEA wrapper to use as an operator (hybrid mode).

    Returns:
        Tuple of (refined pool, final snapshot data)
    """
    from src.geakg.aco import MetaACOConfig, MetaACOSelector, OperatorMode
    from src.geakg.bindings import BindingRegistry
    from src.geakg.conditions import ExecutionContext
    from src.geakg.execution import evaluate_operator_path_with_stats
    from src.geakg.layers.l1.hook import L0SynthesisHook
    from src.geakg.layers.l0.topology_generator import L0MetaGraphGenerator, create_default_metagraph_for_pool
    from src.geakg.layers.l0.metagraph import InstantiatedGraph
    from src.geakg.layers.l0.patterns import create_hybrid_meta_graph

    config = config or IterativeRefinementConfig()

    logger.info(f"[ITERATIVE] Starting refinement with {pool.total_operators} operators")
    logger.info(f"[ITERATIVE] Training on {len(instance_pool)} instances")

    # Setup
    BindingRegistry.reset()
    registry = BindingRegistry()
    bindings = registry.get_domain("tsp")

    best_avg_gap = float("inf")
    best_operator_path = []
    best_role_path = []
    history = []
    metagraph_reasoning = ""

    # Track validation errors during generation
    validation_stats = ValidationStats()

    # Selector may not be initialized if token limit is reached before any ACO round
    selector = None

    # For pheromone transfer between rounds
    prev_role_pheromones: dict[tuple[str, str], float] = {}
    prev_operator_pheromones: dict[tuple[str, str], float] = {}

    # For pruning: accumulate improvement history across rounds
    # Maps operator_id -> (improvements, total_uses)
    global_improvement_history: dict[str, tuple[int, int]] = {}

    # Global iteration counter for pruning (persists across rounds)
    global_iteration = 0

    # Generate metagraph ONCE before the loop (topology is fixed)
    # This ensures pheromone transfer works correctly across rounds
    domain_name = domain_config.name if hasattr(domain_config, "name") else "TSP"
    if use_llm_metagraph:
        logger.info("[ITERATIVE] LLM generating metagraph topology (L2/L3)...")
        metagraph_generator = L0MetaGraphGenerator(
            llm_client=llm_client,
            pool=pool,
            domain=domain_name,
        )
        meta_graph = metagraph_generator.generate()

        if meta_graph is None:
            logger.warning("[ITERATIVE] LLM metagraph failed, using default")
            meta_graph = create_default_metagraph_for_pool(pool)
        else:
            metagraph_reasoning = meta_graph.llm_reasoning or ""
            logger.info(f"[ITERATIVE] LLM metagraph: {len(meta_graph.nodes)} roles, {len(meta_graph.edges)} edges")
    else:
        logger.info("[ITERATIVE] Using predefined metagraph structure")
        meta_graph = create_hybrid_meta_graph()

    # Visualize the metagraph topology
    if output_dir:
        from src.geakg.visualization import visualize_metagraph_topology

        topo_path = Path(output_dir) / "metagraph_topology.png"
        visualize_metagraph_topology(
            meta_graph=meta_graph,
            output_path=topo_path,
            title=f"MetaGraph for {domain_name}",
        )
        logger.info(f"\033[96m[ITERATIVE] 📊 MetaGraph topology: {topo_path.absolute()}\033[0m")

    # ACO config (fixed across rounds)
    aco_config = MetaACOConfig(
        n_ants=config.n_ants,
        operator_mode=OperatorMode.DYNAMIC,
        enable_synthesis=False,
        enable_conditions=True,
        enable_incompatibility_tracking=True,
        # Pruning configuration (aggressive to remove slow _base operators)
        pruning_check_interval=5,  # Check every 5 iterations
        pruning_improvement_threshold=0.03,  # Prune if improvement < 3%
        pruning_min_uses=5,  # Only 5 uses before considering
        pruning_grace_period=0,  # No grace period - any operator can be pruned
    )

    def get_token_str() -> str:
        """Get formatted token usage string."""
        stats = llm_client.stats
        return f"tokens={stats.total_tokens:,} (in={stats.prompt_tokens:,}, out={stats.completion_tokens:,})"

    def check_token_limit() -> bool:
        """Check if token limit has been reached."""
        if max_tokens is None:
            return False
        return llm_client.stats.total_tokens >= max_tokens

    # If max_tokens is set, ignore max_rounds (use infinite loop until token limit)
    # If max_tokens is not set, use max_rounds as the limit
    max_rounds_effective = 999999 if max_tokens else config.max_rounds

    # === INITIAL OPERATOR GENERATION (before first ACO round) ===
    if config.initial_operators_per_category > 0:
        import random as rnd
        logger.info(f"\n[ITERATIVE] === Initial Generation: {config.initial_operators_per_category} operator(s) per category ===")
        domain_name = domain_config.name if hasattr(domain_config, "name") else "tsp"

        # Build validation instances
        validation_instances = [
            {**inst.instance_data, "optimal": inst.optimal}
            for inst in instance_pool.instances
        ]

        # Group roles by category
        roles_by_category: dict[str, list[str]] = {
            "construction": [],
            "local_search": [],
            "perturbation": [],
        }
        for role in pool.roles:
            cat = _get_role_category(role)
            if cat in roles_by_category:
                roles_by_category[cat].append(role)

        # Generate for each category (pick random role from category)
        for category in ["construction", "local_search", "perturbation"]:
            if check_token_limit():
                logger.info(f"[ITERATIVE] Token limit reached - stopping initial generation")
                break

            available_roles = roles_by_category.get(category, [])
            if not available_roles:
                continue

            for op_idx in range(config.initial_operators_per_category):
                if check_token_limit():
                    break

                # Pick a random role from this category
                role = rnd.choice(available_roles)

                # Create a "weak spot" for this role to trigger generation
                weak_spot = WeakSpot(
                    role=role,
                    category=category,
                    reason=f"Initial generation for {category}",
                    priority=1.0,
                )

                logger.info(f"[ITERATIVE] Generating initial {category} operator ({role})")

                new_op, error_type = generate_contextual_operator(
                    weak_spot=weak_spot,
                    llm_client=llm_client,
                    domain=domain_name,
                    max_refine_attempts=2,
                    instances=validation_instances,
                    ctx_factory=ctx_factory,
                    pool=pool,
                )

                # Track validation stats
                validation_stats.total_generated += 1
                if new_op:
                    validation_stats.successful += 1
                    pool.add_operator(new_op)
                    logger.info(f"[ITERATIVE] ✓ Added: {new_op.name} [{get_token_str()}]")
                else:
                    # Track specific error type
                    if error_type == "timeout":
                        validation_stats.timeout_errors += 1
                    elif error_type == "syntax":
                        validation_stats.syntax_errors += 1
                    elif error_type == "runtime":
                        validation_stats.runtime_errors += 1
                    elif error_type == "invalid_result":
                        validation_stats.invalid_result_errors += 1
                    logger.info(f"[ITERATIVE] ✗ Generation failed for {role} ({error_type}) [{get_token_str()}]")

        logger.info(f"[ITERATIVE] Initial generation complete. Pool size: {pool.total_operators}")

    for round_num in range(max_rounds_effective):
        # Check token limit at start of each round
        if check_token_limit():
            logger.info(f"[ITERATIVE] Token limit reached ({llm_client.stats.total_tokens:,} >= {max_tokens:,}) - stopping")
            break
        rounds_display = f"{round_num + 1}" if max_tokens else f"{round_num + 1}/{config.max_rounds}"
        logger.info(f"\n[ITERATIVE] === Round {rounds_display} === [{get_token_str()}]")
        logger.info(f"[ITERATIVE] Pool size: {pool.total_operators} operators")

        # Create synthesis hook with current pool (updates each round as operators are added)
        synthesis_hook = L0SynthesisHook(pool)

        # Register LLaMEA wrapper if in hybrid mode
        if llamea_wrapper is not None:
            from src.geakg.layers.l1.hook import L0OperatorRecord
            record = L0OperatorRecord(
                operator_id=llamea_wrapper.name,
                role=llamea_wrapper.role,
                code="# LLaMEA wrapper - code loaded dynamically",
                compiled_fn=llamea_wrapper,  # The wrapper IS the callable
            )
            synthesis_hook.registry.register(record)
            logger.info(f"[ITERATIVE] Registered LLaMEA wrapper: {llamea_wrapper.name}")

        synthesis_hook.register_operators_to_bindings(bindings)

        # Instantiate graph with current bindings (operators may have changed)
        instantiated = InstantiatedGraph(meta_graph, bindings)
        selector = MetaACOSelector(instantiated, aco_config, synthesis_hook=synthesis_hook)

        # Transfer pheromones from previous round (if any)
        if prev_role_pheromones or prev_operator_pheromones:
            selector.transfer_pheromones_from(prev_role_pheromones, prev_operator_pheromones)
            logger.info("[ITERATIVE] Transferred pheromones from previous round")

        # Transfer improvement history from previous rounds (for pruning)
        selector._operator_improvement_history = global_improvement_history.copy()

        # Run ACO training - SIMPLE VERSION
        # Each iteration: build path, evaluate on ALL instances, track avg_gap
        import time

        start = time.time()
        iteration = 0
        round_best_avg_gap = float("inf")
        round_best_operator_path: list[str] = []
        round_best_role_path: list[str] = []
        stagnation = 0

        # Import base operator names once (used for highlighting generated operators)
        from src.geakg.layers.l1.base_operators import ALL_ROLES
        base_op_names = {f"{role}_base" for role in ALL_ROLES}

        last_log_time = start
        log_interval = 10.0

        # Accumulate operator deltas during this round for fitness tracking
        round_operator_deltas: dict[str, list[float]] = {}

        def evaluate_on_all_instances(op_path: list[str]) -> tuple[float, dict[str, float], dict[str, list[float]]]:
            """Evaluate path on all instances, return (avg_gap, per_instance_gaps, operator_deltas)."""
            gaps = {}
            all_deltas: dict[str, list[float]] = {}
            for inst in instance_pool.instances:
                fitness, op_deltas = evaluate_operator_path_with_stats(
                    op_path, inst.instance_data, domain_config, synthesis_hook
                )
                # Merge deltas from this instance
                for op, deltas in op_deltas.items():
                    if op not in all_deltas:
                        all_deltas[op] = []
                    all_deltas[op].extend(deltas)
                # Calculate gap
                if inst.optimal and inst.optimal > 0:
                    gap = 100 * (fitness - inst.optimal) / inst.optimal
                else:
                    gap = fitness / inst.dimension
                gaps[inst.instance_id] = gap
            avg_gap = sum(gaps.values()) / len(gaps) if gaps else float("inf")
            return avg_gap, gaps, all_deltas

        while time.time() - start < config.aco_timeout:
            iteration += 1
            global_iteration += 1
            elapsed = time.time() - start

            # Construct solutions
            iteration_best_avg_gap = float("inf")
            iteration_best_ant = None

            for _ in range(aco_config.n_ants):
                # Use first instance dimension for path construction
                ant = selector.construct_solution(problem_size=instance_pool.instances[0].dimension)
                if not ant.operator_path:
                    continue

                # Evaluate on ALL instances
                avg_gap, per_instance, op_deltas = evaluate_on_all_instances(ant.operator_path)
                ant.gap = avg_gap

                # Accumulate operator deltas for this round
                for op, deltas in op_deltas.items():
                    if op not in round_operator_deltas:
                        round_operator_deltas[op] = []
                    round_operator_deltas[op].extend(deltas)

                    # Register each delta for pruning history
                    # delta > 0 means improvement (cost decreased)
                    for delta in deltas:
                        selector.update_operator_improvement(op, improved=(delta > 0))

                if avg_gap < iteration_best_avg_gap:
                    iteration_best_avg_gap = avg_gap
                    iteration_best_ant = ant

            # Update context for conditions
            context = ExecutionContext(
                generations_without_improvement=stagnation,
                population_diversity=0.5,  # Simplified
                current_fitness=iteration_best_avg_gap,
                best_fitness=round_best_avg_gap,
            )
            selector.set_execution_context(context)

            # Check if this iteration found a better path
            if iteration_best_ant and iteration_best_avg_gap < round_best_avg_gap:
                # Re-evaluate to get per-instance gaps for logging
                _, per_instance, _ = evaluate_on_all_instances(iteration_best_ant.operator_path)

                old_avg = round_best_avg_gap
                round_best_avg_gap = iteration_best_avg_gap
                round_best_operator_path = iteration_best_ant.operator_path.copy()
                round_best_role_path = iteration_best_ant.role_path.copy()
                stagnation = 0

                selector.record_successful_path(round_best_role_path, round_best_operator_path)

                # Log improvement
                new_ops = [op for op in round_best_operator_path if op not in base_op_names]
                gaps_str = " | ".join(f"{k}:{v:.1f}%" for k, v in sorted(per_instance.items()))

                if old_avg == float("inf"):
                    msg = f"[ITERATIVE] [{elapsed:.1f}s] avg_gap={round_best_avg_gap:.2f}% ({gaps_str})"
                else:
                    msg = f"[ITERATIVE] [{elapsed:.1f}s] avg_gap: {old_avg:.2f}% → {round_best_avg_gap:.2f}% ({gaps_str})"

                if new_ops:
                    logger.info(f"\033[95m{msg} ★ Uses: {new_ops}\033[0m")
                else:
                    logger.info(msg)
            else:
                stagnation += 1

            # Update pheromones
            if iteration_best_ant:
                selector.update_pheromones_for_path(
                    iteration_best_ant.role_path,
                    iteration_best_avg_gap,
                    operator_path=iteration_best_ant.operator_path,
                )
                selector.update_operator_pheromones(iteration_best_ant, iteration_best_avg_gap)

            # Check for operator pruning (remove underperforming operators)
            pruned = selector.check_and_prune_operators(global_iteration)
            if pruned:
                # Remove from pool permanently
                for op_name in pruned:
                    pool.remove_operator(op_name)
                logger.info(f"[ITERATIVE] Pruned {len(pruned)} operators: {pruned}")

            # Periodic progress log
            now = time.time()
            if now - last_log_time >= log_interval:
                elapsed = now - start
                avg_str = f"{round_best_avg_gap:.2f}%" if round_best_avg_gap < float("inf") else "N/A"
                logger.info(f"[ITERATIVE] [{elapsed:.1f}s] iter={iteration}, avg_gap={avg_str}, stag={stagnation}")
                last_log_time = now

        # Get final per-instance gaps for this round
        if round_best_operator_path:
            _, final_gaps, _ = evaluate_on_all_instances(round_best_operator_path)
            gaps_str = ", ".join(f"{k}: {v:.2f}%" for k, v in sorted(final_gaps.items()))
        else:
            gaps_str = "N/A"

        # Propagate operator deltas to pool as fitness scores
        # Store average delta (negative = good, reduces cost)
        fitness_updates = []
        for op_name, deltas in round_operator_deltas.items():
            operator = pool.get_operator_by_name(op_name)
            if operator and deltas:
                avg_delta = sum(deltas) / len(deltas)
                operator.fitness_scores.append(avg_delta)
                sign = "-" if avg_delta < 0 else "+"
                fitness_updates.append(f"{op_name}: {sign}{abs(avg_delta):.1f}")

        logger.info(f"[ITERATIVE] Round {round_num + 1} complete: avg_gap={round_best_avg_gap:.2f}%")
        if fitness_updates:
            logger.info(f"[ITERATIVE]   Operator deltas: {', '.join(fitness_updates)}")
        logger.info(f"[ITERATIVE]   Per-instance: {gaps_str}")

        # Save state for transfer to next round
        prev_role_pheromones = selector.pheromones.copy()
        prev_operator_pheromones = selector.get_operator_pheromones()
        global_improvement_history = selector._operator_improvement_history.copy()
        # Note: global_synth_operator_iteration is already updated in-place above

        # Visualize best path for this round
        if output_dir and round_best_role_path:
            from src.geakg.visualization import visualize_best_path

            viz_path = Path(output_dir) / f"best_path_round_{round_num + 1}.png"
            visualize_best_path(
                role_path=round_best_role_path,
                operator_path=round_best_operator_path,
                output_path=viz_path,
                title=f"Best Path - Round {round_num + 1}",
                gap=round_best_avg_gap,
                meta_graph=meta_graph,
            )
            logger.info(f"\033[96m[ITERATIVE] 📊 Path visualization: {viz_path.absolute()}\033[0m")

            # Generate best_program for this round
            program_path = Path(output_dir) / f"best_program_round_{round_num + 1}.py"
            generate_best_program(
                pool=pool,
                role_path=round_best_role_path,
                operator_path=round_best_operator_path,
                output_path=program_path,
                gap=round_best_avg_gap,
                round_num=round_num + 1,
                domain=domain_config.name if hasattr(domain_config, "name") else "tsp",
            )

        # Analyze snapshot
        analysis = analyze_snapshot(
            selector, pool, min_effective_pheromone=config.min_effective_pheromone
        )

        history.append({
            "round": round_num + 1,
            "gap": round_best_avg_gap,
            "weak_spots": len(analysis.weak_spots),
            "role_diversity": analysis.role_diversity.copy(),
            # Feromonas de este round (para análisis de evolución)
            "pheromones": {
                "role_level": {
                    f"{src}->{tgt}": tau for (src, tgt), tau in selector.pheromones.items()
                },
                "operator_level": {
                    f"{r}:{op}": tau for (r, op), tau in selector.get_operator_pheromones().items()
                },
            },
        })

        # Update best gap
        if round_best_avg_gap < best_avg_gap:
            best_avg_gap = round_best_avg_gap

        # Check termination conditions (only early-stop if no weak spots)
        if not analysis.has_weak_spots():
            logger.info("[ITERATIVE] No weak spots found - stopping")
            break

        # Skip operator generation on last round (they won't be tested)
        # Only applies when using max_rounds (not token-limited mode)
        if not max_tokens and round_num == config.max_rounds - 1:
            logger.info("[ITERATIVE] Last round - skipping operator generation (no ACO round to test them)")
            continue

        # Generate operators for weak spots
        weak_spots = analysis.get_top_weak_spots(config.weak_spots_per_round)
        logger.info(f"[ITERATIVE] Addressing {len(weak_spots)} weak spots")

        domain_name = domain_config.name if hasattr(domain_config, "name") else "tsp"

        for spot in weak_spots:
            logger.info(f"[ITERATIVE] Generating for {spot.role}: {spot.reason}")

            # Build instances list for validation
            validation_instances = [
                {**inst.instance_data, "optimal": inst.optimal}
                for inst in instance_pool.instances
            ]

            # Generate with AFO + Design-Space + validation/refinement
            new_op, error_type = generate_contextual_operator(
                weak_spot=spot,
                llm_client=llm_client,
                domain=domain_name,
                max_refine_attempts=2,
                instances=validation_instances,
                ctx_factory=ctx_factory,
                pool=pool,
            )

            # Track validation stats
            validation_stats.total_generated += 1
            if new_op:
                validation_stats.successful += 1
                pool.add_operator(new_op)
                logger.info(f"[ITERATIVE] ✓ Added: {new_op.name} [{get_token_str()}]")
            else:
                # Track specific error type
                if error_type == "timeout":
                    validation_stats.timeout_errors += 1
                elif error_type == "syntax":
                    validation_stats.syntax_errors += 1
                elif error_type == "runtime":
                    validation_stats.runtime_errors += 1
                elif error_type == "invalid_result":
                    validation_stats.invalid_result_errors += 1
                logger.info(f"[ITERATIVE] ✗ Generation failed for {spot.role} ({error_type}) [{get_token_str()}]")

            # Check token limit after each operator generation
            if check_token_limit():
                logger.info(f"[ITERATIVE] Token limit reached ({llm_client.stats.total_tokens:,} >= {max_tokens:,}) - stopping operator generation")
                break

    # Build final snapshot
    # Handle case where no ACO rounds were executed (selector is None)
    if selector is not None:
        pheromones = {
            "role_level": {
                f"{src}->{tgt}": tau for (src, tgt), tau in selector.pheromones.items()
            },
            "operator_level": {
                f"{r}:{op}": tau for (r, op), tau in selector.get_operator_pheromones().items()
            },
        }
        operator_stats = selector.get_operator_stats()
        successful_paths = selector.get_successful_paths()
        best_path = {
            "roles": selector.best_role_path,
            "operators": selector.best_operator_path,
        }
    else:
        pheromones = {"role_level": {}, "operator_level": {}}
        operator_stats = {}
        successful_paths = []
        best_path = {"roles": [], "operators": []}

    final_snapshot = {
        "domain": domain_config.name if hasattr(domain_config, "name") else "tsp",
        "best_gap": best_avg_gap,
        "total_operators": pool.total_operators,
        "refinement_rounds": len(history),
        "history": history,
        # L1: Role transition pheromones (transferable structure)
        "pheromones": pheromones,
        # L2: Metagraph topology (edges and weights)
        "metagraph": {
            "name": meta_graph.name,
            "edges": [
                {
                    "source": src,
                    "target": tgt,
                    "weight": edge.weight,
                    "conditions": [
                        {"type": c.condition_type.value, "threshold": c.threshold}
                        for c in edge.conditions
                    ] if edge.conditions else [],
                }
                for (src, tgt), edge in meta_graph.edges.items()
            ],
            "reasoning": metagraph_reasoning,
        },
        # Operator statistics and successful paths
        "operator_stats": operator_stats,
        "successful_paths": successful_paths,
        # Best path info
        "best_path": best_path,
        # Token usage
        "token_usage": {
            "total": llm_client.get_stats().total_tokens,
            "input": llm_client.get_stats().prompt_tokens,
            "output": llm_client.get_stats().completion_tokens,
        },
        # Validation statistics (errors during operator generation)
        "validation_stats": validation_stats.to_dict(),
    }

    # Generate final best_program.py
    if output_dir and selector is not None and selector.best_operator_path:
        final_program_path = Path(output_dir) / "best_program.py"
        generate_best_program(
            pool=pool,
            role_path=selector.best_role_path,
            operator_path=selector.best_operator_path,
            output_path=final_program_path,
            gap=best_avg_gap,
            round_num=None,  # Final program, no round number
            domain=domain_config.name if hasattr(domain_config, "name") else "tsp",
        )

    logger.info(f"\n[ITERATIVE] Complete: {pool.total_operators} operators, best gap {best_avg_gap:.2f}% [{get_token_str()}]")

    # Log validation statistics
    if validation_stats.total_generated > 0:
        success_rate = 100 * validation_stats.successful / validation_stats.total_generated
        logger.info(f"[ITERATIVE] Validation stats: {validation_stats.successful}/{validation_stats.total_generated} successful ({success_rate:.1f}%)")
        if validation_stats.timeout_errors > 0:
            logger.info(f"[ITERATIVE]   - Timeouts (infinite loops): {validation_stats.timeout_errors}")
        if validation_stats.syntax_errors > 0:
            logger.info(f"[ITERATIVE]   - Syntax errors: {validation_stats.syntax_errors}")
        if validation_stats.runtime_errors > 0:
            logger.info(f"[ITERATIVE]   - Runtime errors: {validation_stats.runtime_errors}")
        if validation_stats.invalid_result_errors > 0:
            logger.info(f"[ITERATIVE]   - Invalid results: {validation_stats.invalid_result_errors}")

    # Save LLaMEA code to pool if hybrid mode was used
    if llamea_wrapper is not None:
        code = llamea_wrapper.get_code_for_pool()
        if code:
            from src.geakg.layers.l1.pool import Operator

            # Remove ALL operators for the LLaMEA role (LLaMEA is the only operator for this role)
            # This prevents slow base operators and LLM-generated operators from competing
            role = llamea_wrapper.role
            existing_ops = pool.get_operators_for_role(role)
            removed_names = [op.name for op in existing_ops]
            for op_name in removed_names:
                pool.remove_operator(op_name)
            if removed_names:
                logger.info(f"[ITERATIVE] Removed {len(removed_names)} operators from {role}: {removed_names}")

            # Add LLaMEA operator as the only operator for this role
            llamea_op = Operator(
                name=llamea_wrapper.name,
                code=code,
                role=llamea_wrapper.role,
                design_choices={"type": "llamea_evolved", "model": llamea_wrapper.model},
                interaction_effects="LLaMEA-evolved operator",
            )
            pool.add_operator(llamea_op)
            logger.info(f"[ITERATIVE] Saved LLaMEA code to pool: {llamea_wrapper.name} (only operator for {role})")

            # Clean pheromones: remove entries for operators no longer in pool
            # and set LLaMEA operator pheromone to max
            if "operator_level" in final_snapshot.get("pheromones", {}):
                op_pheromones = final_snapshot["pheromones"]["operator_level"]

                # Get all valid operator names from pool
                valid_ops = set()
                for r, ops in pool.operators_by_role.items():
                    for op in ops:
                        valid_ops.add(f"{r}:{op.name}")

                # Remove pheromones for operators not in pool
                keys_to_remove = [k for k in op_pheromones.keys() if k not in valid_ops]
                for k in keys_to_remove:
                    del op_pheromones[k]
                if keys_to_remove:
                    logger.info(f"[ITERATIVE] Cleaned {len(keys_to_remove)} stale pheromone entries")

                # Set LLaMEA operator pheromone to max (1.0)
                llamea_key = f"{role}:{llamea_wrapper.name}"
                op_pheromones[llamea_key] = 1.0
                logger.info(f"[ITERATIVE] Set pheromone for {llamea_key} = 1.0")

    return pool, final_snapshot


def save_snapshot_for_transfer(snapshot: dict, path: str) -> None:
    """Save snapshot in format compatible with transfer modules.

    Args:
        snapshot: Snapshot data from iterative refinement
        path: Output file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    logger.info(f"[ITERATIVE] Saved snapshot to {path}")


def load_snapshot_for_transfer(path: str) -> dict:
    """Load a snapshot for transfer learning.

    Args:
        path: Path to snapshot file

    Returns:
        Snapshot dictionary
    """
    with open(path) as f:
        return json.load(f)
