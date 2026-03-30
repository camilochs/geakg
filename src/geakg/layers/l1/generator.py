"""L1 Generator: Offline operator pool generation with AFO principle.

This module implements the L1 (operator generation) phase that generates a pool of
validated operators before runtime. Key features:

1. AFO (Always-From-Original): Variants are generated from base A₀, not iteratively
2. Design-Space Prompting: Orthogonal axes guide diverse generation
3. Token Budget Control: Number of variants depends on available tokens
4. Mini F-Race: Statistical selection eliminates weak candidates

Architecture:
    L0: MetaGraph Topology (offline, LLM) - src/geakg/layers/l0/
    L1: Operator Generation (offline, LLM) - src/geakg/layers/l1/  <-- THIS FILE
    L2: ACO Training (offline, ACO) - src/geakg/layers/l2/
    Online: Symbolic Executor - src/geakg/online/
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from src.geakg.layers.l1.base_operators import ALL_ROLES, BASE_OPERATORS, get_role_category
from src.geakg.layers.l1.design_space import (
    design_point_to_key,
    format_design_point,
    sample_design_point_for_category,
)
from src.geakg.layers.l1.pool import Operator, OperatorPool
from src.geakg.layers.l1.prompts import DESIGN_SPACE_RESPONSE_SCHEMA, build_design_space_prompt
from src.geakg.layers.l1.racing import mini_frace, rank_operators

if TYPE_CHECKING:
    from src.llm.client import OllamaClient, OpenAIClient


# Estimated tokens per variant (prompt + response)
TOKENS_PER_VARIANT = 1500


@dataclass
class L1Config:
    """Configuration for L1 operator generation."""

    max_tokens: int = 100_000
    """Maximum total tokens to spend on generation."""

    pool_size_per_role: int = 5
    """How many operators to keep per role after racing."""

    racing_instances: int = 10
    """Number of instances to use for F-Race selection."""

    significance: float = 0.05
    """p-value threshold for F-Race elimination."""

    temperature: float = 0.7
    """LLM temperature for generation (higher = more creative)."""

    max_refine_attempts: int = 2
    """Maximum refinement attempts per variant."""

    seed: int = 42
    """Random seed for reproducibility."""

    roles: list[str] = field(default_factory=lambda: ALL_ROLES.copy())
    """Which roles to generate operators for."""


class L1Generator:
    """Offline operator pool generator using AFO + Design-Space Prompting.

    Usage:
        generator = L1Generator(llm_client, domain="tsp")
        pool = generator.generate(instances, max_tokens=50000)
        pool.save("pools/tsp_pool.json")
    """

    def __init__(
        self,
        llm_client: "OllamaClient | OpenAIClient",
        domain: str,
        config: L1Config | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            llm_client: LLM client for generation
            domain: Target domain (e.g., "tsp", "vrp")
            config: Generation configuration
        """
        self.llm_client = llm_client
        self.domain = domain
        self.config = config or L1Config()
        self.rng = random.Random(self.config.seed)
        self.tokens_used = 0
        self.used_combinations: set[tuple] = set()

    def generate(
        self,
        instances: list[Any],
        evaluate_fn: Callable[[Operator, Any], float],
        ctx_factory: Callable[[Any], Any] | None = None,
    ) -> OperatorPool:
        """Generate operator pool using AFO + Mini F-Race.

        Args:
            instances: Problem instances for validation and racing
            evaluate_fn: Function to evaluate operator fitness
            ctx_factory: Optional factory for creating domain context

        Returns:
            OperatorPool with selected operators per role
        """
        logger.info(
            f"[L1] Starting generation for {self.domain} with {self.config.max_tokens} token budget"
        )

        # Calculate variants per role based on budget
        n_roles = len(self.config.roles)
        tokens_per_role = self.config.max_tokens // n_roles
        variants_per_role = max(1, tokens_per_role // TOKENS_PER_VARIANT)

        logger.info(
            f"[L1] Generating ~{variants_per_role} variants per role ({n_roles} roles)"
        )

        # Phase 1: Generate variants from A₀ (AFO principle)
        raw_pool: dict[str, list[Operator]] = {}

        for role in self.config.roles:
            logger.info(f"[L1] Generating variants for {role}")
            candidates = self._generate_role_variants(
                role=role,
                max_variants=variants_per_role,
                instances=instances[:3],  # Quick validation on subset
                ctx_factory=ctx_factory,
            )
            raw_pool[role] = candidates
            logger.info(f"[L1] {role}: {len(candidates)} valid candidates")

        # Phase 2: Mini F-Race to select best per role
        logger.info("[L1] Starting F-Race selection")
        final_pool = OperatorPool(
            metadata={
                "domain": self.domain,
                "tokens_used": self.tokens_used,
                "config": {
                    "max_tokens": self.config.max_tokens,
                    "pool_size_per_role": self.config.pool_size_per_role,
                    "racing_instances": self.config.racing_instances,
                },
            }
        )

        racing_instances = instances[: self.config.racing_instances]

        for role, candidates in raw_pool.items():
            if len(candidates) <= self.config.pool_size_per_role:
                # No racing needed
                survivors = candidates
            elif len(racing_instances) >= 3:
                # Use F-Race
                survivors = mini_frace(
                    candidates=candidates,
                    instances=racing_instances,
                    evaluate_fn=evaluate_fn,
                    significance=self.config.significance,
                    min_survivors=self.config.pool_size_per_role,
                )
            else:
                # Simple ranking
                survivors = rank_operators(candidates, racing_instances, evaluate_fn)

            # Keep top N
            for op in survivors[: self.config.pool_size_per_role]:
                final_pool.add_operator(op)

        logger.info(
            f"[L1] Complete: {final_pool.total_operators} operators, "
            f"{self.tokens_used} tokens used"
        )

        return final_pool

    def _generate_role_variants(
        self,
        role: str,
        max_variants: int,
        instances: list[Any],
        ctx_factory: Callable[[Any], Any] | None = None,
    ) -> list[Operator]:
        """Generate variants for a single role using AFO.

        Args:
            role: Target role name
            max_variants: Maximum variants to generate
            instances: Instances for quick validation
            ctx_factory: Context factory for validation

        Returns:
            List of valid operators (including base A₀)
        """
        category = get_role_category(role)
        base_code = BASE_OPERATORS[role]

        # Start with base operator (A₀)
        base_op = Operator(
            name=f"{role}_base",
            code=base_code.strip(),
            role=role,
            design_choices={},
            interaction_effects="Base operator (A₀)",
        )
        candidates = [base_op]

        for i in range(max_variants):
            if self.tokens_used >= self.config.max_tokens:
                logger.info("[L1] Token budget exhausted")
                break

            # Sample unique design point
            design_point = self._sample_unique_design_point(category)
            if design_point is None:
                logger.debug("[L1] All design combinations exhausted")
                break

            # Generate variant
            variant = self._generate_variant(role, category, base_code, design_point)

            if variant is None:
                continue

            # Quick validation
            if self._validate_operator(variant, instances, ctx_factory):
                candidates.append(variant)
                logger.info(f"[L1]   ✓ Created: {variant.name}")
                logger.info(f"[L1]   Code:\n{variant.code}")
            else:
                # Try refinement
                refined = self._refine_variant(variant, instances, ctx_factory)
                if refined:
                    candidates.append(refined)
                    logger.info(f"[L1]   ✓ Created (refined): {refined.name}")
                    logger.info(f"[L1]   Code:\n{refined.code}")

        return candidates

    def _sample_unique_design_point(self, category: str) -> dict[str, str] | None:
        """Sample a design point not yet used.

        Args:
            category: Operator category for constraints

        Returns:
            Unique design point or None if exhausted
        """
        max_attempts = 50
        for _ in range(max_attempts):
            point = sample_design_point_for_category(self.rng, category)
            key = design_point_to_key(point)
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
        """Generate a single variant using Design-Space Prompting.

        Args:
            role: Target role
            category: Operator category
            base_code: Base operator code (A₀)
            design_point: Selected design choices

        Returns:
            Generated operator or None if failed
        """
        prompt = build_design_space_prompt(
            role=role,
            role_category=category,
            original_code=base_code,
            design_point=design_point,
            domain=self.domain,
        )

        try:
            response = self.llm_client.query(
                prompt=prompt,
                temperature=self.config.temperature,
                json_schema=DESIGN_SPACE_RESPONSE_SCHEMA,
            )
            self.tokens_used += TOKENS_PER_VARIANT

            # Parse response
            parsed = self._parse_response(response.content)
            if parsed is None:
                return None

            return Operator(
                name=parsed.get("name", f"{category}_variant"),
                code=self._clean_code(parsed.get("code", "")),
                role=role,
                design_choices=parsed.get("design_choices", design_point),
                interaction_effects=parsed.get("structural_changes", parsed.get("interaction_effects", "")),
            )

        except Exception as e:
            logger.warning(f"[L1] Generation failed: {e}")
            self.tokens_used += TOKENS_PER_VARIANT // 2  # Partial cost
            return None

    def _parse_response(self, content: str) -> dict | None:
        """Parse LLM response, handling various formats.

        Args:
            content: Raw LLM response

        Returns:
            Parsed dictionary or None
        """
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
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
                            return json.loads(content[start : i + 1])
                        except json.JSONDecodeError:
                            break

        # Extract code separately
        code_match = re.search(r"```python\s*([\s\S]*?)\s*```", content)
        if code_match:
            return {
                "name": "extracted_operator",
                "code": code_match.group(1),
                "design_choices": {},
                "interaction_effects": "Extracted from non-JSON response",
            }

        return None

    def _clean_code(self, code: str) -> str:
        """Clean up generated code.

        Args:
            code: Raw code string

        Returns:
            Cleaned code
        """
        # Remove markdown formatting
        code = re.sub(r"```python\s*", "", code)
        code = re.sub(r"```\s*", "", code)
        return code.strip()

    def _validate_operator(
        self,
        operator: Operator,
        instances: list[Any],
        ctx_factory: Callable[[Any], Any] | None = None,
    ) -> bool:
        """Validate that operator compiles and runs without errors.

        Args:
            operator: Operator to validate
            instances: Test instances
            ctx_factory: Context factory

        Returns:
            True if valid
        """
        # Check compilation
        try:
            compiled = compile(operator.code, "<string>", "exec")
            namespace: dict = {}
            exec(compiled, namespace)

            # Find the function
            func_name = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    func_name = name
                    break

            if func_name is None:
                return False

            func = namespace[func_name]

        except Exception as e:
            logger.debug(f"[L1] Compilation failed: {e}")
            return False

        # Test execution on instances
        if not instances or ctx_factory is None:
            return True

        try:
            instance = instances[0]
            ctx = ctx_factory(instance)

            # Create test solution
            n = instance.get("dimension", 10) if isinstance(instance, dict) else 10
            test_solution = list(range(n))

            # Execute
            result = func(test_solution, ctx)

            # Basic validation
            if not isinstance(result, list):
                return False
            if len(result) != n:
                return False
            if set(result) != set(range(n)):
                return False

            return True

        except Exception as e:
            logger.debug(f"[L1] Execution failed: {e}")
            return False

    def _refine_variant(
        self,
        operator: Operator,
        instances: list[Any],
        ctx_factory: Callable[[Any], Any] | None = None,
    ) -> Operator | None:
        """Attempt to fix errors in a variant.

        Args:
            operator: Operator with errors
            instances: Test instances
            ctx_factory: Context factory

        Returns:
            Refined operator or None
        """
        from src.geakg.layers.l1.prompts import build_refinement_prompt

        for attempt in range(self.config.max_refine_attempts):
            # Collect errors
            errors = self._collect_errors(operator, instances, ctx_factory)
            if not errors:
                return operator

            prompt = build_refinement_prompt(
                code=operator.code,
                errors=errors,
                design_point=operator.design_choices,
            )

            try:
                response = self.llm_client.query(
                    prompt=prompt,
                    temperature=0.3,  # Lower temperature for fixes
                )
                self.tokens_used += TOKENS_PER_VARIANT // 2

                parsed = self._parse_response(response.content)
                if parsed and "code" in parsed:
                    operator.code = self._clean_code(parsed["code"])

                    if self._validate_operator(operator, instances, ctx_factory):
                        return operator

            except Exception as e:
                logger.debug(f"[L1] Refinement failed: {e}")

        return None

    def _collect_errors(
        self,
        operator: Operator,
        instances: list[Any],
        ctx_factory: Callable[[Any], Any] | None = None,
    ) -> list[str]:
        """Collect error messages from validation.

        Args:
            operator: Operator to test
            instances: Test instances
            ctx_factory: Context factory

        Returns:
            List of error messages
        """
        errors = []

        # Check compilation
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

        # Find function
        func = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("_"):
                func = obj
                break

        if func is None:
            errors.append("No function definition found")
            return errors

        # Test execution
        if instances and ctx_factory:
            try:
                instance = instances[0]
                ctx = ctx_factory(instance)
                n = instance.get("dimension", 10) if isinstance(instance, dict) else 10
                test_solution = list(range(n))
                result = func(test_solution, ctx)

                if not isinstance(result, list):
                    errors.append(f"Return type is {type(result).__name__}, expected list")
                elif len(result) != n:
                    errors.append(f"Returned list length {len(result)}, expected {n}")
                elif set(result) != set(range(n)):
                    errors.append("Result is not a valid permutation of [0..n-1]")

            except IndexError as e:
                errors.append(f"IndexError: {e} - check array bounds")
            except Exception as e:
                errors.append(f"Runtime error: {type(e).__name__}: {e}")

        return errors


# Backward compatibility aliases
L0Config = L1Config
L0Generator = L1Generator
