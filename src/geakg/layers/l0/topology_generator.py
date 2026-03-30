"""L0 Topology Generator: Generate MetaGraph structure from L1 operator pool.

This generator receives an L1 operator pool (with operators already assigned to roles)
and asks the LLM to generate the topology (transitions) between roles.

Architecture:
    L0: MetaGraph Topology (offline, LLM) - src/geakg/layers/l0/  <-- THIS FILE
    L1: Operator Generation (offline, LLM) - src/geakg/layers/l1/
    L2: ACO Training (offline, ACO) - src/geakg/layers/l2/
    Online: Symbolic Executor - src/geakg/online/
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.geakg.layers.l1.pool import OperatorPool
from src.geakg.layers.l0.metagraph import MetaGraph, MetaEdge
from src.geakg.layers.l0.roles import AbstractRole
from src.geakg.layers.l0.conditions import EdgeCondition, ConditionType

if TYPE_CHECKING:
    from src.llm.client import OllamaClient, OpenAIClient
    from src.geakg.core.role_schema import RoleSchema


# =============================================================================
# PROMPT FOR L0 METAGRAPH GENERATION
# =============================================================================

L0_METAGRAPH_PROMPT = """Design transition edges for {domain} metaheuristic. Connect the EXACT roles listed below.

## AVAILABLE ROLES (use these EXACT names)
{roles_by_category}

## RULES
1. const → ls: Start optimization (weight 0.9)
2. ls → ls: Chain different local searches (weight 0.5-0.7)
3. ls → pert: Escape local optimum (weight 0.5, add stagnation condition)
4. pert → ls: Re-optimize after perturbation (weight 0.9)

FORBIDDEN: const→const, const→pert, pert→const, pert→pert

CRITICAL: Every role must have at least one incoming AND one outgoing edge (except const which only needs outgoing).

## OUTPUT FORMAT
```json
{{
  "name": "{domain}_ils",
  "edges": [
    {{"source": "<const_role>", "target": "<ls_role>", "weight": 0.9}},
    {{"source": "<ls_role>", "target": "<another_ls_role>", "weight": 0.6}},
    {{"source": "<ls_role>", "target": "<pert_role>", "weight": 0.5, "condition": {{"when": "stagnation", "threshold": 3, "boost": 2.0}}}},
    {{"source": "<pert_role>", "target": "<ls_role>", "weight": 0.9}}
  ],
  "reasoning": "brief explanation"
}}
```

Generate edges connecting ALL roles listed above. Use ONLY the exact role names provided."""


# JSON Schema for structured output
L0_METAGRAPH_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "weight": {"type": "number", "minimum": 0.1, "maximum": 0.95},
                    "condition": {
                        "type": "object",
                        "properties": {
                            "when": {"type": "string"},
                            "threshold": {"type": "number"},
                            "boost": {"type": "number"},
                        },
                    },
                },
                "required": ["source", "target", "weight"],
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["name", "edges"],
}


def extract_balanced_json(text: str) -> str | None:
    """Extract a balanced JSON object from text.

    Finds the first '{' and tracks nesting depth, respecting string literals
    and escape sequences. Returns the substring from the opening '{' to its
    matching '}', or None if no balanced object is found.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


class L0MetaGraphGenerator:
    """Generate MetaGraph edges for L0 operator pool.

    Unlike MetaGraphGenerator which generates both roles and edges,
    this generator takes roles as given (from L0 pool) and only generates edges.
    """

    def __init__(
        self,
        llm_client: "OllamaClient | OpenAIClient",
        pool: OperatorPool,
        domain: str = "TSP",
        max_retries: int = 3,
    ) -> None:
        """Initialize generator.

        Args:
            llm_client: LLM client for queries
            pool: L0 operator pool with operators by role
            domain: Problem domain (e.g., "TSP", "VRP")
            max_retries: Maximum generation attempts
        """
        self.llm_client = llm_client
        self.pool = pool
        self.domain = domain.upper()
        self.max_retries = max_retries

    def _format_roles_with_operators(self) -> str:
        """Format pool roles organized by category for prompt."""
        # Group roles by category
        constructions = sorted([r for r in self.pool.roles if r.startswith("const_")])
        local_searches = sorted([r for r in self.pool.roles if r.startswith("ls_")])
        perturbations = sorted([r for r in self.pool.roles if r.startswith("pert_")])

        lines = []

        # Construction roles
        lines.append("### CONSTRUCTION (only outgoing edges)")
        for role in constructions:
            lines.append(f"  - {role}")
        lines.append("")

        # Local search roles
        lines.append("### LOCAL_SEARCH (incoming and outgoing edges)")
        for role in local_searches:
            lines.append(f"  - {role}")
        lines.append("")

        # Perturbation roles
        lines.append("### PERTURBATION (incoming and outgoing edges)")
        for role in perturbations:
            lines.append(f"  - {role}")

        return "\n".join(lines)

    def _get_category(self, role: str) -> str:
        """Get category from role name."""
        if role.startswith("const_"):
            return "construction"
        elif role.startswith("ls_"):
            return "local_search"
        elif role.startswith("pert_"):
            return "perturbation"
        return "unknown"

    def generate(self) -> MetaGraph | None:
        """Generate MetaGraph with edges for L0 pool roles.

        Returns:
            MetaGraph or None if generation failed
        """
        logger.info(f"[L0-MetaGraph] Generating edges for {len(self.pool.roles)} roles")

        roles_desc = self._format_roles_with_operators()
        prompt = L0_METAGRAPH_PROMPT.format(
            domain=self.domain,
            roles_by_category=roles_desc,
        )

        for attempt in range(self.max_retries):
            logger.info(f"[L0-MetaGraph] Attempt {attempt + 1}/{self.max_retries}")

            try:
                response = self.llm_client.query(
                    prompt,
                    temperature=0.0,
                    json_schema=L0_METAGRAPH_JSON_SCHEMA,
                    agent_name="L0MetaGraphGenerator",
                    context={"domain": self.domain},
                )

                logger.debug(f"[L0-MetaGraph] Raw response (first 500 chars): {response.content[:500]}")

                result = self._parse_response(response.content)

                if result is None:
                    logger.warning("[L0-MetaGraph] Failed to parse response")
                    logger.warning(f"[L0-MetaGraph] Response content: {response.content[:1000]}")
                    continue

                mg, warnings = result

                if warnings:
                    logger.warning(f"[L0-MetaGraph] Warnings: {warnings}")

                # Validate minimum structure (at least some edges to start)
                if len(mg.edges) < 3:
                    logger.warning(f"[L0-MetaGraph] Too few edges ({len(mg.edges)}), need at least 3")
                    continue

                # Ensure all roles from pool are in graph
                pool_roles = set(self.pool.roles)
                graph_roles = set(mg.nodes.keys())
                missing = pool_roles - graph_roles

                if missing:
                    logger.warning(f"[L0-MetaGraph] Roles not connected: {missing}")
                    # Add isolated roles
                    for role in missing:
                        try:
                            mg.add_role(AbstractRole(role))
                        except ValueError:
                            logger.warning(f"[L0-MetaGraph] Unknown role: {role}")

                # Validate graph connectivity (all roles must be reachable)
                ils_valid, ils_errors = self._validate_ils_structure(mg)
                if not ils_valid:
                    logger.warning(f"[L0-MetaGraph] Graph structure incomplete: {ils_errors}")
                    # Try to auto-repair missing connections
                    mg = self._repair_missing_connections(mg)
                    ils_valid, ils_errors = self._validate_ils_structure(mg)
                    if not ils_valid:
                        logger.warning(f"[L0-MetaGraph] Auto-repair failed: {ils_errors}")
                        continue  # Retry with LLM

                logger.info(
                    f"[L0-MetaGraph] Generated: {len(mg.nodes)} roles, {len(mg.edges)} edges"
                )
                return mg

            except Exception as e:
                logger.warning(f"[L0-MetaGraph] Error: {e}")
                continue

        logger.error("[L0-MetaGraph] Generation failed after max retries")
        return None

    def _parse_response(self, content: str) -> tuple[MetaGraph, list[str]] | None:
        """Parse LLM response into MetaGraph.

        Returns:
            Tuple of (MetaGraph, warnings) or None if parsing failed
        """
        warnings = []

        # Save for debugging
        os.makedirs("logs", exist_ok=True)
        with open("logs/last_l0_metagraph_response.json", "w") as f:
            f.write(content)

        try:
            # Clean up response - remove markdown code blocks
            cleaned = content.strip()
            # Remove leading ```json or ``` with any whitespace
            cleaned = re.sub(r"^```json\s*\n?", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^```\s*\n?", "", cleaned)
            # Remove trailing ```
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            # Remove ellipsis patterns
            cleaned = re.sub(r",\s*\.\.\.", "", cleaned)
            cleaned = re.sub(r"\.\.\.,?\s*", "", cleaned)

            # Extract JSON
            json_str = self._extract_balanced_json(cleaned)
            if not json_str:
                logger.warning("[L0-MetaGraph] No valid JSON found")
                return None

            data = json.loads(json_str)

            mg = MetaGraph(
                name=data.get("name", f"l0_{self.domain.lower()}_graph"),
                description=data.get("reasoning", ""),
                llm_reasoning=data.get("reasoning", ""),
            )

            # Parse edges
            valid_roles = set(self.pool.roles)

            for edge_data in data.get("edges", []):
                source_str = edge_data.get("source", "")
                target_str = edge_data.get("target", "")
                weight = float(edge_data.get("weight", 0.5))
                cond_data = edge_data.get("condition")

                # Validate roles exist in pool
                if source_str not in valid_roles:
                    warnings.append(f"Unknown source role: {source_str}")
                    continue
                if target_str not in valid_roles:
                    warnings.append(f"Unknown target role: {target_str}")
                    continue

                # Filter forbidden edges
                if source_str.startswith("pert_") and target_str.startswith("const_"):
                    warnings.append(f"Forbidden: {source_str} → {target_str}")
                    continue
                if source_str.startswith("const_") and target_str.startswith("pert_"):
                    warnings.append(f"Forbidden: {source_str} → {target_str}")
                    continue

                # Parse conditions
                conditions = []
                condition_boost = 2.0
                if cond_data and isinstance(cond_data, dict):
                    cond_type_str = cond_data.get("when", "")
                    threshold = float(cond_data.get("threshold", 0))
                    condition_boost = max(1.0, float(cond_data.get("boost", 2.0)))

                    cond_type_map = {
                        "stagnation": ConditionType.STAGNATION,
                        "diversity_low": ConditionType.DIVERSITY_LOW,
                        "gap_to_best": ConditionType.GAP_TO_BEST,
                    }
                    if cond_type_str in cond_type_map:
                        conditions.append(
                            EdgeCondition(
                                condition_type=cond_type_map[cond_type_str],
                                threshold=threshold,
                                reason=f"L0: {cond_type_str} >= {threshold}",
                            )
                        )

                try:
                    source = AbstractRole(source_str)
                    target = AbstractRole(target_str)

                    mg.add_edge(
                        MetaEdge(
                            source=source,
                            target=target,
                            weight=weight,
                            conditions=conditions,
                            condition_boost=condition_boost,
                            reasoning=data.get("reasoning", ""),
                        )
                    )
                except ValueError as e:
                    warnings.append(f"Invalid role: {e}")
                    continue

            return mg, warnings

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[L0-MetaGraph] Parse error: {e}")
            return None

    def _repair_missing_connections(self, mg: MetaGraph) -> MetaGraph:
        """Auto-repair missing connections in the metagraph.

        Adds default edges for isolated roles to ensure graph connectivity.

        Args:
            mg: MetaGraph to repair

        Returns:
            Repaired MetaGraph
        """
        existing_edges = set(mg.edges.keys())

        # Get roles by category
        local_searches = sorted([r for r in self.pool.roles if r.startswith("ls_")])
        perturbations = sorted([r for r in self.pool.roles if r.startswith("pert_")])
        constructions = sorted([r for r in self.pool.roles if r.startswith("const_")])

        if not local_searches:
            return mg  # Nothing to connect to

        # Use first local search as default target
        default_ls = local_searches[0]

        # Fix isolated perturbations
        for pert in perturbations:
            has_incoming = any(
                (AbstractRole(src), AbstractRole(pert)) in existing_edges
                for src in local_searches
            )
            has_outgoing = any(
                (AbstractRole(pert), AbstractRole(tgt)) in existing_edges
                for tgt in local_searches
            )

            # Add missing incoming edge (ls → pert)
            if not has_incoming:
                try:
                    mg.add_edge(MetaEdge(
                        source=AbstractRole(default_ls),
                        target=AbstractRole(pert),
                        weight=0.5,
                        conditions=[EdgeCondition(
                            condition_type=ConditionType.STAGNATION,
                            threshold=3.0,
                            reason="Auto-repair: escape stagnation",
                        )],
                        condition_boost=2.0,
                        reasoning="Auto-repaired missing edge",
                    ))
                    logger.info(f"[L0-MetaGraph] Auto-added: {default_ls} → {pert}")
                except (ValueError, KeyError):
                    pass

            # Add missing outgoing edge (pert → ls)
            if not has_outgoing:
                try:
                    mg.add_edge(MetaEdge(
                        source=AbstractRole(pert),
                        target=AbstractRole(default_ls),
                        weight=0.9,
                        reasoning="Auto-repaired missing edge",
                    ))
                    logger.info(f"[L0-MetaGraph] Auto-added: {pert} → {default_ls}")
                except (ValueError, KeyError):
                    pass

        # Fix constructions without outgoing edges
        for const in constructions:
            has_outgoing = any(
                (AbstractRole(const), AbstractRole(tgt)) in existing_edges
                for tgt in local_searches
            )
            if not has_outgoing:
                try:
                    mg.add_edge(MetaEdge(
                        source=AbstractRole(const),
                        target=AbstractRole(default_ls),
                        weight=0.9,
                        reasoning="Auto-repaired missing edge",
                    ))
                    logger.info(f"[L0-MetaGraph] Auto-added: {const} → {default_ls}")
                except (ValueError, KeyError):
                    pass

        return mg

    def _validate_ils_structure(self, mg: MetaGraph) -> tuple[bool, list[str]]:
        """Validate that metagraph has no isolated components.

        Checks:
        - All perturbation roles are connected (have incoming AND outgoing edges)
        - At least one path from construction to local search

        Args:
            mg: MetaGraph to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        existing_edges = set(mg.edges.keys())

        # Get roles by category
        perturbations = [r for r in self.pool.roles if r.startswith("pert_")]
        local_searches = [r for r in self.pool.roles if r.startswith("ls_")]

        # Check perturbation roles are not isolated
        for pert in perturbations:
            has_incoming = any(
                (AbstractRole(src), AbstractRole(pert)) in existing_edges
                for src in local_searches
            )
            has_outgoing = any(
                (AbstractRole(pert), AbstractRole(tgt)) in existing_edges
                for tgt in local_searches
            )

            if not has_incoming:
                errors.append(f"{pert} has no incoming edges (isolated)")
            if not has_outgoing:
                errors.append(f"{pert} has no outgoing edges (dead end)")

        return len(errors) == 0, errors

    def _extract_balanced_json(self, text: str) -> str | None:
        """Extract balanced JSON object from text."""
        return extract_balanced_json(text)


def create_default_metagraph_for_pool(pool: OperatorPool) -> MetaGraph:
    """Create a default MetaGraph without LLM for testing.

    This creates a standard ILS-style graph connecting all roles from the pool.

    Args:
        pool: L0 operator pool

    Returns:
        MetaGraph with standard ILS edges
    """
    mg = MetaGraph(
        name="default_ils_graph",
        description="Default ILS structure for L0 pool",
    )

    roles = list(pool.roles)

    # Categorize roles
    constructions = [r for r in roles if r.startswith("const_")]
    local_searches = [r for r in roles if r.startswith("ls_")]
    perturbations = [r for r in roles if r.startswith("pert_")]

    # Construction → Local Search
    for const in constructions:
        for ls in local_searches:
            try:
                mg.add_edge(MetaEdge(
                    source=AbstractRole(const),
                    target=AbstractRole(ls),
                    weight=0.90,
                    reasoning="Start optimization after construction",
                ))
            except ValueError:
                continue

    # Local Search → Local Search (chain)
    for i, ls1 in enumerate(local_searches):
        for ls2 in local_searches[i + 1:]:
            try:
                mg.add_edge(MetaEdge(
                    source=AbstractRole(ls1),
                    target=AbstractRole(ls2),
                    weight=0.60,
                    reasoning="Chain complementary LS",
                ))
                mg.add_edge(MetaEdge(
                    source=AbstractRole(ls2),
                    target=AbstractRole(ls1),
                    weight=0.60,
                    reasoning="Chain complementary LS",
                ))
            except ValueError:
                continue

    # Local Search → Perturbation (escape)
    for ls in local_searches:
        for pert in perturbations:
            try:
                mg.add_edge(MetaEdge(
                    source=AbstractRole(ls),
                    target=AbstractRole(pert),
                    weight=0.55,
                    conditions=[EdgeCondition(
                        condition_type=ConditionType.STAGNATION,
                        threshold=3.0,
                        reason="Escape after stagnation",
                    )],
                    condition_boost=2.0,
                    reasoning="Escape local optimum",
                ))
            except ValueError:
                continue

    # Perturbation → Local Search (re-optimize)
    for pert in perturbations:
        for ls in local_searches:
            try:
                mg.add_edge(MetaEdge(
                    source=AbstractRole(pert),
                    target=AbstractRole(ls),
                    weight=0.90,
                    reasoning="Re-optimize after perturbation",
                ))
            except ValueError:
                continue

    return mg


# =============================================================================
# PROMPT FOR L0 NAS METAGRAPH GENERATION
# =============================================================================

L0_NAS_METAGRAPH_PROMPT = """Design transition edges for a Neural Architecture Search meta-graph.
Connect the EXACT roles listed below following the category transition rules.

## AVAILABLE ROLES (use these EXACT names)
{roles_description}

## CATEGORY TRANSITION RULES
- topology → {{activation, topology}} (structure defined, then activations or refine)
- activation → {{activation, training}} (try combos, then training)
- training → {{training, regularization}} (combine training choices, then regularize)
- regularization → {{regularization, evaluation}} (combine regularization, then evaluate)
- evaluation → {{topology, training, activation}} (feedback loops for redesign)

## RULES
1. topology → activation: After defining structure, choose activations (weight 0.7-0.9)
2. topology → topology: Refine topology (weight 0.3-0.5)
3. activation → activation: Experiment with combos (weight 0.4-0.6)
4. activation → training: Move to training config (weight 0.7-0.8)
5. training → training: Combine optimizer + schedule + augmentation (weight 0.5-0.8)
6. training → regularization: Add regularization (weight 0.6-0.8)
7. regularization → regularization: Combine regularizations (weight 0.5-0.7)
8. regularization → evaluation: Ready to evaluate (weight 0.7-0.9)
9. evaluation → topology/training/activation: Feedback loops (weight 0.3-0.5)
   - evaluation → topology may use stagnation condition

FORBIDDEN: evaluation → evaluation (no self-loops in eval), topology → training (skip activation), topology → regularization (skip)

CRITICAL: Every role must have at least 1 incoming AND 1 outgoing edge.
Exception: topology roles only need outgoing edges (they are entry points).

## OUTPUT FORMAT
```json
{{
  "name": "nas_llm_generated",
  "edges": [
    {{"source": "<role>", "target": "<role>", "weight": 0.8}},
    {{"source": "<role>", "target": "<role>", "weight": 0.5, "condition": {{"when": "stagnation", "threshold": 3, "boost": 2.0}}}}
  ],
  "reasoning": "brief explanation of design choices"
}}
```

Generate edges connecting ALL roles listed above. Use ONLY the exact role names provided."""


L0_NAS_METAGRAPH_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "weight": {"type": "number", "minimum": 0.1, "maximum": 0.95},
                    "condition": {
                        "type": "object",
                        "properties": {
                            "when": {"type": "string"},
                            "threshold": {"type": "number"},
                            "boost": {"type": "number"},
                        },
                    },
                },
                "required": ["source", "target", "weight"],
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["name", "edges"],
}


class L0NASMetaGraphGenerator:
    """Generate MetaGraph topology for NAS domain via LLM.

    Uses NASRoleSchema (18 roles, 5 categories) and category transition rules
    to prompt the LLM for a valid NAS meta-graph structure.

    This makes Case Study 1 (NAS) consistent with Case Study 2 (TSP):
    both use LLM to generate L0 topology, demonstrating that GEAKG is
    truly generative.
    """

    def __init__(
        self,
        llm_client: "OllamaClient | OpenAIClient",
        schema: "RoleSchema",
        max_retries: int = 3,
    ) -> None:
        self.llm_client = llm_client
        self.schema = schema
        self.max_retries = max_retries
        self.tokens_used: int = 0
        self.generation_time_s: float = 0.0
        self.reasoning: str = ""

    def generate(self) -> MetaGraph | None:
        """Generate a NAS MetaGraph with edges designed by the LLM.

        Returns:
            MetaGraph or None if generation failed after max retries.
        """
        import time as _time

        all_roles = self.schema.get_all_roles()
        logger.info(f"[L0-NAS-MetaGraph] Generating edges for {len(all_roles)} roles")

        roles_desc = self.schema.get_role_description_for_llm()
        prompt = L0_NAS_METAGRAPH_PROMPT.format(roles_description=roles_desc)

        t0 = _time.time()

        for attempt in range(self.max_retries):
            logger.info(f"[L0-NAS-MetaGraph] Attempt {attempt + 1}/{self.max_retries}")

            try:
                response = self.llm_client.query(
                    prompt,
                    temperature=0.0,
                    json_schema=L0_NAS_METAGRAPH_JSON_SCHEMA,
                    agent_name="L0NASMetaGraphGenerator",
                    context={"domain": "NAS"},
                )

                self.tokens_used += (
                    response.tokens_generated + response.prompt_tokens
                )

                logger.debug(
                    f"[L0-NAS-MetaGraph] Raw response (first 500 chars): "
                    f"{response.content[:500]}"
                )

                result = self._parse_response(response.content)
                if result is None:
                    logger.warning("[L0-NAS-MetaGraph] Failed to parse response")
                    continue

                mg, warnings = result
                if warnings:
                    logger.warning(f"[L0-NAS-MetaGraph] Warnings: {warnings}")

                if len(mg.edges) < 10:
                    logger.warning(
                        f"[L0-NAS-MetaGraph] Too few edges ({len(mg.edges)}), "
                        f"need at least 10 for 18 roles"
                    )
                    continue

                # Validate and repair
                valid, errors = self._validate_nas_structure(mg)
                if not valid:
                    logger.warning(
                        f"[L0-NAS-MetaGraph] Structure incomplete: {errors}"
                    )
                    mg = self._repair_nas_connections(mg)
                    valid, errors = self._validate_nas_structure(mg)
                    if not valid:
                        logger.warning(
                            f"[L0-NAS-MetaGraph] Auto-repair failed: {errors}"
                        )
                        continue

                self.generation_time_s = _time.time() - t0
                logger.info(
                    f"[L0-NAS-MetaGraph] Generated: {len(mg.nodes)} roles, "
                    f"{len(mg.edges)} edges, {self.tokens_used} tokens"
                )
                return mg

            except Exception as e:
                logger.warning(f"[L0-NAS-MetaGraph] Error: {e}")
                continue

        self.generation_time_s = _time.time() - t0
        logger.error("[L0-NAS-MetaGraph] Generation failed after max retries")
        return None

    def _parse_response(self, content: str) -> tuple[MetaGraph, list[str]] | None:
        """Parse LLM response into a NAS MetaGraph."""
        warnings: list[str] = []

        os.makedirs("logs", exist_ok=True)
        with open("logs/last_l0_nas_metagraph_response.json", "w") as f:
            f.write(content)

        try:
            cleaned = content.strip()
            cleaned = re.sub(r"^```json\s*\n?", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^```\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            cleaned = re.sub(r",\s*\.\.\.", "", cleaned)
            cleaned = re.sub(r"\.\.\.,?\s*", "", cleaned)

            json_str = extract_balanced_json(cleaned)
            if not json_str:
                logger.warning("[L0-NAS-MetaGraph] No valid JSON found")
                return None

            data = json.loads(json_str)

            self.reasoning = data.get("reasoning", "")

            mg = MetaGraph(
                name=data.get("name", "nas_llm_generated"),
                description=data.get("reasoning", ""),
                llm_reasoning=data.get("reasoning", ""),
            )
            mg.role_schema = self.schema

            # Add all roles from schema
            for role_id in self.schema.get_all_roles():
                meta = self.schema.get_role_metadata(role_id)
                mg.add_role_generic(
                    role_id=role_id,
                    description=meta.get("description", role_id),
                    category=meta.get("category", ""),
                    expected_cost=meta.get("expected_cost", "O(n)"),
                    exploration_bias=meta.get("exploration_bias", 0.5),
                )

            # Parse edges
            valid_roles = set(self.schema.get_all_roles())
            category_transitions = self.schema.get_category_transitions()

            for edge_data in data.get("edges", []):
                source_str = edge_data.get("source", "")
                target_str = edge_data.get("target", "")
                weight = float(edge_data.get("weight", 0.5))
                cond_data = edge_data.get("condition")

                if source_str not in valid_roles:
                    warnings.append(f"Unknown source role: {source_str}")
                    continue
                if target_str not in valid_roles:
                    warnings.append(f"Unknown target role: {target_str}")
                    continue

                # Validate category transition
                src_cat = self.schema.get_role_category(source_str)
                tgt_cat = self.schema.get_role_category(target_str)
                allowed_targets = category_transitions.get(src_cat, [])
                if tgt_cat not in allowed_targets and source_str != target_str:
                    warnings.append(
                        f"Invalid transition: {source_str}({src_cat}) -> "
                        f"{target_str}({tgt_cat})"
                    )
                    continue

                # Forbid evaluation self-loops
                if src_cat == "evaluation" and tgt_cat == "evaluation":
                    warnings.append(
                        f"Forbidden: eval->eval: {source_str} -> {target_str}"
                    )
                    continue

                # Parse conditions
                conditions: list[EdgeCondition] = []
                condition_boost = 2.0
                if cond_data and isinstance(cond_data, dict):
                    cond_type_str = cond_data.get("when", "")
                    threshold = float(cond_data.get("threshold", 0))
                    condition_boost = max(1.0, float(cond_data.get("boost", 2.0)))

                    cond_type_map = {
                        "stagnation": ConditionType.STAGNATION,
                        "diversity_low": ConditionType.DIVERSITY_LOW,
                        "gap_to_best": ConditionType.GAP_TO_BEST,
                    }
                    if cond_type_str in cond_type_map:
                        conditions.append(
                            EdgeCondition(
                                condition_type=cond_type_map[cond_type_str],
                                threshold=threshold,
                                reason=f"L0-NAS: {cond_type_str} >= {threshold}",
                            )
                        )

                mg.add_edge(
                    MetaEdge(
                        source=source_str,
                        target=target_str,
                        weight=weight,
                        conditions=conditions,
                        condition_boost=condition_boost,
                        reasoning=data.get("reasoning", ""),
                    )
                )

            return mg, warnings

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[L0-NAS-MetaGraph] Parse error: {e}")
            return None

    def _validate_nas_structure(
        self, mg: MetaGraph
    ) -> tuple[bool, list[str]]:
        """Validate NAS MetaGraph structure.

        Checks:
        - All roles have incoming edges (except topology = entry)
        - All roles have outgoing edges (except eval_full which can be terminal)
        - Evaluation has at least 1 feedback loop
        - No invalid category transitions
        """
        errors: list[str] = []
        all_roles = set(self.schema.get_all_roles())
        entry_categories = set(self.schema.get_entry_categories())
        category_transitions = self.schema.get_category_transitions()

        for role in all_roles:
            cat = self.schema.get_role_category(role)
            incoming = mg.get_incoming_edges(role)
            outgoing = mg.get_outgoing_edges(role)

            # Entry roles only need outgoing
            if cat in entry_categories:
                if not outgoing:
                    errors.append(f"{role} (entry) has no outgoing edges")
            else:
                if not incoming:
                    errors.append(f"{role} has no incoming edges")
                # eval_full can be terminal (no outgoing)
                if not outgoing and role != "eval_full":
                    errors.append(f"{role} has no outgoing edges")

        # Check evaluation feedback loops
        eval_roles = self.schema.get_roles_by_category("evaluation")
        has_feedback = False
        for eval_role in eval_roles:
            for edge in mg.get_outgoing_edges(eval_role):
                tgt_cat = self.schema.get_role_category(edge.target_str)
                if tgt_cat in ("topology", "training", "activation"):
                    has_feedback = True
                    break
            if has_feedback:
                break

        if not has_feedback:
            errors.append("No feedback loop from evaluation to earlier categories")

        # Validate transitions
        for (src, tgt), edge in mg.edges.items():
            if src not in all_roles or tgt not in all_roles:
                continue
            src_cat = self.schema.get_role_category(src)
            tgt_cat = self.schema.get_role_category(tgt)
            allowed = category_transitions.get(src_cat, [])
            if tgt_cat not in allowed and src != tgt:
                errors.append(f"Invalid transition: {src}({src_cat}) -> {tgt}({tgt_cat})")

        return len(errors) == 0, errors

    def _repair_nas_connections(self, mg: MetaGraph) -> MetaGraph:
        """Auto-repair missing connections in NAS MetaGraph.

        For roles without incoming: add edge from a predecessor category role.
        For roles without outgoing: add edge to a successor category role.
        """
        category_transitions = self.schema.get_category_transitions()
        entry_categories = set(self.schema.get_entry_categories())

        # Build reverse map: category -> predecessor categories
        predecessors: dict[str, list[str]] = {}
        for src_cat, tgt_cats in category_transitions.items():
            for tgt_cat in tgt_cats:
                predecessors.setdefault(tgt_cat, []).append(src_cat)

        all_roles = self.schema.get_all_roles()

        for role in all_roles:
            cat = self.schema.get_role_category(role)
            incoming = mg.get_incoming_edges(role)
            outgoing = mg.get_outgoing_edges(role)

            # Fix missing incoming (skip entry roles)
            if not incoming and cat not in entry_categories:
                pred_cats = predecessors.get(cat, [])
                for pred_cat in pred_cats:
                    pred_roles = self.schema.get_roles_by_category(pred_cat)
                    if pred_roles:
                        src_role = pred_roles[0]
                        mg.add_edge(MetaEdge(
                            source=src_role,
                            target=role,
                            weight=0.5,
                            reasoning="Auto-repaired missing incoming edge",
                        ))
                        logger.info(
                            f"[L0-NAS-MetaGraph] Auto-added: {src_role} -> {role}"
                        )
                        break

            # Fix missing outgoing (skip eval_full)
            if not outgoing and role != "eval_full":
                succ_cats = category_transitions.get(cat, [])
                for succ_cat in succ_cats:
                    if succ_cat == cat:
                        continue  # Prefer cross-category
                    succ_roles = self.schema.get_roles_by_category(succ_cat)
                    if succ_roles:
                        tgt_role = succ_roles[0]
                        mg.add_edge(MetaEdge(
                            source=role,
                            target=tgt_role,
                            weight=0.5,
                            reasoning="Auto-repaired missing outgoing edge",
                        ))
                        logger.info(
                            f"[L0-NAS-MetaGraph] Auto-added: {role} -> {tgt_role}"
                        )
                        break

        # Ensure at least one evaluation feedback loop
        eval_roles = self.schema.get_roles_by_category("evaluation")
        has_feedback = False
        for eval_role in eval_roles:
            for edge in mg.get_outgoing_edges(eval_role):
                tgt_cat = self.schema.get_role_category(edge.target_str)
                if tgt_cat in ("topology", "training", "activation"):
                    has_feedback = True
                    break
            if has_feedback:
                break

        if not has_feedback and eval_roles:
            topo_roles = self.schema.get_roles_by_category("topology")
            if topo_roles:
                mg.add_edge(MetaEdge(
                    source=eval_roles[0],
                    target=topo_roles[0],
                    weight=0.4,
                    conditions=[EdgeCondition(
                        condition_type=ConditionType.STAGNATION,
                        threshold=3.0,
                        reason="Feedback: redesign topology on stagnation",
                    )],
                    condition_boost=2.0,
                    reasoning="Auto-repaired: evaluation feedback loop",
                ))
                logger.info(
                    f"[L0-NAS-MetaGraph] Auto-added feedback: "
                    f"{eval_roles[0]} -> {topo_roles[0]}"
                )

        return mg

    def get_stats(self) -> dict[str, Any]:
        """Get generation statistics."""
        return {
            "tokens_used": self.tokens_used,
            "time_s": self.generation_time_s,
            "reasoning": self.reasoning,
        }
