"""LLM-based AKG Generator with symbolic validation.

The LLM constructs the Algorithmic Knowledge Graph structure:
- Categorizes operators
- Defines valid transitions (edges)
- Assigns initial weights
- The symbolic validator ensures structural correctness
"""

from typing import Any
from pydantic import BaseModel, Field
from loguru import logger

from src.geakg.conditions import parse_condition_from_dict
from src.geakg.graph import AlgorithmicKnowledgeGraph
from src.geakg.nodes import OperatorNode, OperatorCategory, AKGEdge, EdgeType


class OperatorInfo(BaseModel):
    """Basic operator information provided to LLM."""
    id: str
    name: str
    description: str
    category: str = ""  # Optional: construction, local_search, perturbation


class ProposedEdge(BaseModel):
    """Edge proposed by LLM.

    Level 3 extension: Edges can include conditions that control WHEN
    the transition should be taken.
    """
    source: str
    target: str
    weight: float = Field(ge=0.0, le=1.0)
    reason: str = ""

    # Level 3: Conditional transitions
    # conditions is a list of condition dicts from LLM
    # e.g., [{"when": "stagnation", "threshold": 3}]
    conditions: list[dict] = Field(default_factory=list)
    condition_boost: float = Field(ge=0.0, le=5.0, default=1.0)


class ProposedAKG(BaseModel):
    """AKG structure proposed by LLM."""
    operator_categories: dict[str, str]  # operator_id -> category
    edges: list[ProposedEdge] = Field(default_factory=list)
    operator_quality: dict[str, float] = Field(default_factory=dict)  # operator_id -> quality score
    reasoning: str = ""


class ValidationResult(BaseModel):
    """Result of validating a proposed AKG."""
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AKGValidator:
    """Validates proposed AKG structures against structural rules."""

    VALID_CATEGORIES = {"construction", "local_search", "perturbation"}

    def __init__(self, operator_ids: list[str]) -> None:
        """Initialize validator.

        Args:
            operator_ids: List of valid operator IDs
        """
        self.operator_ids = set(operator_ids)

    def validate(self, proposed: ProposedAKG) -> ValidationResult:
        """Validate a proposed AKG structure.

        Args:
            proposed: Proposed AKG from LLM

        Returns:
            Validation result with errors/warnings
        """
        errors = []
        warnings = []

        # Rule 1: Most operators must be categorized (allow some missing for small LLMs)
        categorized = set(proposed.operator_categories.keys())
        missing = self.operator_ids - categorized
        if len(missing) > len(self.operator_ids) * 0.2:  # Allow up to 20% missing
            errors.append(f"Operators not categorized: {missing}")
        elif missing:
            warnings.append(f"Some operators not categorized (will use default): {missing}")

        unknown = categorized - self.operator_ids
        if unknown:
            errors.append(f"Unknown operators: {unknown}")

        # Rule 2: Categories must be valid
        for op_id, category in proposed.operator_categories.items():
            if category not in self.VALID_CATEGORIES:
                errors.append(f"Invalid category '{category}' for {op_id}. Valid: {self.VALID_CATEGORIES}")

        # Rule 3: Must have at least one construction operator
        constructions = [op for op, cat in proposed.operator_categories.items()
                        if cat == "construction"]
        if not constructions:
            errors.append("Must have at least one construction operator")

        # Rule 4: Edges must reference valid operators (if provided)
        for edge in proposed.edges:
            if edge.source not in self.operator_ids:
                errors.append(f"Edge source '{edge.source}' not a valid operator")
            if edge.target not in self.operator_ids:
                errors.append(f"Edge target '{edge.target}' not a valid operator")

        # Rule 5: Quality scores must be valid (if provided)
        for op_id, quality in proposed.operator_quality.items():
            if op_id not in self.operator_ids:
                warnings.append(f"Quality score for unknown operator: {op_id}")
            if not (0.0 <= quality <= 1.0):
                errors.append(f"Quality score for {op_id} must be 0.0-1.0, got {quality}")

        # Rule 6: ILS cycle must exist (CRITICAL for algorithm effectiveness)
        ils_errors = self._validate_ils_cycle(proposed)
        errors.extend(ils_errors)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _validate_ils_cycle(self, proposed: ProposedAKG) -> list[str]:
        """Validate ILS cycle - now just warnings since _fix_diversity_edges guarantees it.

        The _fix_diversity_edges phase will add any missing edges automatically,
        so we don't need to fail validation here.

        Args:
            proposed: Proposed AKG structure

        Returns:
            List of errors (empty - we handle missing edges in fix phase)
        """
        # No longer fail on missing ILS edges - _fix_diversity_edges will handle it
        # Just log a warning for visibility
        local_search_ops = {op for op, cat in proposed.operator_categories.items()
                          if cat == "local_search"}
        perturbation_ops = {op for op, cat in proposed.operator_categories.items()
                          if cat == "perturbation"}

        if perturbation_ops:
            has_escape = any(
                e.source in local_search_ops and e.target in perturbation_ops
                for e in proposed.edges
            )
            has_reopt = any(
                e.source in perturbation_ops and e.target in local_search_ops
                for e in proposed.edges
            )

            if not has_escape or not has_reopt:
                logger.warning("LLM didn't provide complete ILS cycle - will be fixed in diversity phase")

        return []  # No errors - diversity phase will fix

    def _get_diversity_gaps(self, proposed: ProposedAKG) -> tuple[set[str], set[str]]:
        """Find operators missing escape/re-optimize edges.

        Returns:
            Tuple of (ls_without_escape, pert_without_reopt)
        """
        local_search_ops = {op for op, cat in proposed.operator_categories.items()
                          if cat == "local_search"}
        perturbation_ops = {op for op, cat in proposed.operator_categories.items()
                          if cat == "perturbation"}

        # Find which local_search operators have escape edges
        ls_with_escape = {
            e.source for e in proposed.edges
            if e.source in local_search_ops and e.target in perturbation_ops
        }

        # Find which perturbation operators have re-optimize edges
        pert_with_reopt = {
            e.source for e in proposed.edges
            if e.source in perturbation_ops and e.target in local_search_ops
        }

        ls_without_escape = local_search_ops - ls_with_escape
        pert_without_reopt = perturbation_ops - pert_with_reopt

        return ls_without_escape, pert_without_reopt

    def _find_reachable(self, start_nodes: list[str], edges: list[ProposedEdge]) -> set[str]:
        """Find all nodes reachable from start nodes."""
        adj = {}
        for edge in edges:
            if edge.source not in adj:
                adj[edge.source] = []
            adj[edge.source].append(edge.target)

        reachable = set(start_nodes)
        frontier = list(start_nodes)

        while frontier:
            node = frontier.pop()
            for neighbor in adj.get(node, []):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    frontier.append(neighbor)

        return reachable


# Prompt template for AKG construction with algorithmic knowledge
# Used when categories are NOT pre-defined
AKG_CONSTRUCTION_PROMPT_WITH_CATEGORIZATION = '''You are an expert in combinatorial optimization. Analyze these operators and define VALID transitions for Iterated Local Search (ILS).

OPERATORS:
{operators}

TASK 1 - Categorize each operator into ONE of these categories:
- construction: Creates initial solution (starting point)
- local_search: Improves solution locally (two_opt, three_opt, or_opt, etc.)
- perturbation: Escapes local optima (double_bridge, ruin_recreate, etc.)

TASK 2 - Define transitions between operators (see below)

{transition_instructions}

**CRITICAL: In compatible_pairs, use the EXACT operator IDs (e.g., "two_opt", "greedy_nearest_neighbor").**
**DO NOT use category names like "local_search" or "construction" as source/target.**

Respond in JSON:
{{
  "operator_categories": {{
    "greedy_nearest_neighbor": "construction",
    "two_opt": "local_search",
    "double_bridge": "perturbation",
    ...
  }},
  "compatible_pairs": [
    ["greedy_nearest_neighbor", "two_opt", 0.90],
    ["two_opt", "three_opt", 0.75],
    ["three_opt", "double_bridge", 0.65],
    ...
  ],
  "reasoning": "explain the ILS cycle you designed"
}}

Use ONLY the exact operator IDs from the list above. Provide 50-60 pairs.'''

# Prompt when categories ARE pre-defined (preferred)
AKG_CONSTRUCTION_PROMPT = '''You are an expert in combinatorial optimization. Define VALID transitions for Iterated Local Search (ILS).

OPERATORS WITH THEIR CATEGORIES:
{operators}

{transition_instructions}

**CRITICAL: Use the EXACT operator IDs from the list above (e.g., "two_opt", "greedy_nearest_neighbor").**
**DO NOT use category names like "local_search" or "construction" as source/target.**

Respond in JSON:
{{
  "compatible_pairs": [
    ["greedy_nearest_neighbor", "two_opt", 0.90],
    ["two_opt", "three_opt", 0.75],
    ["three_opt", "double_bridge", 0.65],
    ...
  ],
  "reasoning": "explain the ILS cycle you designed"
}}

Use ONLY the exact operator IDs from the list above. {n_pairs_instruction}'''

# Number of pairs instruction varies by mode
N_PAIRS_BASIC = "Provide 50-60 pairs."
N_PAIRS_LEARNED = "Provide AT LEAST 60-80 pairs for full connectivity."

# Shared transition instructions (basic version)
ILS_TRANSITION_INSTRUCTIONS = '''The ILS cycle requires these transition types:

1. construction → local_search (start optimization)
2. local_search → local_search (chain different LS operators)
3. local_search → perturbation (ESCAPE local optimum!) ← CRITICAL
4. perturbation → local_search (RE-OPTIMIZE after perturbation!) ← CRITICAL
5. perturbation → construction (restart from scratch)

**MANDATORY DIVERSITY REQUIREMENTS** (YOUR OUTPUT WILL BE REJECTED IF NOT MET):

1. EVERY local_search operator MUST have at least 1 edge TO a perturbation operator
   - This is the "escape route" from local optima
   - If you have 7 local_search operators, you need AT LEAST 7 escape edges

2. EVERY perturbation operator MUST have at least 1 edge TO a local_search operator
   - This is the "re-optimization" after perturbation
   - If you have 3 perturbation operators, you need AT LEAST 3 re-optimize edges

3. DO NOT centralize through one operator
   - Spread edges across ALL operators evenly
   - Each operator should have 2-4 outgoing edges, not 0 or 10

Quality scores:
- 0.9-0.95: Excellent (proven ILS pattern)
- 0.7-0.85: Good (commonly used)
- 0.5-0.65: Acceptable
- NO self-loops
- NO bad transitions (< 0.5)

Provide 50-60 pairs to ensure EVERY operator has sufficient connections.'''


# Enhanced instructions for LLM-learned weights (more specific guidance)
ILS_TRANSITION_INSTRUCTIONS_LEARNED_WEIGHTS = '''The ILS cycle requires these transition types with SPECIFIC weights based on optimization literature:

**WEIGHT ASSIGNMENT GUIDELINES (based on metaheuristic research):**

You must assign weights based on YOUR knowledge of combinatorial optimization. Consider:
- How often this transition appears in successful algorithms (Lin-Kernighan, ILS, LKH, etc.)
- Whether the operators are complementary (explore different neighborhoods)
- The risk/reward tradeoff of the transition

SPECIFIC WEIGHT RANGES BY TRANSITION TYPE:
1. construction → local_search: 0.85-0.95 (always beneficial to start optimizing)
2. local_search → local_search (SAME family, e.g., both k-opt): 0.40-0.55 (redundant)
3. local_search → local_search (DIFFERENT family): 0.70-0.85 (complementary)
4. local_search → perturbation: 0.55-0.70 (escape - necessary but risky)
5. perturbation → local_search: 0.85-0.95 (re-optimize - always good after disruption)
6. perturbation → construction: 0.30-0.45 (restart - loses progress)
7. perturbation → perturbation: 0.25-0.40 (chaining perturbations - usually harmful)

**OPERATOR-SPECIFIC CONSIDERATIONS:**
- two_opt is the most reliable LS operator - transitions TO two_opt should be higher
- three_opt is more powerful but slower - use after two_opt converges
- or_opt and insert are fine-tuning operators - use late in sequence
- double_bridge is the classic ILS perturbation - high weight FROM LS operators
- ruin_recreate is aggressive - lower weight, use when stuck

**MANDATORY REQUIREMENTS:**
1. EVERY local_search operator MUST have at least 1 edge TO a perturbation operator
2. EVERY perturbation operator MUST have at least 1 edge TO a local_search operator
3. Each operator should have 2-4 outgoing edges (minimum 2!)
4. NO self-loops
5. Weights must be in range [0.3, 0.95] - no extreme values

**CRITICAL: You MUST provide AT LEAST 60 pairs to ensure full connectivity.**
**If you provide fewer than 60 pairs, the AKG will be rejected.**

Provide 60-80 pairs with THOUGHTFUL weights based on the above guidelines.'''


# Level 3: Conditional Transitions - LLM assigns control policies
ILS_TRANSITION_INSTRUCTIONS_CONDITIONAL = '''The ILS cycle requires these transition types with CONDITIONS that control WHEN to take each transition.

**LEVEL 3: CONDITIONAL TRANSITIONS**

You can add CONDITIONS to edges that specify WHEN the transition should be taken.
This encodes expert control knowledge like "escape to perturbation after stagnation".

CONDITION TYPES:
- "stagnation": Generations without improvement (e.g., "after 3 generations stuck")
- "gap_to_best": Gap to best solution exceeds threshold (e.g., "if gap > 5%")
- "diversity_low": Population diversity below threshold (e.g., "when diversity < 0.3")
- "improvement_rate": Recent success rate (e.g., "if last 3 ops failed")

**EDGE FORMAT WITH CONDITIONS:**
["source", "target", weight, {"when": "condition_type", "threshold": value, "boost": multiplier}]

The "boost" is a probability multiplier (1.0-3.0) applied when the condition is met.

**EXAMPLES OF CONDITIONAL TRANSITIONS:**

1. ESCAPE edges (local_search → perturbation) should have stagnation conditions:
   ["two_opt", "double_bridge", 0.55, {"when": "stagnation", "threshold": 3, "boost": 2.0}]
   → "Use double_bridge after 3 generations without improvement, 2x more likely"

2. AGGRESSIVE perturbation when stuck:
   ["three_opt", "ruin_recreate", 0.45, {"when": "gap_to_best", "threshold": 0.05, "boost": 2.5}]
   → "Try ruin_recreate when gap > 5%, 2.5x boost"

3. INTENSIFY when improving:
   ["two_opt", "three_opt", 0.75]
   → No condition - always available for chaining local search

4. RESTART when severely stuck:
   ["double_bridge", "greedy_nearest_neighbor", 0.35, {"when": "stagnation", "threshold": 10, "boost": 3.0}]
   → "Restart construction after 10 generations stuck"

**CONTROL POLICY GUIDELINES:**

1. ESCAPE edges (LS → Perturbation):
   - Add stagnation conditions (threshold 2-5)
   - Higher boost (1.5-2.5) to encourage escape when stuck

2. AGGRESSIVE moves (→ ruin_recreate, restart):
   - Add gap_to_best or high stagnation thresholds
   - Only use when severely stuck

3. FINE-TUNING edges (or_opt, insert):
   - No conditions (always available)
   - Lower boost (1.0)

4. RESTART edges (→ construction):
   - High stagnation threshold (8-15)
   - Last resort when all else fails

**MANDATORY REQUIREMENTS:**
1. EVERY local_search MUST have at least 1 edge TO perturbation (with stagnation condition)
2. EVERY perturbation MUST have at least 1 edge TO local_search (unconditional)
3. Include 15-25 conditional edges (escape, aggressive, restart)
4. Include 40-55 unconditional edges (normal transitions)
5. Weights in range [0.3, 0.95], boosts in range [1.0, 3.0]

Provide 60-80 pairs total, with clear control policies for escaping local optima.'''


# Prompt for asking LLM to connect missing operators
AKG_CONNECTIVITY_PROMPT = '''Fix these connectivity issues in the ILS graph:

{disconnected_sources}

ALL OPERATORS: {all_operators}

ILS CYCLE RULES:
- Operators needing INCOMING edges: add edges FROM other operators TO them
  Example: "two_opt -> double_bridge" (local_search can escape to perturbation)
- Operators needing OUTGOING edges: add edges FROM them TO other operators
  Example: "three_opt -> ruin_recreate" (local_search can escape to perturbation)

KEY TRANSITIONS TO ENABLE:
- local_search → perturbation (ESCAPE local optima!)
- perturbation → local_search (RE-OPTIMIZE after perturbation!)

Respond with JSON:
{{
  "compatible_pairs": [
    ["source", "target", quality_score],
    ...
  ]
}}

Add edges to fix ALL listed connectivity issues.'''


# Prompt for adding diversity edges - simple and direct
AKG_DIVERSITY_PROMPT = '''Add these edges to the graph. Copy exactly:

{{
  "compatible_pairs": [
{edge_list}
  ]
}}'''


AKG_CORRECTION_PROMPT = '''Your previous AKG proposal had errors:

{errors}

{warnings}

Please fix these issues and provide a corrected AKG.

Previous proposal:
{previous}

Respond with the corrected JSON in the same format.'''


class LLMAKGGenerator:
    """Generates AKG using LLM with strict symbolic validation.

    The LLM constructs the topology, but the validator ensures:
    - All operators are categorized correctly
    - ILS cycle exists (local_search ↔ perturbation)
    - No invalid edges or self-loops

    AKG Levels:
    - Level 1 (Topology): LLM decides which operators can connect
    - Level 2 (Weights): use_learned_weights=True → LLM assigns weights
    - Level 3 (Conditions): use_conditions=True → LLM assigns conditions

    Weight modes:
    - use_learned_weights=False (default): LLM proposes topology, Stage 4 applies expert weights
    - use_learned_weights=True: LLM proposes both topology AND weights (no Stage 4 override)
    """

    def __init__(
        self,
        llm_client: Any,
        operators: list[OperatorInfo],
        max_retries: int = 3,
        use_learned_weights: bool = False,
        use_conditions: bool = False,
    ) -> None:
        """Initialize generator.

        Args:
            llm_client: LLM client for queries
            operators: List of operators (with optional pre-defined categories)
            max_retries: Maximum correction attempts
            use_learned_weights: If True, use LLM-assigned weights instead of expert weights
            use_conditions: If True, prompt LLM to assign conditions to edges (Level 3)
        """
        self.llm_client = llm_client
        self.operators = operators
        self.max_retries = max_retries
        self.validator = AKGValidator([op.id for op in operators])
        self.use_learned_weights = use_learned_weights
        self.use_conditions = use_conditions

        # Check if categories are pre-defined by user
        self.categories_predefined = all(op.category for op in operators)

    def generate(self, max_connectivity_iterations: int = 3) -> AlgorithmicKnowledgeGraph | None:
        """Generate AKG using LLM with strict ILS validation.

        The LLM constructs the topology. The validator ensures:
        - ILS cycle exists (local_search ↔ perturbation edges)
        - All operators are properly categorized
        - No invalid transitions

        Args:
            max_connectivity_iterations: Max iterations to fix connectivity

        Returns:
            Valid AKG or None if failed after retries
        """
        weight_mode = "LLM-learned" if self.use_learned_weights else "expert-defined"
        conditions_mode = "with conditions" if self.use_conditions else "no conditions"
        logger.info(f"Generating AKG with LLM (weights: {weight_mode}, {conditions_mode})")

        # Select transition instructions based on mode
        # Level 3 (conditions) takes precedence over Level 2 (weights)
        if self.use_conditions:
            transition_instructions = ILS_TRANSITION_INSTRUCTIONS_CONDITIONAL
            n_pairs_instruction = N_PAIRS_LEARNED  # Same as learned weights
        elif self.use_learned_weights:
            transition_instructions = ILS_TRANSITION_INSTRUCTIONS_LEARNED_WEIGHTS
            n_pairs_instruction = N_PAIRS_LEARNED
        else:
            transition_instructions = ILS_TRANSITION_INSTRUCTIONS
            n_pairs_instruction = N_PAIRS_BASIC

        # Build operator description for prompt
        if self.categories_predefined:
            # Categories provided by user - LLM only needs to create edges
            op_desc = "\n".join([
                f"- {op.id} [{op.category}]: {op.description}"
                for op in self.operators
            ])
            prompt = AKG_CONSTRUCTION_PROMPT.format(
                operators=op_desc,
                transition_instructions=transition_instructions,
                n_pairs_instruction=n_pairs_instruction
            )
        else:
            # LLM must categorize operators too
            op_desc = "\n".join([
                f"- {op.id}: {op.description}"
                for op in self.operators
            ])
            prompt = AKG_CONSTRUCTION_PROMPT_WITH_CATEGORIZATION.format(
                operators=op_desc,
                transition_instructions=transition_instructions
            )

        proposed = None
        for attempt in range(self.max_retries):
            logger.info(f"AKG generation attempt {attempt + 1}/{self.max_retries}")

            # Query LLM with temperature=0 for deterministic MetaGraph generation
            response = self.llm_client.query(
                prompt,
                temperature=0.0,
                agent_name="LLMAKGGenerator",
                context={"stage": "akg_generation"},
            )

            # Parse response
            proposed = self._parse_response(response.content)
            if proposed is None:
                logger.warning("Failed to parse LLM response")
                continue

            # Validate
            result = self.validator.validate(proposed)
            logger.debug(f"Validation: valid={result.valid}, errors={result.errors}, warnings={result.warnings}")

            if result.valid:
                if result.warnings:
                    logger.warning(f"AKG valid with warnings: {result.warnings}")
                logger.info("AKG generation successful")

                # Build initial AKG
                akg = self._build_akg(proposed)

                # Iteratively fix connectivity
                akg = self._fix_connectivity(akg, proposed, max_connectivity_iterations)

                # Fix diversity (ensure ALL operators have escape/reopt edges)
                akg = self._fix_diversity_edges(akg, proposed)

                # Stage 4: Apply weights
                if self.use_learned_weights:
                    # Keep LLM-assigned weights (already in AKG from _build_akg)
                    logger.info("Using LLM-learned weights (Stage 4 skipped)")
                    self._log_weight_statistics(akg, proposed)
                else:
                    # Apply transition-type based weights (override LLM weights)
                    akg = self._apply_transition_weights(akg, proposed)

                return akg

            # Build correction prompt
            errors_str = "\n".join(f"- ERROR: {e}" for e in result.errors)
            warnings_str = "\n".join(f"- WARNING: {w}" for w in result.warnings)

            prompt = AKG_CORRECTION_PROMPT.format(
                errors=errors_str,
                warnings=warnings_str if result.warnings else "None",
                previous=response.content
            )

        logger.error("AKG generation failed after max retries")
        return None

    def _fix_connectivity(
        self,
        akg: AlgorithmicKnowledgeGraph,
        proposed: ProposedAKG,
        max_iterations: int
    ) -> AlgorithmicKnowledgeGraph:
        """Iteratively ask LLM to connect disconnected operators.

        Args:
            akg: Current AKG
            proposed: Original proposal with categories
            max_iterations: Maximum iterations

        Returns:
            AKG with improved connectivity
        """
        operator_ids = set(akg.nodes.keys())

        for iteration in range(max_iterations):
            # Find operators without outgoing edges (excluding constructions that should start)
            sources_with_outgoing = {e.source for e in akg.edges.values()}
            targets_with_incoming = {e.target for e in akg.edges.values()}

            # Disconnected: no outgoing edges (can't continue from here)
            disconnected_sources = operator_ids - sources_with_outgoing

            # Isolated: no incoming edges (can't reach them)
            # Exclude construction operators (they're start points)
            constructions = {
                op_id for op_id, cat in proposed.operator_categories.items()
                if cat == "construction"
            }
            isolated_targets = (operator_ids - targets_with_incoming) - constructions

            # Check if we're done
            if not disconnected_sources and not isolated_targets:
                logger.info(f"AKG fully connected after {iteration} iterations")
                break

            if not disconnected_sources and len(isolated_targets) <= 2:
                # Some isolated nodes are acceptable (rarely used operators)
                logger.info(f"AKG acceptable: {len(isolated_targets)} isolated operators")
                break

            logger.info(f"Connectivity iteration {iteration + 1}: "
                       f"{len(disconnected_sources)} disconnected, {len(isolated_targets)} isolated")

            # Build categories string for context
            cat_str = "\n".join([
                f"- {op_id}: {cat}" for op_id, cat in proposed.operator_categories.items()
            ])

            # Prepare lists for prompt
            all_ops = ", ".join(sorted(operator_ids))

            # Build detailed description of what's needed
            needs_incoming = sorted(isolated_targets)
            needs_outgoing = sorted(disconnected_sources)

            disconnected_desc = []
            if needs_incoming:
                disconnected_desc.append(f"Need INCOMING edges (unreachable): {', '.join(needs_incoming)}")
            if needs_outgoing:
                disconnected_desc.append(f"Need OUTGOING edges (dead-ends): {', '.join(needs_outgoing)}")

            # Ask LLM to connect missing operators
            prompt = AKG_CONNECTIVITY_PROMPT.format(
                disconnected_sources="\n".join(disconnected_desc),
                all_operators=all_ops
            )

            response = self.llm_client.query(
                prompt,
                temperature=0.0,
                agent_name="LLMAKGGenerator",
                context={"stage": "connectivity_fix"},
            )
            new_edges = self._parse_connectivity_response(response.content)

            if not new_edges:
                logger.warning("LLM didn't provide new edges, stopping iteration")
                break

            # Add new edges (filter self-loops)
            added = 0
            for edge in new_edges:
                # Skip self-loops
                if edge.source == edge.target:
                    continue
                if edge.source in operator_ids and edge.target in operator_ids:
                    # Check if edge already exists
                    edge_key = f"{edge.source}->{edge.target}"
                    if edge_key not in akg.edges:
                        akg.add_edge(AKGEdge(
                            source=edge.source,
                            target=edge.target,
                            edge_type=EdgeType.SEQUENTIAL,
                            weight=edge.weight,
                        ))
                        added += 1

            logger.info(f"Added {added} new edges from LLM")

            if added == 0:
                break

        return akg

    def _fix_diversity_edges(
        self,
        akg: AlgorithmicKnowledgeGraph,
        proposed: ProposedAKG,
        max_llm_attempts: int = 2
    ) -> AlgorithmicKnowledgeGraph:
        """Ensure ALL local_search and perturbation operators have escape/reopt edges.

        Strategy:
        1. Try LLM to get weights for missing edges
        2. If LLM fails, use symbolic fallback with default weight

        Args:
            akg: Current AKG
            proposed: Original proposal with categories
            max_llm_attempts: Maximum LLM attempts before fallback

        Returns:
            AKG with full diversity (guaranteed)
        """
        local_search_ops = sorted([op for op, cat in proposed.operator_categories.items()
                                   if cat == "local_search"])
        perturbation_ops = sorted([op for op, cat in proposed.operator_categories.items()
                                   if cat == "perturbation"])

        if not perturbation_ops:
            logger.info("No perturbation operators, skipping diversity fix")
            return akg

        # Calculate required edges
        required_edges = []

        # Find LS operators without escape edges
        ls_with_escape = {e.source for e in akg.edges.values()
                        if e.source in local_search_ops and e.target in perturbation_ops}
        for i, ls_op in enumerate(local_search_ops):
            if ls_op not in ls_with_escape:
                # Distribute across perturbation operators
                target = perturbation_ops[i % len(perturbation_ops)]
                required_edges.append((ls_op, target, "escape"))

        # Find perturbation operators without re-opt edges
        pert_with_reopt = {e.source for e in akg.edges.values()
                          if e.source in perturbation_ops and e.target in local_search_ops}
        for i, pert_op in enumerate(perturbation_ops):
            if pert_op not in pert_with_reopt:
                # Distribute across local_search operators
                target = local_search_ops[i % len(local_search_ops)]
                required_edges.append((pert_op, target, "reopt"))

        if not required_edges:
            logger.info("Diversity complete: all operators have escape/reopt edges")
            return akg

        logger.info(f"Need {len(required_edges)} diversity edges: "
                   f"{sum(1 for _, _, t in required_edges if t == 'escape')} escape, "
                   f"{sum(1 for _, _, t in required_edges if t == 'reopt')} reopt")

        # Try LLM first to get better weights
        edge_weights = {}
        for attempt in range(max_llm_attempts):
            edge_list = ",\n".join([
                f'    ["{src}", "{tgt}", 0.75]'
                for src, tgt, _ in required_edges
            ])
            prompt = AKG_DIVERSITY_PROMPT.format(edge_list=edge_list)

            try:
                response = self.llm_client.query(
                    prompt,
                    temperature=0.0,
                    agent_name="LLMAKGGenerator",
                    context={"stage": "diversity_fix"},
                )
                new_edges = self._parse_connectivity_response(response.content)

                if new_edges:
                    for edge in new_edges:
                        key = (edge.source, edge.target)
                        edge_weights[key] = edge.weight
                    logger.info(f"LLM provided weights for {len(new_edges)} edges")
                    break
            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")

        # Add all required edges (with LLM weight or default)
        added = 0
        for src, tgt, edge_type in required_edges:
            edge_key = f"{src}->{tgt}"
            if edge_key in akg.edges:
                continue

            # Use LLM weight if available, otherwise default
            weight = edge_weights.get((src, tgt), 0.75)

            akg.add_edge(AKGEdge(
                source=src,
                target=tgt,
                edge_type=EdgeType.SEQUENTIAL,
                weight=weight,
            ))
            added += 1
            logger.debug(f"Added {edge_type} edge: {src} -> {tgt} (w={weight:.2f})")

        logger.info(f"Added {added} diversity edges (guaranteed)")

        # Verify
        ls_with_escape = {e.source for e in akg.edges.values()
                        if e.source in local_search_ops and e.target in perturbation_ops}
        pert_with_reopt = {e.source for e in akg.edges.values()
                          if e.source in perturbation_ops and e.target in local_search_ops}

        ls_missing = set(local_search_ops) - ls_with_escape
        pert_missing = set(perturbation_ops) - pert_with_reopt

        if ls_missing or pert_missing:
            # This should never happen with the new logic
            logger.error(f"Diversity still incomplete! LS={ls_missing}, Pert={pert_missing}")
        else:
            logger.info("Full diversity achieved!")

        return akg

    def _parse_connectivity_response(self, content: str) -> list[ProposedEdge]:
        """Parse LLM response for connectivity edges."""
        import json
        import re

        edges = []
        try:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())

                for pair in data.get("compatible_pairs", []):
                    if isinstance(pair, list) and len(pair) >= 2:
                        edges.append(ProposedEdge(
                            source=pair[0],
                            target=pair[1],
                            weight=float(pair[2]) if len(pair) > 2 else 0.6,
                            reason=""
                        ))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse connectivity response: {e}")

        return edges

    def _parse_response(self, content: str) -> ProposedAKG | None:
        """Parse LLM response into ProposedAKG."""
        import json
        import re

        # Extract JSON from response
        try:
            # Try to find JSON block
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())

                # Use pre-defined categories if available, otherwise parse from LLM
                if self.categories_predefined:
                    # Use user-provided categories
                    operator_categories = {op.id: op.category for op in self.operators}
                else:
                    # Parse operator_categories from LLM - handle both formats
                    raw_categories = data.get("operator_categories", {})
                    operator_categories = {}

                    # Check if format is {category: [op1, op2, ...]} (LLM sometimes returns this)
                    first_value = list(raw_categories.values())[0] if raw_categories else None
                    if isinstance(first_value, list):
                        # Inverted format: {category: [operators]}
                        for cat, ops in raw_categories.items():
                            for op in ops:
                                operator_categories[op] = cat
                    else:
                        # Normal format: {operator: category}
                        operator_categories = raw_categories

                # Parse edges from compatible_pairs (new format) or edges (old format)
                edges = []

                # NEW FORMAT: compatible_pairs = [["src", "tgt", weight], ...]
                # Level 3 format: ["src", "tgt", weight, {"when": "stagnation", "threshold": 3}]
                compatible_pairs = data.get("compatible_pairs", [])
                for pair in compatible_pairs:
                    if isinstance(pair, list) and len(pair) >= 2:
                        src = pair[0]
                        tgt = pair[1]
                        weight = float(pair[2]) if len(pair) > 2 else 0.7

                        # Level 3: Parse condition if present (4th element)
                        conditions = []
                        condition_boost = 1.0
                        if len(pair) > 3 and isinstance(pair[3], dict):
                            cond_data = pair[3]
                            conditions = [cond_data]
                            condition_boost = float(cond_data.get("boost", 1.5))

                        edges.append(ProposedEdge(
                            source=src,
                            target=tgt,
                            weight=weight,
                            reason="",
                            conditions=conditions,
                            condition_boost=condition_boost,
                        ))

                # OLD FORMAT: edges = [{source, target, weight}, ...]
                for e in data.get("edges", []):
                    edges.append(ProposedEdge(
                        source=e["source"],
                        target=e["target"],
                        weight=float(e.get("weight", 0.5)),
                        reason=e.get("reason", "")
                    ))

                # Parse quality scores (optional)
                quality = {}
                for op_id, score in data.get("operator_quality", {}).items():
                    try:
                        quality[op_id] = float(score)
                    except (ValueError, TypeError):
                        quality[op_id] = 0.5

                return ProposedAKG(
                    operator_categories=operator_categories,
                    edges=edges,
                    operator_quality=quality,
                    reasoning=data.get("reasoning", "")
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse response: {e}")

        return None

    def _log_weight_statistics(
        self,
        akg: AlgorithmicKnowledgeGraph,
        proposed: ProposedAKG
    ) -> None:
        """Log statistics about LLM-assigned weights for analysis.

        Args:
            akg: AKG with LLM weights
            proposed: Original proposal with categories
        """
        categories = proposed.operator_categories

        # Group weights by transition type
        weight_by_type: dict[str, list[float]] = {}

        for edge in akg.edges.values():
            src_cat = categories.get(edge.source, "unknown")
            tgt_cat = categories.get(edge.target, "unknown")
            transition_type = f"{src_cat} → {tgt_cat}"

            if transition_type not in weight_by_type:
                weight_by_type[transition_type] = []
            weight_by_type[transition_type].append(edge.weight)

        # Log statistics
        logger.info("LLM-assigned weight statistics by transition type:")
        for t_type, weights in sorted(weight_by_type.items()):
            avg = sum(weights) / len(weights)
            min_w = min(weights)
            max_w = max(weights)
            logger.info(f"  {t_type}: avg={avg:.2f}, min={min_w:.2f}, max={max_w:.2f}, n={len(weights)}")

    def _build_akg(self, proposed: ProposedAKG) -> AlgorithmicKnowledgeGraph:
        """Build AKG using LLM-provided compatible pairs.

        Uses edges from proposed.edges directly.
        """
        akg = AlgorithmicKnowledgeGraph()

        # Map category strings to enum (3 categories only)
        cat_map = {
            "construction": OperatorCategory.CONSTRUCTION,
            "local_search": OperatorCategory.LOCAL_SEARCH,
            "perturbation": OperatorCategory.PERTURBATION,
        }

        # Add operator nodes
        operator_ids = set()
        for op in self.operators:
            operator_ids.add(op.id)
            category = cat_map.get(
                proposed.operator_categories.get(op.id, "local_search"),
                OperatorCategory.LOCAL_SEARCH
            )
            node = OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=category,
            )
            akg.add_node(node)

        # Add LLM-provided edges
        logger.info(f"Using {len(proposed.edges)} LLM-provided compatible pairs")
        for edge in proposed.edges:
            if edge.source in operator_ids and edge.target in operator_ids:
                # Level 3: Parse conditions if present
                conditions = [
                    parse_condition_from_dict(c)
                    for c in edge.conditions
                ]

                akg.add_edge(AKGEdge(
                    source=edge.source,
                    target=edge.target,
                    edge_type=EdgeType.SEQUENTIAL,
                    weight=edge.weight,
                    conditions=conditions,
                    condition_boost=edge.condition_boost,
                ))

        return akg

    def _apply_transition_weights(
        self,
        akg: AlgorithmicKnowledgeGraph,
        proposed: ProposedAKG
    ) -> AlgorithmicKnowledgeGraph:
        """Apply weights based on transition type instead of LLM weights.

        Weight scheme based on ILS effectiveness (3 categories):
        - construction → LS:      0.90 (always good to start optimizing)
        - LS → LS (same op):      0.30 (redundant, penalize)
        - LS → LS (different):    0.70 (complementary techniques)
        - LS → perturbation:      0.60 (risky but necessary for escape)
        - perturbation → LS:      0.90 (always re-optimize after perturbation)
        - perturbation → construction: 0.40 (restart, lose progress)
        - perturbation → perturbation: 0.35 (chaining - usually bad)
        - construction → construction: 0.25 (redundant)
        - other:                  0.50 (default)

        Args:
            akg: AKG with LLM weights
            proposed: Original proposal with categories

        Returns:
            AKG with transition-based weights
        """
        categories = proposed.operator_categories

        # Collect edges to replace (since AKGEdge is frozen)
        edges_to_replace = []

        for edge_key, edge in akg.edges.items():
            src_cat = categories.get(edge.source, "")
            tgt_cat = categories.get(edge.target, "")

            # Determine weight based on transition type
            if src_cat == "construction" and tgt_cat == "local_search":
                new_weight = 0.90
            elif src_cat == "local_search" and tgt_cat == "local_search":
                # Same operator = redundant, different = complementary
                if edge.source == edge.target:
                    new_weight = 0.30  # Self-loop (shouldn't exist, but penalize)
                else:
                    new_weight = 0.70
            elif src_cat == "local_search" and tgt_cat == "perturbation":
                new_weight = 0.60  # Escape - risky but necessary
            elif src_cat == "perturbation" and tgt_cat == "local_search":
                new_weight = 0.90  # Re-optimize - always good
            elif src_cat == "perturbation" and tgt_cat == "construction":
                new_weight = 0.40  # Restart - loses progress
            elif src_cat == "perturbation" and tgt_cat == "perturbation":
                new_weight = 0.35  # Chaining perturbations - usually bad
            elif src_cat == "construction" and tgt_cat == "construction":
                new_weight = 0.25  # Redundant constructions
            else:
                new_weight = 0.50  # Default

            # Create new edge with updated weight
            new_edge = AKGEdge(
                source=edge.source,
                target=edge.target,
                edge_type=edge.edge_type,
                weight=new_weight,
            )
            edges_to_replace.append((edge_key, new_edge))

        # Replace edges
        for edge_key, new_edge in edges_to_replace:
            akg.edges[edge_key] = new_edge

        logger.info("Applied transition-type based weights")
        return akg


# Prompt template for SELECTIVE AKG construction (exclusion-based)
SELECTIVE_AKG_PROMPT = '''You are an expert in combinatorial optimization algorithms.

Given these {n_operators} operators, IDENTIFY AND EXCLUDE THE WORST ONES, then construct an Algorithmic Knowledge Graph (AKG) with the remaining operators.

OPERATORS:
{operators}

YOUR TASK:
1. EXCLUDE only the operators you consider redundant, ineffective, or problematic
2. KEEP all other operators (err on the side of keeping more)
3. Categorize the remaining operators into: construction, local_search, or perturbation
4. Define valid transitions (edges) between operators with weights (0.0-1.0)

RULES:
- construction: Creates initial solution from scratch (e.g., nearest neighbor, insertion)
- local_search: Improves solution by exploring neighborhood (e.g., 2-opt, 3-opt)
- perturbation: Escapes local optima by disrupting solution (e.g., double bridge)

- You can exclude UP TO 15 operators maximum (keep at least {min_operators})
- Only exclude operators that are truly redundant or harmful
- When in doubt, KEEP the operator
- Higher weight = better/more common transition

Respond in this exact JSON format:
{{
  "excluded_operators": ["op3", "op4", ...],
  "exclusion_reasons": "Why you excluded those specific operators",
  "selected_operators": ["op1", "op2", ...],
  "operator_categories": {{
    "op1": "category",
    "op2": "category",
    ...
  }},
  "edges": [
    {{"source": "op1", "target": "op2", "weight": 0.8, "reason": "why this transition"}},
    ...
  ],
  "reasoning": "Overall design rationale"
}}

Be conservative with exclusions - it's better to keep a mediocre operator than to exclude a useful one.'''


class SelectiveProposedAKG(BaseModel):
    """AKG structure proposed by LLM with operator selection."""
    selected_operators: list[str]
    excluded_operators: list[str] = Field(default_factory=list)
    exclusion_reasons: str = ""
    operator_categories: dict[str, str]  # operator_id -> category
    edges: list[ProposedEdge]
    reasoning: str = ""


class SelectiveAKGValidator:
    """Validates selectively proposed AKG structures."""

    VALID_CATEGORIES = {"construction", "local_search", "perturbation"}
    MIN_CONSTRUCTION = 1  # At least 1 to start
    MIN_LOCAL_SEARCH = 1  # At least 1 to improve
    MAX_EXCLUSIONS = 15   # Can exclude at most 15 operators

    def __init__(self, operator_ids: list[str]) -> None:
        self.all_operator_ids = set(operator_ids)
        self.min_operators = max(15, len(operator_ids) - self.MAX_EXCLUSIONS)

    def validate(self, proposed: SelectiveProposedAKG) -> ValidationResult:
        errors = []
        warnings = []

        selected = set(proposed.selected_operators)

        # Rule 1: Selected operators must be from valid set
        unknown = selected - self.all_operator_ids
        if unknown:
            errors.append(f"Unknown operators selected: {unknown}")

        # Rule 2: All selected operators must be categorized
        categorized = set(proposed.operator_categories.keys())
        missing = selected - categorized
        if missing:
            errors.append(f"Selected operators not categorized: {missing}")

        extra = categorized - selected
        if extra:
            warnings.append(f"Categorized operators not in selected list: {extra}")

        # Rule 3: Categories must be valid
        for op_id, category in proposed.operator_categories.items():
            if category not in self.VALID_CATEGORIES:
                errors.append(f"Invalid category '{category}' for {op_id}")

        # Rule 4: Minimum total operators
        if len(selected) < self.min_operators:
            errors.append(f"Must keep at least {self.min_operators} operators, got {len(selected)}")

        # Rule 5: Minimum operators per category
        constructions = [op for op, cat in proposed.operator_categories.items()
                        if cat == "construction" and op in selected]
        local_searches = [op for op, cat in proposed.operator_categories.items()
                         if cat == "local_search" and op in selected]

        if len(constructions) < self.MIN_CONSTRUCTION:
            errors.append(f"Need at least {self.MIN_CONSTRUCTION} construction operators, got {len(constructions)}")
        if len(local_searches) < self.MIN_LOCAL_SEARCH:
            errors.append(f"Need at least {self.MIN_LOCAL_SEARCH} local_search operators, got {len(local_searches)}")

        # Rule 6: Edges must reference selected operators
        for edge in proposed.edges:
            if edge.source not in selected:
                errors.append(f"Edge source '{edge.source}' not in selected operators")
            if edge.target not in selected:
                errors.append(f"Edge target '{edge.target}' not in selected operators")

        # Rule 7: Construction must have outgoing edges
        if not errors:
            construction_has_outgoing = False
            for edge in proposed.edges:
                if proposed.operator_categories.get(edge.source) == "construction":
                    if proposed.operator_categories.get(edge.target) != "construction":
                        construction_has_outgoing = True
                        break
            if not construction_has_outgoing and constructions:
                errors.append("Construction operators must have edges to non-construction operators")

        # Info: Report selection stats
        n_selected = len(selected)
        n_total = len(self.all_operator_ids)
        if n_selected < n_total:
            warnings.append(f"Selected {n_selected}/{n_total} operators ({100*n_selected/n_total:.0f}%)")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class SelectiveLLMAKGGenerator:
    """Generates AKG using LLM with operator selection capability."""

    def __init__(self, llm_client: Any, operators: list[OperatorInfo], max_retries: int = 3) -> None:
        self.llm_client = llm_client
        self.operators = operators
        self.max_retries = max_retries
        self.validator = SelectiveAKGValidator([op.id for op in operators])
        self.selection_stats: dict = {}

    def generate(self) -> AlgorithmicKnowledgeGraph | None:
        """Generate AKG using LLM with operator selection."""
        op_desc = "\n".join([
            f"- {op.id}: {op.description}"
            for op in self.operators
        ])

        min_operators = max(20, len(self.operators) - 10)  # Keep at least 20, or exclude max 10

        prompt = SELECTIVE_AKG_PROMPT.format(
            n_operators=len(self.operators),
            operators=op_desc,
            min_operators=min_operators
        )

        for attempt in range(self.max_retries):
            logger.info(f"Selective AKG generation attempt {attempt + 1}/{self.max_retries}")

            response = self.llm_client.query(
                prompt,
                temperature=0.0,
                agent_name="SelectiveAKGGenerator",
                context={"stage": "selective_akg"},
            )
            proposed = self._parse_response(response.content)

            if proposed is None:
                logger.warning("Failed to parse LLM response")
                continue

            result = self.validator.validate(proposed)

            if result.errors:
                logger.warning(f"Selective validation errors: {result.errors}")

            if result.valid:
                if result.warnings:
                    logger.warning(f"AKG valid with warnings: {result.warnings}")

                # Store selection stats
                self.selection_stats = {
                    "total_operators": len(self.operators),
                    "selected": len(proposed.selected_operators),
                    "excluded": len(proposed.excluded_operators),
                    "exclusion_reasons": proposed.exclusion_reasons,
                }
                logger.info(f"Selected {self.selection_stats['selected']}/{self.selection_stats['total_operators']} operators")

                return self._build_akg(proposed)

            # Build correction prompt
            errors_str = "\n".join(f"- ERROR: {e}" for e in result.errors)
            warnings_str = "\n".join(f"- WARNING: {w}" for w in result.warnings)

            prompt = AKG_CORRECTION_PROMPT.format(
                errors=errors_str,
                warnings=warnings_str if result.warnings else "None",
                previous=response.content
            )

        logger.error("Selective AKG generation failed after max retries")
        return None

    def _parse_response(self, content: str) -> SelectiveProposedAKG | None:
        """Parse LLM response into SelectiveProposedAKG."""
        import json
        import re

        try:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())

                edges = []
                for e in data.get("edges", []):
                    edges.append(ProposedEdge(
                        source=e["source"],
                        target=e["target"],
                        weight=float(e.get("weight", 0.5)),
                        reason=e.get("reason", "")
                    ))

                # If selected_operators not provided, infer from operator_categories
                selected = data.get("selected_operators")
                if not selected:
                    selected = list(data.get("operator_categories", {}).keys())

                return SelectiveProposedAKG(
                    selected_operators=selected,
                    excluded_operators=data.get("excluded_operators", []),
                    exclusion_reasons=data.get("exclusion_reasons", ""),
                    operator_categories=data.get("operator_categories", {}),
                    edges=edges,
                    reasoning=data.get("reasoning", "")
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse selective response: {e}")
            logger.debug(f"Content was: {content[:500]}")

        return None

    def _build_akg(self, proposed: SelectiveProposedAKG) -> AlgorithmicKnowledgeGraph:
        """Build actual AKG from validated proposal (only selected operators)."""
        akg = AlgorithmicKnowledgeGraph()

        cat_map = {
            "construction": OperatorCategory.CONSTRUCTION,
            "local_search": OperatorCategory.LOCAL_SEARCH,
            "perturbation": OperatorCategory.PERTURBATION,
        }

        selected_set = set(proposed.selected_operators)

        # Add only selected operator nodes
        for op in self.operators:
            if op.id in selected_set:
                category = cat_map.get(
                    proposed.operator_categories.get(op.id, "local_search"),
                    OperatorCategory.LOCAL_SEARCH
                )
                node = OperatorNode(
                    id=op.id,
                    name=op.name,
                    description=op.description,
                    category=category,
                )
                akg.add_node(node)

        # Add edges
        for edge in proposed.edges:
            if edge.source in selected_set and edge.target in selected_set:
                akg.add_edge(AKGEdge(
                    source=edge.source,
                    target=edge.target,
                    edge_type=EdgeType.SEQUENTIAL,
                    weight=edge.weight,
                ))

        return akg


# =============================================================================
# META-GRAPH GENERATION (Abstract Roles)
# =============================================================================

# Base description of abstract roles (shared by all prompts)
_ROLES_DESCRIPTION = '''## ABSTRACT ROLES (11 total):

### CONSTRUCTION (build initial solutions)
- const_greedy: Greedy construction (e.g., Nearest Neighbor, SPT dispatch)
- const_insertion: Insertion-based construction (e.g., Cheapest Insertion)
- const_savings: Savings/merging construction (e.g., Clarke-Wright)
- const_random: Random construction for diversity

### LOCAL SEARCH (improve solutions)
- ls_intensify_small: Simple, fast moves (e.g., 2-opt, swap)
- ls_intensify_medium: More complex moves (e.g., 3-opt, or-opt)
- ls_intensify_large: Expensive thorough moves (e.g., Lin-Kernighan)
- ls_chain: Chained/VND exploration of multiple neighborhoods

### PERTURBATION (escape local optima)
- pert_escape_small: Mild perturbation (e.g., double bridge)
- pert_escape_large: Strong perturbation (e.g., ruin-recreate)
- pert_adaptive: History-based adaptive perturbation'''

_CONDITIONS_DESCRIPTION = '''## CONDITIONAL TRANSITIONS (Level 3):
Some edges should have CONDITIONS that control WHEN to take them:
- "stagnation": After N generations without improvement
- "diversity_low": When population diversity drops below threshold

Format for conditional edges:
["source_role", "target_role", weight, {{"when": "condition", "threshold": value, "boost": multiplier}}]'''

_RESPONSE_FORMAT = '''## RESPONSE FORMAT:
Return ONLY valid JSON, no markdown, no comments, no ellipsis (...).

{{
  "name": "your_algorithm_name",
  "roles": ["const_greedy", "ls_intensify_small", "pert_escape_small"],
  "edges": [
    ["const_greedy", "ls_intensify_small", 0.90],
    ["ls_intensify_small", "pert_escape_small", 0.45, {{"when": "stagnation", "threshold": 3, "boost": 2.0}}],
    ["pert_escape_small", "ls_intensify_small", 0.90]
  ],
  "reasoning": "Your explanation here"
}}

IMPORTANT:
- Do NOT use "..." or ellipsis in the JSON
- Do NOT wrap in markdown code blocks
- Return COMPLETE JSON with ALL edges'''


# =============================================================================
# MetaGraph prompt (Hybrid-style with 8+ roles)
# =============================================================================
_META_GRAPH_PROMPT_HYBRID_TEMPLATE = '''Design a SPARSE MetaGraph for <<DOMAIN>> optimization.

ROLES (pick 5-7 total):
CONST (pick 1-2): const_greedy, const_insertion, const_savings, const_random
LS (pick 2-3): ls_intensify_small, ls_intensify_medium, ls_intensify_large, ls_chain
PERT (pick 1-2): pert_escape_small, pert_escape_large, pert_adaptive

EDGE RULES:
- ALLOWED: const→ls, ls→ls, ls→pert, pert→ls
- FORBIDDEN: const→const, const→pert, pert→const, pert→pert
- MAX EDGES: 12-18 total (sparse graph, not fully connected!)
- Each role should have 2-4 outgoing edges, not more

EDGE FORMAT: {"source": "X", "target": "Y", "weight": 0.1-0.9}

WEIGHT GUIDELINES:
- const→ls: 0.7-0.9 (primary entry to intensification)
- ls→pert: 0.5-0.7 (encourage escape from local optima)
- pert→ls: 0.7-0.9 (return to intensification after escape)
- ls→ls: 0.2-0.4 (LOW weight - avoid getting stuck in LS loops)

IMPORTANT: Create a SPARSE graph. Do NOT connect every role to every other role.

Seed: <<SEED>>

Return JSON: {"name": "...", "roles": [...], "edges": [...]}'''


def _get_hybrid_prompt() -> str:
    """Get Hybrid prompt with random seed for variability."""
    import random
    seed = random.randint(1000, 9999)
    return _META_GRAPH_PROMPT_HYBRID_TEMPLATE.replace("<<SEED>>", str(seed))


# For backwards compatibility
META_GRAPH_PROMPT_HYBRID = _META_GRAPH_PROMPT_HYBRID_TEMPLATE.replace("<<SEED>>", "42")


# Default prompt (Hybrid)
META_GRAPH_PROMPT = META_GRAPH_PROMPT_HYBRID


# JSON Schema for structured output (Ollama format parameter)
METAGRAPH_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "roles": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "const_greedy", "const_insertion", "const_savings", "const_random",
                    "ls_intensify_small", "ls_intensify_medium", "ls_intensify_large", "ls_chain",
                    "pert_escape_small", "pert_escape_large", "pert_adaptive"
                ]
            },
            "minItems": 5,
            "maxItems": 7
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "weight": {"type": "number", "minimum": 0.1, "maximum": 0.9}
                },
                "required": ["source", "target", "weight"]
            },
            "minItems": 10,
            "maxItems": 18
        }
    },
    "required": ["name", "roles", "edges"]
}


class MetaGraphGenerator:
    """Generates MetaGraph using LLM with abstract roles.

    Unlike LLMAKGGenerator which uses concrete operators, this generates
    a domain-agnostic graph using abstract roles that can be instantiated
    for any domain.

    The generator creates a rich hybrid-style graph with multiple constructions,
    full LS ladder, and multiple perturbations for MMAS to explore diverse paths.
    """

    def __init__(
        self,
        llm_client: Any,
        max_retries: int = 3,
        use_conditions: bool = True,
        domain: str = "combinatorial optimization",
    ) -> None:
        """Initialize generator.

        Args:
            llm_client: LLM client for queries
            max_retries: Maximum correction attempts
            use_conditions: If True, prompt LLM to assign conditions (Level 3)
            domain: Problem domain (e.g., "tsp", "jssp", "vrp")
        """
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.use_conditions = use_conditions
        self.domain = domain.upper()

    def _get_prompt(self) -> str:
        """Get prompt with random seed for variability."""
        prompt = _get_hybrid_prompt()
        prompt = prompt.replace("<<DOMAIN>>", self.domain)
        return prompt

    def generate(self) -> "MetaGraph | None":
        """Generate MetaGraph using LLM.

        Returns:
            MetaGraph or None if generation failed
        """
        from .meta_graph import MetaGraph, MetaEdge
        from .roles import AbstractRole
        from .conditions import EdgeCondition, ConditionType

        logger.info(f"Generating MetaGraph with LLM (domain={self.domain})")

        for attempt in range(self.max_retries):
            logger.info(f"MetaGraph generation attempt {attempt + 1}/{self.max_retries}")

            prompt = self._get_prompt()  # Fresh prompt with new seed each attempt
            response = self.llm_client.query(
                prompt,
                temperature=0.0,  # Deterministic MetaGraph generation
                json_schema=METAGRAPH_JSON_SCHEMA,
                use_cache=False,  # Always fresh generation due to seed
                agent_name="MetaGraphGenerator",
                context={"domain": self.domain},
            )
            result = self._parse_response(response.content)

            if result is None:
                logger.warning("Failed to parse MetaGraph response")
                continue

            mg, warnings = result
            if warnings:
                logger.warning(f"MetaGraph warnings: {warnings}")

            # Validate structure
            validation_warnings = mg.validate_transitions()
            if validation_warnings:
                logger.warning(f"Transition warnings: {validation_warnings}")

            # Ensure construction roles exist
            constructions = mg.get_construction_roles()
            if not constructions:
                logger.warning("No construction roles, adding const_greedy")
                mg.add_role(AbstractRole.CONST_GREEDY)

            logger.info(f"MetaGraph generated: {len(mg.nodes)} roles, {len(mg.edges)} edges")
            return mg

        logger.error("MetaGraph generation failed after max retries")
        return None

    def _extract_balanced_json(self, text: str) -> str | None:
        """Extract a balanced JSON object from text.

        Finds the first '{' and matches it with its corresponding '}'.
        Handles nested objects and arrays correctly.
        """
        start = text.find('{')
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

        return None  # Unbalanced

    def _remove_unreachable_nodes(self, mg: "MetaGraph") -> "tuple[MetaGraph, list[str]]":
        """Remove nodes that are not reachable from any construction role.

        Args:
            mg: MetaGraph to clean

        Returns:
            Tuple of (cleaned MetaGraph, list of removed node names)
        """
        from .meta_graph import MetaGraph, MetaEdge
        from .roles import AbstractRole

        # Find all construction roles (entry points)
        construction_roles = set()
        for role_value, node in mg.nodes.items():
            if node.is_construction():
                construction_roles.add(role_value)

        if not construction_roles:
            # No construction roles - can't determine reachability
            return mg, []

        # BFS to find all reachable nodes from construction
        reachable = set(construction_roles)
        frontier = list(construction_roles)

        while frontier:
            current = frontier.pop(0)
            for (src, tgt) in mg.edges:
                if src == current and tgt not in reachable:
                    reachable.add(tgt)
                    frontier.append(tgt)

        # Find unreachable nodes
        all_nodes = set(mg.nodes.keys())
        unreachable = all_nodes - reachable

        if not unreachable:
            return mg, []

        # Create new MetaGraph without unreachable nodes
        new_mg = MetaGraph(
            name=mg.name,
            description=mg.description,
            llm_reasoning=mg.llm_reasoning,
        )

        # Copy only reachable nodes and their edges
        for (src, tgt), edge in mg.edges.items():
            if src in reachable and tgt in reachable:
                new_mg.add_edge(edge)

        logger.info(f"Removed {len(unreachable)} unreachable nodes: {unreachable}")
        return new_mg, list(unreachable)

    def _parse_response(self, content: str) -> "tuple[MetaGraph, list[str]] | None":
        """Parse LLM response into MetaGraph.

        Returns:
            Tuple of (MetaGraph, warnings) or None if parsing failed
        """
        import json
        import re
        from .meta_graph import MetaGraph, MetaEdge
        from .roles import AbstractRole
        from .conditions import EdgeCondition, ConditionType

        warnings = []

        # Log raw response for debugging
        logger.debug(f"Raw LLM response:\n{content[:500]}...")

        # Save full JSON to file for debugging L3 conditions
        import os
        debug_dir = "logs"
        os.makedirs(debug_dir, exist_ok=True)
        with open(f"{debug_dir}/last_metagraph_response.json", "w") as f:
            f.write(content)

        try:
            # Clean up common LLM issues
            cleaned = content

            # Remove markdown code blocks
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)

            # Remove ellipsis that LLMs love to add
            cleaned = re.sub(r',\s*\.\.\.', '', cleaned)
            cleaned = re.sub(r'\.\.\.,?\s*', '', cleaned)

            # Remove JavaScript-style comments that GPT sometimes adds
            cleaned = re.sub(r'//[^\n]*\n', '\n', cleaned)

            # Find JSON object with balanced braces
            json_str = self._extract_balanced_json(cleaned)
            if not json_str:
                logger.warning("No valid JSON object found in response")
                logger.debug(f"Cleaned content: {cleaned[:300]}")
                return None

            logger.debug(f"Extracted JSON: {json_str[:300]}...")

            data = json.loads(json_str)

            mg = MetaGraph(
                name=data.get("name", "llm_meta_graph"),
                description=data.get("reasoning", ""),
                llm_reasoning=data.get("reasoning", ""),
            )

            # Add roles mentioned in the response
            roles_mentioned = set(data.get("roles", []))

            # Parse edges (supports both array and object format)
            for edge_data in data.get("edges", []):
                # Handle object format: {"source": ..., "target": ..., "weight": ...}
                if isinstance(edge_data, dict):
                    source_str = edge_data.get("source", "")
                    target_str = edge_data.get("target", "")
                    weight = float(edge_data.get("weight", 0.5))
                    cond_data = edge_data.get("condition")
                # Handle array format: ["source", "target", weight, {condition}]
                elif isinstance(edge_data, list) and len(edge_data) >= 3:
                    source_str = edge_data[0]
                    target_str = edge_data[1]
                    weight = float(edge_data[2])
                    cond_data = edge_data[3] if len(edge_data) > 3 else None
                else:
                    warnings.append(f"Invalid edge format: {edge_data}")
                    continue

                # Validate role values
                try:
                    source = AbstractRole(source_str)
                    target = AbstractRole(target_str)
                except ValueError as e:
                    warnings.append(f"Unknown role: {e}")
                    continue

                # Filter forbidden edges
                # 1. perturbation → construction (destroys progress)
                if source_str.startswith("pert_") and target_str.startswith("const_"):
                    warnings.append(f"Forbidden edge filtered: {source_str} → {target_str} (pert→const)")
                    continue
                # 2. construction → perturbation (perturb what? need LS first)
                if source_str.startswith("const_") and target_str.startswith("pert_"):
                    warnings.append(f"Forbidden edge filtered: {source_str} → {target_str} (const→pert)")
                    continue

                # Track roles used
                roles_mentioned.add(source_str)
                roles_mentioned.add(target_str)

                # Parse conditions
                conditions = []
                condition_boost = 2.0
                if cond_data and isinstance(cond_data, dict):
                    cond_type_str = cond_data.get("when", "")
                    threshold = float(cond_data.get("threshold", 0))
                    # Ensure boost >= 1.0 (clamp invalid values)
                    raw_boost = float(cond_data.get("boost", 2.0))
                    condition_boost = max(1.0, raw_boost)

                    # Map condition type
                    cond_type_map = {
                        "stagnation": ConditionType.STAGNATION,
                        "diversity_low": ConditionType.DIVERSITY_LOW,
                        "gap_to_best": ConditionType.GAP_TO_BEST,
                        "improvement_rate": ConditionType.IMPROVEMENT_RATE,
                    }
                    if cond_type_str in cond_type_map:
                        conditions.append(EdgeCondition(
                            condition_type=cond_type_map[cond_type_str],
                            threshold=threshold,
                            reason=f"LLM: {cond_type_str} >= {threshold}",
                        ))

                mg.add_edge(MetaEdge(
                    source=source,
                    target=target,
                    weight=weight,
                    conditions=conditions,
                    condition_boost=condition_boost,
                    reasoning=data.get("reasoning", ""),
                ))

            # Remove unreachable nodes (nodes with no path from construction)
            mg, removed = self._remove_unreachable_nodes(mg)
            if removed:
                warnings.append(f"Removed unreachable nodes: {removed}")

            return mg, warnings

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse MetaGraph response: {e}")
            return None


class RandomAKGGenerator:
    """Generates random AKG structure for ablation baseline."""

    def __init__(self, operators: list[OperatorInfo], edge_probability: float = 0.3) -> None:
        """Initialize generator.

        Args:
            operators: List of operators
            edge_probability: Probability of creating edge between any two operators
        """
        self.operators = operators
        self.edge_probability = edge_probability

    def generate(self) -> AlgorithmicKnowledgeGraph:
        """Generate random but structurally valid AKG."""
        import random

        akg = AlgorithmicKnowledgeGraph()

        categories = list(OperatorCategory)

        # Randomly assign categories (ensuring at least one construction)
        op_categories = {}
        for i, op in enumerate(self.operators):
            if i == 0:
                # First operator is always construction
                cat = OperatorCategory.CONSTRUCTION
            else:
                cat = random.choice(categories)
            op_categories[op.id] = cat

        # Ensure at least one construction
        constructions = [op_id for op_id, cat in op_categories.items()
                        if cat == OperatorCategory.CONSTRUCTION]
        if not constructions:
            # Make a random one construction
            random_op = random.choice(list(op_categories.keys()))
            op_categories[random_op] = OperatorCategory.CONSTRUCTION
            constructions = [random_op]

        # Add nodes
        for op in self.operators:
            node = OperatorNode(
                id=op.id,
                name=op.name,
                description=op.description,
                category=op_categories[op.id],
            )
            akg.add_node(node)

        # Add random edges
        non_constructions = [op.id for op in self.operators
                           if op_categories[op.id] != OperatorCategory.CONSTRUCTION]

        # Ensure construction -> others edges exist
        for c_op in constructions:
            for other in non_constructions:
                if random.random() < 0.5:  # Higher probability for construction edges
                    akg.add_edge(AKGEdge(
                        source=c_op,
                        target=other,
                        edge_type=EdgeType.SEQUENTIAL,
                        weight=random.random(),
                    ))

        # Random edges between non-construction operators
        for op1 in self.operators:
            for op2 in self.operators:
                if op1.id == op2.id:
                    continue
                if op_categories[op2.id] == OperatorCategory.CONSTRUCTION:
                    continue  # Don't create edges TO construction
                if random.random() < self.edge_probability:
                    akg.add_edge(AKGEdge(
                        source=op1.id,
                        target=op2.id,
                        edge_type=EdgeType.SEQUENTIAL,
                        weight=random.random(),
                    ))

        return akg
