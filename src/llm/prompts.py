"""Prompt templates for LLM reasoning."""

from typing import Any

from pydantic import BaseModel, Field


class DecisionOutcome(BaseModel):
    """Record of a past decision and its outcome."""

    operator: str
    fitness_before: float
    fitness_after: float
    improved: bool

    @property
    def change_percent(self) -> float:
        """Calculate percentage change in fitness (negative = improvement for minimization)."""
        if self.fitness_before == 0:
            return 0.0
        return 100 * (self.fitness_after - self.fitness_before) / self.fitness_before


class SearchState(BaseModel):
    """Current state of the evolutionary search."""

    generation: int = 0
    max_generations: int = 50
    best_fitness: float = float("inf")
    generations_without_improvement: int = 0
    last_operator_used: str | None = None
    last_operator_improved: bool = False
    population_diversity: float = 0.5  # 0-1 scale
    evaluations_used: int = 0
    max_evaluations: int = 100
    # Decision history for learning
    recent_decisions: list[DecisionOutcome] = Field(default_factory=list)


class PromptContext(BaseModel):
    """Context for building prompts."""

    problem_type: str = "tsp"
    problem_size: int = 0
    problem_description: str = ""
    current_operators: list[str] = Field(default_factory=list)
    valid_operations: list[tuple[str, float, str]] = Field(default_factory=list)  # (id, score, description)
    similar_trajectories: list[dict[str, Any]] = Field(default_factory=list)
    previous_feedback: str | None = None
    iteration: int = 0
    # New: search state for informed decisions
    search_state: SearchState | None = None


SYSTEM_PROMPT = """You are an expert algorithm designer specializing in combinatorial optimization.
Your task is to propose the next operator for building an optimization algorithm.

RULES:
1. Only propose operators from the valid operations list
2. Consider the problem characteristics when choosing
3. Build on the current algorithm structure
4. Provide reasoning for your choice

RESPONSE FORMAT:
You must respond with exactly this format:
Operation: <operator_id>
Reasoning: <brief explanation>

Do not include any other text or formatting."""


OPERATOR_SELECTION_TEMPLATE = """PROBLEM: {problem_type} with {problem_size} elements
{problem_description}

CURRENT ALGORITHM:
{current_algorithm}
{search_state_section}
VALID OPERATIONS (sorted by preference):
{valid_operations}

{similar_trajectories}

{feedback_section}

Propose the next operation to add to the algorithm.
Remember: respond with exactly "Operation: <id>" followed by "Reasoning: <explanation>"."""


SEARCH_STATE_TEMPLATE = """
STATUS: {status_summary}
{decision_history}
STRATEGY GUIDE:
- If STUCK/STAGNANT: Use perturbation operators (double_bridge, ruin_recreate, random_segment_shuffle) to escape local optima
- If IMPROVING: Continue with similar operators or try refinement (two_opt, three_opt, or_opt)
- If LOW DIVERSITY: Use exploratory operators (large_neighborhood_search, adaptive_mutation) to diversify
- LEARN FROM HISTORY: Repeat operators that improved fitness, avoid those that didn't help

RECOMMENDATION: {hint}
"""


def _build_decision_history(decisions: list[DecisionOutcome]) -> str:
    """Build decision history section for prompt.

    Args:
        decisions: List of recent decisions (max 5)

    Returns:
        Formatted decision history string
    """
    if not decisions:
        return ""

    lines = ["\nRECENT DECISIONS (learn from these):"]
    for d in decisions[-5:]:  # Last 5 decisions
        if d.improved:
            result = f"✓ improved {abs(d.change_percent):.1f}%"
        else:
            change = d.change_percent
            if abs(change) < 0.1:
                result = "→ no change"
            else:
                result = f"✗ worsened {abs(change):.1f}%"
        lines.append(f"  - {d.operator}: {result}")

    return "\n".join(lines)


def _build_search_state_section(state: SearchState) -> str:
    """Build simplified search state section for prompt.

    Args:
        state: Current search state

    Returns:
        Formatted search state string (simplified for small LLMs)
    """
    # Simple status summary
    if state.generations_without_improvement == 0:
        status_summary = "Improving"
    elif state.generations_without_improvement < 3:
        status_summary = "Stable"
    else:
        status_summary = f"Stuck ({state.generations_without_improvement} gens)"

    # Build decision history
    decision_history = _build_decision_history(state.recent_decisions)

    # Hint based on history first, then fallback to status
    hint = None

    # Check if any operator worked well recently
    if state.recent_decisions:
        good_ops = [d.operator for d in state.recent_decisions if d.improved]
        bad_ops = [d.operator for d in state.recent_decisions if not d.improved and d.change_percent > 0.1]

        if good_ops:
            hint = f"{good_ops[-1]} worked well, try it again or similar"
        elif bad_ops and state.generations_without_improvement >= 3:
            hint = f"Avoid {bad_ops[-1]}, try perturbation operators"

    # Fallback hints based on status
    if hint is None:
        if state.generations_without_improvement >= 5:
            hint = "Use double_bridge or ruin_recreate to escape"
        elif state.population_diversity < 0.3:
            hint = "Population converging, try random_segment_shuffle"
        elif state.last_operator_improved and state.last_operator_used:
            hint = f"{state.last_operator_used} worked, try similar"
        else:
            hint = "Try two_opt or three_opt for refinement"

    return SEARCH_STATE_TEMPLATE.format(
        status_summary=status_summary,
        decision_history=decision_history,
        hint=hint,
    )


def build_operator_selection_prompt(context: PromptContext) -> str:
    """Build prompt for operator selection.

    Args:
        context: Prompt context with all required information

    Returns:
        Formatted prompt string
    """
    # Format current algorithm
    if context.current_operators:
        current_algorithm = " -> ".join(context.current_operators)
    else:
        current_algorithm = "(empty - need to start with a construction operator)"

    # Format search state section
    if context.search_state is not None:
        search_state_section = _build_search_state_section(context.search_state)
    else:
        search_state_section = ""

    # Format valid operations with scores AND descriptions for reasoning
    valid_ops_lines = []
    for item in context.valid_operations[:10]:
        if len(item) == 3:
            op, score, desc = item
            valid_ops_lines.append(f"  - {op} (score: {score:.2f}) - {desc}")
        else:
            op, score = item[0], item[1]
            valid_ops_lines.append(f"  - {op} (score: {score:.2f})")
    valid_operations = "\n".join(valid_ops_lines)

    # Format similar trajectories
    if context.similar_trajectories:
        traj_lines = ["SIMILAR SUCCESSFUL TRAJECTORIES:"]
        for i, traj in enumerate(context.similar_trajectories[:3], 1):
            ops = " -> ".join(traj.get("operators", []))
            fitness = traj.get("fitness", "N/A")
            traj_lines.append(f"  {i}. {ops} (fitness: {fitness})")
        similar_trajectories = "\n".join(traj_lines)
    else:
        similar_trajectories = ""

    # Format feedback section
    if context.previous_feedback:
        feedback_section = f"PREVIOUS ATTEMPT FEEDBACK:\n{context.previous_feedback}\n"
    else:
        feedback_section = ""

    return OPERATOR_SELECTION_TEMPLATE.format(
        problem_type=context.problem_type.upper(),
        problem_size=context.problem_size,
        problem_description=context.problem_description,
        current_algorithm=current_algorithm,
        search_state_section=search_state_section,
        valid_operations=valid_operations,
        similar_trajectories=similar_trajectories,
        feedback_section=feedback_section,
    )


def build_full_prompt(context: PromptContext) -> str:
    """Build full prompt with system instructions.

    Args:
        context: Prompt context

    Returns:
        Complete prompt string
    """
    user_prompt = build_operator_selection_prompt(context)
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}"


# Alternative prompts for different tasks

ALGORITHM_EXPLANATION_TEMPLATE = """Explain the following algorithm in simple terms:

Algorithm: {algorithm}

Provide:
1. What each step does
2. Why this combination might work well
3. Potential improvements"""


TRAJECTORY_COMPARISON_TEMPLATE = """Compare these two algorithm trajectories:

Trajectory A: {trajectory_a}
Fitness A: {fitness_a}

Trajectory B: {trajectory_b}
Fitness B: {fitness_b}

Explain why one performs better than the other."""
