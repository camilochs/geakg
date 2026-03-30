"""Context builder for LLM prompts.

Builds rich context for LLM queries by combining:
- Problem information
- Current algorithm state
- Similar trajectories from lookup
- Valid operations from constraint engine
"""

from typing import Any

from pydantic import BaseModel, Field

from src.geakg.graph import AlgorithmicKnowledgeGraph
from src.constraints.mask_generator import MaskGenerator
from src.llm.prompts import PromptContext
from src.retrieval.lookup import TrajectoryLookup, TrajectoryRecord


class ContextConfig(BaseModel):
    """Configuration for context building."""

    max_trajectories: int = Field(default=3, description="Max similar trajectories")
    max_valid_ops: int = Field(default=10, description="Max valid operations to show")
    include_reasoning: bool = Field(default=True, description="Include trajectory reasoning")


class ContextBuilder:
    """Builds context for LLM operator selection.

    Combines information from multiple sources:
    1. Problem features (type, size, description)
    2. Current algorithm state
    3. Similar successful trajectories (from lookup)
    4. Valid operations (from mask generator)
    """

    def __init__(
        self,
        akg: AlgorithmicKnowledgeGraph,
        trajectory_lookup: TrajectoryLookup,
        config: ContextConfig | None = None,
    ) -> None:
        """Initialize context builder.

        Args:
            akg: Algorithmic Knowledge Graph
            trajectory_lookup: Trajectory lookup database
            config: Configuration options
        """
        self.akg = akg
        self.lookup = trajectory_lookup
        self.config = config or ContextConfig()
        self.mask_generator = MaskGenerator(akg)

    def build_context(
        self,
        problem_type: str,
        problem_size: int,
        current_operators: list[str],
        problem_description: str = "",
        previous_feedback: str | None = None,
    ) -> PromptContext:
        """Build complete context for LLM query.

        Args:
            problem_type: Type of problem (tsp, jssp, vrp)
            problem_size: Problem dimension
            current_operators: Current algorithm operators
            problem_description: Human-readable problem description
            previous_feedback: Feedback from previous rejected proposal

        Returns:
            PromptContext ready for prompt building
        """
        # Get valid operations with scores and descriptions
        mask = self.mask_generator.generate_mask(
            current_operators,
            problem_context={"dimension": problem_size, "domain": problem_type},
        )
        # Add descriptions from AKG nodes
        valid_operations = []
        for op_id, score in mask.ranked_operators[: self.config.max_valid_ops]:
            node = self.akg.get_node(op_id)
            desc = node.description if node else ""
            valid_operations.append((op_id, score, desc))

        # Get similar trajectories
        similar_trajectories = self._get_similar_trajectories(
            problem_type, problem_size, current_operators
        )

        return PromptContext(
            problem_type=problem_type,
            problem_size=problem_size,
            problem_description=problem_description,
            current_operators=current_operators,
            valid_operations=valid_operations,
            similar_trajectories=similar_trajectories,
            previous_feedback=previous_feedback,
        )

    def _get_similar_trajectories(
        self,
        problem_type: str,
        problem_size: int,
        current_operators: list[str],
    ) -> list[dict[str, Any]]:
        """Get similar trajectories for context.

        Args:
            problem_type: Problem type
            problem_size: Problem size
            current_operators: Current operators

        Returns:
            List of trajectory dicts for prompt
        """
        # First try: lookup by type and size
        records = self.lookup.lookup(
            problem_type=problem_type,
            problem_size=problem_size,
            max_results=self.config.max_trajectories,
        )

        # If not enough, try similar operator sequences
        if len(records) < self.config.max_trajectories and current_operators:
            similar = self.lookup.lookup_similar(
                problem_type=problem_type,
                operators=current_operators,
                max_results=self.config.max_trajectories - len(records),
            )
            # Combine, avoiding duplicates
            seen_ids = {r.id for r in records}
            for s in similar:
                if s.id not in seen_ids:
                    records.append(s)

        # Convert to prompt format
        return [self._record_to_dict(r) for r in records]

    def _record_to_dict(self, record: TrajectoryRecord) -> dict[str, Any]:
        """Convert trajectory record to prompt dict.

        Args:
            record: Trajectory record

        Returns:
            Dict for prompt inclusion
        """
        return {
            "operators": record.operators,
            "fitness": record.fitness,
            "size": record.problem_size,
        }

    def add_successful_trajectory(
        self,
        operators: list[str],
        fitness: float,
        problem_type: str,
        problem_size: int,
        trajectory_id: str | None = None,
    ) -> None:
        """Add a successful trajectory to the lookup database.

        Args:
            operators: Operator sequence
            fitness: Achieved fitness
            problem_type: Problem type
            problem_size: Problem size
            trajectory_id: Optional ID (auto-generated if None)
        """
        from src.geakg.graph import Trajectory

        if trajectory_id is None:
            trajectory_id = f"traj_{self.lookup.count_trajectories() + 1:04d}"

        traj = Trajectory(
            id=trajectory_id,
            operators=operators,
            problem_type=problem_type,
            problem_size=problem_size,
            fitness=fitness,
        )

        self.lookup.add_trajectory(traj)

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about stored trajectories.

        Returns:
            Statistics dict
        """
        return {
            "total_trajectories": self.lookup.count_trajectories(),
            "tsp_trajectories": self.lookup.count_trajectories("tsp"),
            "jssp_trajectories": self.lookup.count_trajectories("jssp"),
            "vrp_trajectories": self.lookup.count_trajectories("vrp"),
        }
