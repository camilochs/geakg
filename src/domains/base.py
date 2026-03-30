"""Base classes for optimization problem domains."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

# Type variables for generic domain implementation
InstanceT = TypeVar("InstanceT", bound="ProblemInstance")
SolutionT = TypeVar("SolutionT", bound="Solution")


class ProblemInstance(BaseModel):
    """Base class for problem instances."""

    name: str
    dimension: int = Field(gt=0)


class Solution(BaseModel):
    """Base class for solutions."""

    cost: float = Field(ge=0)
    is_valid: bool = True


class ProblemFeatures(BaseModel):
    """Features extracted from a problem instance for retrieval."""

    dimension: int
    # Subclasses add domain-specific features


class OptimizationDomain(ABC, Generic[InstanceT, SolutionT]):
    """Abstract base class for optimization problem domains.

    Each domain (TSP, JSSP, VRP) implements this interface to provide:
    - Instance loading and parsing
    - Solution evaluation
    - Solution validation
    - Feature extraction for retrieval
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Domain name (e.g., 'tsp', 'jssp', 'vrp')."""
        ...

    @abstractmethod
    def load_instance(self, path: Path) -> InstanceT:
        """Load a problem instance from file.

        Args:
            path: Path to instance file

        Returns:
            Loaded problem instance
        """
        ...

    @abstractmethod
    def evaluate_solution(self, solution: SolutionT, instance: InstanceT) -> float:
        """Evaluate a solution's cost/fitness.

        Args:
            solution: The solution to evaluate
            instance: The problem instance

        Returns:
            Cost/fitness value (lower is better for minimization)
        """
        ...

    @abstractmethod
    def validate_solution(self, solution: SolutionT, instance: InstanceT) -> bool:
        """Check if a solution is valid.

        Args:
            solution: The solution to validate
            instance: The problem instance

        Returns:
            True if solution is valid, False otherwise
        """
        ...

    @abstractmethod
    def get_features(self, instance: InstanceT) -> ProblemFeatures:
        """Extract features from problem instance for retrieval.

        Args:
            instance: The problem instance

        Returns:
            Extracted features
        """
        ...

    @abstractmethod
    def random_solution(self, instance: InstanceT) -> SolutionT:
        """Generate a random valid solution.

        Args:
            instance: The problem instance

        Returns:
            Random valid solution
        """
        ...

    def compare_solutions(
        self, sol1: SolutionT, sol2: SolutionT, instance: InstanceT
    ) -> int:
        """Compare two solutions.

        Args:
            sol1: First solution
            sol2: Second solution
            instance: Problem instance

        Returns:
            -1 if sol1 is better, 1 if sol2 is better, 0 if equal
        """
        cost1 = self.evaluate_solution(sol1, instance)
        cost2 = self.evaluate_solution(sol2, instance)

        if cost1 < cost2:
            return -1
        elif cost1 > cost2:
            return 1
        return 0


class DomainConfig(ABC):
    """Configuration for operator synthesis and validation.

    Each domain must implement this to enable:
    - Code generation with domain-specific prompts
    - Test case generation
    - Solution validation and fitness evaluation

    This allows adding new domains without modifying agents.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Domain identifier (e.g., 'tsp', 'jssp')."""
        ...

    @property
    @abstractmethod
    def generator_prompt(self) -> str:
        """Prompt template for LLM code generation (from src.prompts)."""
        ...

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Human-readable domain name for prompt (e.g., 'TSP', 'JSSP')."""
        ...

    @property
    @abstractmethod
    def domain_guidance(self) -> str:
        """Optional domain-specific guidance to include in prompt."""
        ...

    @property
    @abstractmethod
    def function_signature(self) -> str:
        """Function signature for generated operators.

        Example for TSP:
        def operator_name(tour: list[int], distance_matrix: list[list[float]]) -> list[int]:
        """
        ...

    @property
    @abstractmethod
    def min_function_args(self) -> int:
        """Minimum number of arguments for operator function."""
        ...

    @abstractmethod
    def generate_test_case(self, size: int) -> dict[str, Any]:
        """Generate a random test case for validation.

        Args:
            size: Instance size (e.g., number of cities for TSP)

        Returns:
            Dictionary with input_solution structure for this domain
        """
        ...

    @abstractmethod
    def validate_solution(self, result: Any, original: dict[str, Any]) -> bool:
        """Validate that operator output is a valid solution.

        Args:
            result: Output from the operator
            original: Original input solution dict

        Returns:
            True if result is valid, False otherwise
        """
        ...

    @abstractmethod
    def evaluate_fitness(self, solution: Any, instance: dict[str, Any]) -> float:
        """Calculate fitness/cost of a solution.

        Args:
            solution: The solution to evaluate
            instance: Instance data dict

        Returns:
            Fitness value (lower is better for minimization)
        """
        ...

    @abstractmethod
    def extract_operator_args(self, input_solution: dict[str, Any]) -> tuple:
        """Extract arguments to pass to operator function.

        DEPRECATED: Use create_context() for agnostic operators.

        Args:
            input_solution: Input solution dictionary

        Returns:
            Tuple of arguments for operator function
        """
        ...

    @abstractmethod
    def create_context(self, instance_data: dict[str, Any]) -> Any:
        """Create a DomainContext for agnostic operator execution.

        The context implements the 5-method protocol (cost, delta, neighbors,
        evaluate, valid) that allows operators to work across domains.

        Args:
            instance_data: Instance data dictionary

        Returns:
            DomainContext instance for this domain
        """
        ...
