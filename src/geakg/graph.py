"""Algorithmic Knowledge Graph implementation."""

from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from src.geakg.nodes import (
    AKGEdge,
    AKGNode,
    DataStructureNode,
    EdgeType,
    NodeType,
    OperatorCategory,
    OperatorNode,
    PropertyNode,
)


class Trajectory(BaseModel):
    """A sequence of operators forming an algorithm."""

    id: str
    operators: list[str]  # List of operator node IDs
    problem_type: str  # tsp, jssp, vrp
    problem_size: int
    fitness: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class AlgorithmicKnowledgeGraph:
    """The Algorithmic Knowledge Graph (AKG).

    A directed graph where:
    - Nodes represent operators, data structures, and problem properties
    - Edges represent valid transitions and compatibility relationships

    The AKG enables:
    - Structured navigation of the algorithm design space
    - Constraint-based validation of LLM proposals
    - Experience-based edge weight updates
    """

    def __init__(self) -> None:
        """Initialize an empty AKG."""
        self.graph = nx.DiGraph()
        self.nodes: dict[str, AKGNode] = {}
        self.edges: dict[tuple[str, str], AKGEdge] = {}
        self.trajectory_history: list[Trajectory] = []

    def add_node(self, node: AKGNode) -> None:
        """Add a node to the graph.

        Args:
            node: The node to add (OperatorNode, DataStructureNode, or PropertyNode)
        """
        self.nodes[node.id] = node
        self.graph.add_node(
            node.id,
            name=node.name,
            node_type=node.node_type.value,
            description=node.description,
        )

        # Add operator-specific attributes
        if isinstance(node, OperatorNode):
            self.graph.nodes[node.id]["category"] = node.category.value
            self.graph.nodes[node.id]["preconditions"] = node.preconditions
            self.graph.nodes[node.id]["effects"] = node.effects

    def add_edge(self, edge: AKGEdge) -> None:
        """Add an edge to the graph.

        Args:
            edge: The edge to add
        """
        if edge.source not in self.nodes:
            raise ValueError(f"Source node '{edge.source}' not found in graph")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node '{edge.target}' not found in graph")

        self.edges[(edge.source, edge.target)] = edge
        self.graph.add_edge(
            edge.source,
            edge.target,
            edge_type=edge.edge_type.value,
            weight=edge.weight,
        )

    def get_node(self, node_id: str) -> AKGNode | None:
        """Get a node by ID.

        Args:
            node_id: The node ID

        Returns:
            The node, or None if not found
        """
        return self.nodes.get(node_id)

    def get_operator_nodes(self) -> list[OperatorNode]:
        """Get all operator nodes.

        Returns:
            List of operator nodes
        """
        return [n for n in self.nodes.values() if isinstance(n, OperatorNode)]

    def get_operators_by_category(self, category: OperatorCategory) -> list[OperatorNode]:
        """Get operators of a specific category.

        Args:
            category: The operator category

        Returns:
            List of matching operator nodes
        """
        return [
            n
            for n in self.nodes.values()
            if isinstance(n, OperatorNode) and n.category == category
        ]

    def get_valid_transitions(
        self,
        current_node_id: str | None,
        edge_type: EdgeType = EdgeType.SEQUENTIAL,
    ) -> list[str]:
        """Get valid next nodes from the current node.

        Args:
            current_node_id: Current node ID, or None for start
            edge_type: Type of edge to follow

        Returns:
            List of valid target node IDs
        """
        if current_node_id is None:
            # Return construction operators as valid starting points
            return [
                n.id
                for n in self.get_operators_by_category(OperatorCategory.CONSTRUCTION)
            ]

        # Get successors with matching edge type
        valid = []
        for successor in self.graph.successors(current_node_id):
            edge_data = self.graph.get_edge_data(current_node_id, successor)
            if edge_data and edge_data.get("edge_type") == edge_type.value:
                valid.append(successor)

        return valid

    def get_valid_operations_mask(
        self, current_operators: list[str]
    ) -> dict[str, bool]:
        """Generate a mask of valid next operations.

        Args:
            current_operators: List of operators in current algorithm

        Returns:
            Dict mapping operator IDs to validity
        """
        last_op = current_operators[-1] if current_operators else None
        valid_next = set(self.get_valid_transitions(last_op))

        return {
            op.id: op.id in valid_next for op in self.get_operator_nodes()
        }

    def update_edge_weight(
        self,
        source: str,
        target: str,
        delta: float,
        learning_rate: float = 0.1,
    ) -> None:
        """Update edge weight based on experience.

        Args:
            source: Source node ID
            target: Target node ID
            delta: Fitness improvement (positive = good transition)
            learning_rate: Learning rate for weight update
        """
        key = (source, target)
        if key not in self.edges:
            return

        edge = self.edges[key]
        new_weight = edge.weight + learning_rate * delta
        new_weight = max(0.0, min(1.0, new_weight))  # Clamp to [0, 1]

        # Create new edge with updated weight (edges are frozen)
        updated_edge = AKGEdge(
            source=edge.source,
            target=edge.target,
            edge_type=edge.edge_type,
            weight=new_weight,
            metadata=edge.metadata,
        )
        self.edges[key] = updated_edge
        self.graph.edges[source, target]["weight"] = new_weight

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Record a successful trajectory.

        Args:
            trajectory: The trajectory to record
        """
        self.trajectory_history.append(trajectory)

        # Update edge weights based on fitness
        if len(trajectory.operators) < 2:
            return

        # Normalize fitness to [-1, 1] for weight updates
        # (Assumes higher fitness is better, normalized elsewhere)
        normalized_delta = trajectory.fitness / 100.0  # Simple normalization

        for i in range(len(trajectory.operators) - 1):
            self.update_edge_weight(
                trajectory.operators[i],
                trajectory.operators[i + 1],
                normalized_delta,
            )

    def get_trajectories_by_problem(
        self,
        problem_type: str,
        min_fitness: float | None = None,
        max_count: int = 10,
    ) -> list[Trajectory]:
        """Get recorded trajectories for a problem type.

        Args:
            problem_type: The problem type (tsp, jssp, vrp)
            min_fitness: Minimum fitness threshold
            max_count: Maximum number of trajectories to return

        Returns:
            List of matching trajectories, sorted by fitness
        """
        matching = [
            t
            for t in self.trajectory_history
            if t.problem_type == problem_type
            and (min_fitness is None or t.fitness >= min_fitness)
        ]

        # Sort by fitness (descending) and return top max_count
        matching.sort(key=lambda t: t.fitness, reverse=True)
        return matching[:max_count]

    def to_graphml(self, path: Path) -> None:
        """Save graph to GraphML format.

        Args:
            path: Output file path
        """
        nx.write_graphml(self.graph, str(path))

    @classmethod
    def from_graphml(cls, path: Path) -> "AlgorithmicKnowledgeGraph":
        """Load graph from GraphML format.

        Args:
            path: Input file path

        Returns:
            Loaded AKG instance
        """
        akg = cls()
        akg.graph = nx.read_graphml(str(path))

        # Reconstruct node objects from graph attributes
        for node_id, data in akg.graph.nodes(data=True):
            node_type = NodeType(data.get("node_type", "operator"))

            if node_type == NodeType.OPERATOR:
                node = OperatorNode(
                    id=node_id,
                    name=data.get("name", node_id),
                    description=data.get("description", ""),
                    category=OperatorCategory(data.get("category", "construction")),
                    preconditions=data.get("preconditions", []),
                    effects=data.get("effects", []),
                )
            elif node_type == NodeType.DATA_STRUCTURE:
                node = DataStructureNode(
                    id=node_id,
                    name=data.get("name", node_id),
                    description=data.get("description", ""),
                )
            else:
                node = PropertyNode(
                    id=node_id,
                    name=data.get("name", node_id),
                    description=data.get("description", ""),
                )

            akg.nodes[node_id] = node

        # Reconstruct edge objects
        for source, target, data in akg.graph.edges(data=True):
            edge = AKGEdge(
                source=source,
                target=target,
                edge_type=EdgeType(data.get("edge_type", "sequential")),
                weight=float(data.get("weight", 0.5)),
            )
            akg.edges[(source, target)] = edge

        return akg

    def __len__(self) -> int:
        """Return number of nodes in the graph."""
        return len(self.nodes)

    def __repr__(self) -> str:
        """String representation."""
        n_operators = len(self.get_operator_nodes())
        n_edges = len(self.edges)
        return f"AKG(operators={n_operators}, edges={n_edges})"
