"""Simple lookup-based trajectory retrieval.

Phase 1 implementation: Direct lookup by problem type and size range.
Embeddings will be added in Phase 2 only if needed.
"""

import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.geakg.graph import Trajectory


class TrajectoryRecord(BaseModel):
    """Record of a trajectory in the database."""

    id: str
    problem_type: str
    problem_size: int
    size_category: str  # small, medium, large
    operators: list[str]
    fitness: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_trajectory(cls, traj: Trajectory) -> "TrajectoryRecord":
        """Create record from trajectory.

        Args:
            traj: Trajectory object

        Returns:
            TrajectoryRecord
        """
        # Determine size category
        if traj.problem_size < 100:
            size_cat = "small"
        elif traj.problem_size < 500:
            size_cat = "medium"
        else:
            size_cat = "large"

        return cls(
            id=traj.id,
            problem_type=traj.problem_type,
            problem_size=traj.problem_size,
            size_category=size_cat,
            operators=traj.operators,
            fitness=traj.fitness,
            metadata=traj.metadata,
        )


class TrajectoryLookup:
    """Simple trajectory lookup by problem type and size.

    This is the Phase 1 implementation that uses direct indexing
    rather than embeddings. Fast and sufficient for initial experiments.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize trajectory lookup.

        Args:
            db_path: Path to SQLite database. If None, uses in-memory.
        """
        if db_path:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        else:
            self._db = sqlite3.connect(":memory:", check_same_thread=False)

        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                id TEXT PRIMARY KEY,
                problem_type TEXT NOT NULL,
                problem_size INTEGER NOT NULL,
                size_category TEXT NOT NULL,
                operators TEXT NOT NULL,
                fitness REAL NOT NULL,
                metadata TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)

        # Create indices for fast lookup
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_problem_type
            ON trajectories(problem_type)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_size_category
            ON trajectories(size_category)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_fitness
            ON trajectories(fitness)
        """)

        self._db.commit()

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the database.

        Args:
            trajectory: Trajectory to add
        """
        import json

        record = TrajectoryRecord.from_trajectory(trajectory)

        self._db.execute(
            """
            INSERT OR REPLACE INTO trajectories
            (id, problem_type, problem_size, size_category, operators, fitness, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.problem_type,
                record.problem_size,
                record.size_category,
                json.dumps(record.operators),
                record.fitness,
                json.dumps(record.metadata),
            ),
        )
        self._db.commit()

    def add_trajectories(self, trajectories: list[Trajectory]) -> None:
        """Add multiple trajectories.

        Args:
            trajectories: List of trajectories to add
        """
        for traj in trajectories:
            self.add_trajectory(traj)

    def lookup(
        self,
        problem_type: str,
        problem_size: int | None = None,
        min_fitness: float | None = None,
        max_results: int = 5,
    ) -> list[TrajectoryRecord]:
        """Look up trajectories by problem type and size.

        Args:
            problem_type: Type of problem (tsp, jssp, vrp)
            problem_size: Problem size (optional, used for size category)
            min_fitness: Minimum fitness threshold (optional)
            max_results: Maximum number of results

        Returns:
            List of matching trajectory records, sorted by fitness
        """
        import json

        query = "SELECT * FROM trajectories WHERE problem_type = ?"
        params: list[Any] = [problem_type]

        # Filter by size category if size provided
        if problem_size is not None:
            if problem_size < 100:
                size_cat = "small"
            elif problem_size < 500:
                size_cat = "medium"
            else:
                size_cat = "large"
            query += " AND size_category = ?"
            params.append(size_cat)

        # Filter by minimum fitness
        if min_fitness is not None:
            query += " AND fitness >= ?"
            params.append(min_fitness)

        # Order by fitness (lower is better for minimization)
        query += " ORDER BY fitness ASC LIMIT ?"
        params.append(max_results)

        cursor = self._db.execute(query, params)
        results = []

        for row in cursor.fetchall():
            results.append(
                TrajectoryRecord(
                    id=row[0],
                    problem_type=row[1],
                    problem_size=row[2],
                    size_category=row[3],
                    operators=json.loads(row[4]),
                    fitness=row[5],
                    metadata=json.loads(row[6]) if row[6] else {},
                )
            )

        return results

    def lookup_similar(
        self,
        problem_type: str,
        operators: list[str],
        max_results: int = 5,
    ) -> list[TrajectoryRecord]:
        """Look up trajectories with similar operator sequences.

        Uses Jaccard similarity on operator sets.

        Args:
            problem_type: Type of problem
            operators: Current operator sequence
            max_results: Maximum results

        Returns:
            Similar trajectories sorted by similarity
        """
        import json

        # Get all trajectories of this problem type
        cursor = self._db.execute(
            "SELECT * FROM trajectories WHERE problem_type = ?",
            (problem_type,),
        )

        current_set = set(operators)
        scored = []

        for row in cursor.fetchall():
            traj_operators = json.loads(row[4])
            traj_set = set(traj_operators)

            # Jaccard similarity
            intersection = len(current_set & traj_set)
            union = len(current_set | traj_set)
            similarity = intersection / union if union > 0 else 0

            scored.append((
                similarity,
                TrajectoryRecord(
                    id=row[0],
                    problem_type=row[1],
                    problem_size=row[2],
                    size_category=row[3],
                    operators=traj_operators,
                    fitness=row[5],
                    metadata=json.loads(row[6]) if row[6] else {},
                ),
            ))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored[:max_results]]

    def get_best_trajectories(
        self,
        problem_type: str,
        top_k: int = 10,
    ) -> list[TrajectoryRecord]:
        """Get the best performing trajectories.

        Args:
            problem_type: Problem type
            top_k: Number of top trajectories

        Returns:
            Best trajectories by fitness
        """
        return self.lookup(problem_type, max_results=top_k)

    def count_trajectories(self, problem_type: str | None = None) -> int:
        """Count trajectories in database.

        Args:
            problem_type: Optional filter by problem type

        Returns:
            Count of trajectories
        """
        if problem_type:
            cursor = self._db.execute(
                "SELECT COUNT(*) FROM trajectories WHERE problem_type = ?",
                (problem_type,),
            )
        else:
            cursor = self._db.execute("SELECT COUNT(*) FROM trajectories")

        return cursor.fetchone()[0]

    def clear(self) -> None:
        """Clear all trajectories."""
        self._db.execute("DELETE FROM trajectories")
        self._db.commit()

    def close(self) -> None:
        """Close database connection."""
        self._db.close()
