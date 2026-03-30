"""Online Phase: Symbolic Execution without LLM.

This module contains the runtime components that execute search
using learned knowledge (L2) without any LLM calls:

- symbolic_executor: Main search engine using pheromones and rules
- operator_selector: Selection of operators based on learned preferences
- execution_context: State tracking during execution

Architecture:
    L0: MetaGraph Topology (offline, LLM) - src/geakg/layers/l0/
    L1: Operator Generation (offline, LLM) - src/geakg/layers/l1/
    L2: ACO Training (offline, ACO) - src/geakg/layers/l2/
    Online: Symbolic Executor - src/geakg/online/  <-- YOU ARE HERE

Key principle: NO LLM CALLS during online execution. All intelligence
comes from the pre-trained GEAKG (L0 topology + L1 operators + L2 knowledge).

Usage:
    from src.geakg.online import SymbolicExecutor

    # Load trained GEAKG
    snapshot = GEAKGSnapshot.load("path/to/snapshot.json")

    # Execute on new instance
    executor = SymbolicExecutor(snapshot)
    solution = executor.optimize(instance)
"""

# Will be populated as we migrate symbolic_executor.py
__all__ = []
