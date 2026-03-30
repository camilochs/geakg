"""Domain Adapters for Transfer Learning.

Adapters translate domain-specific representations to/from the abstract
role-based representation used by GEAKG. This enables transferring
knowledge learned on one domain (e.g., TSP) to another (e.g., VRP).

Available adapters:
- VRPAdapter: Vehicle Routing Problem
- JSSPAdapter: Job Shop Scheduling Problem
- PFSPAdapter: Permutation Flow Shop Problem
- LOPAdapter: Linear Ordering Problem
- QAPAdapter: Quadratic Assignment Problem
- SOPAdapter: Sequential Ordering Problem

Usage:
    from src.geakg.transfer.adapters import VRPAdapter

    adapter = VRPAdapter()
    vrp_solution = adapter.run_with_knowledge(vrp_instance, tsp_knowledge)
"""

from src.geakg.transfer.adapters.vrp_adapter import VRPAdapter
from src.geakg.transfer.adapters.jssp_adapter import JSSPAdapter
from src.geakg.transfer.adapters.pfsp_adapter import PFSPAdapter
from src.geakg.transfer.adapters.lop_adapter import LOPAdapter
from src.geakg.transfer.adapters.qap_adapter import QAPAdapter
from src.geakg.transfer.adapters.sop_adapter import SOPAdapter

__all__ = [
    "VRPAdapter",
    "JSSPAdapter",
    "PFSPAdapter",
    "LOPAdapter",
    "QAPAdapter",
    "SOPAdapter",
]
