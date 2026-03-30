"""Problem domain implementations (TSP, JSSP, VRP).

This module provides:
1. Domain classes for problem-specific logic (TSPDomain, JSSPDomain, etc.)
2. DomainConfig registry for operator synthesis configuration

To add a new domain:
1. Create domain file (e.g., src/domains/qap.py)
2. Implement DomainConfig subclass with prompts, validation, fitness
3. Register in DOMAIN_CONFIGS below
"""

from src.domains.base import DomainConfig, OptimizationDomain, ProblemFeatures, ProblemInstance, Solution
from src.domains.jssp import (
    JSSPDomain,
    JSSPFeatures,
    JSSPInstance,
    JSSPSolution,
    create_sample_jssp_instance,
)
from src.domains.tsp import (
    TSP_CONFIG,
    TSPDomain,
    TSPInstance,
    TSPLIB_OPTIMA,
)
from src.domains.sop import (
    SOPDomain,
    SOPInstance,
    SOPSolution,
    SOPFeatures,
    create_sample_sop_instance,
)

# =============================================================================
# DOMAIN CONFIG REGISTRY
# =============================================================================
# Register domain configurations for operator synthesis here.
# Each domain must implement DomainConfig interface.

DOMAIN_CONFIGS: dict[str, DomainConfig] = {
    "tsp": TSP_CONFIG,
    # "jssp": JSSP_CONFIG,  # TODO: Implement
    # "vrp": VRP_CONFIG,    # TODO: Implement
    # "bpp": BPP_CONFIG,    # TODO: Implement
}


def get_domain_config(domain: str) -> DomainConfig:
    """Get the DomainConfig for a domain.

    Args:
        domain: Domain name (e.g., 'tsp', 'jssp')

    Returns:
        DomainConfig instance for the domain

    Raises:
        ValueError: If domain is not registered
    """
    if domain not in DOMAIN_CONFIGS:
        available = list(DOMAIN_CONFIGS.keys())
        raise ValueError(
            f"Unknown domain: '{domain}'. "
            f"Available domains: {available}. "
            f"To add a new domain, implement DomainConfig in src/domains/{domain}.py "
            f"and register it in DOMAIN_CONFIGS."
        )
    return DOMAIN_CONFIGS[domain]


__all__ = [
    # Base classes
    "OptimizationDomain",
    "ProblemInstance",
    "ProblemFeatures",
    "Solution",
    "DomainConfig",
    # Registry
    "DOMAIN_CONFIGS",
    "get_domain_config",
    # TSP
    "TSPDomain",
    "TSPInstance",
    "TSP_CONFIG",
    "TSPLIB_OPTIMA",
    # JSSP
    "JSSPDomain",
    "JSSPInstance",
    "JSSPSolution",
    "JSSPFeatures",
    "create_sample_jssp_instance",
    # SOP
    "SOPDomain",
    "SOPInstance",
    "SOPSolution",
    "SOPFeatures",
    "create_sample_sop_instance",
]
