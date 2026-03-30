"""CaseStudy: Bundles all domain-specific configuration for GEAKG.

A CaseStudy packages together:
- RoleSchema (role vocabulary + transitions)
- Domain configuration
- Representation type
- Base operators (A₀)
- MetaGraph factory

Two built-in case studies:
- CaseStudy.optimization(domain="tsp") — Combinatorial optimization
- CaseStudy.nas(dataset="cifar10") — Neural Architecture Search
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from src.geakg.core.role_schema import RoleSchema
    from src.geakg.layers.l0.metagraph import MetaGraph
    from src.geakg.representations import RepresentationType


@dataclass
class CaseStudy:
    """A complete case study configuration for GEAKG.

    Bundles all the pieces needed to run the GEAKG pipeline
    on a specific domain/problem type.

    Attributes:
        name: Case study identifier (e.g., "optimization", "nas").
        role_schema: The role vocabulary for this case study.
        representation_type: Solution representation type.
        base_operators: Dict mapping role_id -> operator code string (A₀).
        meta_graph_factory: Callable that creates a MetaGraph for this case study.
        domain_config: Domain-specific configuration (DomainConfig or NASDomainConfig).
        description: Human-readable description.
    """

    name: str
    role_schema: "RoleSchema"
    representation_type: "RepresentationType"
    base_operators: dict[str, str]
    meta_graph_factory: Callable[[], "MetaGraph"]
    domain_config: Any = None
    description: str = ""

    def create_meta_graph(self) -> "MetaGraph":
        """Create a MetaGraph for this case study."""
        return self.meta_graph_factory()

    def get_base_operator(self, role: str) -> str:
        """Get the base operator code for a role.

        Args:
            role: Role ID.

        Returns:
            Python code string.

        Raises:
            KeyError: If role not found in this case study.
        """
        if role not in self.base_operators:
            raise KeyError(
                f"Unknown role '{role}' for case study '{self.name}'. "
                f"Available: {list(self.base_operators.keys())}"
            )
        return self.base_operators[role].strip()

    def get_all_roles(self) -> list[str]:
        """Get all role IDs from the schema."""
        return self.role_schema.get_all_roles()

    def validate(self) -> list[str]:
        """Validate that the case study is complete and consistent.

        Returns:
            List of warning messages (empty if valid).
        """
        warnings = []

        # Check all roles have base operators
        for role in self.role_schema.get_all_roles():
            if role not in self.base_operators:
                warnings.append(f"Role '{role}' has no base operator")

        # Check base operators don't have extra roles
        for role in self.base_operators:
            if not self.role_schema.is_valid_role(role):
                warnings.append(f"Base operator '{role}' not in schema")

        return warnings

    # ---- Factory methods ----

    @staticmethod
    def optimization(
        domain: str = "tsp",
        pattern: str = "hybrid",
    ) -> CaseStudy:
        """Create a case study for combinatorial optimization.

        Args:
            domain: Problem domain ("tsp", "jssp", "vrp", "bpp").
            pattern: MetaGraph pattern ("ils", "vns", "hybrid").

        Returns:
            Configured CaseStudy for optimization.
        """
        from src.geakg.core.schemas.optimization import OptimizationRoleSchema
        from src.geakg.layers.l0.patterns import (
            create_hybrid_meta_graph,
            create_ils_meta_graph,
            create_vns_meta_graph,
        )
        from src.geakg.layers.l1.base_operators import BASE_OPERATORS
        from src.geakg.representations import RepresentationType

        schema = OptimizationRoleSchema()

        # Select MetaGraph pattern
        factories = {
            "ils": create_ils_meta_graph,
            "vns": create_vns_meta_graph,
            "hybrid": create_hybrid_meta_graph,
        }
        factory = factories.get(pattern, create_hybrid_meta_graph)

        return CaseStudy(
            name=f"optimization_{domain}",
            role_schema=schema,
            representation_type=RepresentationType.PERMUTATION,
            base_operators=dict(BASE_OPERATORS),
            meta_graph_factory=factory,
            description=f"Combinatorial optimization ({domain}) with {pattern} pattern",
        )

    @staticmethod
    def nas(
        dataset: str = "cifar10",
        proxy_epochs: int = 20,
        proxy_data_fraction: float = 0.25,
        use_gpu: bool = True,
    ) -> CaseStudy:
        """Create a case study for Neural Architecture Search.

        Args:
            dataset: Target dataset ("cifar10", "cifar100", "imdb", "sst2").
            proxy_epochs: Epochs for proxy evaluation.
            proxy_data_fraction: Fraction of data for proxy evaluation.
            use_gpu: Whether to use GPU for evaluation.

        Returns:
            Configured CaseStudy for NAS.
        """
        from src.geakg.core.schemas.nas import NASRoleSchema
        from src.geakg.layers.l0.patterns import create_nas_meta_graph
        from src.geakg.layers.l1.base_operators_nas import NAS_BASE_OPERATORS
        from src.geakg.representations import RepresentationType
        from src.domains.nas.config import NASDomainConfig

        schema = NASRoleSchema()
        domain_config = NASDomainConfig(
            dataset=dataset,
            proxy_epochs=proxy_epochs,
            proxy_data_fraction=proxy_data_fraction,
            use_gpu=use_gpu,
        )

        def _nas_factory() -> "MetaGraph":
            return create_nas_meta_graph(schema)

        return CaseStudy(
            name=f"nas_{dataset}",
            role_schema=schema,
            representation_type=RepresentationType.ARCHITECTURE_DAG,
            base_operators=dict(NAS_BASE_OPERATORS),
            meta_graph_factory=_nas_factory,
            domain_config=domain_config,
            description=f"Neural Architecture Search on {dataset}",
        )

    @staticmethod
    def nas_benchmark(
        dataset: str = "cifar10",
        nasbench_path: str | None = None,
        use_proxy: bool = False,
        seed: int | None = None,
    ) -> CaseStudy:
        """Create a case study for NAS-Bench-201 benchmark.

        Uses CellArchitecture (6 edges, 5 ops) with tabular lookup.
        Same NASRoleSchema (18 roles, 5 categories) but with
        cell-specific base operators.

        Args:
            dataset: Target dataset ("cifar10", "cifar100", "ImageNet16-120").
            nasbench_path: Path to NAS-Bench-201 data. If None, uses env var.
            use_proxy: If True, use proxy evaluator instead of benchmark.
            seed: Random seed for reproducibility.

        Returns:
            Configured CaseStudy for NAS-Bench-201.
        """
        from src.geakg.core.schemas.nas import NASRoleSchema
        from src.geakg.layers.l0.patterns import create_nas_meta_graph
        from src.geakg.layers.l1.base_operators_nas_bench import NAS_BENCH_BASE_OPERATORS
        from src.geakg.representations import RepresentationType
        from src.domains.nas.config import NASBenchConfig

        schema = NASRoleSchema()
        domain_config = NASBenchConfig(
            dataset=dataset,
            nasbench_path=nasbench_path,
            use_proxy=use_proxy,
            seed=seed,
        )

        def _nas_bench_factory() -> "MetaGraph":
            return create_nas_meta_graph(schema)

        return CaseStudy(
            name=f"nas_bench_{dataset}",
            role_schema=schema,
            representation_type=RepresentationType.ARCHITECTURE_DAG,
            base_operators=dict(NAS_BENCH_BASE_OPERATORS),
            meta_graph_factory=_nas_bench_factory,
            domain_config=domain_config,
            description=f"NAS-Bench-201 benchmark on {dataset}",
        )
