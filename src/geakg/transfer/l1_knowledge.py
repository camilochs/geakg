"""L1 Knowledge Extraction and Transfer.

Extrae el conocimiento estructural de las decisiones L1 (selección de roles)
de un AKG entrenado para transferirlo a otros dominios.

Dos niveles de abstracción:
1. RoleCategory: 3 categorías (construction, local_search, perturbation)
2. AbstractRole: 11 roles específicos (const_greedy, ls_intensify_small, etc.)

El nivel de roles preserva más conocimiento estructural (escaleras de intensificación).
"""

from dataclasses import dataclass, field
from enum import Enum


class RoleCategory(Enum):
    """Categorías abstractas de roles (nivel más alto)."""
    CONSTRUCTION = "construction"
    LOCAL_SEARCH = "local_search"
    PERTURBATION = "perturbation"

    @classmethod
    def from_role(cls, role: str) -> "RoleCategory":
        """Inferir categoría desde nombre de rol."""
        if role.startswith("const"):
            return cls.CONSTRUCTION
        elif role.startswith("ls"):
            return cls.LOCAL_SEARCH
        elif role.startswith("pert"):
            return cls.PERTURBATION
        else:
            # Default a local search para roles desconocidos
            return cls.LOCAL_SEARCH


# Los 11 roles abstractos que usamos en NS-SE
ABSTRACT_ROLES = [
    # Construction (4)
    "const_greedy",
    "const_insertion",
    "const_savings",
    "const_random",
    # Local Search (4)
    "ls_intensify_small",
    "ls_intensify_medium",
    "ls_intensify_large",
    "ls_chain",
    # Perturbation (3)
    "pert_escape_small",
    "pert_escape_large",
    "pert_adaptive",
]


@dataclass
class TransitionPattern:
    """Un patrón de transición entre categorías."""
    source: RoleCategory
    target: RoleCategory
    strength: float  # 0.0 a 1.0 (normalizado)

    def __repr__(self) -> str:
        return f"{self.source.value} → {self.target.value}: {self.strength:.2f}"


@dataclass
class L1Knowledge:
    """Conocimiento estructural extraído de L1.

    Representa los patrones de secuenciación de roles aprendidos,
    abstraídos a nivel de categoría para ser domain-agnostic.
    """

    # Patrones de transición entre categorías (normalizado)
    transition_matrix: dict[tuple[RoleCategory, RoleCategory], float] = field(
        default_factory=dict
    )

    # Roles específicos que fueron exitosos (para referencia)
    successful_roles: set[str] = field(default_factory=set)

    # Metadata
    source_domain: str = ""
    total_iterations: int = 0

    def get_transition_strength(
        self,
        source: RoleCategory,
        target: RoleCategory
    ) -> float:
        """Obtener fuerza de transición normalizada."""
        return self.transition_matrix.get((source, target), 0.0)

    def get_preferred_next_category(
        self,
        current: RoleCategory
    ) -> RoleCategory:
        """Obtener la categoría preferida después de current."""
        best_cat = None
        best_strength = -1.0

        for target in RoleCategory:
            strength = self.get_transition_strength(current, target)
            if strength > best_strength:
                best_strength = strength
                best_cat = target

        return best_cat or RoleCategory.LOCAL_SEARCH

    def get_transition_distribution(
        self,
        current: RoleCategory
    ) -> dict[RoleCategory, float]:
        """Obtener distribución de probabilidad para siguiente categoría."""
        distribution = {}
        total = 0.0

        for target in RoleCategory:
            strength = self.get_transition_strength(current, target)
            distribution[target] = strength
            total += strength

        # Normalizar a probabilidades
        if total > 0:
            for cat in distribution:
                distribution[cat] /= total
        else:
            # Uniforme si no hay datos
            uniform = 1.0 / len(RoleCategory)
            for cat in distribution:
                distribution[cat] = uniform

        return distribution

    def to_dict(self) -> dict:
        """Serializar a diccionario."""
        return {
            "transition_matrix": {
                f"{src.value}->{tgt.value}": strength
                for (src, tgt), strength in self.transition_matrix.items()
            },
            "successful_roles": list(self.successful_roles),
            "source_domain": self.source_domain,
            "total_iterations": self.total_iterations,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "L1Knowledge":
        """Deserializar desde diccionario."""
        knowledge = cls()
        knowledge.source_domain = data.get("source_domain", "")
        knowledge.total_iterations = data.get("total_iterations", 0)
        knowledge.successful_roles = set(data.get("successful_roles", []))

        for key, strength in data.get("transition_matrix", {}).items():
            if "->" in key:
                src_str, tgt_str = key.split("->")
                src = RoleCategory(src_str)
                tgt = RoleCategory(tgt_str)
                knowledge.transition_matrix[(src, tgt)] = strength

        return knowledge


def extract_l1_knowledge(snapshot: dict) -> L1Knowledge:
    """Extraer conocimiento L1 de un snapshot AKG.

    Analiza las feromonas de nivel de rol y las abstrae a
    patrones de transición entre categorías.

    Args:
        snapshot: AKG snapshot dict

    Returns:
        L1Knowledge con patrones extraídos
    """
    knowledge = L1Knowledge()
    knowledge.source_domain = snapshot.get("domain", "unknown")

    # Obtener feromonas de rol
    pheromones = snapshot.get("pheromones", {})
    role_pheromones = pheromones.get("role_level", {})

    if not role_pheromones:
        return knowledge

    # Agregar transiciones por categoría
    category_totals: dict[tuple[RoleCategory, RoleCategory], float] = {}
    category_counts: dict[tuple[RoleCategory, RoleCategory], int] = {}

    for transition, value in role_pheromones.items():
        if "->" not in transition:
            continue

        src_role, tgt_role = transition.split("->")
        src_cat = RoleCategory.from_role(src_role)
        tgt_cat = RoleCategory.from_role(tgt_role)

        key = (src_cat, tgt_cat)
        category_totals[key] = category_totals.get(key, 0.0) + value
        category_counts[key] = category_counts.get(key, 0) + 1

        # Registrar roles exitosos (feromonas > umbral)
        if value > 1.0:
            knowledge.successful_roles.add(src_role)
            knowledge.successful_roles.add(tgt_role)

    # Calcular promedios por categoría
    for key in category_totals:
        if category_counts[key] > 0:
            avg = category_totals[key] / category_counts[key]
            knowledge.transition_matrix[key] = avg

    # Normalizar matriz de transición (por fila)
    for src_cat in RoleCategory:
        row_total = sum(
            knowledge.transition_matrix.get((src_cat, tgt), 0.0)
            for tgt in RoleCategory
        )
        if row_total > 0:
            for tgt_cat in RoleCategory:
                key = (src_cat, tgt_cat)
                if key in knowledge.transition_matrix:
                    knowledge.transition_matrix[key] /= row_total

    return knowledge


def print_l1_knowledge(knowledge: L1Knowledge) -> None:
    """Imprimir conocimiento L1 de forma legible."""
    print(f"=== L1 Knowledge from {knowledge.source_domain} ===\n")

    print("Transition Matrix (normalized):")
    print("           ", end="")
    for tgt in RoleCategory:
        print(f"{tgt.value[:8]:>10}", end="")
    print()

    for src in RoleCategory:
        print(f"{src.value[:10]:<10}", end=" ")
        for tgt in RoleCategory:
            strength = knowledge.get_transition_strength(src, tgt)
            print(f"{strength:>10.2f}", end="")
        print()

    print(f"\nPreferred transitions:")
    for src in RoleCategory:
        preferred = knowledge.get_preferred_next_category(src)
        strength = knowledge.get_transition_strength(src, preferred)
        print(f"  {src.value} → {preferred.value} ({strength:.2f})")


@dataclass
class L1RoleKnowledge:
    """Conocimiento L1 a nivel de roles (11 roles, no 3 categorías).

    Preserva el conocimiento estructural completo:
    - Escaleras de intensificación (small → medium → large → chain)
    - Cuándo perturbar (después de chain, no de small)
    - Transiciones específicas aprendidas
    """

    # Matriz de transición entre roles (normalizada por fila)
    transition_matrix: dict[tuple[str, str], float] = field(default_factory=dict)

    # Roles que aparecieron en el entrenamiento
    available_roles: set[str] = field(default_factory=set)

    # Metadata
    source_domain: str = ""

    def get_transition_strength(self, source: str, target: str) -> float:
        """Obtener fuerza de transición normalizada."""
        return self.transition_matrix.get((source, target), 0.0)

    def get_transition_distribution(self, current: str) -> dict[str, float]:
        """Obtener distribución de probabilidad para siguiente rol."""
        distribution = {}
        total = 0.0

        for target in self.available_roles:
            strength = self.get_transition_strength(current, target)
            distribution[target] = strength
            total += strength

        # Normalizar a probabilidades
        if total > 0:
            for role in distribution:
                distribution[role] /= total
        else:
            # Uniforme si no hay datos para este rol
            uniform = 1.0 / len(self.available_roles) if self.available_roles else 0
            for role in self.available_roles:
                distribution[role] = uniform

        return distribution

    def get_preferred_next_role(self, current: str) -> str | None:
        """Obtener el rol preferido después de current."""
        best_role = None
        best_strength = -1.0

        for target in self.available_roles:
            strength = self.get_transition_strength(current, target)
            if strength > best_strength:
                best_strength = strength
                best_role = target

        return best_role

    def to_dict(self) -> dict:
        """Serializar a diccionario."""
        return {
            "transition_matrix": {
                f"{src}->{tgt}": strength
                for (src, tgt), strength in self.transition_matrix.items()
            },
            "available_roles": list(self.available_roles),
            "source_domain": self.source_domain,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "L1RoleKnowledge":
        """Deserializar desde diccionario."""
        knowledge = cls()
        knowledge.source_domain = data.get("source_domain", "")
        knowledge.available_roles = set(data.get("available_roles", []))

        for key, strength in data.get("transition_matrix", {}).items():
            if "->" in key:
                src, tgt = key.split("->")
                knowledge.transition_matrix[(src, tgt)] = strength

        return knowledge


def extract_l1_role_knowledge(snapshot: dict) -> L1RoleKnowledge:
    """Extraer conocimiento L1 a nivel de roles (no categorías).

    Preserva las transiciones específicas entre los 11 roles,
    manteniendo información como la escalera de intensificación.

    Args:
        snapshot: AKG snapshot dict

    Returns:
        L1RoleKnowledge con transiciones entre roles
    """
    knowledge = L1RoleKnowledge()
    knowledge.source_domain = snapshot.get("domain", "unknown")

    # Obtener feromonas de rol
    pheromones = snapshot.get("pheromones", {})
    role_pheromones = pheromones.get("role_level", {})

    if not role_pheromones:
        return knowledge

    # Extraer transiciones y roles
    for transition, value in role_pheromones.items():
        if "->" not in transition:
            continue

        src_role, tgt_role = transition.split("->")
        knowledge.available_roles.add(src_role)
        knowledge.available_roles.add(tgt_role)
        knowledge.transition_matrix[(src_role, tgt_role)] = value

    # Normalizar por fila (cada rol source)
    for src in knowledge.available_roles:
        row_total = sum(
            knowledge.transition_matrix.get((src, tgt), 0.0)
            for tgt in knowledge.available_roles
        )
        if row_total > 0:
            for tgt in knowledge.available_roles:
                key = (src, tgt)
                if key in knowledge.transition_matrix:
                    knowledge.transition_matrix[key] /= row_total

    return knowledge


def print_l1_role_knowledge(knowledge: L1RoleKnowledge) -> None:
    """Imprimir conocimiento L1 a nivel de roles."""
    print(f"=== L1 Role Knowledge from {knowledge.source_domain} ===\n")
    print(f"Roles: {len(knowledge.available_roles)}")

    # Mostrar transiciones más fuertes
    transitions = [
        (src, tgt, strength)
        for (src, tgt), strength in knowledge.transition_matrix.items()
        if strength > 0.05  # Solo mostrar transiciones significativas
    ]
    transitions.sort(key=lambda x: -x[2])

    print(f"\nTop transitions (>{5}%):")
    for src, tgt, strength in transitions[:15]:
        print(f"  {src} → {tgt}: {strength:.0%}")
