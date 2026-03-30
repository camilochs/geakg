"""Symbolic Rules: Domain-agnostic knowledge transfer.

Extrae y aplica reglas simbólicas de búsqueda aprendidas por NS-SE.

Las reglas son condicionales abstractas que capturan:
- CUÁNDO cambiar de estrategia (no solo probabilidades)
- CÓMO escalar intensificación
- CUÁNDO perturbar (escapar de óptimos locales)

Principios NS-SE:
- Abstracto: Reglas sobre ROLES, no operadores específicos
- Transferible: Mismas reglas aplican a cualquier dominio
- Interpretable: Conocimiento simbólico explícito
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class SearchPhase(Enum):
    """Fases de búsqueda abstractas."""
    CONSTRUCTION = auto()
    INTENSIFICATION = auto()
    PERTURBATION = auto()


class IntensificationLevel(Enum):
    """Niveles de intensificación (escalera)."""
    SMALL = 1
    MEDIUM = 2
    LARGE = 3
    CHAIN = 4

    def next_level(self) -> "IntensificationLevel | None":
        """Siguiente nivel en la escalera."""
        levels = list(IntensificationLevel)
        idx = levels.index(self)
        if idx < len(levels) - 1:
            return levels[idx + 1]
        return None

    def prev_level(self) -> "IntensificationLevel | None":
        """Nivel anterior en la escalera."""
        levels = list(IntensificationLevel)
        idx = levels.index(self)
        if idx > 0:
            return levels[idx - 1]
        return None

    @classmethod
    def from_role(cls, role: str) -> "IntensificationLevel | None":
        """Inferir nivel desde nombre de rol."""
        if "small" in role:
            return cls.SMALL
        elif "medium" in role:
            return cls.MEDIUM
        elif "large" in role:
            return cls.LARGE
        elif "chain" in role:
            return cls.CHAIN
        return None


@dataclass
class SearchState:
    """Estado actual de la búsqueda.

    Captura el contexto necesario para aplicar reglas.
    """
    # Fase y nivel actual
    phase: SearchPhase = SearchPhase.CONSTRUCTION
    intensification_level: IntensificationLevel = IntensificationLevel.SMALL

    # Métricas de progreso
    iterations_since_improvement: int = 0
    total_improvements: int = 0
    current_cost: float = float("inf")
    best_cost: float = float("inf")

    # Historial reciente (para detectar patrones)
    recent_deltas: list[float] = field(default_factory=list)
    max_recent_history: int = 10

    # Contadores de fase
    perturbations_without_improvement: int = 0
    escalations_at_current_level: int = 0

    def record_iteration(self, new_cost: float, improved: bool) -> None:
        """Registrar resultado de una iteración."""
        delta = self.current_cost - new_cost
        self.recent_deltas.append(delta)
        if len(self.recent_deltas) > self.max_recent_history:
            self.recent_deltas.pop(0)

        self.current_cost = new_cost

        if improved:
            self.iterations_since_improvement = 0
            self.total_improvements += 1
            self.best_cost = min(self.best_cost, new_cost)
            self.perturbations_without_improvement = 0
        else:
            self.iterations_since_improvement += 1

    def is_stagnant(self, threshold: int = 5) -> bool:
        """¿Estamos estancados?"""
        return self.iterations_since_improvement >= threshold

    def is_improving(self) -> bool:
        """¿Hubo mejora reciente?"""
        return self.iterations_since_improvement == 0

    def average_recent_improvement(self) -> float:
        """Mejora promedio reciente."""
        if not self.recent_deltas:
            return 0.0
        return sum(self.recent_deltas) / len(self.recent_deltas)


@dataclass
class SymbolicRule(ABC):
    """Regla simbólica abstracta."""

    name: str
    description: str
    priority: int = 0  # Mayor = más prioritaria

    @abstractmethod
    def applies(self, state: SearchState) -> bool:
        """¿Esta regla aplica en el estado actual?"""
        ...

    @abstractmethod
    def action(self, state: SearchState) -> tuple[SearchPhase, IntensificationLevel | None]:
        """Acción a tomar: (nueva_fase, nuevo_nivel)."""
        ...


@dataclass
class RuleStartWithSmall(SymbolicRule):
    """REGLA 1: Después de construcción, empezar con intensificación pequeña."""

    name: str = "start_with_small"
    description: str = "Después de construcción → intensificación pequeña"
    priority: int = 100

    def applies(self, state: SearchState) -> bool:
        return state.phase == SearchPhase.CONSTRUCTION

    def action(self, state: SearchState) -> tuple[SearchPhase, IntensificationLevel | None]:
        return (SearchPhase.INTENSIFICATION, IntensificationLevel.SMALL)


@dataclass
class RuleClimbLadder(SymbolicRule):
    """REGLA 2: Subir escalera mientras mejora."""

    name: str = "climb_ladder"
    description: str = "Si mejorando, subir al siguiente nivel de intensificación"
    priority: int = 80
    min_improvements_to_climb: int = 2

    def applies(self, state: SearchState) -> bool:
        if state.phase != SearchPhase.INTENSIFICATION:
            return False
        if state.intensification_level == IntensificationLevel.CHAIN:
            return False  # Ya en el tope
        # Aplicar si estamos mejorando consistentemente
        return (state.is_improving() and
                state.escalations_at_current_level >= self.min_improvements_to_climb)

    def action(self, state: SearchState) -> tuple[SearchPhase, IntensificationLevel | None]:
        next_level = state.intensification_level.next_level()
        return (SearchPhase.INTENSIFICATION, next_level)


@dataclass
class RuleStayAtLevel(SymbolicRule):
    """REGLA 2b: Quedarse en nivel actual si sigue mejorando."""

    name: str = "stay_at_level"
    description: str = "Continuar en nivel actual mientras mejore"
    priority: int = 70

    def applies(self, state: SearchState) -> bool:
        if state.phase != SearchPhase.INTENSIFICATION:
            return False
        return state.is_improving()

    def action(self, state: SearchState) -> tuple[SearchPhase, IntensificationLevel | None]:
        return (SearchPhase.INTENSIFICATION, state.intensification_level)


@dataclass
class RulePerturbWhenStagnant(SymbolicRule):
    """REGLA 3: Perturbar cuando estancado en niveles altos."""

    name: str = "perturb_when_stagnant"
    description: str = "Si estancado en chain/large, perturbar"
    priority: int = 90
    stagnation_threshold: int = 5

    def applies(self, state: SearchState) -> bool:
        if state.phase != SearchPhase.INTENSIFICATION:
            return False
        # Solo perturbar desde niveles altos
        if state.intensification_level not in [IntensificationLevel.LARGE,
                                                 IntensificationLevel.CHAIN]:
            return False
        return state.is_stagnant(self.stagnation_threshold)

    def action(self, state: SearchState) -> tuple[SearchPhase, IntensificationLevel | None]:
        return (SearchPhase.PERTURBATION, None)


@dataclass
class RuleDescendLadder(SymbolicRule):
    """REGLA 3b: Bajar escalera si estancado en niveles bajos."""

    name: str = "descend_ladder"
    description: str = "Si estancado en small/medium, subir nivel antes de perturbar"
    priority: int = 85
    stagnation_threshold: int = 3

    def applies(self, state: SearchState) -> bool:
        if state.phase != SearchPhase.INTENSIFICATION:
            return False
        if state.intensification_level in [IntensificationLevel.LARGE,
                                            IntensificationLevel.CHAIN]:
            return False
        return state.is_stagnant(self.stagnation_threshold)

    def action(self, state: SearchState) -> tuple[SearchPhase, IntensificationLevel | None]:
        # Subir nivel para probar más antes de perturbar
        next_level = state.intensification_level.next_level()
        return (SearchPhase.INTENSIFICATION, next_level or state.intensification_level)


@dataclass
class RuleRestartAfterPerturbation(SymbolicRule):
    """REGLA 4: Después de perturbar, reiniciar desde small."""

    name: str = "restart_after_perturbation"
    description: str = "Después de perturbación → reiniciar en intensificación pequeña"
    priority: int = 100

    def applies(self, state: SearchState) -> bool:
        return state.phase == SearchPhase.PERTURBATION

    def action(self, state: SearchState) -> tuple[SearchPhase, IntensificationLevel | None]:
        return (SearchPhase.INTENSIFICATION, IntensificationLevel.SMALL)


@dataclass
class RuleGiveUp(SymbolicRule):
    """REGLA 5: Si muchas perturbaciones sin mejora, reconstruir."""

    name: str = "give_up_and_reconstruct"
    description: str = "Si perturbaciones repetidas fallan, reconstruir"
    priority: int = 95
    max_failed_perturbations: int = 3

    def applies(self, state: SearchState) -> bool:
        return state.perturbations_without_improvement >= self.max_failed_perturbations

    def action(self, state: SearchState) -> tuple[SearchPhase, IntensificationLevel | None]:
        return (SearchPhase.CONSTRUCTION, None)


class AcceptanceCriterion(Enum):
    """Criterios de aceptación de soluciones."""
    STRICT = auto()      # Solo acepta mejoras
    PERTURBATION = auto() # Acepta cualquier resultado (escapar)
    PROBABILISTIC = auto() # Acepta peor con cierta probabilidad


@dataclass
class SymbolicRuleEngine:
    """Motor de reglas simbólicas.

    Ejecuta reglas en orden de prioridad para decidir
    la siguiente acción de búsqueda.

    Incluye política de aceptación derivada de las reglas:
    - En INTENSIFICATION: solo acepta mejoras (greedy)
    - En PERTURBATION: acepta cualquier resultado (escapar de óptimo local)

    Usa feromonas del snapshot para seleccionar roles dinámicamente.
    """

    rules: list[SymbolicRule] = field(default_factory=list)
    state: SearchState = field(default_factory=SearchState)

    # Configuración extraída del snapshot
    stagnation_threshold: int = 5
    climb_threshold: int = 2
    max_failed_perturbations: int = 3

    # Feromonas para selección de roles (extraídas del snapshot)
    role_pheromones: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.rules:
            self.rules = self._default_rules()
        # Ordenar por prioridad (mayor primero)
        self.rules.sort(key=lambda r: -r.priority)

    def _default_rules(self) -> list[SymbolicRule]:
        """Reglas por defecto basadas en conocimiento TSP."""
        return [
            RuleStartWithSmall(),
            RuleClimbLadder(min_improvements_to_climb=self.climb_threshold),
            RuleStayAtLevel(),
            RulePerturbWhenStagnant(stagnation_threshold=self.stagnation_threshold),
            RuleDescendLadder(stagnation_threshold=max(2, self.stagnation_threshold - 2)),
            RuleRestartAfterPerturbation(),
            RuleGiveUp(max_failed_perturbations=self.max_failed_perturbations),
        ]

    def decide_next_action(self) -> tuple[SearchPhase, IntensificationLevel | None]:
        """Decidir siguiente acción basada en reglas.

        Returns:
            (fase, nivel) - La siguiente fase y nivel de intensificación
        """
        for rule in self.rules:
            if rule.applies(self.state):
                return rule.action(self.state)

        # Default: quedarse donde está
        return (self.state.phase, self.state.intensification_level)

    def update_state(self, new_cost: float, improved: bool) -> None:
        """Actualizar estado después de una iteración."""
        old_phase = self.state.phase
        old_level = self.state.intensification_level

        self.state.record_iteration(new_cost, improved)

        # Actualizar contadores específicos
        if improved:
            self.state.escalations_at_current_level += 1

        if old_phase == SearchPhase.PERTURBATION and not improved:
            self.state.perturbations_without_improvement += 1

    def transition_to(self, phase: SearchPhase, level: IntensificationLevel | None) -> None:
        """Transicionar a nueva fase/nivel."""
        old_phase = self.state.phase

        self.state.phase = phase
        if level is not None:
            if level != self.state.intensification_level:
                self.state.escalations_at_current_level = 0
            self.state.intensification_level = level

        # Reset contadores en transiciones de fase
        if phase != old_phase:
            if phase == SearchPhase.INTENSIFICATION:
                self.state.iterations_since_improvement = 0

    def get_role_for_current_state(self) -> str:
        """Obtener nombre de rol abstracto para el estado actual.

        Usa feromonas del snapshot para seleccionar el mejor rol
        dentro de la categoría correspondiente a la fase actual.
        """
        phase = self.state.phase
        level = self.state.intensification_level

        if phase == SearchPhase.CONSTRUCTION:
            # Seleccionar entre roles de construcción según feromonas
            construction_roles = [
                "const_greedy",
                "const_random",
                "const_insertion",
                "const_savings",
            ]
            return self._select_role_by_pheromone(construction_roles, "const_greedy")

        elif phase == SearchPhase.PERTURBATION:
            # Seleccionar entre roles de perturbación según feromonas
            perturbation_roles = [
                "pert_escape_small",
                "pert_escape_large",
                "pert_adaptive",
            ]
            return self._select_role_by_pheromone(perturbation_roles, "pert_escape_small")

        else:
            # Intensification - seleccionar entre todos los roles ls_ según feromonas
            # Los niveles dan preferencia pero las feromonas deciden
            local_search_roles = [
                "ls_intensify_small",
                "ls_intensify_medium",
                "ls_intensify_large",
                "ls_chain",
            ]

            # Default basado en nivel actual
            level_to_default = {
                IntensificationLevel.SMALL: "ls_intensify_small",
                IntensificationLevel.MEDIUM: "ls_intensify_medium",
                IntensificationLevel.LARGE: "ls_intensify_large",
                IntensificationLevel.CHAIN: "ls_chain",
            }
            default = level_to_default.get(level, "ls_intensify_small")

            return self._select_role_by_pheromone(local_search_roles, default)

    def _select_role_by_pheromone(self, candidates: list[str], default: str) -> str:
        """Seleccionar rol de la lista según feromonas.

        Usa selección proporcional basada en feromonas del snapshot.
        Si no hay feromonas, devuelve el default.
        """
        import random

        if not self.role_pheromones or not candidates:
            return default

        # Calcular pesos basados en feromonas salientes de cada rol
        weights = []
        for role in candidates:
            # Sumar feromonas de edges que salen de este rol
            role_weight = sum(
                v for k, v in self.role_pheromones.items()
                if k.startswith(f"{role}->")
            )
            # Mínimo peso para evitar roles con 0
            weights.append(max(0.1, role_weight))

        # Selección proporcional
        total = sum(weights)
        if total <= 0:
            return default

        r = random.random() * total
        cumsum = 0.0
        for role, w in zip(candidates, weights):
            cumsum += w
            if r <= cumsum:
                return role

        return default


def extract_symbolic_rules(snapshot: dict) -> SymbolicRuleEngine:
    """Extraer reglas simbólicas de un snapshot AKG.

    Analiza las feromonas para inferir parámetros de las reglas:
    - Umbral de estancamiento
    - Cuándo subir escalera
    - Ratios intensificación/perturbación

    Args:
        snapshot: AKG snapshot dict

    Returns:
        SymbolicRuleEngine configurado con reglas extraídas
    """
    pheromones = snapshot.get("pheromones", {}).get("role_level", {})

    if not pheromones:
        return SymbolicRuleEngine()

    # Analizar ratios para inferir umbrales

    # 1. ¿Cuánto prefiere intensificar vs perturbar en cada nivel?
    ratios = {}
    for level in ["small", "medium", "large", "chain"]:
        role = f"ls_intensify_{level}" if level != "chain" else "ls_chain"

        # Suma de feromonas hacia intensificación
        ls_sum = sum(v for k, v in pheromones.items()
                     if k.startswith(role) and "->ls_" in k)
        # Suma hacia perturbación
        pert_sum = sum(v for k, v in pheromones.items()
                       if k.startswith(role) and "->pert_" in k)

        if pert_sum > 0:
            ratios[level] = ls_sum / pert_sum
        else:
            ratios[level] = float("inf")

    # 2. Inferir umbral de estancamiento
    # Si ratio es alto, necesita más estancamiento para perturbar
    # Heurística: umbral ∝ log(ratio) para chain/large
    import math
    chain_ratio = ratios.get("chain", 1.0)
    if chain_ratio > 1 and chain_ratio < float('inf'):
        stagnation_threshold = min(10, max(3, int(math.log(chain_ratio) * 2)))
    else:
        stagnation_threshold = 3

    # 3. Umbral para subir escalera
    # Basado en cuántas mejoras típicas antes de subir
    climb_threshold = 2  # Default conservador

    # 4. Max perturbaciones fallidas
    # Si pert→ls tiene feromona alta, las perturbaciones suelen funcionar
    pert_to_ls = sum(v for k, v in pheromones.items()
                     if k.startswith("pert_") and "->ls_" in k)
    if pert_to_ls > 5:
        max_failed = 5  # Perturbaciones suelen funcionar, dar más chances
    else:
        max_failed = 2  # Perturbaciones menos efectivas, reconstruir antes

    return SymbolicRuleEngine(
        stagnation_threshold=stagnation_threshold,
        climb_threshold=climb_threshold,
        max_failed_perturbations=max_failed,
        role_pheromones=pheromones,  # Pasar feromonas para selección de roles
    )


def extract_success_frequency(snapshot: dict) -> dict[str, int]:
    """Extraer frecuencia de operadores en paths exitosos.

    Args:
        snapshot: AKG snapshot dict

    Returns:
        Dict {operator_name: frequency_count}
    """
    from collections import Counter

    frequency: Counter[str] = Counter()

    # Contar operadores en successful_paths
    paths = snapshot.get("successful_paths", [])
    for path in paths:
        operators = path.get("operators", [])
        frequency.update(operators)

    # También contar el best_path (con peso extra)
    best_path = snapshot.get("best_path", {})
    if best_path:
        operators = best_path.get("operators", [])
        # Dar peso doble al best path
        frequency.update(operators)
        frequency.update(operators)

    return dict(frequency)


def print_symbolic_rules(engine: SymbolicRuleEngine) -> None:
    """Imprimir reglas del motor de forma legible."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              MOTOR DE REGLAS SIMBÓLICAS                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"Configuración:")
    print(f"  - Umbral estancamiento: {engine.stagnation_threshold} iteraciones")
    print(f"  - Mejoras para subir: {engine.climb_threshold}")
    print(f"  - Max perturbaciones fallidas: {engine.max_failed_perturbations}")
    print()
    print("Reglas (por prioridad):")
    for rule in engine.rules:
        print(f"  [{rule.priority:3d}] {rule.name}")
        print(f"        {rule.description}")
