"""Symbolic Executor: Domain-agnostic search guided by symbolic rules.

Principios de Butler Lampson aplicados:
- "Keep Secrets": El executor no conoce detalles del dominio
- "Separate Policy from Mechanism": Reglas (policy) separadas de ejecución (mechanism)
- "Design for Iteration": Interfaz mínima para extensibilidad

El executor recibe:
- Operadores adaptados (cualquier dominio)
- Motor de reglas (extraído de cualquier AKG)
- Funciones de evaluación (inyectadas por el dominio)

Y ejecuta la búsqueda siguiendo las reglas simbólicas.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .symbolic_rules import (
    SymbolicRuleEngine,
    SearchPhase,
    IntensificationLevel,
)


class Solution(Protocol):
    """Protocolo mínimo para una solución."""
    @property
    def cost(self) -> float: ...


class Operator(Protocol):
    """Protocolo mínimo para un operador adaptado."""
    @property
    def role(self) -> Any: ...

    @property
    def adapted_fn(self) -> Callable | None: ...


@dataclass
class ExecutionResult:
    """Resultado de la ejecución del motor simbólico."""
    best_solution: Any
    best_cost: float
    initial_cost: float
    iterations: int
    improvements: int
    elapsed_time: float
    history: list[dict] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """Mejora absoluta sobre la solución inicial."""
        return self.initial_cost - self.best_cost

    @property
    def improvement_percent(self) -> float:
        """Mejora porcentual sobre la solución inicial."""
        if self.initial_cost == 0:
            return 0.0
        return 100 * self.improvement / self.initial_cost


@dataclass
class SymbolicExecutor:
    """Executor de búsqueda guiado por reglas simbólicas.

    Domain-agnostic: funciona con cualquier dominio que provea:
    - evaluate_fn: Solution → float
    - copy_fn: Solution → Solution
    - operators agrupados por rol

    Modos de selección:
    - global_mode=False (default): Selecciona rol → luego operador dentro del rol
    - global_mode=True: Selecciona directamente entre TODOS los operadores usando
      feromonas combinadas (role_level * operator_level)
    - ablation_mode=True: Selección ALEATORIA (sin feromonas ni reglas).
      Usado para demostrar que el conocimiento transferido aporta valor.
    """

    rule_engine: SymbolicRuleEngine

    # Funciones inyectadas por el dominio
    evaluate_fn: Callable[[Any], float] | None = None
    copy_fn: Callable[[Any], Any] | None = None

    # Feromonas de operadores (extraídas del snapshot)
    operator_pheromones: dict[str, float] = field(default_factory=dict)

    # Bonus por frecuencia en paths exitosos (extraído del snapshot)
    # Formato: {"operator_name": frequency_count}
    success_frequency: dict[str, int] = field(default_factory=dict)

    # Modo de selección
    global_mode: bool = True  # True = selección global entre todos los operadores
    ablation_mode: bool = False  # True = selección aleatoria (ablation study)

    # Parámetros ACO (extraídos del config de entrenamiento)
    # Estos valores deben coincidir con los usados en aco.py durante training
    alpha: float = 1.0  # Exponente para feromonas (τ^α) - default de ACOConfig

    # Adaptación local de feromonas durante ejecución
    local_adaptation: bool = False  # True = adaptar feromonas según éxito local
    evaporation_rate: float = 0.1   # ρ: tasa de evaporación (0-1)
    reward_factor: float = 1.0      # Factor de recompensa por mejora

    # Configuración
    verbose: bool = True

    # Estado interno para adaptación local (no configurable)
    _local_pheromones: dict[str, float] = field(default_factory=dict)
    _initialized_local: bool = field(default=False)

    def execute(
        self,
        operators: list[Operator],
        initial_solution: Any,
        initial_cost: float,
        time_limit: float = 60.0,
        instance: Any = None,
    ) -> ExecutionResult:
        """Ejecutar búsqueda guiada por reglas simbólicas.

        Args:
            operators: Lista de operadores adaptados
            initial_solution: Solución inicial
            initial_cost: Costo de la solución inicial
            time_limit: Límite de tiempo en segundos
            instance: Instancia del problema (pasada a operadores)

        Returns:
            ExecutionResult con la mejor solución encontrada
        """
        # Agrupar operadores por rol
        ops_by_role = self._group_operators_by_role(operators)

        # Inicializar estado
        current = self._copy(initial_solution)
        current_cost = initial_cost
        best = self._copy(initial_solution)
        best_cost = initial_cost

        # Inicializar feromonas locales (copia de las del snapshot)
        if self.local_adaptation and not self._initialized_local:
            self._local_pheromones = dict(self.operator_pheromones)
            self._initialized_local = True

        # Inicializar motor de reglas
        self.rule_engine.state.current_cost = current_cost
        self.rule_engine.state.best_cost = best_cost

        history = [{"time": 0.0, "cost": best_cost, "event": "initial"}]

        if self.verbose:
            print(f"\nInitial cost: {best_cost:.2f}")
            mode_str = "ABLATION (random)" if self.ablation_mode else "symbolic"
            if self.local_adaptation:
                mode_str += " + local_adaptation"
            print(f"Running {mode_str} executor for {time_limit}s...")
            if not self.ablation_mode:
                print(f"Rules: stagnation={self.rule_engine.stagnation_threshold}")

        start_time = time.time()
        iterations = 0
        improvements = 0

        while time.time() - start_time < time_limit:
            iterations += 1
            elapsed = time.time() - start_time

            if self.ablation_mode:
                # MODO ABLATION: Selección completamente aleatoria
                # No usa reglas simbólicas ni feromonas
                selected_op, target_role = self._select_operator_random(operators)
                if selected_op is None:
                    continue
                phase = SearchPhase.INTENSIFICATION  # Dummy phase for logging
                level = None
            else:
                # 1. Decidir acción usando reglas simbólicas
                phase, level = self.rule_engine.decide_next_action()
                self.rule_engine.transition_to(phase, level)

                if self.global_mode:
                    # MODO GLOBAL: Seleccionar directamente entre todos los operadores
                    selected_op, target_role = self._select_operator_global(operators, phase)
                    if selected_op is None:
                        continue
                else:
                    # MODO POR ROL: Selecciona rol → luego operador
                    # 2. Obtener rol para el estado actual
                    target_role = self.rule_engine.get_role_for_current_state()

                    # 3. Seleccionar operador del rol (usando feromonas)
                    role_ops = self._get_operators_for_role(ops_by_role, target_role, phase)
                    if not role_ops:
                        continue

                    selected_op = self._select_operator_by_pheromone(role_ops, target_role)

            # 4. Aplicar operador
            if selected_op.adapted_fn is None:
                continue

            try:
                result = selected_op.adapted_fn(current, instance)

                # Validar solución antes de evaluar (evita aceptar soluciones corruptas)
                if hasattr(instance, 'valid') and not instance.valid(result):
                    continue  # Rechazar solución inválida

                new_cost = self._evaluate(result, instance)

                # 5. Decidir aceptación basada en fase (NO random SA)
                improved = new_cost < best_cost

                if improved:
                    best = self._copy(result)
                    best_cost = new_cost
                    current = self._copy(result)
                    current_cost = new_cost
                    improvements += 1

                    history.append({
                        "time": elapsed,
                        "cost": best_cost,
                        "event": "improvement",
                        "phase": phase.name,
                        "level": level.name if level else "-",
                        "role": target_role,
                    })

                    if self.verbose:
                        print(f"  [{elapsed:5.1f}s] #{improvements}: {best_cost:.2f} via {phase.name}/{target_role}")

                elif phase == SearchPhase.PERTURBATION:
                    # En perturbación: aceptar cualquier resultado (escapar)
                    current = self._copy(result)
                    current_cost = new_cost

                elif new_cost < current_cost:
                    # Aceptar mejora sobre current (no best)
                    current = self._copy(result)
                    current_cost = new_cost

                # 6. Actualizar estado del motor de reglas (solo si no es ablation)
                if not self.ablation_mode:
                    self.rule_engine.update_state(new_cost, improved)

                # 7. Adaptar feromonas localmente si está habilitado
                if self.local_adaptation:
                    self._update_local_pheromones(
                        selected_op, target_role, improved, new_cost, best_cost
                    )

            except Exception:
                pass

        elapsed = time.time() - start_time
        history.append({"time": elapsed, "cost": best_cost, "event": "final"})

        if self.verbose:
            print(f"\nCompleted {iterations} iterations in {elapsed:.1f}s")
            print(f"Found {improvements} improvements")

        return ExecutionResult(
            best_solution=best,
            best_cost=best_cost,
            initial_cost=initial_cost,
            iterations=iterations,
            improvements=improvements,
            elapsed_time=elapsed,
            history=history,
        )

    def _group_operators_by_role(self, operators: list[Operator]) -> dict[str, list]:
        """Agrupar operadores por rol."""
        ops_by_role: dict[str, list] = {}
        for op in operators:
            role = op.role.value if hasattr(op.role, 'value') else str(op.role)
            if role not in ops_by_role:
                ops_by_role[role] = []
            ops_by_role[role].append(op)
        return ops_by_role

    def _select_operator_random(self, operators: list[Operator]) -> tuple[Any | None, str]:
        """Seleccionar operador ALEATORIAMENTE (modo ablation).

        Sin feromonas, sin reglas simbólicas. Selección uniforme entre todos
        los operadores disponibles con adapted_fn válido.

        Args:
            operators: Lista completa de operadores

        Returns:
            (operador_seleccionado, rol_del_operador) o (None, "") si no hay candidatos
        """
        # Filtrar solo operadores con función adaptada válida
        valid_ops = [op for op in operators if op.adapted_fn is not None]

        if not valid_ops:
            return None, ""

        # Selección uniforme aleatoria
        selected = random.choice(valid_ops)
        role = selected.role.value if hasattr(selected.role, 'value') else str(selected.role)

        return selected, role

    def _get_operators_for_role(
        self,
        ops_by_role: dict[str, list],
        target_role: str,
        phase: SearchPhase
    ) -> list:
        """Obtener operadores para un rol, con fallback por fase."""
        role_ops = ops_by_role.get(target_role, [])
        if role_ops:
            return role_ops

        # Fallback: cualquier operador de la misma fase
        if phase == SearchPhase.CONSTRUCTION:
            prefix = "const_"
        elif phase == SearchPhase.PERTURBATION:
            prefix = "pert_"
        else:
            prefix = "ls_"

        for role, ops in ops_by_role.items():
            if role.startswith(prefix) and ops:
                return ops

        return []

    def _select_operator_global(
        self,
        operators: list[Operator],
        phase: SearchPhase
    ) -> tuple[Any | None, str]:
        """Seleccionar operador globalmente usando feromonas combinadas.

        Combina feromonas de rol (role_level) y operador (operator_level)
        para calcular el peso de cada operador.

        peso(op) = τ_role(rol_previo → rol_op) * τ_operator(rol_op:op_id)

        Args:
            operators: Lista completa de operadores
            phase: Fase actual de búsqueda

        Returns:
            (operador_seleccionado, rol_del_operador) o (None, "") si no hay candidatos
        """
        if not operators:
            return None, ""

        # Filtrar por fase (solo operadores compatibles con la fase actual)
        phase_prefix = {
            SearchPhase.CONSTRUCTION: "const_",
            SearchPhase.PERTURBATION: "pert_",
            SearchPhase.INTENSIFICATION: "ls_",
        }.get(phase, "ls_")

        candidates = []
        for op in operators:
            role = op.role.value if hasattr(op.role, 'value') else str(op.role)
            if role.startswith(phase_prefix):
                candidates.append((op, role))

        if not candidates:
            # Fallback: usar todos los operadores
            candidates = [
                (op, op.role.value if hasattr(op.role, 'value') else str(op.role))
                for op in operators
            ]

        if not candidates:
            return None, ""

        if len(candidates) == 1:
            return candidates[0]

        # Calcular pesos usando fórmula ACO: τ^α * η^β
        # donde τ = feromona combinada (role * operator)
        # y η = bonus por frecuencia en paths exitosos
        role_pheromones = self.rule_engine.role_pheromones
        weights = []

        for op, role in candidates:
            op_id = getattr(op, 'operator_id', None) or getattr(op, 'name', str(op))

            # τ_role: feromona de transición hacia este rol (suma de entrantes)
            tau_role = sum(
                v for k, v in role_pheromones.items()
                if k.endswith(f"->{role}")
            )
            tau_role = max(0.1, tau_role)

            # τ_operator: feromona específica del operador
            # Usar feromona local si adaptación activa, sino del snapshot
            tau_op = self._get_effective_pheromone(role, op_id)
            tau_op = max(0.1, tau_op)

            # η (eta): heurística basada en frecuencia en paths exitosos
            # Operadores que aparecen más en paths exitosos tienen mayor eta
            freq = self.success_frequency.get(op_id, 0)
            eta = 1.0 + freq  # Base 1.0 + bonus por frecuencia

            # Fórmula ACO: prob ∝ τ^α * η
            tau = tau_role * tau_op
            weight = (tau ** self.alpha) * eta
            weights.append(weight)

        # Selección proporcional (como en ACO)
        total = sum(weights)
        if total <= 0:
            selected = random.choice(candidates)
            return selected

        r = random.random() * total
        cumsum = 0.0
        for (op, role), w in zip(candidates, weights):
            cumsum += w
            if r <= cumsum:
                return op, role

        return candidates[-1]

    def _select_operator_by_pheromone(self, operators: list, role: str) -> Any:
        """Seleccionar operador usando feromonas (selección proporcional).

        Args:
            operators: Lista de operadores del rol
            role: Nombre del rol actual

        Returns:
            Operador seleccionado
        """
        if not operators:
            raise ValueError("No operators to select from")

        if len(operators) == 1:
            return operators[0]

        if not self.operator_pheromones:
            # Sin feromonas: selección aleatoria
            return random.choice(operators)

        # Calcular pesos basados en feromonas
        weights = []
        for op in operators:
            # Obtener ID del operador
            op_id = getattr(op, 'operator_id', None) or getattr(op, 'name', str(op))
            # Usar feromona local si adaptación activa, sino del snapshot
            tau = self._get_effective_pheromone(role, op_id)
            weights.append(max(0.1, tau))  # Mínimo 0.1 para dar chance a todos

        # Selección proporcional
        total = sum(weights)
        if total <= 0:
            return random.choice(operators)

        r = random.random() * total
        cumsum = 0.0
        for op, w in zip(operators, weights):
            cumsum += w
            if r <= cumsum:
                return op

        return operators[-1]

    def _evaluate(self, solution: Any, instance: Any) -> float:
        """Evaluar una solución."""
        if self.evaluate_fn:
            return self.evaluate_fn(solution, instance)

        # Fallback: intentar acceder a .cost
        if hasattr(solution, 'cost') and solution.cost > 0:
            return solution.cost

        raise ValueError("No evaluate_fn provided and solution has no .cost")

    def _copy(self, solution: Any) -> Any:
        """Copiar una solución."""
        if self.copy_fn:
            return self.copy_fn(solution)

        # Fallback: copia superficial
        if hasattr(solution, '__class__') and hasattr(solution, '__dict__'):
            import copy
            return copy.deepcopy(solution)

        return solution

    def _update_local_pheromones(
        self,
        operator: Any,
        role: str,
        improved: bool,
        new_cost: float,
        best_cost: float,
    ) -> None:
        """Actualizar feromonas locales basado en el resultado.

        Implementa actualización estilo ACO:
        - Evaporación: τ = (1 - ρ) * τ
        - Recompensa si mejora: τ = τ + Δτ

        Args:
            operator: Operador que se aplicó
            role: Rol del operador
            improved: Si hubo mejora sobre el mejor conocido
            new_cost: Costo de la nueva solución
            best_cost: Mejor costo conocido
        """
        op_id = getattr(operator, 'operator_id', None) or getattr(operator, 'name', str(operator))
        pheromone_key = f"{role}:{op_id}"

        # Obtener feromona actual (local o del snapshot)
        current_tau = self._local_pheromones.get(
            pheromone_key,
            self.operator_pheromones.get(pheromone_key, 1.0)
        )

        # Evaporación
        new_tau = (1 - self.evaporation_rate) * current_tau

        # Recompensa si mejoró
        if improved and best_cost > 0:
            # Δτ proporcional a la calidad de la mejora
            delta_tau = self.reward_factor * (1.0 / best_cost)
            new_tau += delta_tau

        # Mantener en rango razonable [0.1, 20.0]
        new_tau = max(0.1, min(20.0, new_tau))

        self._local_pheromones[pheromone_key] = new_tau

    def _get_effective_pheromone(self, role: str, op_id: str) -> float:
        """Obtener feromona efectiva (local si adaptación activa, sino snapshot)."""
        pheromone_key = f"{role}:{op_id}"

        if self.local_adaptation and self._local_pheromones:
            return self._local_pheromones.get(
                pheromone_key,
                self.operator_pheromones.get(pheromone_key, 0.1)
            )

        return self.operator_pheromones.get(pheromone_key, 0.1)
