"""Execution Persistence: Save programs, AKG snapshots, and LLM logs.

This module provides domain-agnostic persistence for NS-GE execution artifacts:
1. Best program as executable Python file (with synthesized operators inline)
2. Complete AKG snapshot (MetaGraph + operators + pheromones)
3. LLM interaction logs (L1-synthesized + cache hits)

Usage:
    from src.geakg.persistence import ExecutionPersistence

    persistence = ExecutionPersistence(
        output_dir="experiments/nsgge/results",
        domain="tsp",
    )

    # During execution: LLM interactions are logged automatically via callback
    # At end: export all artifacts
    persistence.export_all(
        selector=selector,
        meta_graph=meta_graph,
        synthesis_hook=synthesis_hook,
        domain_config=domain_config,
        metadata={"gap": gap, "fitness": best_fitness}
    )
"""

import inspect
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

if TYPE_CHECKING:
    from src.geakg.aco import MetaACOSelector
    from src.geakg.meta_graph import MetaGraph
    from src.domains.base import DomainConfig
    from src.llm.client import LLMResponse


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class LLMInteraction:
    """Single LLM interaction record."""

    timestamp: str
    level: str  # "L1-L3" or "synthesized"
    agent: str
    prompt: str
    response: str
    latency_ms: float
    tokens: int
    from_cache: bool
    iteration: int = 0
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "agent": self.agent,
            "iteration": self.iteration,
            "context": self.context,
            "prompt": self.prompt,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "from_cache": self.from_cache,
        }


# =============================================================================
# Main Persistence Class
# =============================================================================


class ExecutionPersistence:
    """Manages persistence of NS-GE execution artifacts.

    Provides:
    1. LLM interaction logging via callback
    2. Best program export as executable Python
    3. AKG snapshot export as JSON
    4. LLM logs export as JSON

    Attributes:
        output_dir: Base output directory.
        domain: Problem domain (tsp, jssp, vrp, bpp).
        session_id: Unique session identifier (timestamp by default).
    """

    def __init__(
        self,
        output_dir: str = "experiments/nsgge/results",
        domain: str = "tsp",
        session_id: str | None = None,
        backend: str = "unknown",
    ) -> None:
        """Initialize persistence manager.

        Args:
            output_dir: Base output directory.
            domain: Problem domain.
            session_id: Session ID (timestamp if None).
            backend: LLM backend name for directory naming.
        """
        self.output_dir = Path(output_dir)
        self.domain = domain
        self.backend = backend
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # LLM interaction log
        self._interactions: list[LLMInteraction] = []
        self._current_iteration = 0

        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_tokens = 0

        logger.info(
            f"[Persistence] Initialized for domain={domain}, session={self.session_id}"
        )

    @property
    def session_dir(self) -> Path:
        """Get the session-specific output directory."""
        return self.output_dir / f"{self.session_id}_{self.backend}"

    def set_iteration(self, iteration: int) -> None:
        """Update current iteration for logging context."""
        self._current_iteration = iteration

    # =========================================================================
    # LLM Interaction Callback (for automatic logging)
    # =========================================================================

    def create_llm_callback(self) -> Callable:
        """Create callback function for LLM client.

        Returns:
            Callback function to pass to LLM client.
        """

        def callback(
            prompt: str,
            response: "LLMResponse",
            agent: str | None = None,
            context: dict | None = None,
        ) -> None:
            self.log_llm_interaction(
                prompt=prompt,
                response=response.content,
                agent=agent or "Unknown",
                latency_ms=response.latency_ms,
                tokens=response.tokens_generated,
                from_cache=response.from_cache,
                context=context,
            )

        return callback

    def log_llm_interaction(
        self,
        prompt: str,
        response: str,
        agent: str,
        latency_ms: float = 0.0,
        tokens: int = 0,
        from_cache: bool = False,
        context: dict | None = None,
        level: str | None = None,
    ) -> None:
        """Log an LLM interaction.

        Args:
            prompt: The prompt sent to LLM.
            response: The response from LLM.
            agent: Agent name (CodeGenerator, MetaGraphGenerator, etc.).
            latency_ms: Response latency in milliseconds.
            tokens: Number of tokens generated.
            from_cache: Whether response was from cache.
            context: Additional context (bottleneck_type, target_role, etc.).
            level: Explicit level ("L1-L3" or "synthesized"), auto-detected if None.
        """
        # Auto-detect level from agent name
        if level is None:
            if agent in ("MetaGraphGenerator", "L1Generator", "L2Generator", "L3Generator"):
                level = "L1-L3"
            else:
                level = "synthesized"

        interaction = LLMInteraction(
            timestamp=datetime.now().isoformat(),
            level=level,
            agent=agent,
            prompt=prompt,
            response=response,
            latency_ms=latency_ms,
            tokens=tokens,
            from_cache=from_cache,
            iteration=self._current_iteration,
            context=context or {},
        )

        self._interactions.append(interaction)

        # Update statistics
        if from_cache:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        self._total_tokens += tokens

        logger.debug(
            f"[Persistence] Logged LLM interaction: agent={agent}, "
            f"level={level}, cache={from_cache}"
        )

    # =========================================================================
    # Export Methods
    # =========================================================================

    def export_all(
        self,
        selector: "MetaACOSelector",
        meta_graph: "MetaGraph",
        synthesis_hook: "DynamicSynthesisHook | None",
        domain_config: "DomainConfig",
        metadata: dict[str, Any],
    ) -> Path:
        """Export all artifacts to session directory.

        Args:
            selector: MetaACOSelector with best path and pheromones.
            meta_graph: The MetaGraph used.
            synthesis_hook: Synthesis hook with synthesized operators (optional).
            domain_config: Domain configuration.
            metadata: Additional metadata (gap, fitness, elapsed, etc.).

        Returns:
            Path to session directory with all artifacts.
        """
        # Create session directory
        self.session_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Persistence] Exporting artifacts to {self.session_dir}")

        # Export each artifact
        self.export_best_program(
            selector=selector,
            synthesis_hook=synthesis_hook,
            domain_config=domain_config,
            metadata=metadata,
        )

        self.export_akg_snapshot(
            selector=selector,
            meta_graph=meta_graph,
            synthesis_hook=synthesis_hook,
            metadata=metadata,
        )

        self.export_llm_logs()

        logger.info(f"[Persistence] All artifacts exported to {self.session_dir}")
        return self.session_dir

    def export_best_program(
        self,
        selector: "MetaACOSelector",
        synthesis_hook: "DynamicSynthesisHook | None",
        domain_config: "DomainConfig",
        metadata: dict[str, Any],
        filename: str = "best_program.py",
    ) -> Path:
        """Export best program as executable Python file.

        Args:
            selector: MetaACOSelector with best path.
            synthesis_hook: Synthesis hook with synthesized operators.
            domain_config: Domain configuration.
            metadata: Metadata for header.
            filename: Output filename.

        Returns:
            Path to exported file.
        """
        from src.geakg.persistence_templates import get_program_template

        output_path = self.session_dir / filename

        # Get best operator path
        best_operator_path = selector.best_operator_path
        best_role_path = selector.best_role_path
        best_fitness = selector.best_fitness

        # Collect operator code
        generic_operators = self._collect_generic_operators(best_operator_path)
        synth_operators = {}
        if synthesis_hook and synthesis_hook.registry:
            synth_operators = self._collect_synth_operators(
                best_operator_path, synthesis_hook.registry
            )

        # Generate program content
        template = get_program_template(self.domain)
        # Handle both single-instance (gap, fitness) and multi-instance (aggregate_gap)
        gap = metadata.get("gap", metadata.get("aggregate_gap", 0.0))
        # In multi-instance mode, best_fitness is the gap %, not absolute cost
        # Use fitness from metadata if available, otherwise use selector's value
        fitness = metadata.get("fitness", best_fitness)
        content = template.format(
            domain=self.domain.upper(),
            timestamp=datetime.now().isoformat(),
            gap=gap,
            fitness=fitness,
            evals=metadata.get("total_evals", 0),
            elapsed=metadata.get("elapsed", 0.0),
            operator_path=repr(best_operator_path),
            role_path=repr(best_role_path),
            generic_operators_code=self._format_operators_code(generic_operators),
            synth_operators_code=self._format_operators_code(synth_operators, is_synth=True),
            operator_registry=self._format_operator_registry(
                generic_operators, synth_operators, best_operator_path
            ),
        )

        output_path.write_text(content)
        logger.info(f"[Persistence] Best program exported to {output_path}")
        return output_path

    def export_akg_snapshot(
        self,
        selector: "MetaACOSelector",
        meta_graph: "MetaGraph",
        synthesis_hook: "DynamicSynthesisHook | None",
        metadata: dict[str, Any],
        filename: str = "akg_snapshot.json",
    ) -> Path:
        """Export complete AKG snapshot as JSON.

        Args:
            selector: MetaACOSelector with pheromones.
            meta_graph: The MetaGraph.
            synthesis_hook: Synthesis hook with synthesized operators.
            metadata: Additional metadata.
            filename: Output filename.

        Returns:
            Path to exported file.
        """
        output_path = self.session_dir / filename

        # Build snapshot
        snapshot = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "domain": self.domain,
            "session_id": self.session_id,
            "meta_graph": self._serialize_meta_graph(meta_graph),
            "operators": self._serialize_operators(selector, synthesis_hook),
            "pheromones": self._serialize_pheromones(selector),
            "best_path": {
                "roles": selector.best_role_path,
                "operators": selector.best_operator_path,
                "fitness": selector.best_fitness,
            },
            "metadata": metadata,
        }

        output_path.write_text(json.dumps(snapshot, indent=2, default=str))
        logger.info(f"[Persistence] AKG snapshot exported to {output_path}")
        return output_path

    def export_llm_logs(self, filename: str = "llm_interactions.json") -> Path:
        """Export LLM interaction logs as JSON.

        Args:
            filename: Output filename.

        Returns:
            Path to exported file.
        """
        # Ensure session directory exists
        self.session_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.session_dir / filename

        # Count by level
        l1_l3_count = sum(1 for i in self._interactions if i.level == "L1-L3")
        synth_count = sum(1 for i in self._interactions if i.level == "synthesized")

        logs = {
            "session_id": self.session_id,
            "domain": self.domain,
            "total_calls": len(self._interactions),
            "total_tokens": self._total_tokens,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "by_level": {
                "L1_L3_metagraph": l1_l3_count,
                "synthesized_synthesis": synth_count,
            },
            "interactions": [i.to_dict() for i in self._interactions],
        }

        output_path.write_text(json.dumps(logs, indent=2, default=str))
        logger.info(
            f"[Persistence] LLM logs exported to {output_path} "
            f"({len(self._interactions)} interactions)"
        )
        return output_path

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _collect_generic_operators(
        self, operator_path: list[str]
    ) -> dict[str, str]:
        """Collect source code for generic operators used in path.

        Includes ALL public functions from the permutation module to ensure
        the exported program is completely standalone. This is necessary because
        operators may depend on each other (e.g., vnd_generic uses swap, insert, invert).

        Args:
            operator_path: List of operator IDs.

        Returns:
            Dict mapping operator_id/function_name to source code.
        """
        from src.geakg.generic_operators import PERMUTATION_OPERATORS
        from src.geakg.generic_operators import permutation as perm_module

        # Build lookup dict from RepresentationOperators
        generic_ops_lookup = {
            op.operator_id: op.function
            for op in PERMUTATION_OPERATORS.get_all_operators()
        }

        # Determine which operators from the path are generic (not synthesized)
        generic_in_path = set()
        for op_id in operator_path:
            if op_id.startswith("synth_") or self._is_synth_operator(op_id):
                continue
            generic_in_path.add(op_id)

        operators = {}

        # Collect ALL public functions from permutation module
        # This ensures all dependencies are available (e.g., swap, insert, invert for vnd_generic)
        for name in dir(perm_module):
            # Skip private functions and non-functions
            if name.startswith("_"):
                continue

            obj = getattr(perm_module, name)
            if not callable(obj):
                continue

            # Skip classes and imports
            if isinstance(obj, type):
                continue

            # Skip functions from other modules
            if hasattr(obj, "__module__") and obj.__module__ != perm_module.__name__:
                continue

            try:
                source = inspect.getsource(obj)
                operators[name] = source
            except (OSError, TypeError):
                pass

        return operators

    def _collect_synth_operators(
        self, operator_path: list[str], registry: Any
    ) -> dict[str, str]:
        """Collect source code for synthesized synthesized operators.

        Args:
            operator_path: List of operator IDs.
            registry: DynamicBindingRegistry with synthesized operators.

        Returns:
            Dict mapping operator_id to source code.
        """
        operators = {}
        for op_id in operator_path:
            if op_id.startswith("synth_") or self._is_synth_operator(op_id):
                record = registry.get(op_id)
                if record:
                    operators[op_id] = record.code
        return operators

    def _is_synth_operator(self, op_id: str) -> bool:
        """Check if operator is synthesized synthesized (has hex suffix)."""
        if op_id.startswith("synth_"):
            return True
        # Check for hex suffix pattern (e.g., _abc123)
        if "_" in op_id:
            suffix = op_id.split("_")[-1]
            if len(suffix) >= 4:
                try:
                    int(suffix, 16)
                    return True
                except ValueError:
                    pass
        return False

    def _format_operators_code(
        self, operators: dict[str, str], is_synth: bool = False
    ) -> str:
        """Format operator code for embedding in template.

        Args:
            operators: Dict mapping operator_id to source code.
            is_synth: Whether these are synthesized operators.

        Returns:
            Formatted code string.
        """
        if not operators:
            return "# No operators in this category\npass"

        lines = []
        for op_id, code in operators.items():
            if is_synth:
                lines.append(f"# synthesized Synthesized: {op_id}")
            lines.append(code)
            lines.append("")  # Blank line between operators

        return "\n".join(lines)

    def _format_operator_registry(
        self, generic: dict[str, str], synth: dict[str, str], operator_path: list[str]
    ) -> str:
        """Format operator registry for template with agnostic wrappers.

        All operators in OPERATORS use unified signature: (solution, ctx)
        Generic operators are wrapped with lambdas to adapt their signatures.
        synthesized operators already use agnostic signature.

        Following Lampson's "Keep interfaces stable" - all operators have same interface.

        Args:
            generic: All generic operators (code for dependencies).
            synth: synthesized operators.
            operator_path: The actual best operator path.

        Returns:
            Registry entries as string with wrapper lambdas.
        """
        from src.geakg.generic_operators import PERMUTATION_OPERATORS

        # Build mapping from operator_id to function name
        op_id_to_fn_name = {
            op.operator_id: op.function.__name__
            for op in PERMUTATION_OPERATORS.get_all_operators()
        }

        # Operators that take only (solution) - wrap to (solution, ctx)
        simple_ops = {
            "swap", "segment_reverse", "segment_shuffle", "partial_restart",
        }

        # Construction operators need special handling
        construction_ops = {
            "random_permutation", "greedy_by_fitness", "random_insertion", "pairwise_merge"
        }

        entries = []
        seen_ops = set()  # Avoid duplicates

        # Only register operators that are in the path
        for op_id in operator_path:
            # Skip synthesized operators here (handled separately)
            if op_id.startswith("synth_") or self._is_synth_operator(op_id):
                continue

            # Skip if already added (path may have repeated operators)
            if op_id in seen_ops:
                continue
            seen_ops.add(op_id)

            # Find the function name for this operator
            fn_name = op_id_to_fn_name.get(op_id) or self._extract_function_name(
                generic.get(op_id, ""), op_id
            )

            # Wrap generic operators to agnostic signature
            if op_id in simple_ops:
                # Simple ops: (solution) -> wrap to (solution, ctx)
                entries.append(f'    "{op_id}": lambda s, ctx, fn={fn_name}: fn(s),')
            elif op_id in construction_ops:
                # Construction: returns new solution, ignores input
                if op_id == "random_permutation":
                    entries.append(f'    "{op_id}": lambda s, ctx: random_permutation_construct(len(s)),')
                elif op_id == "greedy_by_fitness":
                    # Uses ctx.evaluate for fitness
                    entries.append(
                        f'    "{op_id}": lambda s, ctx: greedy_by_fitness('
                        f'len(s), lambda p, c: ctx.evaluate(p + [c]) if ctx.valid(p + [c]) else float("inf")),'
                    )
                else:
                    entries.append(f'    "{op_id}": lambda s, ctx, fn={fn_name}: fn(len(s)),')
            elif op_id in {"variable_depth_search", "vnd_generic"}:
                # Local search ops that need ctx.evaluate - keep direct reference
                entries.append(f'    "{op_id}": _wrap_{op_id},')
            else:
                # Fallback: assume simple (solution) signature
                entries.append(f'    "{op_id}": lambda s, ctx, fn={fn_name}: fn(s),')

        # Add synthesized operators (already agnostic)
        for op_id in synth:
            fn_name = self._extract_function_name(synth[op_id], op_id)
            entries.append(f'    "{op_id}": {fn_name},  # synthesized agnostic')

        return "\n".join(entries)

    def _extract_function_name(self, code: str, fallback: str) -> str:
        """Extract function name from source code.

        Args:
            code: Source code.
            fallback: Fallback name if extraction fails.

        Returns:
            Function name.
        """
        for line in code.split("\n"):
            if line.strip().startswith("def "):
                # Extract name between 'def ' and '('
                start = line.find("def ") + 4
                end = line.find("(")
                if end > start:
                    return line[start:end].strip()
        return fallback

    def _serialize_meta_graph(self, meta_graph: "MetaGraph") -> dict:
        """Serialize MetaGraph to dict.

        Args:
            meta_graph: MetaGraph to serialize.

        Returns:
            Dict representation.
        """
        try:
            roles = [
                node.role.value if hasattr(node.role, "value") else str(node.role)
                for node in meta_graph.nodes.values()
            ]

            edges = []
            for (src, tgt), edge in meta_graph.edges.items():
                src_str = src.value if hasattr(src, "value") else str(src)
                tgt_str = tgt.value if hasattr(tgt, "value") else str(tgt)
                edge_dict = {
                    "source": src_str,
                    "target": tgt_str,
                    "weight": edge.weight,
                }
                if edge.conditions:
                    edge_dict["conditions"] = [
                        {"type": c.condition_type.value, "threshold": c.threshold}
                        for c in edge.conditions
                    ]
                edges.append(edge_dict)

            return {
                "name": getattr(meta_graph, "name", "MetaGraph"),
                "roles": roles,
                "edges": edges,
                "reasoning": getattr(meta_graph, "llm_reasoning", ""),
            }
        except Exception as e:
            logger.warning(f"[Persistence] Error serializing MetaGraph: {e}")
            return {"error": str(e)}

    def _serialize_operators(
        self, selector: "MetaACOSelector", synthesis_hook: Any
    ) -> dict:
        """Serialize operators to dict.

        Args:
            selector: MetaACOSelector.
            synthesis_hook: Synthesis hook.

        Returns:
            Dict with generic and synthesized operators.
        """
        result = {"generic": {}, "synthesized_synth": []}

        # Generic operators by role
        try:
            if hasattr(selector, "graph") and selector.graph:
                for role, bindings in selector.graph.bindings.bindings.items():
                    role_str = role.value if hasattr(role, "value") else str(role)
                    result["generic"][role_str] = [b.operator_id for b in bindings]
        except Exception as e:
            logger.warning(f"[Persistence] Error serializing generic operators: {e}")

        # synthesized operators
        if synthesis_hook and hasattr(synthesis_hook, "registry"):
            try:
                for op_id, record in synthesis_hook.registry._operators.items():
                    result["synthesized_synth"].append({
                        "operator_id": op_id,
                        "role": record.role,
                        "code": record.code,
                        "p_value": record.p_value,
                        "effect_size": record.effect_size,
                        "usage_count": record.usage_count,
                        "created_at": record.created_at,
                    })
            except Exception as e:
                logger.warning(f"[Persistence] Error serializing synthesized operators: {e}")

        return result

    def _serialize_pheromones(self, selector: "MetaACOSelector") -> dict:
        """Serialize pheromones to dict.

        Args:
            selector: MetaACOSelector with pheromones.

        Returns:
            Dict with role-level and operator-level pheromones.
        """
        result = {"role_level": {}, "operator_level": {}}

        try:
            # Role-level pheromones
            for (src, tgt), tau in selector.pheromones.items():
                src_str = src.value if hasattr(src, "value") else str(src)
                tgt_str = tgt.value if hasattr(tgt, "value") else str(tgt)
                result["role_level"][f"{src_str}->{tgt_str}"] = tau

            # Operator-level pheromones
            op_pheromones = selector.get_operator_pheromones()
            for (role, op), tau in op_pheromones.items():
                role_str = role.value if hasattr(role, "value") else str(role)
                result["operator_level"][f"{role_str}:{op}"] = tau

        except Exception as e:
            logger.warning(f"[Persistence] Error serializing pheromones: {e}")

        return result
