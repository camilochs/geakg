#!/usr/bin/env python3
"""
Script para crear un snapshot optimizado para instancias grandes.
Elimina operadores _base de roles costosos (ls_*) para evitar O(n²).
"""
import json
import sys
from pathlib import Path

def clean_snapshot(input_path: str, output_path: str):
    """Crea un snapshot limpio sin operadores _base problemáticos."""

    with open(input_path) as f:
        data = json.load(f)

    # Roles donde _base es O(n²) y debemos preferir LLM
    costly_roles = [
        "ls_intensify_small",
        "ls_intensify_medium",
        "ls_intensify_large",
        "ls_chain",
    ]

    # Operadores a eliminar
    operators_to_remove = []
    for role in costly_roles:
        operators_to_remove.append(f"{role}:{role}_base")

    print("=== Limpiando snapshot para instancias grandes ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    # Mostrar feromonas antes
    op_pheromones = data["pheromones"]["operator_level"]
    print("Feromonas antes:")
    for k, v in sorted(op_pheromones.items(), key=lambda x: -x[1]):
        marker = " [ELIMINAR]" if k in operators_to_remove else ""
        print(f"  {v:.4f} | {k}{marker}")

    # Eliminar operadores _base de roles costosos
    removed = []
    for op_key in operators_to_remove:
        if op_key in op_pheromones:
            del op_pheromones[op_key]
            removed.append(op_key)

    # Eliminar de operators si existe
    if "operators" in data:
        original_count = len(data["operators"])
        data["operators"] = [
            op for op in data["operators"]
            if f"{op.get('role', '')}:{op.get('name', '')}" not in operators_to_remove
        ]
        print(f"\nOperadores eliminados de lista: {original_count - len(data['operators'])}")

    print(f"\n=== Eliminados {len(removed)} operadores _base ===")
    for op in removed:
        print(f"  - {op}")

    print("\nFeromonas después:")
    for k, v in sorted(op_pheromones.items(), key=lambda x: -x[1]):
        print(f"  {v:.4f} | {k}")

    # Guardar
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Snapshot guardado en: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python clean_snapshot_for_large.py <input_snapshot> [output_snapshot]")
        print()
        print("Ejemplo:")
        print("  python clean_snapshot_for_large.py knowledge/nsse_50k_gpt5_2_llamea/akg_snapshot.json")
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        # Default: añadir _large al nombre
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_large{p.suffix}")

    clean_snapshot(input_path, output_path)
