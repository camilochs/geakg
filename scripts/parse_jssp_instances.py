#!/usr/bin/env python3
"""Parse OR-Library JSSP instances into individual files."""

import re
from pathlib import Path


# Óptimos conocidos (de la literatura)
JSSP_OPTIMA = {
    "ft06": 55,
    "ft10": 930,
    "ft20": 1165,
    "la01": 666,
    "la02": 655,
    "la03": 597,
    "la04": 590,
    "la05": 593,
    "la06": 926,
    "la07": 890,
    "la08": 863,
    "la09": 951,
    "la10": 958,
    "abz5": 1234,
    "abz6": 943,
    "abz7": 656,
    "abz8": 665,
    "abz9": 678,
    "orb01": 1059,
    "orb02": 888,
    "orb03": 1005,
    "orb04": 1005,
    "orb05": 887,
}


def parse_jobshop_file(filepath: Path) -> dict:
    """Parse the OR-Library jobshop file."""
    with open(filepath) as f:
        content = f.read()

    instances = {}

    # Split by instance markers
    parts = re.split(r'\+{20,}\s*\n\s*instance\s+(\w+)\s*\n\s*\+{20,}', content)

    # parts[0] is the header, then alternating (name, content)
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break

        name = parts[i].strip()
        instance_content = parts[i + 1].strip()

        # Parse instance content
        lines = [l.strip() for l in instance_content.split('\n') if l.strip()]

        # Skip description line
        # Find the line with dimensions (n_jobs n_machines)
        dim_line = None
        data_start = 0
        for j, line in enumerate(lines):
            if re.match(r'^\d+\s+\d+$', line):
                dim_line = line
                data_start = j + 1
                break

        if not dim_line:
            continue

        n_jobs, n_machines = map(int, dim_line.split())

        # Parse job lines
        processing_times = []
        machine_assignments = []

        for j in range(data_start, data_start + n_jobs):
            if j >= len(lines):
                break
            values = list(map(int, lines[j].split()))

            machines = []
            times = []
            for k in range(0, len(values), 2):
                machines.append(values[k])
                times.append(values[k + 1])

            machine_assignments.append(machines)
            processing_times.append(times)

        if len(processing_times) == n_jobs:
            instances[name] = {
                "n_jobs": n_jobs,
                "n_machines": n_machines,
                "processing_times": processing_times,
                "machine_assignments": machine_assignments,
                "optimal": JSSP_OPTIMA.get(name),
            }

    return instances


def save_instance(name: str, data: dict, output_dir: Path):
    """Save instance in standard format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{name}.txt"

    with open(filepath, 'w') as f:
        # Header comment
        if data.get("optimal"):
            f.write(f"# {name} - optimal: {data['optimal']}\n")
        else:
            f.write(f"# {name}\n")

        # Dimensions
        f.write(f"{data['n_jobs']} {data['n_machines']}\n")

        # Job lines: machine time machine time ...
        for job_idx in range(data['n_jobs']):
            line_parts = []
            for op_idx in range(data['n_machines']):
                machine = data['machine_assignments'][job_idx][op_idx]
                time = data['processing_times'][job_idx][op_idx]
                line_parts.append(f"{machine} {time}")
            f.write(" ".join(line_parts) + "\n")

    return filepath


def main():
    input_file = Path("data/instances/jssp/jobshop1.txt")
    output_dir = Path("data/instances/jssp/parsed")

    print("Parsing OR-Library JSSP instances...")
    instances = parse_jobshop_file(input_file)

    print(f"Found {len(instances)} instances")

    # Save each instance
    saved = []
    for name, data in instances.items():
        filepath = save_instance(name, data, output_dir)
        opt_str = f" (opt: {data.get('optimal', '?')})" if data.get('optimal') else ""
        print(f"  {name}: {data['n_jobs']}x{data['n_machines']}{opt_str}")
        saved.append(name)

    print(f"\nSaved {len(saved)} instances to {output_dir}")

    # List some key instances for benchmarking
    print("\nRecommended for benchmarking:")
    key_instances = ["ft06", "ft10", "la01", "la02", "la06", "abz5", "abz6"]
    for name in key_instances:
        if name in instances:
            data = instances[name]
            print(f"  {name}: {data['n_jobs']}x{data['n_machines']} (opt: {data.get('optimal', '?')})")


if __name__ == "__main__":
    main()
