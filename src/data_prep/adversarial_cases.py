# src/data_prep/adversarial_cases.py
#
# Generate adversarial tasks by pairing probes with the most relevant failure taxonomy category.

import json
import re


def parse_failures(failure_file):
    """Extract failure mode codes and descriptions from failure_taxonomy.md."""
    failures = {}
    with open(failure_file, "r", encoding="utf-8") as f:
        text = f.read()
        for line in text.splitlines():
            match = re.search(r"\*\*(F\d+\.\d+):\s*(.+?)\*\*", line)
            if match:
                code = match.group(1)
                desc = match.group(2)
                failures[code] = f"{code}: {desc}"
    return failures


def parse_probes(probe_file):
    """Extract probe IDs and descriptions from probe_library.md."""
    probes = []
    with open(probe_file, "r", encoding="utf-8") as f:
        text = f.read()
        for line in text.splitlines():
            match = re.search(r"\*\*([A-Z]+-\d+)\*\*.*?\|\s*\*\*(.+?)\*\*", line)
            if match:
                probe_id = match.group(1)
                desc = match.group(2)
                probes.append((probe_id, desc))
    return probes


def build_adversarial_cases(probe_file, failure_file, output_file):
    failures = parse_failures(failure_file)
    probes = parse_probes(probe_file)

    # Mapping from probe prefix to relevant failure codes
    mapping = {
        "SR": ["F1.1", "F1.4"],
        "CD": ["F2.1", "F2.3"],
        "TU": ["F3.1", "F3.3"],
        "GS": ["F4.1", "F4.2", "F4.3"],
    }

    tasks = []
    for i, (probe_id, desc) in enumerate(probes):
        prefix = probe_id.split("-")[0]
        codes = mapping.get(prefix, ["F1.1"])  # default fallback
        failure_code = codes[i % len(codes)]
        failure_tag = failures.get(failure_code, "Unknown failure")

        # Negative case: mishandled
        tasks.append(
            {
                "prospect_input": f"{probe_id}: {desc}",
                "agent_output": f"Agent mishandles: {failure_tag}",
                "label": 0,
                "failure_mode_tag": failure_tag,
                "metadata": "Adversarial case (auto-generated)",
            }
        )

        # Positive case: correctly handled
        tasks.append(
            {
                "prospect_input": f"{probe_id}: {desc}",
                "agent_output": f"Agent correctly addresses: {failure_tag}",
                "label": 1,
                "failure_mode_tag": failure_tag,
                "metadata": "Adversarial case (auto-generated)",
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)


if __name__ == "__main__":
    build_adversarial_cases(
        "docs/probe_library.md",
        "docs/failure_taxonomy.md",
        "data/tasks/adversarial_cases.json",
    )
