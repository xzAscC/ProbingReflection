#!/usr/bin/env python
"""Generate evaluation report from steering experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument(
        "--output-dir",
        default="outputs/steering_experiments",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    results: dict[str, dict[str, dict]] = {}
    for dataset_dir in output_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        results[dataset_name] = {}
        for condition_dir in dataset_dir.iterdir():
            if not condition_dir.is_dir():
                continue
            condition_name = condition_dir.name
            eval_file = condition_dir / "evaluation.json"
            if eval_file.exists():
                with open(eval_file) as f:
                    results[dataset_name][condition_name] = json.load(f)

    report_lines = [
        "# Steering Experiment Results",
        "",
        "## Summary",
        "",
        "| Dataset | Condition | Accuracy | Correct | Total |",
        "|---------|-----------|----------|---------|-------|",
    ]

    for dataset, conditions in sorted(results.items()):
        for condition, data in sorted(conditions.items()):
            accuracy = data.get("overall_accuracy", 0) * 100
            correct = data.get("correct_count", 0)
            total = data.get("total_samples", 0)
            report_lines.append(
                f"| {dataset} | {condition} | {accuracy:.1f}% | {correct} | {total} |"
            )

    report_lines.extend(
        [
            "",
            "## Detailed Results",
            "",
        ]
    )

    for dataset, conditions in sorted(results.items()):
        report_lines.append(f"### {dataset}")
        report_lines.append("")
        for condition, data in sorted(conditions.items()):
            accuracy = data.get("overall_accuracy", 0) * 100
            report_lines.append(f"- **{condition}**: {accuracy:.1f}% accuracy")
        report_lines.append("")

    report_path = output_dir / "evaluation_report.md"
    report_path.write_text("\n".join(report_lines))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
