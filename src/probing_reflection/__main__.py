"""Entry point for running inference and evaluation from command line."""

import argparse
import json
import sys
from pathlib import Path

from probing_reflection import InferenceConfig
from probing_reflection.evaluation import evaluate
from probing_reflection.inference import run_inference
from probing_reflection.types import EvaluationConfig, EvaluationReport


def format_report(report: EvaluationReport) -> str:
    """Format evaluation report for console output.

    Args:
        report: The evaluation report to format.

    Returns:
        Formatted string for console display.
    """
    lines = [
        "=== Evaluation Report ===",
        f"Overall Accuracy: {report['overall_accuracy']:.2%} "
        f"({report['correct_count']}/{report['total_samples']})",
        "",
        "By Subject:",
    ]
    for subj, acc in sorted(report["per_subject_accuracy"].items()):
        lines.append(f"  {subj}: {acc:.2%}")
    lines.append("")
    lines.append("By Level:")
    for level, acc in sorted(report["per_level_accuracy"].items()):
        lines.append(f"  Level {level}: {acc:.2%}")
    return "\n".join(lines)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="probing_reflection",
        description="Probing and steering self-reflection in language models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    inference_parser = subparsers.add_parser(
        "inference",
        help="Run inference on dataset",
    )
    inference_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process",
    )

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model outputs",
    )
    eval_parser.add_argument(
        "jsonl_path",
        help="Path to JSONL file containing model outputs",
    )
    eval_parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen3.5-27B",
        help="Judge model name (default: Qwen/Qwen3.5-27B)",
    )
    eval_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Save report to file (JSON format)",
    )
    eval_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    eval_parser.add_argument(
        "--confidence-threshold",
        "-c",
        type=float,
        default=0.7,
        help="Minimum confidence to accept verdict (default: 0.7)",
    )

    return parser


def handle_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation with parsed arguments.

    Args:
        args: Parsed command line arguments.
    """
    config = EvaluationConfig(
        judge_model_name=args.model,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
    )

    print(f"Evaluating {args.jsonl_path} with model {args.model}...")
    report = evaluate(args.jsonl_path, config)

    print(format_report(report))

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {args.output}")


def handle_inference(args: argparse.Namespace) -> None:
    """Run inference with parsed arguments.

    Args:
        args: Parsed command line arguments.
    """
    limit = args.limit
    config = InferenceConfig(batch_size=limit) if limit is not None else InferenceConfig()

    print(f"Running inference on {limit if limit else 'all'} samples...")
    run_inference(config)
    print(f"Results saved to {config.output_path}")


def main() -> None:
    """Parse arguments and dispatch to appropriate subcommand."""
    parser = create_parser()

    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] not in ("inference", "evaluate")):
        sys.argv.insert(1, "inference")

    args = parser.parse_args()

    if args.command == "evaluate":
        handle_evaluate(args)
    else:
        handle_inference(args)


if __name__ == "__main__":
    main()
