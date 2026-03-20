"""Entry point for running inference from command line."""

import sys

from probing_reflection import InferenceConfig
from probing_reflection.inference import run_inference


def main() -> None:
    """Parse --limit argument and run inference."""
    limit: int | None = None
    args = sys.argv[1:]

    i = 0
    while i < len(args):
        if args[i] == "--limit" and i + 1 < len(args):
            try:
                limit = int(args[i + 1])
            except ValueError:
                print(f"Invalid limit value: {args[i + 1]}")
                sys.exit(1)
            i += 2
        else:
            i += 1

    config = InferenceConfig(batch_size=limit) if limit is not None else InferenceConfig()

    print(f"Running inference on {limit if limit else 'all'} samples...")
    run_inference(config)
    print(f"Results saved to {config.output_path}")


if __name__ == "__main__":
    main()
