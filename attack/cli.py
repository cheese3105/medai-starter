"""CLI for the MED-AI prompt-injection benchmark."""

from __future__ import annotations

import argparse
import json
import sys

from attack.attacks import ATTACKS
from attack.benchmark import run_attack_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark prompt injection attacks on MED-AI")
    parser.add_argument("--target-config", default="attack/configs/v0-attack.yaml")
    parser.add_argument("--attacks", nargs="+", choices=sorted(ATTACKS), default=list(ATTACKS))
    parser.add_argument("--tasks", nargs="+", choices=["sentiment", "spam"], default=["sentiment", "spam"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-limit", type=int, default=8)
    parser.add_argument("--injected-limit", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="attack/output/smoke-v0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = run_attack_benchmark(
            target_config=args.target_config,
            attack_names=args.attacks,
            task_names=args.tasks,
            split=args.split,
            target_limit=args.target_limit,
            injected_limit=args.injected_limit,
            sample_size=args.sample_size,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Attack benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps(result["metrics"], indent=2))
    print(f"Wrote {result['case_count']} cases to {result['output_dir']}")


if __name__ == "__main__":
    main()
