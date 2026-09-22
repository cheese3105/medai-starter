"""Compare defense effectiveness across all strategies.

Runs the attack benchmark with each defense strategy and generates
a comparative report showing which defenses work best.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from attack.benchmark import run_attack_benchmark
from attack.defenses import DEFENSES


def run_defense_comparison(
    *,
    target_config: str,
    attacks: list[str],
    tasks: list[str],
    defenses: list[str] | None,
    split: str,
    target_limit: int,
    injected_limit: int,
    sample_size: int,
    seed: int,
    output_base_dir: str,
) -> dict[str, Any]:
    """Run attack benchmark with multiple defense strategies."""

    if defenses is None:
        defenses = list(DEFENSES.keys())

    unknown = set(defenses) - set(DEFENSES.keys())
    if unknown:
        raise ValueError(f"Unknown defenses: {', '.join(sorted(unknown))}")

    base_path = Path(output_base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    print("=" * 80)
    print("DEFENSE COMPARISON BENCHMARK")
    print("=" * 80)
    print(f"Target config: {target_config}")
    print(f"Attacks: {', '.join(attacks)}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Defenses: {', '.join(defenses)}")
    print(f"Sample size: {sample_size} per task")
    print(f"Seed: {seed}")
    print("=" * 80)

    for defense in defenses:
        print(f"\n{'#' * 80}")
        print(f"# Running with defense: {defense.upper()}")
        print(f"{'#' * 80}\n")

        output_dir = str(base_path / f"defense-{defense}")
        start_time = time.time()

        try:
            result = run_attack_benchmark(
                target_config=target_config,
                attack_names=attacks,
                task_names=tasks,
                split=split,
                target_limit=target_limit,
                injected_limit=injected_limit,
                sample_size=sample_size,
                seed=seed,
                output_dir=output_dir,
                defense=defense,
            )

            elapsed = time.time() - start_time
            result["elapsed_seconds"] = elapsed
            result["success"] = True
            results[defense] = result

            print(f"\n✓ Completed {defense} in {elapsed:.1f}s")
            print(f"  Output: {output_dir}")

        except Exception as exc:
            elapsed = time.time() - start_time
            print(f"\n✗ Failed {defense} after {elapsed:.1f}s: {exc}")
            results[defense] = {
                "success": False,
                "error": str(exc),
                "elapsed_seconds": elapsed,
            }

    # Generate comparison summary
    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}\n")

    comparison = generate_comparison_report(results)
    comparison_path = base_path / "comparison_summary.json"
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\nComparison summary written to: {comparison_path}")

    # Print summary table
    print_comparison_table(comparison)

    # Generate markdown report
    markdown_path = base_path / "COMPARISON_REPORT.md"
    generate_markdown_report(comparison, markdown_path)
    print(f"Markdown report written to: {markdown_path}")

    return comparison


def generate_comparison_report(results: dict[str, Any]) -> dict[str, Any]:
    """Extract key metrics for comparison."""

    comparison = {
        "defenses": {},
        "summary": {},
    }

    for defense, result in results.items():
        if not result.get("success", False):
            comparison["defenses"][defense] = {
                "status": "failed",
                "error": result.get("error", "Unknown error"),
            }
            continue

        metrics = result["metrics"]

        # Aggregate attack metrics
        attacks = metrics.get("attacks", [])
        if not attacks:
            comparison["defenses"][defense] = {
                "status": "no_data",
            }
            continue

        avg_asv = sum(a["asv"] for a in attacks) / len(attacks)
        avg_mr = sum(a["matching_rate"] for a in attacks) / len(attacks)
        avg_target_acc = sum(a["target_accuracy_under_attack"] for a in attacks) / len(attacks)

        pna_t = metrics["pna_t"]["value"]

        comparison["defenses"][defense] = {
            "status": "success",
            "pna_t": round(pna_t, 4),
            "avg_attack_success_rate": round(avg_asv, 4),
            "avg_matching_rate": round(avg_mr, 4),
            "avg_target_preservation": round(avg_target_acc, 4),
            "target_accuracy_drop": round(pna_t - avg_target_acc, 4),
            "elapsed_seconds": round(result.get("elapsed_seconds", 0), 2),
            "case_count": result.get("case_count", 0),
            "by_attack": {
                attack["attack_method"]: {
                    "asv": round(attack["asv"], 4),
                    "target_acc": round(attack["target_accuracy_under_attack"], 4),
                }
                for attack in attacks
            },
        }

    # Calculate summary statistics
    successful = {k: v for k, v in comparison["defenses"].items()
                  if v.get("status") == "success"}

    if successful:
        comparison["summary"] = {
            "best_defense_by_asv": min(successful.items(),
                                        key=lambda x: x[1]["avg_attack_success_rate"])[0],
            "best_defense_by_target_preservation": max(successful.items(),
                                                        key=lambda x: x[1]["avg_target_preservation"])[0],
            "baseline_asv": successful.get("none", {}).get("avg_attack_success_rate"),
            "best_asv": min(v["avg_attack_success_rate"] for v in successful.values()),
            "improvement_vs_baseline": None,
        }

        if "none" in successful:
            baseline = successful["none"]["avg_attack_success_rate"]
            best = comparison["summary"]["best_asv"]
            if baseline > 0:
                improvement = ((baseline - best) / baseline) * 100
                comparison["summary"]["improvement_vs_baseline"] = round(improvement, 2)

    return comparison


def print_comparison_table(comparison: dict[str, Any]) -> None:
    """Print ASCII comparison table."""

    defenses = comparison["defenses"]
    successful = {k: v for k, v in defenses.items() if v.get("status") == "success"}

    if not successful:
        print("No successful defense runs to compare.")
        return

    print("\n" + "=" * 100)
    print(f"{'Defense':<20} {'ASR ↓':<10} {'Target Acc ↑':<12} {'MR':<10} {'Time (s)':<10}")
    print("=" * 100)

    # Sort by ASR (lower is better)
    sorted_defenses = sorted(successful.items(),
                             key=lambda x: x[1]["avg_attack_success_rate"])

    for defense, metrics in sorted_defenses:
        print(f"{defense:<20} "
              f"{metrics['avg_attack_success_rate']:<10.3f} "
              f"{metrics['avg_target_preservation']:<12.3f} "
              f"{metrics['avg_matching_rate']:<10.3f} "
              f"{metrics['elapsed_seconds']:<10.1f}")

    print("=" * 100)
    print("\nMetrics:")
    print("  ASR ↓  = Attack Success Rate (lower is better)")
    print("  Target Acc ↑ = Target task accuracy under attack (higher is better)")
    print("  MR = Matching Rate with clean predictions")
    print()

    summary = comparison.get("summary", {})
    if summary:
        print("Best Performers:")
        print(f"  • Lowest ASR: {summary.get('best_defense_by_asv')} "
              f"({successful[summary['best_defense_by_asv']]['avg_attack_success_rate']:.3f})")
        print(f"  • Best Target Preservation: {summary.get('best_defense_by_target_preservation')} "
              f"({successful[summary['best_defense_by_target_preservation']]['avg_target_preservation']:.3f})")

        if summary.get("improvement_vs_baseline"):
            print(f"  • Improvement vs baseline: {summary['improvement_vs_baseline']:.1f}% ASR reduction")


def generate_markdown_report(comparison: dict[str, Any], output_path: Path) -> None:
    """Generate detailed markdown comparison report."""

    defenses = comparison["defenses"]
    successful = {k: v for k, v in defenses.items() if v.get("status") == "success"}

    lines = [
        "# Defense Strategy Comparison Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
    ]

    summary = comparison.get("summary", {})
    if summary:
        lines.extend([
            f"- **Best defense (lowest ASR)**: `{summary.get('best_defense_by_asv')}`",
            f"- **Best defense (target preservation)**: `{summary.get('best_defense_by_target_preservation')}`",
            f"- **Baseline ASR**: {summary.get('baseline_asv', 'N/A'):.3f}" if summary.get('baseline_asv') else "- **Baseline ASR**: N/A",
            f"- **Best ASR achieved**: {summary.get('best_asv', 'N/A'):.3f}" if summary.get('best_asv') else "- **Best ASR achieved**: N/A",
        ])

        if summary.get("improvement_vs_baseline"):
            lines.append(f"- **Improvement vs baseline**: {summary['improvement_vs_baseline']:.1f}% ASR reduction")

    lines.extend([
        "",
        "## Overall Comparison",
        "",
        "| Defense | ASR ↓ | Target Acc ↑ | Acc Drop | MR | Time (s) |",
        "|---------|-------|--------------|----------|-----|----------|",
    ])

    sorted_defenses = sorted(successful.items(),
                             key=lambda x: x[1]["avg_attack_success_rate"])

    for defense, metrics in sorted_defenses:
        lines.append(
            f"| `{defense}` "
            f"| {metrics['avg_attack_success_rate']:.3f} "
            f"| {metrics['avg_target_preservation']:.3f} "
            f"| {metrics['target_accuracy_drop']:.3f} "
            f"| {metrics['avg_matching_rate']:.3f} "
            f"| {metrics['elapsed_seconds']:.1f} |"
        )

    lines.extend([
        "",
        "**Metrics:**",
        "- **ASR** (Attack Success Rate): Lower is better — % of attacks that successfully hijacked the model",
        "- **Target Acc** (Target Accuracy): Higher is better — % of attacks where model still answered correctly",
        "- **Acc Drop**: Accuracy drop from clean baseline",
        "- **MR** (Matching Rate): Agreement with clean injected predictions",
        "",
        "## Defense Details",
        "",
    ])

    for defense, metrics in sorted_defenses:
        lines.extend([
            f"### {defense}",
            "",
            f"- **Attack Success Rate**: {metrics['avg_attack_success_rate']:.3f}",
            f"- **Target Preservation**: {metrics['avg_target_preservation']:.3f}",
            f"- **PNA-T (clean)**: {metrics['pna_t']:.3f}",
            f"- **Elapsed time**: {metrics['elapsed_seconds']:.1f}s",
            "",
            "**By Attack Method:**",
            "",
        ])

        for attack_method, attack_metrics in metrics.get("by_attack", {}).items():
            lines.append(
                f"- `{attack_method}`: ASR={attack_metrics['asv']:.3f}, "
                f"Target Acc={attack_metrics['target_acc']:.3f}"
            )

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare defense strategies against prompt injection attacks"
    )
    parser.add_argument("--target-config", default="attack/configs/v0-attack.yaml")
    parser.add_argument(
        "--attacks", nargs="+",
        default=["naive", "escape_characters", "context_ignoring", "fake_completion", "combined"],
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=["sentiment", "spam", "duplicate", "hate", "nli"],
    )
    parser.add_argument(
        "--defenses", nargs="+", choices=sorted(DEFENSES),
        help="Specific defenses to compare (default: all)",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-limit", type=int, default=20)
    parser.add_argument("--injected-limit", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="attack/output/defense-comparison")

    args = parser.parse_args()

    try:
        run_defense_comparison(
            target_config=args.target_config,
            attacks=args.attacks,
            tasks=args.tasks,
            defenses=args.defenses,
            split=args.split,
            target_limit=args.target_limit,
            injected_limit=args.injected_limit,
            sample_size=args.sample_size,
            seed=args.seed,
            output_base_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Defense comparison failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
