"""Generate results tables across all datasets and methods.

Usage: python -m eval.analyze --results-dir results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None  # type: ignore[assignment]


def _fmt(val, fmt=".3f"):
    """Format a numeric value, or return '-' if None."""
    if val is None:
        return "-"
    return f"{val:{fmt}}"


def _load_results(results_dir: str) -> dict[str, dict]:
    """Load all *_eval.json files from results directory."""
    results: dict[str, dict] = {}
    for path in sorted(Path(results_dir).glob("*_eval.json")):
        with open(path) as f:
            data = json.load(f)
        # Key by filename stem (e.g., "clamber_eval" or "d2c_clamber_eval")
        results[path.stem] = data
    return results


def main_results_table(all_results: dict[str, dict]) -> None:
    """Print the main results table across datasets."""
    # Group by dataset
    datasets = {}
    for name, data in all_results.items():
        ds = data.get("dataset", name)
        if ds not in datasets:
            datasets[ds] = {}
        # Infer method name from filename (e.g., "d2c_clamber" -> "d2c")
        method = name.replace(f"_{ds}", "").replace("_eval", "") or ds
        datasets[ds][method] = data

    # Build table rows
    headers = ["Method"]
    ds_names = sorted(datasets.keys())
    for ds in ds_names:
        if ds != "qulac":
            headers.extend([f"{ds} F1", f"{ds} JQ", f"{ds} SS"])
        else:
            headers.extend([f"{ds} JQ", f"{ds} SS"])

    rows = []
    # Collect all method names
    all_methods = set()
    for ds_data in datasets.values():
        all_methods.update(ds_data.keys())

    for method in sorted(all_methods):
        row = [method]
        for ds in ds_names:
            data = datasets.get(ds, {}).get(method)
            if data is None:
                n_cols = 2 if ds == "qulac" else 3
                row.extend(["-"] * n_cols)
                continue

            if ds != "qulac":
                cn = data.get("clarification_need")
                row.append(_fmt(cn.get("f1") if cn else None))

            jq = data.get("judge_quality")
            row.append(_fmt(jq.get("mean") if jq else None, ".2f"))

            ss = data.get("semantic_similarity")
            row.append(_fmt(ss.get("mean") if ss else None))

        rows.append(row)

    if tabulate:
        print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        # Simple fallback
        print("\n" + "\t".join(headers))
        print("-" * (len(headers) * 12))
        for row in rows:
            print("\t".join(str(x) for x in row))


def per_ambiguity_breakdown(all_results: dict[str, dict]) -> None:
    """Print per-ambiguity-type breakdown (CLAMBER only)."""
    clamber_results = {k: v for k, v in all_results.items() if v.get("dataset") == "clamber"}
    if not clamber_results:
        return

    print("\n--- Per-Ambiguity-Type Breakdown (CLAMBER) ---")
    for name, data in clamber_results.items():
        per_ex = data.get("per_example", [])
        if not per_ex:
            continue

        # We don't have ambiguity_type in per_example by default,
        # so this is a placeholder for when it's added
        print(f"\n  Method: {name}")
        scores_by_type: dict[str, list[float]] = {}
        for ex in per_ex:
            # Future: group by ambiguity_type
            ss = ex.get("semantic_similarity", 0)
            scores_by_type.setdefault("all", []).append(ss)

        for atype, scores in sorted(scores_by_type.items()):
            avg = sum(scores) / len(scores) if scores else 0
            print(f"    {atype}: SS={avg:.3f} (n={len(scores)})")


def score_distribution(all_results: dict[str, dict]) -> None:
    """Print judge score distribution for each method."""
    print("\n--- Judge Score Distribution ---")
    for name, data in sorted(all_results.items()):
        jq = data.get("judge_quality")
        if not jq:
            continue
        dist = jq.get("distribution", {})
        ds = data.get("dataset", "?")
        total = sum(dist.values())
        print(f"\n  {name} ({ds}, n={total}):")
        for score in range(1, 6):
            count = dist.get(score, dist.get(str(score), 0))
            bar = "#" * count
            pct = count / total * 100 if total else 0
            print(f"    {score}: {count:>4} ({pct:5.1f}%) {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze D2C evaluation results")
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory containing *_eval.json files",
    )
    args = parser.parse_args()

    all_results = _load_results(args.results_dir)
    if not all_results:
        print(f"No results found in {args.results_dir}/")
        return

    print(f"Loaded {len(all_results)} result files from {args.results_dir}/")

    main_results_table(all_results)
    per_ambiguity_breakdown(all_results)
    score_distribution(all_results)


if __name__ == "__main__":
    main()
