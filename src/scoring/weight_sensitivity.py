"""
weight_sensitivity.py — Layer 1 Weight Sensitivity Analysis

Monte Carlo perturbation of group weights to assess rank stability of
the top-10% highest-risk buildings.

Method
------
1. Load group scores (score_terrain, score_vegetation, score_fire_weather,
   score_fire_history) from the risk_scores table.
2. For 500 iterations:
   a. Perturb each of the 4 group weights by ±5% uniform noise.
   b. Re-normalise to sum=1.
   c. Recompute composite_score = weighted sum of group scores.
   d. Identify the top 10% buildings by composite score.
3. Record how many of the original top-10% buildings remain in the top 10%
   across all 500 perturbations (rank stability rate).

Outputs
-------
  outputs/validation/weight_sensitivity.json
    - base_weights, n_iterations, perturbation_pct
    - top10_stability_rate (fraction of original top-10% always in top 10%)
    - stability distribution (histogram of per-building stability counts)
  outputs/validation/weight_sensitivity.png
    - Histogram of per-building rank stability

This is a diagnostic-only analysis; no scoring logic is modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path


def _load_group_scores(db_path: Path) -> pd.DataFrame:
    """Load per-building group scores from risk_scores table."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("""
        SELECT building_id,
               score_terrain, score_vegetation,
               score_fire_weather, score_fire_history,
               composite_score
        FROM risk_scores
        ORDER BY building_id
    """).df()
    con.close()
    return df


def _get_base_weights(cfg: dict) -> dict[str, float]:
    """Get redistributed weights (same logic as engine.py)."""
    cfg_weights = cfg["scoring"]["weights"]
    present = ["terrain", "vegetation", "fire_weather", "fire_history"]
    total = sum(cfg_weights[g] for g in present)
    return {g: cfg_weights[g] / total for g in present}


def run_sensitivity(
    db_path: Path | None = None,
    n_iterations: int = 500,
    perturbation_pct: float = 0.05,
    seed: int = 42,
) -> dict:
    """Run Monte Carlo weight perturbation analysis.

    Args:
        db_path: Path to DuckDB file. Auto-resolved if None.
        n_iterations: Number of MC samples.
        perturbation_pct: Max relative perturbation (±5% = 0.05).
        seed: Random seed.

    Returns:
        Results dict with stability metrics.
    """
    cfg = load_config()
    if db_path is None:
        root = resolve_path(".")
        db_path = root / cfg["pipeline"]["paths"]["db"]

    base_weights = _get_base_weights(cfg)
    groups = list(base_weights.keys())
    score_cols = [f"score_{g}" for g in groups]

    print(f"[weight_sensitivity] Loading group scores from {db_path.name} ...")
    df = _load_group_scores(db_path)
    n_buildings = len(df)
    print(f"  {n_buildings:,} buildings loaded")

    scores_matrix = df[score_cols].values  # (n_buildings, 4)
    base_w = np.array([base_weights[g] for g in groups])

    # Original top 10%
    base_composite = scores_matrix @ base_w
    n_top = int(np.ceil(n_buildings * 0.10))
    original_top10_idx = set(np.argsort(base_composite)[::-1][:n_top])

    print(f"  Base weights: {dict(zip(groups, base_w.round(4)))}")
    print(f"  Top 10% = {n_top:,} buildings")
    print(f"  Running {n_iterations} MC iterations (±{perturbation_pct*100:.0f}% perturbation) ...")

    rng = np.random.RandomState(seed)

    # Count how many times each original top-10% building stays in top 10%
    stability_counts = np.zeros(n_buildings, dtype=int)

    for i in range(n_iterations):
        # Perturb: multiply each weight by (1 + uniform(-pct, +pct))
        noise = rng.uniform(-perturbation_pct, perturbation_pct, size=len(groups))
        perturbed_w = base_w * (1.0 + noise)
        # Re-normalise
        perturbed_w = perturbed_w / perturbed_w.sum()

        composite = scores_matrix @ perturbed_w
        top10_idx = set(np.argsort(composite)[::-1][:n_top])

        # For each original top-10% building, check if still in top 10%
        for idx in original_top10_idx:
            if idx in top10_idx:
                stability_counts[idx] += 1

    # Analyse stability for original top-10% buildings only
    top10_stability = stability_counts[list(original_top10_idx)]

    # "Always in top 10%" = present in all n_iterations
    always_stable = int((top10_stability == n_iterations).sum())
    always_stable_rate = always_stable / n_top

    # Distribution of stability fractions
    stability_fractions = top10_stability / n_iterations
    mean_stability = float(stability_fractions.mean())
    median_stability = float(np.median(stability_fractions))
    p25_stability = float(np.percentile(stability_fractions, 25))
    p75_stability = float(np.percentile(stability_fractions, 75))

    # Histogram bins
    bins = np.linspace(0, 1, 21)  # 5% bins
    hist_counts, _ = np.histogram(stability_fractions, bins=bins)

    print("\n  Results:")
    print(f"    Always in top 10% (100% stable): {always_stable:,} / {n_top:,} "
          f"({always_stable_rate:.1%})")
    print(f"    Mean stability rate: {mean_stability:.3f}")
    print(f"    Median stability rate: {median_stability:.3f}")
    print(f"    IQR: [{p25_stability:.3f}, {p75_stability:.3f}]")

    results = {
        "base_weights": {g: round(w, 4) for g, w in zip(groups, base_w)},
        "n_iterations": n_iterations,
        "perturbation_pct": perturbation_pct,
        "n_buildings_total": n_buildings,
        "n_top10": n_top,
        "top10_always_stable_count": always_stable,
        "top10_always_stable_rate": round(always_stable_rate, 4),
        "stability_mean": round(mean_stability, 4),
        "stability_median": round(median_stability, 4),
        "stability_p25": round(p25_stability, 4),
        "stability_p75": round(p75_stability, 4),
        "stability_histogram": {
            "bin_edges": [round(b, 2) for b in bins.tolist()],
            "counts": hist_counts.tolist(),
        },
    }

    return results, stability_fractions


def write_outputs(
    results: dict,
    stability_fractions: np.ndarray,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write JSON and PNG outputs."""
    if out_dir is None:
        out_dir = resolve_path("outputs/validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / "weight_sensitivity.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # PNG — rank stability histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(stability_fractions, bins=20, range=(0, 1),
            color="#4575b4", edgecolor="white", alpha=0.85)
    ax.axvline(results["stability_mean"], color="#d73027", lw=2, ls="--",
               label=f"Mean = {results['stability_mean']:.2f}")
    ax.axvline(results["stability_median"], color="#fdae61", lw=2, ls="-.",
               label=f"Median = {results['stability_median']:.2f}")
    ax.set_xlabel("Rank Stability (fraction of iterations in top 10%)", fontsize=11)
    ax.set_ylabel("Number of buildings", fontsize=11)
    ax.set_title(
        f"Weight Sensitivity: Top 10% Rank Stability\n"
        f"({results['n_iterations']} iterations, "
        f"\u00b1{results['perturbation_pct']*100:.0f}% weight perturbation)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    png_path = out_dir / "weight_sensitivity.png"
    fig.savefig(str(png_path), dpi=150)
    plt.close(fig)

    return json_path, png_path


def main() -> None:
    """Run weight sensitivity analysis and write outputs."""
    results, stability_fractions = run_sensitivity()
    json_path, png_path = write_outputs(results, stability_fractions)
    print(f"\n  JSON: {json_path}")
    print(f"  PNG:  {png_path}")
    print("[weight_sensitivity] Done.")


if __name__ == "__main__":
    main()
