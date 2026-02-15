"""
era5_resolution_diagnostic.py — ERA5 Grid Resolution Diagnostic

Counts how many unique ERA5 grid cells the 84,767 buildings map to,
and reports per-cell building counts.  This quantifies the information
bottleneck introduced by the ~0.1° ERA5-Land grid.

Outputs:
  outputs/validation/era5_resolution_diagnostic.csv
  outputs/validation/era5_resolution_diagnostic.md

No feature extraction logic is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path


def run_diagnostic(db_path: Path | None = None) -> pd.DataFrame:
    """Compute ERA5 grid-cell assignment statistics for all buildings.

    Returns:
        DataFrame with one row per grid cell: cell_id, latitude, longitude,
        n_buildings, pct_buildings.
    """
    if db_path is None:
        cfg = load_config()
        root = resolve_path(".")
        db_path = root / cfg["pipeline"]["paths"]["db"]

    con = duckdb.connect(str(db_path), read_only=True)

    # Load land-valid FWI grid cells
    fwi_cells = con.execute("""
        SELECT cell_id, latitude, longitude
        FROM fwi_grid_stats
        WHERE fwi_season_mean IS NOT NULL
        ORDER BY cell_id
    """).df()

    # Load all building centroids
    buildings = con.execute("""
        SELECT building_id, centroid_lat, centroid_lon
        FROM buildings
        ORDER BY building_id
    """).df()

    con.close()

    # Nearest-neighbour assignment (same logic as fire_weather.py)
    cell_coords = fwi_cells[["longitude", "latitude"]].values
    tree = cKDTree(cell_coords)
    bldg_coords = buildings[["centroid_lon", "centroid_lat"]].values
    _, nearest_idx = tree.query(bldg_coords, k=1, workers=-1)

    # Assign cell_id to each building
    buildings["era5_cell_id"] = fwi_cells["cell_id"].values[nearest_idx]

    # Aggregate per cell
    cell_counts = (
        buildings.groupby("era5_cell_id")
        .size()
        .reset_index(name="n_buildings")
    )
    cell_counts = cell_counts.merge(
        fwi_cells[["cell_id", "latitude", "longitude"]],
        left_on="era5_cell_id",
        right_on="cell_id",
    ).drop(columns=["cell_id"])
    cell_counts["pct_buildings"] = (
        cell_counts["n_buildings"] / cell_counts["n_buildings"].sum() * 100
    )
    cell_counts = cell_counts.sort_values("n_buildings", ascending=False).reset_index(drop=True)

    return cell_counts, len(buildings), len(fwi_cells)


def write_outputs(
    cell_counts: pd.DataFrame,
    n_buildings: int,
    n_cells_total: int,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write CSV and Markdown summary to outputs/validation/."""
    if out_dir is None:
        out_dir = resolve_path("outputs/validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "era5_resolution_diagnostic.csv"
    cell_counts.to_csv(csv_path, index=False, float_format="%.4f")

    # Summary statistics
    n_cells_used = len(cell_counts)
    mean_per_cell = cell_counts["n_buildings"].mean()
    median_per_cell = cell_counts["n_buildings"].median()
    min_per_cell = cell_counts["n_buildings"].min()
    max_per_cell = cell_counts["n_buildings"].max()
    std_per_cell = cell_counts["n_buildings"].std()

    # Markdown summary
    md_lines = [
        "# ERA5 Resolution Diagnostic",
        "",
        "## Summary",
        "",
        f"- **Total buildings**: {n_buildings:,}",
        f"- **Total land-valid ERA5 grid cells**: {n_cells_total}",
        f"- **Grid cells with buildings**: {n_cells_used}",
        "- **Grid resolution**: 0.1° x 0.1° (~10 km)",
        "",
        "## Per-Cell Building Distribution",
        "",
        "| Statistic | Value |",
        "|-----------|-------|",
        f"| Mean buildings per cell | {mean_per_cell:,.0f} |",
        f"| Median buildings per cell | {median_per_cell:,.0f} |",
        f"| Std dev | {std_per_cell:,.0f} |",
        f"| Min | {min_per_cell:,} |",
        f"| Max | {max_per_cell:,} |",
        "",
        "## Implication",
        "",
        f"All {n_buildings:,} buildings share fire-weather and dynamic features from only "
        f"**{n_cells_used} unique ERA5 grid cells**. Buildings within the same cell receive "
        f"identical values for all 5 fire-weather climatology features and all 5 event-day "
        f"dynamic features. This means ERA5-derived features cannot discriminate between "
        f"buildings within the same ~10 km cell — discrimination power comes entirely from "
        f"terrain, vegetation, and fire-history features at the sub-grid scale.",
        "",
        "## Per-Cell Detail",
        "",
        "| Cell ID | Latitude | Longitude | Buildings | % of Total |",
        "|---------|----------|-----------|-----------|------------|",
    ]
    for _, row in cell_counts.iterrows():
        md_lines.append(
            f"| {row['era5_cell_id']} | {row['latitude']:.1f} | "
            f"{row['longitude']:.1f} | {row['n_buildings']:,} | "
            f"{row['pct_buildings']:.1f}% |"
        )

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("*Generated by WildfireRisk-EU ERA5 Resolution Diagnostic*")

    md_path = out_dir / "era5_resolution_diagnostic.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return csv_path, md_path


def main() -> None:
    """Run the ERA5 resolution diagnostic and write outputs."""
    print("[era5_resolution_diagnostic] Computing grid-cell assignment ...")
    cell_counts, n_buildings, n_cells_total = run_diagnostic()

    n_cells_used = len(cell_counts)
    mean_per = cell_counts["n_buildings"].mean()
    print(f"  {n_buildings:,} buildings → {n_cells_used} unique ERA5 cells "
          f"(mean {mean_per:,.0f} buildings/cell)")

    csv_path, md_path = write_outputs(cell_counts, n_buildings, n_cells_total)
    print(f"  CSV: {csv_path}")
    print(f"  MD:  {md_path}")
    print("[era5_resolution_diagnostic] Done.")


if __name__ == "__main__":
    main()
