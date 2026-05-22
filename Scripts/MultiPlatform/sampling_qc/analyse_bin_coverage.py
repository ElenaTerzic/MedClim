#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze STD-centered bin coverage and decide which bins are adequately supported.

THIS SCRIPT READS:
  - bin_coverage_<TAG>_varOK.csv   (the per-bin table) - from bin_coverage_centered_on_STD_DEPTHS.py script

It WRITES:
  1) flags_bins_<TAG>_analysed.csv
     - per (source, period, basin, season, bin): OK/NOT_OK flags for T and S
  2) max_depth_ok_<TAG>_analysed.csv
     - per (source, period, basin, season): deepest OK bin center for T and for S
  3) coverage_fraction_<TAG>_analysed.png
     - per source & period: mean fraction curves vs depth, clipped to data extent
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------
# EDIT ME
# -------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INFILE = Path(str(PROJECT_ROOT / "Stats/Vertical_Sampling/bin_coverage_ALL_SOURCES_STD_CENTERED_varOK.csv"))
OUTDIR = Path(str(PROJECT_ROOT / "Stats/Vertical_Sampling/bin_coverage_diagnostics"))
OUTDIR.mkdir(parents=True, exist_ok=True)

TAG = "ALL_SOURCES_STD_CENTERED_varOK"

# Variable-specific thresholds (same philosophy as your varOK coverage script)
MIN_FRAC_T = 0.10
MIN_FRAC_S = 0.05
MIN_N_T    = 30
MIN_N_S    = 30

# Optional relaxation below a depth (deep ocean often sparse)
RELAX_BELOW_DEPTH = 1500.0
MIN_FRAC_T_DEEP = 0.07
MIN_FRAC_S_DEEP = 0.03
MIN_N_T_DEEP    = 15
MIN_N_S_DEEP    = 15

MAKE_PLOTS = True


# -------------------
# Helpers
# -------------------
def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept both schemas:
      OLD:
        z_center, nT_profiles, nS_profiles, frac_profiles_T, frac_profiles_S
      NEW (varOK):
        bin_center_m, n_T_profiles_in_bin, n_S_profiles_in_bin, frac_T_in_bin, frac_S_in_bin
    Normalize into internal names used below:
      z_center, nT_profiles, nS_profiles, frac_profiles_T, frac_profiles_S
    """
    # new -> old mapping
    new_map = {
        "bin_center_m": "z_center",
        "n_T_profiles_in_bin": "nT_profiles",
        "n_S_profiles_in_bin": "nS_profiles",
        "frac_T_in_bin": "frac_profiles_T",
        "frac_S_in_bin": "frac_profiles_S",
    }

    # if new names exist, rename them
    if "bin_center_m" in df.columns:
        df = df.rename(columns={k: v for k, v in new_map.items() if k in df.columns})

    needed = {
        "source", "period", "basin", "season",
        "z_center", "n_profiles_total",
        "nT_profiles", "nS_profiles",
        "frac_profiles_T", "frac_profiles_S",
    }
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # numeric conversion
    for c in ["z_center", "n_profiles_total", "nT_profiles", "nS_profiles", "frac_profiles_T", "frac_profiles_S"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _thresholds_for_depth(z: float, var: str):
    """Depth-dependent threshold relaxation."""
    if z >= RELAX_BELOW_DEPTH:
        if var == "T":
            return MIN_N_T_DEEP, MIN_FRAC_T_DEEP
        else:
            return MIN_N_S_DEEP, MIN_FRAC_S_DEEP
    else:
        if var == "T":
            return MIN_N_T, MIN_FRAC_T
        else:
            return MIN_N_S, MIN_FRAC_S


def is_ok(row, var="T") -> bool:
    z = float(row["z_center"])
    min_n, min_f = _thresholds_for_depth(z, var)

    if var == "T":
        return (row["nT_profiles"] >= min_n) and (row["frac_profiles_T"] >= min_f)
    else:
        return (row["nS_profiles"] >= min_n) and (row["frac_profiles_S"] >= min_f)


def deepest_ok_per_group(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["source", "period", "basin", "season"]
    rows = []

    for key, g in df.groupby(group_cols, sort=False):
        g = g.sort_values("z_center")
        okT = g.apply(lambda r: is_ok(r, "T"), axis=1).to_numpy(bool)
        okS = g.apply(lambda r: is_ok(r, "S"), axis=1).to_numpy(bool)

        z = g["z_center"].to_numpy(float)
        z_ok_T = float(np.nanmax(z[okT])) if np.any(okT) else np.nan
        z_ok_S = float(np.nanmax(z[okS])) if np.any(okS) else np.nan

        rows.append({
            **dict(zip(group_cols, key)),
            "deepest_ok_bin_T_m": z_ok_T,
            "deepest_ok_bin_S_m": z_ok_S,
        })

    return pd.DataFrame(rows)


def flags_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ok_T"] = out.apply(lambda r: is_ok(r, "T"), axis=1)
    out["ok_S"] = out.apply(lambda r: is_ok(r, "S"), axis=1)
    return out


def plot_coverage(df: pd.DataFrame, out_png: Path):
    """
    Plot mean frac(T) and frac(S) vs depth, aggregated over basin+season
    for each (source, period). Clip x-axis to deepest bin where there is any data.
    """
    # mean across basin/season (each group equally weighted)
    agg = (df.groupby(["source", "period", "z_center"], as_index=False)
             .agg(frac_T=("frac_profiles_T", "mean"),
                  frac_S=("frac_profiles_S", "mean"),
                  nT=("nT_profiles", "sum"),
                  nS=("nS_profiles", "sum")))

    combos = agg[["source", "period"]].drop_duplicates().sort_values(["source", "period"])
    n = len(combos)
    if n == 0:
        return

    fig_h = max(6, 2.2 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, fig_h), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, combos.iterrows()):
        src = row["source"]; per = row["period"]
        g = agg[(agg["source"] == src) & (agg["period"] == per)].sort_values("z_center")

        ax.plot(g["z_center"], g["frac_T"], marker="o", linewidth=1.2, label="T fraction")
        ax.plot(g["z_center"], g["frac_S"], marker="o", linewidth=1.2, label="S fraction")

        # clip to deepest depth where ANY data exist (T or S) in this source|period
        has_any = (g["nT"].to_numpy(int) > 0) | (g["nS"].to_numpy(int) > 0)
        if np.any(has_any):
            zmax_plot = float(g.loc[has_any, "z_center"].max())
            ax.set_xlim(0, zmax_plot)

        ax.set_title(f"{src} | {per}")
        ax.set_ylabel("Mean fraction of profiles with ≥1 sample in bin")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Bin center depth (m)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    df = pd.read_csv(INFILE)
    df = _ensure_schema(df)

    out_flags = OUTDIR / f"flags_bins_{TAG}_analysed.csv"
    out_depth = OUTDIR / f"max_depth_ok_{TAG}_analysed.csv"
    out_png   = OUTDIR / f"coverage_fraction_{TAG}_analysed.png"

    # 1) flags per bin
    df_flags = flags_table(df)
    df_flags.to_csv(out_flags, index=False)
    print("Wrote:", out_flags)

    # 2) deepest OK depth per group
    df_depth = deepest_ok_per_group(df)
    df_depth.to_csv(out_depth, index=False)
    print("Wrote:", out_depth)

    # 3) plot
    if MAKE_PLOTS:
        plot_coverage(df, out_png)
        print("Wrote:", out_png)

    print("\nDone.")


if __name__ == "__main__":
    main()
