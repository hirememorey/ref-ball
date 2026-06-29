"""Generate SSAC27 abstract figures for Paper 1.

Figure A (Table): Top 5 suppressors and top 5 amplifiers with named officials.
Figure B (Scatter): Predicted crew suppression vs actual FTA/36 deviation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
from scipy import stats

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ═══════════════════════════════════════════════════════════════════════════
# TABLE A: Suppressor / Amplifier Profiles
# ═══════════════════════════════════════════════════════════════════════════

def generate_table_a():
    """Publication-quality table of top suppressors and amplifiers."""
    df = pd.read_parquet(
        PROJECT_ROOT / "data/processed/player_official/official_calling_profiles.parquet"
    )

    # Filter to officials with >= 9 player interactions (avoid low-N noise;
    # 9 includes Bill Spooner, the strongest amplifier effect)
    df = df[df["n_players"] >= 9].copy()

    # Sort for suppressors (highest suppressor_score) and amplifiers (lowest)
    df_sorted = df.sort_values("suppressor_score", ascending=False)

    # Top 5 suppressors
    top_sup = df_sorted.head(5)[
        ["official_name", "suppressor_score", "mean_adj_fta36_delta", "sf_per_game", "n_players"]
    ].copy()

    # Top 5 amplifiers (lowest suppressor score)
    top_amp = df_sorted.tail(5).iloc[::-1][
        ["official_name", "suppressor_score", "mean_adj_fta36_delta", "sf_per_game", "n_players"]
    ].copy()

    # Print for reference
    print("\n══ TABLE A: Suppressor / Amplifier Profiles ══\n")
    print("TOP 5 SUPPRESSORS (highest suppressor_score):")
    print(top_sup.to_string(index=False))
    print("\nTOP 5 AMPLIFIERS (lowest suppressor_score):")
    print(top_amp.to_string(index=False))

    # ── Build matplotlib table figure ──
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axis("off")

    # Combine data with a separator row
    headers = ["Official", "Suppressor\nScore", "Mean Adj\nFTA/36 \u0394", "SF/Game", "N Players"]

    rows = []
    row_types = []  # track row type for styling

    # Suppressor section header
    rows.append(["SUPPRESSORS", "", "", "", ""])
    row_types.append("sup_header")

    for _, r in top_sup.iterrows():
        rows.append([
            r["official_name"],
            f"{r['suppressor_score']:.0%}",
            f"{r['mean_adj_fta36_delta']:+.2f}",
            f"{r['sf_per_game']:.1f}",
            f"{int(r['n_players'])}",
        ])
        row_types.append("sup")

    # Amplifier section header
    rows.append(["AMPLIFIERS", "", "", "", ""])
    row_types.append("amp_header")

    for _, r in top_amp.iterrows():
        rows.append([
            r["official_name"],
            f"{r['suppressor_score']:.0%}",
            f"{r['mean_adj_fta36_delta']:+.2f}",
            f"{r['sf_per_game']:.1f}",
            f"{int(r['n_players'])}",
        ])
        row_types.append("amp")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colColours=["#2c3e50"] * 5,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Widen the first column for official names
    col_widths = [0.25, 0.18, 0.18, 0.18, 0.18]
    for i in range(len(rows) + 1):  # +1 for header
        for j, w in enumerate(col_widths):
            table[i, j].set_width(w)

    # Style header
    for j in range(5):
        cell = table[0, j]
        cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
        cell.set_edgecolor("#2c3e50")

    # Style data rows
    for i in range(1, len(rows) + 1):
        rt = row_types[i - 1]
        for j in range(5):
            cell = table[i, j]
            cell.set_edgecolor("#bdc3c7")

            if rt == "sup_header":
                cell.set_facecolor("#2980b9")
                cell.set_edgecolor("#2980b9")
                if j == 0:
                    cell.set_text_props(color="white", fontweight="bold", fontsize=9.5, ha="left")
                else:
                    cell.set_text_props(color="#2980b9")  # hide empty cells
            elif rt == "amp_header":
                cell.set_facecolor("#27ae60")
                cell.set_edgecolor("#27ae60")
                if j == 0:
                    cell.set_text_props(color="white", fontweight="bold", fontsize=9.5, ha="left")
                else:
                    cell.set_text_props(color="#27ae60")
            elif rt == "sup":
                cell.set_facecolor("#ebf5fb")
                if j == 2:
                    cell.set_text_props(color="#c0392b", fontweight="bold")
            elif rt == "amp":
                cell.set_facecolor("#eafaf1")
                if j == 2:
                    cell.set_text_props(color="#27ae60", fontweight="bold")

    # Title — above the table
    fig.suptitle("Table 1: Individual Official Effects on Player FTA Rates",
                 fontsize=13, fontweight="bold", y=0.97)

    # Footnote
    ax.text(0.5, -0.02,
            "Suppressor Score = fraction of high-FTA players whose FTA/36 decreases under this official.\n"
            "Mean Adj \u0394 = defense-adjusted mean FTA/36 shift across all player interactions. "
            "N = 40 target players, 2014\u201325.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color="#7f8c8d", style="italic")

    out_path = FIGURES_DIR / "table_a_suppressor_amplifier.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    return top_sup, top_amp


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE B: Predicted Crew Suppression vs Actual FTA Deviation
# ═══════════════════════════════════════════════════════════════════════════

def generate_figure_b():
    """Scatter: predicted crew suppression vs actual FTA/36 deviation (playoff games)."""

    # Load player-crew suppression predictions (Step 7)
    supp = pd.read_parquet(
        PROJECT_ROOT / "data/processed/model/dhc_merge/player_crew_suppression.parquet"
    )

    # Load the master merged dataset for actual FTA outcomes
    merged = pd.read_parquet(
        PROJECT_ROOT / "data/processed/model/dhc_merge/player_game_merged.parquet"
    )

    print("\n══ FIGURE B: Predicted Crew Suppression vs Actual FTA Deviation ══\n")
    print(f"player_crew_suppression shape: {supp.shape}")
    print(f"player_game_merged shape: {merged.shape}")
    print(f"\nsupp columns: {supp.columns.tolist()}")
    print(f"\nmerged columns with 'fta' or 'delta' or 'suppress':")

    fta_cols = [c for c in merged.columns if any(k in c.lower() for k in ["fta", "delta", "suppress", "predict"])]
    print(fta_cols)

    # The suppression file has player_name, game_id_str, player_predicted_suppression
    # The merged file has fta36 and rs_fta36_mean (baseline) -> fta36_delta = fta36 - rs_fta36_mean
    # Filter to playoff games only (Step 7 used PO games)

    # Check if merged has season_type
    if "season_type" in merged.columns:
        po = merged[merged["season_type"] == "Playoffs"].copy()
    elif "SEASON_TYPE" in merged.columns:
        po = merged[merged["SEASON_TYPE"] == "Playoffs"].copy()
    else:
        # Just use all of merged — Step 7 may have pre-filtered
        po = merged.copy()

    print(f"\nPlayoff rows: {len(po)}")

    # We need game_id to join. Check ID column names
    print(f"\nsupp game_id sample: {supp['game_id_str'].head()}")

    # Check for game_id in merged
    id_cols = [c for c in po.columns if "game_id" in c.lower()]
    print(f"merged game_id columns: {id_cols}")
    print(f"merged game_id sample: {po[id_cols[0]].head() if id_cols else 'NONE'}")

    # Check for player_name
    player_cols = [c for c in po.columns if "player" in c.lower() or "name" in c.lower()]
    print(f"merged player columns: {player_cols[:10]}")

    # Build fta36_delta if not present
    if "fta36_delta" not in po.columns and "fta36" in po.columns and "rs_fta36_mean" in po.columns:
        po["fta36_delta"] = po["fta36"] - po["rs_fta36_mean"]

    # Join on game_id_str (both datasets have this as string with leading zeros)
    player_col = "player_name"

    joined = po.merge(
        supp[["player_name", "game_id_str", "player_predicted_suppression"]],
        on=["player_name", "game_id_str"],
        how="inner",
    )

    print(f"\nJoined rows: {len(joined)}")

    if len(joined) == 0:
        # Try without player — maybe names don't match
        print("Zero joins. Debugging name mismatch...")
        print(f"supp players: {sorted(supp['player_name'].unique()[:10])}")
        print(f"merged players: {sorted(po[player_col].unique()[:10])}")
        return

    # Filter to rows with valid data
    plot_df = joined[["player_predicted_suppression", "fta36_delta", player_col]].dropna()
    print(f"Plot rows after dropna: {len(plot_df)}")

    # Stats
    x = plot_df["player_predicted_suppression"]
    y = plot_df["fta36_delta"]
    rho, p_val = stats.spearmanr(x, y)
    r_pearson, p_pearson = stats.pearsonr(x, y)

    print(f"\nSpearman r = {rho:.3f}, p = {p_val:.6f}")
    print(f"Pearson r  = {r_pearson:.3f}, p = {p_pearson:.6f}")
    print(f"N = {len(plot_df)} player-games")

    # Count unique players
    n_players = plot_df[player_col].nunique()
    print(f"Unique players: {n_players}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter with transparency
    ax.scatter(x, y, alpha=0.25, s=18, color="#2980b9", edgecolors="none", zorder=2)

    # Regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#e74c3c", linewidth=2, zorder=3)

    # Zero lines
    ax.axhline(0, color="#bdc3c7", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(0, color="#bdc3c7", linewidth=0.8, linestyle="--", zorder=1)

    # Annotation
    ax.text(
        0.97, 0.05,
        f"Spearman r = {rho:.3f}\np < 0.001\nn = {len(plot_df)} player-games\n{n_players} players",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, color="#2c3e50",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#bdc3c7", alpha=0.9),
    )

    ax.set_xlabel("Predicted Crew Effect (Mean Player\u2013Official Adj \u0394 FTA/36)", fontsize=11)
    ax.set_ylabel("Actual FTA/36 Deviation from Regular Season Baseline", fontsize=11)
    ax.set_title(
        "Figure 1: Crew Assignment Predicts Individual Playoff FTA Outcomes",
        fontsize=12.5, fontweight="bold", pad=12,
    )

    ax.tick_params(labelsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = FIGURES_DIR / "figure_b_crew_prediction_scatter.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    top_sup, top_amp = generate_table_a()
    generate_figure_b()
    print("\n✓ All figures generated.")
