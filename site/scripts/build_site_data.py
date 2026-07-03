#!/usr/bin/env python3
"""Build static JSON for ref-ball.site from parquet sources."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_DATA = REPO_ROOT / "site" / "src" / "data"

PATHS = {
    "official_profiles": REPO_ROOT / "data/processed/player_official/official_calling_profiles.parquet",
    "defensive_adj": REPO_ROOT / "data/processed/player_official/defensive_adjusted_interactions.parquet",
    "player_official": REPO_ROOT / "data/processed/player_official/player_official_interactions.parquet",
    "ref_profiles": REPO_ROOT / "data/processed/ref_profiles.parquet",
    "crew": REPO_ROOT / "data/processed/crew_assignments.parquet",
}


def slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = slug.replace("'", "")
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def normalize_official_id(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(int(float(value)))


def to_json_value(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def round_or_none(value, digits: int = 3):
    value = to_json_value(value)
    if value is None:
        return None
    return round(float(value), digits)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_frames() -> dict[str, pd.DataFrame]:
    missing = [k for k, p in PATHS.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(", ".join(f"{k} ({PATHS[k]})" for k in missing))
    return {k: pd.read_parquet(p) for k, p in PATHS.items()}


def build_crew_roles(crew: pd.DataFrame) -> dict[str, str]:
    crew = crew.copy()
    crew["official_id"] = crew["official_id"].apply(normalize_official_id)
    counts = crew.groupby(["official_id", "role"]).size().reset_index(name="count")
    roles: dict[str, str] = {}
    for official_id, group in counts.groupby("official_id"):
        if official_id is None:
            continue
        roles[official_id] = str(group.sort_values("count", ascending=False).iloc[0]["role"])
    return roles


def build_official_index(officials: pd.DataFrame, crew_roles: dict[str, str]) -> list[dict]:
    rows = []
    for _, row in officials.iterrows():
        oid = normalize_official_id(row.get("official_id"))
        rows.append({
            "official_id": oid,
            "pbp_name": to_json_value(row.get("official_pbp_name")),
            "official_name": to_json_value(row.get("official_name")),
            "suppressor_score": round_or_none(row.get("suppressor_score")),
            "mean_adj_fta36_delta": round_or_none(row.get("mean_adj_fta36_delta")),
            "mean_raw_fta36_delta": round_or_none(row.get("mean_raw_fta36_delta")),
            "n_players": to_json_value(row.get("n_players")),
            "n_pairs": to_json_value(row.get("n_pairs")),
            "total_games": to_json_value(row.get("total_games")),
            "n_games": to_json_value(row.get("n_games")),
            "sf_per_game": round_or_none(row.get("sf_per_game")),
            "sf_per_game_RS": round_or_none(row.get("sf_per_game_RS")),
            "sf_per_game_PO": round_or_none(row.get("sf_per_game_PO")),
            "sf_per_game_delta": round_or_none(row.get("sf_per_game_delta")),
            "sf_pct_of_fouls": round_or_none(row.get("sf_pct_of_fouls")),
            "rs_po_delta": round_or_none(row.get("rs_po_delta")),
            "n_players_rs": to_json_value(row.get("n_players_rs")),
            "n_players_po": to_json_value(row.get("n_players_po")),
            "crew_role": crew_roles.get(oid),
        })
    return rows


def build_official_detail(
    row: pd.Series,
    defensive_adj: pd.DataFrame,
    ref_profiles: pd.DataFrame,
    crew_roles: dict[str, str],
    prev_id: str | None,
    next_id: str | None,
) -> dict:
    official_id = normalize_official_id(row.get("official_id"))
    pbp_name = to_json_value(row.get("official_pbp_name"))

    player_rows = defensive_adj[defensive_adj["official_pbp_name"] == pbp_name].copy()
    player_rows = player_rows.sort_values("defense_adjusted_fta36_delta", ascending=True)

    player_deltas = []
    for _, p in player_rows.iterrows():
        player_deltas.append({
            "player_name": to_json_value(p.get("player_name")),
            "player_slug": slugify(str(p.get("player_name"))),
            "n_games_with": to_json_value(p.get("n_games_with")),
            "n_games_without": to_json_value(p.get("n_games_without")),
            "fta36_with": round_or_none(p.get("fta36_with")),
            "fta36_without": round_or_none(p.get("fta36_without")),
            "raw_fta36_delta": round_or_none(p.get("raw_fta36_delta")),
            "defense_adjusted_fta36_delta": round_or_none(p.get("defense_adjusted_fta36_delta")),
            "defrtg_with": round_or_none(p.get("defrtg_with")),
            "defrtg_without": round_or_none(p.get("defrtg_without")),
            "defrtg_delta": round_or_none(p.get("defrtg_delta")),
            "adjustment_magnitude": round_or_none(p.get("adjustment_magnitude")),
        })

    ref_row = pd.DataFrame()
    if official_id:
        ref_row = ref_profiles[ref_profiles["official_id"] == int(official_id)]
    if ref_row.empty and pbp_name:
        ref_row = ref_profiles[ref_profiles["caller_official_name"] == pbp_name]

    foul_type_breakdown = {
        "sf_per_game_RS": round_or_none(row.get("sf_per_game_RS")),
        "sf_per_game_PO": round_or_none(row.get("sf_per_game_PO")),
        "sf_per_game_delta": round_or_none(row.get("sf_per_game_delta")),
        "n_shooting_fouls_RS": to_json_value(ref_row.iloc[0]["n_shooting_fouls_RS"]) if not ref_row.empty else None,
        "n_shooting_fouls_PO": to_json_value(ref_row.iloc[0]["n_shooting_fouls_PO"]) if not ref_row.empty else None,
        "note": "Foul-type-specific rates from ref_profiles.parquet; full foul-type classification (Layer 2) pending — see /papers/ssac27.",
    }

    return {
        "official_id": official_id,
        "pbp_name": pbp_name,
        "official_name": to_json_value(row.get("official_name")),
        "headline": {
            "suppressor_score": round_or_none(row.get("suppressor_score")),
            "mean_adj_fta36_delta": round_or_none(row.get("mean_adj_fta36_delta")),
            "mean_raw_fta36_delta": round_or_none(row.get("mean_raw_fta36_delta")),
            "n_players": to_json_value(row.get("n_players")),
            "n_pairs": to_json_value(row.get("n_pairs")),
            "total_games": to_json_value(row.get("total_games")),
            "total_sf": to_json_value(row.get("total_sf")),
            "sf_per_game": round_or_none(row.get("sf_per_game")),
            "sf_per_game_RS": round_or_none(row.get("sf_per_game_RS")),
            "sf_per_game_PO": round_or_none(row.get("sf_per_game_PO")),
            "sf_pct_of_fouls": round_or_none(row.get("sf_pct_of_fouls")),
        },
        "player_deltas": player_deltas,
        "foul_type_breakdown": foul_type_breakdown,
        "context": {
            "rs_po_delta": round_or_none(row.get("rs_po_delta")),
            "n_players_rs": to_json_value(row.get("n_players_rs")),
            "n_players_po": to_json_value(row.get("n_players_po")),
            "crew_role": crew_roles.get(official_id),
        },
        "navigation": {
            "prev_by_suppressor": prev_id,
            "next_by_suppressor": next_id,
        },
    }


def build_player_index(player_official: pd.DataFrame) -> list[dict]:
    grouped = player_official.groupby("player_name").agg(
        n_officials=("official_id", "nunique"),
        total_games_with=("n_games_with_official", "sum"),
        baseline_fta_per_game=("baseline_fta_per_game", "first"),
        baseline_sf_per_game=("baseline_sf_per_game", "first"),
    ).reset_index()
    rows = []
    for _, row in grouped.iterrows():
        name = str(row["player_name"])
        rows.append({
            "player_name": name,
            "slug": slugify(name),
            "n_officials": int(row["n_officials"]),
            "total_games_with": int(row["total_games_with"]),
            "baseline_fta_per_game": round_or_none(row["baseline_fta_per_game"]),
            "baseline_sf_per_game": round_or_none(row["baseline_sf_per_game"]),
        })
    return sorted(rows, key=lambda r: r["player_name"])


def build_player_detail(
    player_name: str,
    player_official: pd.DataFrame,
    defensive_adj: pd.DataFrame,
) -> dict:
    po = player_official[player_official["player_name"] == player_name].copy()
    da = defensive_adj[defensive_adj["player_name"] == player_name].copy()
    baseline_row = po.iloc[0] if not po.empty else None

    pbp_to_id = (
        po.dropna(subset=["official_id"])
        .drop_duplicates("official_pbp_name")
        .set_index("official_pbp_name")["official_id"]
        .apply(normalize_official_id)
        .to_dict()
    )

    official_deltas = []
    for _, row in po.sort_values("fta_delta").iterrows():
        official_deltas.append({
            "official_id": normalize_official_id(row.get("official_id")),
            "pbp_name": to_json_value(row.get("official_pbp_name")),
            "official_name": to_json_value(row.get("official_name")),
            "n_games_with": to_json_value(row.get("n_games_with_official")),
            "fta_with": to_json_value(row.get("fta_with")),
            "fta_without": to_json_value(row.get("fta_without")),
            "fta_per_game_with": round_or_none(row.get("fta_per_game_with")),
            "fta_per_game_without": round_or_none(row.get("fta_per_game_without")),
            "fta_delta": round_or_none(row.get("fta_delta")),
            "sf_with": to_json_value(row.get("sf_with")),
            "sf_without": to_json_value(row.get("sf_without")),
            "sf_per_game_with": round_or_none(row.get("sf_per_game_with")),
            "sf_per_game_without": round_or_none(row.get("sf_per_game_without")),
            "sf_delta": round_or_none(row.get("sf_delta")),
        })

    defense_adjusted = []
    for _, row in da.sort_values("defense_adjusted_fta36_delta").iterrows():
        pbp = to_json_value(row.get("official_pbp_name"))
        defense_adjusted.append({
            "official_id": pbp_to_id.get(pbp),
            "pbp_name": pbp,
            "official_name": to_json_value(row.get("official_name")),
            "n_games_with": to_json_value(row.get("n_games_with")),
            "fta36_with": round_or_none(row.get("fta36_with")),
            "fta36_without": round_or_none(row.get("fta36_without")),
            "raw_fta36_delta": round_or_none(row.get("raw_fta36_delta")),
            "defense_adjusted_fta36_delta": round_or_none(row.get("defense_adjusted_fta36_delta")),
            "defrtg_with": round_or_none(row.get("defrtg_with")),
            "defrtg_delta": round_or_none(row.get("defrtg_delta")),
        })

    return {
        "player_name": player_name,
        "slug": slugify(player_name),
        "baseline": {
            "baseline_fta_per_game": round_or_none(baseline_row["baseline_fta_per_game"]) if baseline_row is not None else None,
            "baseline_sf_per_game": round_or_none(baseline_row["baseline_sf_per_game"]) if baseline_row is not None else None,
            "n_games_without_official": to_json_value(baseline_row["n_games_without_official"]) if baseline_row is not None else None,
        },
        "official_deltas": official_deltas,
        "defense_adjusted": defense_adjusted,
    }


def build_crew_summary(crew: pd.DataFrame) -> list[dict]:
    crew = crew.copy()
    crew["official_id"] = crew["official_id"].apply(normalize_official_id)
    summary = (
        crew.groupby(["official_id", "role"]).size().reset_index(name="count")
    )
    out: dict[str, dict] = {}
    for _, row in summary.iterrows():
        oid = row["official_id"]
        if oid is None:
            continue
        out.setdefault(oid, {"official_id": oid, "roles": {}})
        out[oid]["roles"][row["role"]] = int(row["count"])
    return list(out.values())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


DOWNLOADS = REPO_ROOT / "site" / "public" / "downloads"


def export_downloads(frames: dict[str, pd.DataFrame]) -> None:
    if DOWNLOADS.exists():
        shutil.rmtree(DOWNLOADS)
    DOWNLOADS.mkdir(parents=True)

    exports = [
        ("official_profiles", "official_calling_profiles"),
        ("defensive_adj", "defensive_adjusted_interactions"),
        ("player_official", "player_official_interactions"),
        ("ref_profiles", "ref_profiles"),
        ("crew", "crew_assignments"),
    ]

    for frame_key, file_stem in exports:
        df = frames[frame_key].copy()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str)
        df.to_csv(DOWNLOADS / f"{file_stem}.csv", index=False)
        df.to_json(DOWNLOADS / f"{file_stem}.json", orient="records", indent=2)

    print(f"Exported {len(exports)} datasets → {DOWNLOADS}")


def main() -> int:
    frames = load_frames()
    crew_roles = build_crew_roles(frames["crew"])

    if SITE_DATA.exists():
        shutil.rmtree(SITE_DATA)
    SITE_DATA.mkdir(parents=True)

    officials = frames["official_profiles"].copy()
    officials = officials[officials["official_id"].notna()].copy()
    officials["official_id"] = officials["official_id"].apply(normalize_official_id)
    officials = officials.sort_values("suppressor_score", ascending=False)

    index_rows = build_official_index(officials, crew_roles)
    write_json(SITE_DATA / "officials" / "index.json", index_rows)

    ordered_ids = [r["official_id"] for r in index_rows if r["official_id"]]
    for i, oid in enumerate(ordered_ids):
        row = officials[officials["official_id"] == oid].iloc[0]
        detail = build_official_detail(
            row,
            frames["defensive_adj"],
            frames["ref_profiles"],
            crew_roles,
            ordered_ids[i - 1] if i else None,
            ordered_ids[i + 1] if i < len(ordered_ids) - 1 else None,
        )
        write_json(SITE_DATA / "officials" / f"{oid}.json", detail)

    player_index = build_player_index(frames["player_official"])
    write_json(SITE_DATA / "players" / "index.json", player_index)
    for player in player_index:
        detail = build_player_detail(
            player["player_name"], frames["player_official"], frames["defensive_adj"]
        )
        write_json(SITE_DATA / "players" / f"{player['slug']}.json", detail)

    write_json(SITE_DATA / "crew.json", build_crew_summary(frames["crew"]))

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "github_url": "https://github.com/hirememorey/ref-ball",
        "headlines": {
            "anova_p": 0.000003,
            "attention_load_lift": 3.5,
            "harden_fta_correlation": -0.528,
            "crew_prediction_r": 0.406,
        },
        "counts": {
            "officials": len(index_rows),
            "players": len(player_index),
            "player_official_pairs_raw": int(len(frames["player_official"])),
            "player_official_pairs_adjusted": int(len(frames["defensive_adj"])),
            "crew_rows": int(len(frames["crew"])),
            "games": 13278,
        },
        "license": {"data": "CC-BY-4.0", "code": "MIT"},
        "author": "Harris Gordon",
    }
    write_json(SITE_DATA / "meta.json", meta)

    export_downloads(frames)

    print(f"Wrote {len(ordered_ids)} officials, {len(player_index)} players → {SITE_DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
