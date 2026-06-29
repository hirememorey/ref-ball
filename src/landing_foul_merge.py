"""Merge landing foul classifications with existing v3 foul-type labels.

The v3 classifier tagged mechanism=LANDING on one clip. This script maps those
labels onto landing_foul=YES and all other v3-classified clips to NO, then
merges with a browser-exported landing foul CSV (deduping on game_id+event_id).

Usage:
    python src/landing_foul_merge.py \\
        --landing data/landing_foul_classifications.csv \\
        --v3 data/foul_type_classifications.csv \\
        --output data/landing_foul_ground_truth.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

V3_PATH = config.DATA_DIR / "foul_type_classifications.csv"
DEFAULT_LANDING_PATH = config.DATA_DIR / "landing_foul_classifications.csv"
DEFAULT_OUTPUT_PATH = config.DATA_DIR / "landing_foul_ground_truth.csv"


def load_landing_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    required = {"game_id", "event_id", "landing_foul"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Landing CSV missing columns: {sorted(missing)}")
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["event_id"] = df["event_id"].astype(int)
    df["source"] = "landing_classifier"
    return df


def load_v3_as_landing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["event_id"] = df["event_id"].astype(int)
    df["landing_foul"] = df["mechanism"].apply(lambda m: "YES" if m == "LANDING" else "NO")
    df["note"] = df.apply(
        lambda r: f"v3 mechanism={r.get('mechanism', '')}" if pd.notna(r.get("mechanism")) else "",
        axis=1,
    )
    df["source"] = "v3_foul_type"
    keep = [
        "game_id", "event_id", "period", "clock", "description",
        "landing_foul", "note", "source",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = ""
    return df[keep]


def merge_ground_truth(
    landing_path: Path,
    v3_path: Path,
) -> pd.DataFrame:
    landing = load_landing_csv(landing_path)
    v3 = load_v3_as_landing(v3_path)

    frames = [f for f in (v3, landing) if not f.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["priority"] = combined["source"].map({"landing_classifier": 2, "v3_foul_type": 1}).fillna(0)
    combined = combined.sort_values(["game_id", "event_id", "priority"])
    merged = combined.drop_duplicates(subset=["game_id", "event_id"], keep="last")
    merged = merged.drop(columns=["priority"]).sort_values(["game_id", "event_id"]).reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge landing foul ground truth sources")
    parser.add_argument("--landing", default=str(DEFAULT_LANDING_PATH), help="Landing classifier export CSV")
    parser.add_argument("--v3", default=str(V3_PATH), help="v3 foul type classifications CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Merged output CSV")
    args = parser.parse_args()

    merged = merge_ground_truth(Path(args.landing), Path(args.v3))
    if merged.empty:
        logger.warning("No classifications to merge")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    yes = (merged["landing_foul"] == "YES").sum()
    no = (merged["landing_foul"] == "NO").sum()
    unclear = (merged["landing_foul"] == "UNCLEAR").sum()
    logger.info(
        "Wrote %d rows to %s (YES=%d, NO=%d, UNCLEAR=%d)",
        len(merged), out_path, yes, no, unclear,
    )


if __name__ == "__main__":
    main()
