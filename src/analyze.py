"""Compare foul-type composition between regular season and playoffs.

The core question: what types of shooting fouls disappear in the playoffs?

Reads manual classifications from classifications.csv and compares
per-player RS vs PO distributions across mechanism, severity, body part,
and location axes.

Usage:
    python src/analyze.py
    python src/analyze.py --player "James Harden"
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import config


def load_classifications(path: Path) -> list[dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def split_by_season_type(rows: list[dict]) -> dict[str, list[dict]]:
    """Split classifications into RS and PO buckets.

    The manifest filename encodes season type (_rs vs _po suffix).
    But the CSV itself doesn't carry season_type. We infer it from
    the game_id: playoff game IDs start with 004, regular season with 002.
    """
    rs, po = [], []
    for r in rows:
        gid = str(r.get("game_id", "")).zfill(10)
        if gid.startswith("004"):
            po.append(r)
        elif gid.startswith("002"):
            rs.append(r)
        else:
            # Unknown — skip or assign based on other heuristic
            pass
    return {"RS": rs, "PO": po}


def distribution(rows: list[dict], field: str) -> Counter:
    return Counter(r.get(field, "") for r in rows if r.get(field))


def pct(counter: Counter, total: int) -> dict[str, float]:
    if total == 0:
        return {}
    return {k: v / total * 100 for k, v in counter.items()}


def compare_player(
    player_name: str,
    rs_rows: list[dict],
    po_rows: list[dict],
) -> None:
    print(f"\n{'='*70}")
    print(f"  {player_name}")
    print(f"  RS clips: {len(rs_rows)}  |  PO clips: {len(po_rows)}")
    print(f"{'='*70}")

    if not rs_rows and not po_rows:
        print("  No data.")
        return

    axes = [
        ("mechanism", "Mechanism"),
        ("severity", "Severity"),
        ("body_part", "Body Part"),
        ("location", "Location"),
    ]

    for field, label in axes:
        rs_dist = distribution(rs_rows, field)
        po_dist = distribution(po_rows, field)
        rs_total = sum(rs_dist.values())
        po_total = sum(po_dist.values())
        rs_pct = pct(rs_dist, rs_total)
        po_pct = pct(po_dist, po_total)

        all_keys = sorted(set(rs_dist.keys()) | set(po_dist.keys()))
        if not all_keys:
            continue

        print(f"\n  {label}:")
        print(f"  {'Type':<16} {'RS %':>6} {'PO %':>6} {'Delta':>7}")
        print(f"  {'-'*16} {'-'*6} {'-'*6} {'-'*7}")
        for k in all_keys:
            r = rs_pct.get(k, 0)
            p = po_pct.get(k, 0)
            delta = p - r
            marker = " ***" if abs(delta) >= 10 else ""
            print(f"  {k:<16} {r:>5.1f}% {p:>5.1f}% {delta:>+6.1f}{marker}")

    # Manufactured vs genuine summary
    manufactured = {"ARM-HOOK", "PUMP-JUMP", "RIP-THRU", "DRV-INIT"}
    genuine = {"DRV-FINISH", "CONTEST", "LANDING", "PUTBACK"}

    rs_mfg = sum(1 for r in rs_rows if r.get("mechanism") in manufactured)
    rs_gen = sum(1 for r in rs_rows if r.get("mechanism") in genuine)
    po_mfg = sum(1 for r in po_rows if r.get("mechanism") in manufactured)
    po_gen = sum(1 for r in po_rows if r.get("mechanism") in genuine)

    rs_total = len(rs_rows) if rs_rows else 1
    po_total = len(po_rows) if po_rows else 1

    print(f"\n  Manufactured vs Genuine:")
    print(f"  {'Category':<16} {'RS %':>6} {'PO %':>6} {'Delta':>7}")
    print(f"  {'-'*16} {'-'*6} {'-'*6} {'-'*7}")
    print(f"  {'Manufactured':<16} {rs_mfg/rs_total*100:>5.1f}% {po_mfg/po_total*100:>5.1f}% {po_mfg/po_total*100 - rs_mfg/rs_total*100:>+6.1f}")
    print(f"  {'Genuine':<16} {rs_gen/rs_total*100:>5.1f}% {po_gen/po_total*100:>5.1f}% {po_gen/po_total*100 - rs_gen/rs_total*100:>+6.1f}")


def main():
    parser = argparse.ArgumentParser(description="Compare RS vs PO foul-type composition")
    parser.add_argument("--player", default=None, help="Filter to a single player")
    parser.add_argument(
        "--classifications",
        default=None,
        help="Path to classifications CSV (default: data/processed/classifications.csv)",
    )
    args = parser.parse_args()

    cls_path = Path(args.classifications) if args.classifications else config.PROJECT_ROOT / "data/processed/classifications.csv"

    if not cls_path.exists():
        print(f"No classifications found at {cls_path}")
        print("Run the scraper and classifier first to generate clips, then classify them.")
        return

    rows = load_classifications(cls_path)
    print(f"Loaded {len(rows)} classifications from {cls_path}")

    # Group by player. The CSV doesn't have a player column — it has opponent.
    # For now, we need a player column. Let's check if it exists.
    # If not, we infer from the manifest files.
    if "player" not in rows[0]:
        # TODO: add player column to the classifier export
        print("Note: classifications CSV has no 'player' column.")
        print("If you classified multiple players, merge their exports and add a player column.")
        print("For now, treating all rows as one group.\n")
        split = split_by_season_type(rows)
        compare_player("All players", split["RS"], split["PO"])
        return

    by_player = defaultdict(lambda: {"RS": [], "PO": []})
    for r in rows:
        player = r.get("player", "Unknown")
        gid = str(r.get("game_id", "")).zfill(10)
        if gid.startswith("004"):
            by_player[player]["PO"].append(r)
        elif gid.startswith("002"):
            by_player[player]["RS"].append(r)

    players = [args.player] if args.player else sorted(by_player.keys())
    for p in players:
        if p not in by_player:
            print(f"Player '{p}' not found in classifications.")
            continue
        compare_player(p, by_player[p]["RS"], by_player[p]["PO"])


if __name__ == "__main__":
    main()
