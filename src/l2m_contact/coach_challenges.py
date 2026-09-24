"""Coach's challenges on called fouls, full game, from hoopR NBA play-by-play parquet.

Instant Replay rows (event_type 18) carry the review outcome only as an action code. Codes
were mapped against the nba.com v3 play-by-play `subType` text for 11 games (Sep 2026):

    4 = Coach Challenge Support Ruling
    5 = Coach Challenge Overturn Ruling
    6 = Coach Challenge Ruling Stands
    0 / 1 / 3 = official-initiated review: Support / Overturn / Stands;  2 = altercation

Each challenge is linked to called fouls (event_type 6) at the same period and clock. The
foul text names the calling official, e.g. "(J.Goble)".

    python3 src/l2m_contact/coach_challenges.py 2019-20=path/to/play_by_play_2019-20.parquet ...

Input files: sportsdataverse hoopR-nba-stats-data nba_stats/pbp/parquet/play_by_play_<season>.parquet
Output: data/l2m_contact/coach_challenges.csv
"""

from __future__ import annotations

import sys

import pandas as pd

from paths import TRACKED

OUTCOME = {4: "support", 5: "overturn", 6: "stands"}


def foul_text(row) -> str:
    return " ".join(str(v) for v in (row.home_description, row.visitor_description) if v and str(v) != "None")


def main(pairs: list[str]) -> None:
    rows = []
    for pair in pairs:
        season, path = pair.split("=", 1)
        d = pd.read_parquet(path)
        d["et"] = d.event_type.astype(int)
        d["act"] = d.event_action_type.astype(int)
        for gid, x in d.groupby("game_id"):
            x = x.reset_index(drop=True)
            for i in x.index[(x.et == 18) & x.act.isin(OUTCOME)]:
                r = x.loc[i]
                same = x[(x.period == r.period) & (x.time_quarter == r.time_quarter)]
                fouls = same[same.et == 6]
                mm, ss = r.time_quarter.split(":")
                rows.append({
                    "season": season, "game_id": gid, "period": int(r.period), "clock": r.time_quarter,
                    "outcome": OUTCOME[r.act],
                    "in_l2m_window": int(r.period) >= 4 and int(mm) * 60 + int(ss) <= 120,
                    "n_fouls_same_clock": len(fouls),
                    "foul_desc": " || ".join(foul_text(y) for _, y in fouls.iterrows()),
                })
    df = pd.DataFrame(rows)
    df.to_csv(TRACKED / "coach_challenges.csv", index=False)
    linked = df[df.n_fouls_same_clock > 0]
    outside = linked[~linked.in_l2m_window]
    print(f"challenges {len(df)}; linked to a called foul {len(linked)}; outside last 2 min {len(outside)}")
    print(outside.outcome.value_counts().to_string())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
