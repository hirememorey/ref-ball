"""Do crews differ in letting illegal contact go, holding contact type constant?

Population: L2M foul decisions the league judged to be fouls (CC = called, INC = missed),
excluding take fouls / defensive three seconds / timeouts (not contact judgments).
Outcome: missed (INC=1) vs called (CC=0).
Model: L2-penalized logistic regression. Three official indicators per play (all
three estimated jointly), plus foul type, comment body region, contact wording
(affecting vs other), period (Q4 vs OT), season.

Tests:
  1. Omnibus: in-sample log-loss gain from officials vs the same model without them,
     compared with 500 permutations that shuffle whole crews across games within
     season (keeps crew structure, breaks crew-play link).
  2. Per official: shrunken coefficient; permutation p-value; Benjamini-Hochberg FDR.
  3. Held-out: fit on earlier seasons, score later seasons. Does knowing the crew
     improve prediction of missed calls?

Needs numpy, pandas, scipy, scikit-learn:
    python src/l2m_contact/analyze_crews.py              # rule-based tags
    TAGS=hybrid python src/l2m_contact/analyze_crews.py  # LLM hybrid tags (data/l2m_contact/contact_tags.csv)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from paths import CREW, TRACKED, WORK

ROOT = WORK
TAGS = os.environ.get("TAGS", "rules")      # "rules" (classify_comments.py) or "hybrid" (Luna + Sonnet)
OUT = TRACKED / f"results_{TAGS}"
N_PERM = 500
C_PEN = 0.5          # inverse L2 strength, fixed a priori (not tuned on the test)
MIN_PLAYS = 60       # officials below this get pooled into "other"
SEED = 20260923
TRAIN_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]
TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]


def load() -> pd.DataFrame:
    ev = pd.read_csv(ROOT / "events_tagged.csv", dtype={"game_id": str})
    if TAGS == "hybrid":
        h = pd.read_csv(TRACKED / "contact_tags.csv", dtype={"game_id": str}, usecols=["game_id", "event_index", "final_level"])
        ev = ev.merge(h, on=["game_id", "event_index"], how="left", validate="one_to_one")
        ev["contact_level"] = ev.final_level.fillna(ev.contact_level)
    ev = ev[ev.call_type.str.startswith("Foul")
            & ev.decision.isin(["CC", "INC"])
            & (ev.contact_level != "not_contact_judgment")
            & (ev.call_type != "Foul: Defense 3 Second")].copy()
    ev["missed"] = (ev.decision == "INC").astype(int)
    ev["ot"] = (~ev.period.isin(["Q4"])).astype(int)
    ev["affecting_text"] = (ev.contact_level == "affecting").astype(int)
    # Working officials only: the 4th listed official in playoff/play-in games is the
    # alternate, who does not officiate. Existing file marks role; new file lists in order.
    crew = pd.read_csv(CREW, dtype={"game_id": str})
    crew = crew[crew.role.isin(["crew_chief", "official_2", "official_3"])]
    extra = TRACKED / "crews_extra.csv"
    if extra.exists():
        x = pd.read_csv(extra, dtype={"game_id": str})
        crew = pd.concat([crew, x[x["order"] <= 3]], ignore_index=True)
    crew["game_id"] = crew.game_id.str.zfill(10)
    crews = crew.groupby("game_id").official_name.agg(lambda s: sorted(set(s)))
    ev["crew"] = ev.game_id.map(crews)
    missing = ev.crew.isna()
    print(f"plays: {len(ev)}  missing crew: {missing.sum()} (dropped)")
    return ev[~missing].reset_index(drop=True)


def covariates(ev: pd.DataFrame) -> sparse.csr_matrix:
    cats = pd.get_dummies(ev[["call_type", "contact_region", "season"]].astype(str), drop_first=True)
    X = pd.concat([cats, ev[["ot", "affecting_text"]]], axis=1).astype(float)
    return sparse.csr_matrix(X.values)


def official_matrix(crews: pd.Series, officials: list[str]) -> sparse.csr_matrix:
    idx = {o: i for i, o in enumerate(officials)}
    rows, cols = [], []
    for r, crew in enumerate(crews):
        for o in crew:
            rows.append(r)
            cols.append(idx.get(o, idx["other"]))
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(crews), len(officials)))


def fit(X, y):
    m = LogisticRegression(C=C_PEN, max_iter=2000)
    m.fit(X, y)
    return m


def gain(Xc, Xo, y) -> tuple[float, np.ndarray]:
    base = fit(Xc, y)
    full = fit(sparse.hstack([Xc, Xo]).tocsr(), y)
    g = log_loss(y, base.predict_proba(Xc)[:, 1]) - log_loss(y, full.predict_proba(sparse.hstack([Xc, Xo]).tocsr())[:, 1])
    return g, full.coef_[0][Xc.shape[1]:]


def permuted_crews(ev: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Shuffle whole crews across games within season."""
    out = ev.crew.copy()
    for _, grp in ev.groupby("season"):
        games = grp.game_id.unique()
        game_crew = grp.drop_duplicates("game_id").set_index("game_id").crew
        shuffled = dict(zip(games, rng.permutation(game_crew.loc[games].values)))
        out.loc[grp.index] = grp.game_id.map(shuffled)
    return out


def bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ev = load()
    y = ev.missed.values
    counts = pd.Series([o for c in ev.crew for o in c]).value_counts()
    officials = sorted(counts[counts >= MIN_PLAYS].index) + ["other"]
    Xc = covariates(ev)
    Xo = official_matrix(ev.crew, officials)
    obs_gain, obs_coef = gain(Xc, Xo, y)

    rng = np.random.default_rng(SEED)
    perm_gain, perm_coef = [], []
    for i in range(N_PERM):
        g, c = gain(Xc, official_matrix(permuted_crews(ev, rng), officials), y)
        perm_gain.append(g)
        perm_coef.append(c)
        if (i + 1) % 100 == 0:
            print(f"  permutations {i + 1}/{N_PERM}", flush=True)
    perm_gain, perm_coef = np.array(perm_gain), np.array(perm_coef)
    omnibus_p = (1 + (perm_gain >= obs_gain).sum()) / (N_PERM + 1)

    p_off = (1 + (np.abs(perm_coef) >= np.abs(obs_coef)).sum(axis=0)) / (N_PERM + 1)
    raw = {o: (0, 0) for o in officials}
    for crew, m in zip(ev.crew, y):
        for o in crew:
            k = o if o in raw else "other"
            raw[k] = (raw[k][0] + m, raw[k][1] + 1)
    table = pd.DataFrame({
        "official": officials,
        "plays": [raw[o][1] for o in officials],
        "missed": [raw[o][0] for o in officials],
        "raw_miss_rate": [raw[o][0] / max(raw[o][1], 1) for o in officials],
        "coef_shrunk": obs_coef,
        "perm_p": p_off,
    })
    table = table[table.official != "other"].copy()
    table["fdr_q"] = bh(table.perm_p.values)
    table = table.sort_values("coef_shrunk")
    table.to_csv(OUT / "official_missed_contact.csv", index=False)

    # Held-out prediction
    tr, te = ev.season.isin(TRAIN_SEASONS).values, ev.season.isin(TEST_SEASONS).values
    heldout = {}
    if tr.sum() and te.sum():
        Xfull = sparse.hstack([Xc, Xo]).tocsr()
        base, full = fit(Xc[tr], y[tr]), fit(Xfull[tr], y[tr])
        lb = log_loss(y[te], base.predict_proba(Xc[te])[:, 1])
        lf = log_loss(y[te], full.predict_proba(Xfull[te])[:, 1])
        # game-bootstrap interval on the log-loss difference
        pb, pf = base.predict_proba(Xc[te])[:, 1], full.predict_proba(Xfull[te])[:, 1]
        yt, gt = y[te], ev.game_id.values[te]
        ll = lambda yy, pp: -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))
        diff = ll(yt, pb) - ll(yt, pf)
        per_game = pd.Series(diff).groupby(gt).agg(["sum", "count"])
        boots = []
        for _ in range(2000):
            s = per_game.sample(len(per_game), replace=True, random_state=rng.integers(1 << 31))
            boots.append(s["sum"].sum() / s["count"].sum())
        heldout = {"test_plays": int(te.sum()), "logloss_base": lb, "logloss_crew": lf,
                   "gain": lb - lf, "gain_95ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}

    summary = {
        "plays": int(len(ev)), "missed": int(y.sum()), "games": int(ev.game_id.nunique()),
        "seasons": sorted(ev.season.unique().tolist()),
        "officials_modeled": len(officials) - 1, "min_plays": MIN_PLAYS, "C": C_PEN,
        "omnibus_logloss_gain": obs_gain, "omnibus_perm_p": omnibus_p,
        "perm_gain_95th": float(np.percentile(perm_gain, 95)),
        "officials_fdr_05": int((table.fdr_q < 0.05).sum()),
        "heldout": heldout,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(table.head(8).to_string(index=False))
    print(table.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
