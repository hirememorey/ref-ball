"""Permutation null for the held-out crew gain in analyze_crews.py.

Shuffles whole crews across games within season (train and test alike), refits on
train seasons, scores test seasons. If the observed held-out gain is typical of
shuffled crews, it reflects era/composition drift rather than official behavior.

    python src/l2m_contact/heldout_null.py               # add TAGS=hybrid for the LLM tags
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import log_loss

import analyze_crews as a

N = 500


def heldout_gain(Xc, Xo, y, tr, te) -> float:
    Xf = sparse.hstack([Xc, Xo]).tocsr()
    base, full = a.fit(Xc[tr], y[tr]), a.fit(Xf[tr], y[tr])
    return log_loss(y[te], base.predict_proba(Xc[te])[:, 1]) - log_loss(y[te], full.predict_proba(Xf[te])[:, 1])


def main() -> None:
    ev = a.load()
    y = ev.missed.values
    counts = pd.Series([o for c in ev.crew for o in c]).value_counts()
    officials = sorted(counts[counts >= a.MIN_PLAYS].index) + ["other"]
    Xc = a.covariates(ev)
    tr, te = ev.season.isin(a.TRAIN_SEASONS).values, ev.season.isin(a.TEST_SEASONS).values
    obs = heldout_gain(Xc, a.official_matrix(ev.crew, officials), y, tr, te)
    rng = np.random.default_rng(a.SEED + 1)
    null = []
    for i in range(N):
        null.append(heldout_gain(Xc, a.official_matrix(a.permuted_crews(ev, rng), officials), y, tr, te))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{N}", flush=True)
    null = np.array(null)
    out = {"observed_heldout_gain": obs, "null_mean": float(null.mean()),
           "null_95th": float(np.percentile(null, 95)),
           "perm_p": float((1 + (null >= obs).sum()) / (N + 1)),
           "miss_rate_by_season": ev.groupby("season").missed.mean().round(3).to_dict()}
    (a.OUT / "heldout_null.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
