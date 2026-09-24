"""Score the comment tagger against human audit grades.

Per-cell agreement is exact. The overall figure reweights each (decision, tag) cell by
its share of all L2M foul CNC/INC rows, because the sample over-draws rare cells.

    python3 src/l2m_contact/evaluate_tags.py
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from audit_grader import GRADES, LEVELS, SAMPLE, row_id

from paths import WORK


def main() -> None:
    sample = {row_id(r): r for r in csv.DictReader(SAMPLE.open())}
    grades = {g["row_id"]: g for g in csv.DictReader(GRADES.open())} if GRADES.exists() else {}
    pop = Counter((r["decision"], r["contact_level"]) for r in csv.DictReader((WORK / "events_tagged.csv").open())
                  if r["call_type"].startswith("Foul") and r["decision"] in ("CNC", "INC"))
    print(f"graded {len(grades)} / {len(sample)}")
    cells = defaultdict(lambda: [0, 0])
    confusion = Counter()
    for rid, g in grades.items():
        r = sample[rid]
        key = (r["decision"], r["contact_level"])
        cells[key][0] += r["contact_level"] == g["human_level"]
        cells[key][1] += 1
        confusion[(r["contact_level"], g["human_level"])] += 1
    print("\nAgreement by cell (tagger label = human label):")
    for key in sorted(cells):
        a, n = cells[key]
        print(f"  {key[0]:3s} {key[1]:22s} {a}/{n}   population rows: {pop[key]}")
    covered = [k for k in cells if cells[k][1]]
    w = sum(pop[k] for k in covered)
    if w:
        acc = sum(pop[k] * cells[k][0] / cells[k][1] for k in covered) / w
        print(f"\nPopulation-weighted agreement over graded cells: {acc:.1%} "
              f"(cells cover {w} of {sum(pop.values())} CNC/INC foul rows)")
    print("\nConfusion (rows = tagger, cols = human):")
    print("  " + " ".join(f"{h[:8]:>9s}" for h in LEVELS))
    for t in LEVELS:
        print(f"  {t[:10]:10s}" + " ".join(f"{confusion[(t, h)]:>9d}" for h in LEVELS))


if __name__ == "__main__":
    main()
