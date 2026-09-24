"""Rule-based tagging of L2M comment text. Transparent on purpose: every tag
records the phrase that triggered it, so rules can be audited and revised.

contact_level:
  none        league says no (illegal) contact was made
  marginal    contact occurred and the league calls it marginal/incidental/brush/graze
  affecting   contact that dislodges/displaces or affects RSBQ/FOM/the shot
  legal       legal position/verticality/legal contest, contact level unstated
  unclear     no rule matched
  not_contact_judgment  take fouls, defensive three seconds, timeouts (exclude from contact analyses)

    python3 src/l2m_contact/classify_comments.py      # events.csv -> events_tagged.csv + audit sample
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from pathlib import Path

from paths import TRACKED, WORK

NONE = re.compile(r"\b(avoids? (making |any |illegal |making any |making illegal )*contact"
                  r"|no (illegal )?contact|does not make (any |illegal )*contact"
                  r"|without (making )?(any |illegal )*contact|does not contact|does not (deliver|initiate) contact)\b", re.I)
NOT_CONTACT_JUDGMENT = re.compile(r"\b(take foul|three seconds|3 seconds|timeout)\b", re.I)
MORE_THAN_MARGINAL = re.compile(r"\bmore than (marginal|incidental)\b", re.I)
AFFECT = re.compile(r"\b(dislodg\w* (?!(the ball|ball|it)\b)\w+|displac\w*|alter(s|ed|ing)?\b"
                    r"|affect(s|ed|ing)?\b|impact(s|ed|ing)?\b|(rsbq|sqbr|fom)\b"
                    r"|impedes?|restrict\w*|reroutes?|caus\w* [\w.' -]{1,30}? to (lose|stumble|fall|miss)|clamps?|prevents?"
                    r"|lose (control|possession)|holds?\b|grabs?\b|pushes? off|hooks?\b|should have been called"
                    r"|illegal contact|initiat\w* contact|delivers? contact)", re.I)
NEG = re.compile(r"\b(not|n't|without|no|never|neither|nor|prior to|avoids?|avoiding|no clear)\b", re.I)
MARGINAL = re.compile(r"\b(marginal|incidental|brush\w*|graz\w*|minimal|slight contact|light contact"
                      r"|(briefly|momentarily) (\w+ ){0,2}(grasps?|grabs?|places?|applies|has (his|her) (arm|hand))"
                      r"|immediately releases?|(grasps?|grabs?) and (then )?(immediately )?releases?)\b", re.I)
LEGAL = re.compile(r"\b(legal(ly)? (guarding position|position|contest\w*|screen\w*)|legally \w+"
                   r"|vertical(ity)?|straight up|cleanly|all ball|clean contact with the ball"
                   r"|does not (deliver|initiate|commit) (any )?illegal contact|absorbs?|establish\w*"
                   r"|firms? up|engage and disengage|in (his|her) path before)\b", re.I)


def _clause_before(text: str, start: int) -> str:
    """Text from the start of the current clause/sentence up to `start`."""
    head = text[:start]
    cut = max(head.rfind(". "), head.rfind("; "), head.rfind(", but"), head.rfind(" but "))
    return head[cut + 1:] if cut >= 0 else head


def positive(pattern: re.Pattern, text: str):
    """First match of `pattern` not negated earlier in its clause."""
    for m in pattern.finditer(text):
        if not NEG.search(_clause_before(text, m.start())[-60:]):
            return m
    return None


REGION = [
    ("lower_body", r"lower body|leg|knee|hip|foot|feet|foot to foot"),
    ("arm", r"\barm|elbow|wrist|forearm"),
    ("hand", r"\bhand|finger"),
    ("head", r"\bhead|face|neck"),
    ("body", r"\bbody|torso|chest|shoulder|side|back\b"),
    ("ball", r"\bball\b"),
]


def tag(comment: str) -> dict:
    c = comment or ""
    if NOT_CONTACT_JUDGMENT.search(c):
        return {"contact_level": "not_contact_judgment", "contact_trigger": NOT_CONTACT_JUDGMENT.search(c).group(0).lower(),
                "contact_region": "unstated"}
    m_more = positive(MORE_THAN_MARGINAL, c)
    m_aff = positive(AFFECT, c)
    m_none, m_marg, m_legal = NONE.search(c), MARGINAL.search(c), LEGAL.search(c)
    # Explicit marginal/incidental language outranks weaker affect verbs,
    # unless the league says "more than marginal".
    if m_more:
        level, why = "affecting", m_more.group(0)
    elif m_marg:
        level, why = "marginal", m_marg.group(0)
    elif m_aff:
        level, why = "affecting", m_aff.group(0)
    elif m_none:
        level, why = "none", m_none.group(0)
    elif m_legal:
        level, why = "legal", m_legal.group(0)
    else:
        level, why = "unclear", ""
    region = next((name for name, pat in REGION if re.search(pat, c, re.I)), "unstated")
    return {"contact_level": level, "contact_trigger": why.lower(), "contact_region": region}


def main() -> None:
    with (WORK / "events.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.update(tag(r["comment"]))
    with (WORK / "events_tagged.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fouls = [r for r in rows if r["call_type"].startswith("Foul")]
    print("foul rows:", len(fouls))
    for d in ("CNC", "INC", "CC", "IC"):
        c = Counter(r["contact_level"] for r in fouls if r["decision"] == d)
        print(f"  {d:4s}", dict(c.most_common()))
    # Audit sample: 10 per (decision, level) for human spot-check.
    rng = random.Random(20260923)
    audit = []
    for d in ("CNC", "INC"):
        for lvl in ("none", "marginal", "affecting", "legal", "unclear"):
            grp = [r for r in fouls if r["decision"] == d and r["contact_level"] == lvl]
            audit += rng.sample(grp, min(10, len(grp)))
    # Frozen sample: do not overwrite once human grades exist against it.
    if (TRACKED / "tag_audit_sample.csv").exists():
        print("audit sample already exists; not redrawn")
        return
    with (TRACKED / "tag_audit_sample.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["decision", "call_type", "contact_level", "contact_trigger",
                                          "contact_region", "comment", "clip_url", "human_level"])
        w.writeheader()
        for r in audit:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print("audit sample:", len(audit), "rows -> tag_audit_sample.csv")


if __name__ == "__main__":
    main()
