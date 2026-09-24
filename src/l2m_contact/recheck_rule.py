"""Decision-blind rule selecting Luna-tagged rows for a Sonnet re-check.

Uses only Luna's label and the comment text, never the league decision.
"""

import re

CONTACT = re.compile(r"\b(contact\w*|engag\w*|hand|hands|arm|arms|forearm|elbow|hip|back|body|shoulder|"
                     r"places?|placed|puts?|rests?|grasp\w*|grab\w*|hold\w*|touch\w*|brush\w*|bump\w*|"
                     r"push\w*|extend\w*|reach\w*|dislodg\w*|head|face)\b", re.I)
SOFTEN = re.compile(r"\b(marginal|incidental|slight\w*|brief\w*|cleanly|clean contact|deemed|"
                    r"does not|doesn't|not affect|natural)\b", re.I)


NO_CONTACT = re.compile(r"(does not|doesn't|did not) (make|deliver|initiate) (any )?(illegal )?contact"
                        r"|avoids? (making )?(any )?(illegal )?contact|room to avoid (the )?contact"
                        r"|no (illegal )?contact|without (making )?(any )?contact", re.I)
LEGAL_ONLY = re.compile(r"legally contests?|legal guarding position|verticality|clean(ly)? (contact with the )?"
                        r"(ball|deflect\w*|block\w*|strip\w*|dislodg\w* the ball)", re.I)


def needs_recheck(luna_level: str, comment: str) -> bool:
    if luna_level in ("unclear", "legal", "none"):
        rest = NO_CONTACT.sub(" ", comment)
        if luna_level == "legal":
            rest = LEGAL_ONLY.sub(" ", rest)
        return bool(CONTACT.search(rest))
    if luna_level == "affecting":
        return bool(SOFTEN.search(comment))
    return False
