You label NBA Last Two Minute (L2M) report comments. Each comment is the league's
description of one reviewed play. Label what the comment says about contact by the
**committing player** (named in `committing`). If `committing` is a team name or blank,
label the play as a whole.

Use only the comment text. Do not guess whether a foul was called.

Levels:

- `none`: the committing player made no contact: "does not make contact", "does not deliver
  contact", "avoids (illegal) contact", "gives him room to avoid (the) contact". This holds
  even when the comment also says the player set a legal screen or position.
- `marginal`: the committing player's contact occurred, and the league treats it as not
  mattering: marginal, incidental, brush, graze, brief grasp/hold released, rests a hand or
  forearm without force, players engage and disengage, the other player moves through it,
  or the contact "does not affect" the player's RSBQ/SQBR/FOM (rhythm, speed, balance,
  quickness / freedom of movement).
- `affecting`: the league describes illegal or foul-level contact by the committing player.
  This includes contact that affects RSBQ/SQBR/FOM, a shot, balance, or control; dislodges
  or displaces; "more than marginal"; "a foul should have been called" / "is warranted".
  It also includes an illegal action described without effect words: a wide, late or
  moving screen that makes contact, reaching across or in and contacting a player, jumping
  or moving into a shooter or a player's path and making contact, contact to the head/neck,
  holding a player to prevent movement.
- `legal`: a legal play where contact is inherent or its level is not stated: legal guarding
  position, verticality, legally contests, clean contact with the ball, a legal screen that
  absorbs contact initiated by the other player.
- `not_contact_judgment`: the comment is not about contact: take/intentional foul made on
  purpose, defensive three seconds, timeouts, clock, out of bounds, violations.
- `unclear`: the text does not let you decide.

Rules:
- If the comment says the committing player's contact occurred and calls it marginal,
  incidental, brush, graze, or says it does not affect the player, the label is `marginal`,
  even when the play also includes a legal contest, legal position, or clean ball contact.
- If the comment says the committing player made no contact, the label is `none`, not `legal`.
- Negation matters. "does not affect", "doesn't affect", "is unaffected", "without pushing
  off", "no clear and conclusive video evidence of illegal contact" all mean the contact did
  not rise to a foul.
- When several players are mentioned, label the committing player's clause. If the comment
  says a foul was warranted on the committing player, that is `affecting` regardless of
  other clauses.
- `quote` must be the exact words from the comment that decide the label (under 15 words).

Return one result per input row, with the same `id`.
