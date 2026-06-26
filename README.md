# ref-ball

**What types of shooting fouls disappear in the playoffs?**

A video-classification study of how shooting-foul composition changes from the regular season to the playoffs.

## The question

Some players lose free throws in the playoffs (Harden −1.2 FTA/36, Embiid −1.2, Butler −0.8). Others gain them (LeBron +0.4, Dirk +1.7, Brunson +1.3). The "playoff whistle" narrative says refs swallow their whistles — but 15 of 31 star players *increase* their FTA in the playoffs. The shift is player-specific, not league-wide.

The hypothesis: the *type* of contact that generates a player's free throws determines whether those free throws survive playoff physicality. Genuine rim-finishing contact (LeBron driving through a defender) should persist. Manufactured perimeter contact (Harden's arm hooks, Embiid's rip-throughs) should disappear as refs tighten and defenders become more disciplined.

This can only be tested with video — PBP data records "S.FOUL" for all shooting fouls and cannot distinguish a genuine drive finish from an arm hook.

## Pipeline

```
1. Scrape  →  src/foul_scraper.py    →  data/processed/manifest_{player}_{rs|po}.json
2. Classify →  src/foul_classifier.py →  output/classifier_{player}.html
3. Analyze  →  src/analyze.py         →  RS vs PO foul-type comparison
```

### 1. Scrape clips

Fetches shooting-foul video clips from the NBA Stats API for a player's games:

```bash
make scrape-harden          # 5 RS games
make scrape-harden-po       # 5 PO games
```

### 2. Classify

Generates a self-contained HTML tool for manual video classification:

```bash
make classify-harden
make serve                  # http://localhost:8080/classifier_james_harden.html
```

Click through clips, tag each foul by mechanism / body part / timing / severity / location, then export to CSV.

### 3. Analyze

Compares RS vs PO foul-type composition per player:

```bash
make analyze
make analyze-harden
```

## Foul-type taxonomy

| Axis | Values |
|---|---|
| **Mechanism** | DRV-FINISH, DRV-INIT, ARM-HOOK, CONTEST, LANDING, PUMP-JUMP, RIP-THRU, POST, PUTBACK, OFFBALL, TAKE, AMB |
| **Body Part** | HEAD, ARM, CHEST, SHOULDER, LOWER |
| **Timing** | BEFORE, DURING, AFTER (drive mechanisms only) |
| **Severity** | STRONG, MEDIUM, MARGINAL |
| **Location** | RA, PAINT, MID, PERIM |

The key axis is **manufactured vs genuine**:
- **Manufactured**: ARM-HOOK, PUMP-JUMP, RIP-THRU, DRV-INIT (contact-seeking)
- **Genuine**: DRV-FINISH, CONTEST, LANDING, PUTBACK (real basketball contact)

## Player roster

15 players selected for FTA-shift diversity (see `config.py`):
- Large negative shift: Harden, Embiid, Butler, Fox
- Large positive shift: LeBron, Dirk, Brunson, Durant
- High-volume mixed: Giannis, Luka, SGA, Tatum, Mitchell
- Low-FTA contrast: Curry, Klay

## Alpha test results (from does-harden-choke)

20 Harden clips + 16 Giannis clips (RS only):

| | Harden | Giannis |
|---|---|---|
| Manufactured | 50% | 38% |
| Genuine | 40% | 56% |
| MARGINAL severity | 30% | 0% |
| STRONG severity | 20% | 38% |
| ARM body part | 55% | 38% |

Profiles are directionally different. The next step is scaling to RS + PO clips for the full roster.
