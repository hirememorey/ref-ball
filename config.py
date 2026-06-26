"""Player roster, season range, and project paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PBP_DIR = RAW_DIR / "pbp"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Players to study. Selected for FTA-shift signal diversity (from does-harden-choke):
#   - Large negative FTA shift (lose free throws in playoffs): Harden, Embiid, Butler, Fox
#   - Large positive FTA shift (gain free throws in playoffs): LeBron, Dirk, Brunson, Durant
#   - High-volume FTA stars with mixed signals: Giannis, Luka, SGA, Tatum, Mitchell
#   - Low-FTA contrast: Curry, Klay
#
# Add or remove players here. Each entry needs the NBA Stats API player ID.
PLAYERS = {
    # Large negative FTA shift
    "James Harden":            {"nba_id": 201935},
    "Joel Embiid":             {"nba_id": 203954},
    "Jimmy Butler":            {"nba_id": 202710},
    "De'Aaron Fox":            {"nba_id": 1628368},
    # Large positive FTA shift
    "LeBron James":            {"nba_id": 2544},
    "Dirk Nowitzki":           {"nba_id": 1717},
    "Jalen Brunson":           {"nba_id": 1628973},
    "Kevin Durant":            {"nba_id": 201142},
    # High-volume FTA, mixed signals
    "Giannis Antetokounmpo":   {"nba_id": 203507},
    "Luka Doncic":             {"nba_id": 1629029},
    "Shai Gilgeous-Alexander": {"nba_id": 1628983},
    "Jayson Tatum":            {"nba_id": 1628369},
    "Donovan Mitchell":        {"nba_id": 1628378},
    # Low-FTA contrast
    "Stephen Curry":           {"nba_id": 201939},
    "Klay Thompson":           {"nba_id": 202691},
}

# Seasons: 1996-97 through 2025-26
SEASON_START_YEARS = list(range(1996, 2026))


def year_to_season(start_year: int) -> str:
    """Convert start year (e.g. 2023) to NBA season string (e.g. 2023-24)."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_to_year(season: str) -> int:
    """Convert season string (e.g. 2023-24) to start year."""
    return int(season.split("-")[0])


def player_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("'", "")


def all_seasons() -> list[str]:
    return [year_to_season(y) for y in SEASON_START_YEARS]
