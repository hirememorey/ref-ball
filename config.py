"""Project paths, season range, and configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PBP_DIR = RAW_DIR / "pbp"
RAW_L2M_DIR = RAW_DIR / "l2m"
PROCESSED_DIR = DATA_DIR / "processed"
GAMES_DIR = PROCESSED_DIR / "games"
L2M_EVENTS_PATH = PROCESSED_DIR / "l2m_events.parquet"
L2M_REPORTS_PATH = PROCESSED_DIR / "l2m_reports.parquet"
CREW_ASSIGNMENTS_PATH = PROCESSED_DIR / "crew_assignments.parquet"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

SEASON_START_YEARS = list(range(1996, 2026))

MANUFACTURED_MECHS = {"ARM-HOOK", "PUMP-JUMP", "RIP-THRU", "DRV-INIT"}
GENUINE_MECHS = {"DRV-FINISH", "CONTEST", "LANDING", "PUTBACK"}

ALL_PLAYERS: dict[str, int] = {}


def player_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("'", "")


def year_to_season(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_to_year(season: str) -> int:
    return int(season.split("-")[0])


def all_seasons() -> list[str]:
    return [year_to_season(y) for y in SEASON_START_YEARS]


# L2M JSON API is available from 2018-19 onward (2015-17 return 403).
L2M_SEASONS = [year_to_season(y) for y in range(2018, 2025)]
L2M_ARCHIVE_URLS = {
    season: f"https://official.nba.com/{season}-nba-officiating-last-two-minute-reports/"
    for season in L2M_SEASONS
}
