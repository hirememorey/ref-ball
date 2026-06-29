from config.target_players import ALL_TARGET_PLAYERS, CORE_PLAYERS, EXPANDED_PLAYERS

import importlib
import importlib.util
import sys
from pathlib import Path

_root_config = Path(__file__).resolve().parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("_root_config", _root_config)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name in dir(_mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_mod, _name)

ALL_PLAYERS = ALL_TARGET_PLAYERS
