"""Target player definitions for ref-ball.

Selection rule for expanded set:
  Career RS FTA/36 >= 5.0 AND >= 400 GP (2014-15 onward)

All IDs verified via commonplayerinfo endpoint on 2026-06-28.
"""

CORE_PLAYERS: dict[str, int] = {
    "James Harden": 201935,
    "Luka Doncic": 1629029,
    "Trae Young": 1629027,
    "Shai Gilgeous-Alexander": 1628983,
    "Joel Embiid": 203954,
    "Giannis Antetokounmpo": 203507,
    "Jimmy Butler": 202710,
    "Stephen Curry": 201939,
    "Kevin Durant": 201142,
    "LeBron James": 2544,
    "Nikola Jokic": 203999,
    "Damian Lillard": 203081,
    "DeMar DeRozan": 201942,
    "Chris Paul": 101108,
    "De'Aaron Fox": 1628368,
    "Jalen Brunson": 1628973,
}

EXPANDED_PLAYERS: dict[str, int] = {
    "Anthony Davis": 203076,
    "Dwight Howard": 2730,
    "Lou Williams": 101150,
    "Russell Westbrook": 201566,
    "Devin Booker": 1626164,
    "Montrezl Harrell": 1626149,
    "Julius Randle": 203944,
    "Blake Griffin": 201933,
    "Danilo Gallinari": 201568,
    "Kawhi Leonard": 202695,
    "Rudy Gobert": 203497,
    "Jaren Jackson Jr.": 1628991,
    "Andre Drummond": 203083,
    "Jayson Tatum": 1628369,
    "Kristaps Porzingis": 204001,
    "Karl-Anthony Towns": 1626157,
    "Bam Adebayo": 1628389,
    "Brandon Ingram": 1627742,
    "Donovan Mitchell": 1628378,
    "Paul George": 202331,
    "DeAndre Jordan": 201599,
    "Collin Sexton": 1629012,
    "Jusuf Nurkic": 203994,
    "Hassan Whiteside": 202355,
}

ALL_TARGET_PLAYERS: dict[str, int] = {**CORE_PLAYERS, **EXPANDED_PLAYERS}
