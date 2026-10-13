"""Locations of the MedClim data and output trees, shared by every script.

The repository root is taken from this file's location, so the scripts run from any
working directory and any checkout location.  The data root defaults to the repository
itself (Data/, Stats/, Plots/, Results/ and Diagnostics/ next to Scripts/, as during
manuscript production); set the environment variable MEDCLIM_DATA to keep those five
trees somewhere else.  Inputs that are not part of this repository have their own
entries at the end of the file.

Usage in a script (N = number of directory levels between the script and the repo root):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[N]))
    from Scripts.paths import DATA, PLOTS
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


REPO_ROOT = Path(__file__).resolve().parents[1]

# Data root: the repository by default, or MEDCLIM_DATA.
DATA_ROOT = _env_path("MEDCLIM_DATA", REPO_ROOT)
DATA = DATA_ROOT / "Data"                # source files and preprocessed observation tables
STATS = DATA_ROOT / "Stats"              # climatology tables, cast counts, point metadata
PLOTS = DATA_ROOT / "Plots"              # figures (and the Adriatic readers' long tables)
RESULTS = DATA_ROOT / "Results"          # monthly series, trend fits, robustness tables
DIAGNOSTICS = DATA_ROOT / "Diagnostics"  # per-source diagnostic summaries

# Basins (https://github.com/ElenaTerzic/Basins, Elena Terzić's extension of Arnau Miró's
# Basins package on PyPI) provides the Manca basin polygons.  The package must be installed,
# because the scripts import it, and six mapping scripts additionally parse the basin
# rectangles from its source file Basins/manca.py.  That file is taken from the installed
# package; if MEDCLIM_BASINS names a Basins checkout (its root directory), that checkout's
# copy is used instead.  Without either, the fallback is a sibling checkout of this
# repository's parent directory (../../Basins, i.e. ~/Projects/Basins next to
# ~/Projects/IRB/MedClim).
def _installed_manca() -> Path | None:
    try:
        import Basins.manca as _manca
    except ImportError:
        return None
    return Path(_manca.__file__).resolve()


_parents = REPO_ROOT.parents
_sibling_basins = (_parents[1] if len(_parents) > 1 else _parents[0]) / "Basins"
if os.environ.get("MEDCLIM_BASINS"):
    BASINS_REPO = _env_path("MEDCLIM_BASINS", _sibling_basins)
    MANCA_PY = BASINS_REPO / "Basins" / "manca.py"
else:
    BASINS_REPO = _sibling_basins
    MANCA_PY = _installed_manca() or BASINS_REPO / "Basins" / "manca.py"

# JADMON 2024 Adriatic CTD delivery: two tab-separated files, temperatura_2024.csv and
# salinitet_2024.csv, read only by Scripts/Adriatic/preprocessing/read_JADMON_2024.py.
# Not redistributed with this repository; set MEDCLIM_JADMON_2024 to the directory that
# holds the two files (default: Data/Adriatic/Orig/JADMON_2024 under the data root).
JADMON_2024_RAW = _env_path("MEDCLIM_JADMON_2024", DATA / "Adriatic" / "Orig" / "JADMON_2024")
