"""Constants for the Pass Pocket Safety pipeline."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- Data location -----------------------------------------------------------
# Resolution order: POCKET_DATA_DIR env var (also set by `run.py --data-dir`),
# then the first candidate below that contains tracking/tracking_*.csv.
DATA_ENV_VAR = "POCKET_DATA_DIR"
_DATASET = "nfl-big-data-bowl-regional-event-data"
DATA_CANDIDATES = [
    ROOT / "data_repo" / "data",
    ROOT / "data",
    ROOT / _DATASET / "data",
    ROOT / f"{_DATASET}-main" / "data",
    ROOT.parent / _DATASET / "data",           # dataset cloned next to this repo
    ROOT.parent / f"{_DATASET}-main" / "data",
]


def _has_tracking(d: Path) -> bool:
    return (d / "tracking").is_dir() and any((d / "tracking").glob("tracking_*.csv"))


def _normalise(d: Path) -> Path:
    """Accept the dataset root, its data/ folder, or the tracking/ folder itself."""
    d = Path(d).expanduser().resolve()
    for cand in (d, d / "data", d.parent):
        if _has_tracking(cand):
            return cand
    return d


def resolve_data_dir() -> Path:
    env = os.environ.get(DATA_ENV_VAR)
    if env:
        return _normalise(Path(env))
    for cand in DATA_CANDIDATES:
        if _has_tracking(cand):
            return cand.resolve()
    return DATA_CANDIDATES[0]


def set_data_dir(path: str | os.PathLike | None) -> Path:
    """Point the pipeline at a dataset folder (propagates to worker processes)."""
    global DATA_DIR, TRACKING_DIR
    if path is not None:
        os.environ[DATA_ENV_VAR] = str(Path(path).expanduser().resolve())
    DATA_DIR = resolve_data_dir()
    TRACKING_DIR = DATA_DIR / "tracking"
    return DATA_DIR


DATA_DIR = resolve_data_dir()
TRACKING_DIR = DATA_DIR / "tracking"
OUT_DIR = ROOT / "outputs"
CACHE_PATH = OUT_DIR / "frames.parquet"

FIELD_LENGTH = 120.0
FIELD_WIDTH = 160.0 / 3.0  # 53.33 yd

# Identifier columns: kept on every DataFrame, never part of X.
ID_COLS = ["gameId", "playId", "nflId"]
FRAME_KEYS = ["gameId", "playId", "frameId"]

SNAP_EVENTS = {"ball_snap", "autoevent_ballsnap"}
# First of these after the snap ends the "pocket phase".
END_EVENTS = {"pass_forward", "autoevent_passforward", "qb_sack", "qb_strip_sack", "run", "fumble"}

# Per-player kinematic features (after direction normalization + angle encoding).
PLAYER_FEATS = ["x", "y", "s", "a", "dis", "o_sin", "o_cos", "dir_sin", "dir_cos"]
# Extra QB-relative features for every non-QB slot.
REL_FEATS = ["dist", "closing_speed"]

K_DEF = 7  # nearest defenders to the QB kept per frame
K_OFF = 7  # nearest offensive teammates (blockers) kept per frame

# ---- Target (heuristic label; the dataset has no ground-truth pocket score) ----
HORIZON = 10         # frames (10 Hz -> 1.0 s look-ahead)
D_COLLAPSED = 1.0    # yd: nearest defender this close within horizon -> score 0
D_CLEAN = 6.0        # yd: nearest defender never closer than this -> score 100

RANDOM_STATE = 42
TEST_SIZE = 0.2
