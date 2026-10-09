"""Constants for the Pass Pocket Safety pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_repo" / "data"
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
