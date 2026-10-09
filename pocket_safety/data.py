"""Loading, cleaning and per-frame feature engineering.

Column handling follows Section 1 of the spec:
  * gameId / playId / nflId  -> kept on every DataFrame, never in X
  * time                     -> dropped (redundant with frameId)
  * jerseyNumber             -> kept for display only
  * team                     -> derives is_offense (and plot colour)
  * playDirection            -> used to normalise coordinates, then dropped
  * o / dir                  -> replaced by sin/cos pairs
  * event                    -> used to find the snap / pocket end / QB, then dropped
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Row-level cleaning
# --------------------------------------------------------------------------- #
def load_plays() -> pd.DataFrame:
    return pd.read_csv(C.DATA_DIR / "plays.csv", usecols=["gameId", "playId", "possessionTeam"])


def load_tracking(game_id: int, plays: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load one game's tracking file and apply the row-level transforms."""
    df = pd.read_csv(C.TRACKING_DIR / f"tracking_{game_id}.csv")
    df = df.drop(columns=["time"])
    # pandas < 2.0 keeps the literal string "None"; normalise to NaN everywhere.
    df["event"] = df["event"].replace("None", np.nan)

    # --- playDirection: make every play move left -> right, then drop ----------
    left = df["playDirection"].eq("left").to_numpy()
    df.loc[left, "x"] = C.FIELD_LENGTH - df.loc[left, "x"]
    df.loc[left, "y"] = C.FIELD_WIDTH - df.loc[left, "y"]
    for ang in ("o", "dir"):
        df.loc[left, ang] = (df.loc[left, ang] + 180.0) % 360.0
    df = df.drop(columns=["playDirection"])

    # --- o / dir -> sin/cos -----------------------------------------------------
    for ang in ("o", "dir"):
        rad = np.deg2rad(df[ang].to_numpy())
        df[f"{ang}_sin"] = np.sin(rad)
        df[f"{ang}_cos"] = np.cos(rad)
    # velocity components (NGS convention: dir 0deg = +y, 90deg = +x)
    df["vx"] = df["s"] * df["dir_sin"]
    df["vy"] = df["s"] * df["dir_cos"]

    # --- team -> is_offense -----------------------------------------------------
    plays = load_plays() if plays is None else plays
    df = df.merge(plays, on=["gameId", "playId"], how="left", validate="many_to_one")
    df["is_ball"] = df["team"].eq("football")
    df["is_offense"] = df["team"].eq(df["possessionTeam"])
    df = df.drop(columns=["possessionTeam"])
    return df


def _event_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Snap frame and pocket-end frame per play, from the `event` column."""
    ev = df.loc[df["event"].notna(), ["gameId", "playId", "frameId", "event"]].drop_duplicates()
    snap = (ev[ev["event"].isin(C.SNAP_EVENTS)]
            .groupby(["gameId", "playId"])["frameId"].min().rename("snap_frame"))
    out = snap.to_frame().reset_index()
    ev = ev.merge(out, on=["gameId", "playId"])
    end = (ev[ev["event"].isin(C.END_EVENTS) & (ev["frameId"] > ev["snap_frame"])]
           .groupby(["gameId", "playId"])["frameId"].min().rename("end_frame"))
    out = out.merge(end.reset_index(), on=["gameId", "playId"], how="left")
    last = df.groupby(["gameId", "playId"])["frameId"].max().rename("last_frame").reset_index()
    out = out.merge(last, on=["gameId", "playId"])
    out["end_frame"] = out["end_frame"].fillna(out["last_frame"]).astype(int)
    return out.drop(columns="last_frame")


def _identify_qb(df: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """QB = offensive player closest to the ball in the most post-snap frames.

    Uses the event-derived window [snap + 5, end_frame] (ball is in the QB's
    hands for most of that span, after the centre has released it).
    """
    d = df.merge(windows, on=["gameId", "playId"])
    ball = d.loc[d["is_ball"], C.FRAME_KEYS + ["x", "y"]].rename(columns={"x": "bx", "y": "by"})
    off = d.loc[d["is_offense"], C.FRAME_KEYS + ["nflId", "x", "y", "snap_frame", "end_frame"]]
    off = off.merge(ball, on=C.FRAME_KEYS)
    off["d_ball"] = np.hypot(off["x"] - off["bx"], off["y"] - off["by"])

    # The snapper (centre) is nearest the ball at the snap -> never the QB.
    at_snap = off[off["frameId"] == off["snap_frame"]]
    snapper = (at_snap.loc[at_snap.groupby(["gameId", "playId"])["d_ball"].idxmin(),
                           ["gameId", "playId", "nflId"]].rename(columns={"nflId": "snapper"}))
    off = off.merge(snapper, on=["gameId", "playId"], how="left")
    off = off[(off["nflId"] != off["snapper"])
              & (off["frameId"] >= off["snap_frame"] + 5) & (off["frameId"] <= off["end_frame"])]
    nearest = off.loc[off.groupby(C.FRAME_KEYS)["d_ball"].idxmin(), ["gameId", "playId", "nflId"]]
    qb = (nearest.groupby(["gameId", "playId"])["nflId"]
          .agg(lambda s: s.value_counts().idxmax()).rename("qb_nflId").reset_index())
    return qb


# --------------------------------------------------------------------------- #
# Frame-level features + target
# --------------------------------------------------------------------------- #
def _slot_features(rel: pd.DataFrame, side: str, k: int) -> pd.DataFrame:
    """k nearest players of one side to the QB -> wide, distance-ordered slots."""
    rel = rel.sort_values(C.FRAME_KEYS + ["dist"])
    rel["slot"] = rel.groupby(C.FRAME_KEYS).cumcount()
    rel = rel[rel["slot"] < k]
    cols = C.PLAYER_FEATS + C.REL_FEATS
    wide = rel.pivot(index=C.FRAME_KEYS, columns="slot", values=cols)
    wide.columns = [f"{side}{slot + 1}_{feat}" for feat, slot in wide.columns]
    return wide


def build_game_frames(game_id: int) -> pd.DataFrame:
    """One row per pocket-phase frame of every play in a game."""
    plays = load_plays()
    df = load_tracking(game_id, plays)
    windows = _event_frames(df)
    qb_ids = _identify_qb(df, windows)
    df = df.drop(columns=["event"])  # event fully consumed

    df = df.merge(windows, on=["gameId", "playId"]).merge(qb_ids, on=["gameId", "playId"])
    df = df[(df["frameId"] >= df["snap_frame"]) & (df["frameId"] <= df["end_frame"])]

    # Line of scrimmage = ball x at the snap (plays are normalised L->R).
    los = (df[df["is_ball"] & (df["frameId"] == df["snap_frame"])]
           .set_index(["gameId", "playId"])["x"].rename("los_x"))
    df = df.join(los, on=["gameId", "playId"])

    players = df[~df["is_ball"]]
    is_qb = players["nflId"].eq(players["qb_nflId"])
    qb = players[is_qb].copy()
    others = players[~is_qb]

    # QB block: depth behind LOS, lateral position, kinematics.
    qb["x"] = qb["los_x"] - qb["x"]          # yards behind line of scrimmage
    qb["frames_since_snap"] = qb["frameId"] - qb["snap_frame"]
    qb_block = qb[C.FRAME_KEYS + ["qb_nflId", "frames_since_snap"] + C.PLAYER_FEATS]
    qb_block = qb_block.rename(columns={f: f"qb_{f}" for f in C.PLAYER_FEATS})

    # Everyone else, expressed relative to the QB.
    q = qb[C.FRAME_KEYS + ["vx", "vy"]].rename(columns={"vx": "qvx", "vy": "qvy"})
    q[["qx", "qy"]] = players.loc[is_qb, ["x", "y"]].to_numpy()
    rel = others.merge(q, on=C.FRAME_KEYS)
    dx, dy = rel["x"] - rel["qx"], rel["y"] - rel["qy"]
    rel["dist"] = np.hypot(dx, dy)
    # Closing speed: rate at which the gap to the QB shrinks (>0 = closing).
    ux, uy = dx / rel["dist"].clip(lower=1e-6), dy / rel["dist"].clip(lower=1e-6)
    rel["closing_speed"] = -((rel["vx"] - rel["qvx"]) * ux + (rel["vy"] - rel["qvy"]) * uy)
    rel["x"], rel["y"] = dx, dy

    def_wide = _slot_features(rel[~rel["is_offense"]], "def", C.K_DEF)
    off_wide = _slot_features(rel[rel["is_offense"]], "off", C.K_OFF)

    frames = (qb_block.set_index(C.FRAME_KEYS)
              .join(def_wide, how="inner").join(off_wide, how="inner")
              .reset_index())
    frames = frames.rename(columns={"qb_nflId": "nflId"})  # nflId = the tracked QB
    frames["nflId"] = frames["nflId"].astype("int64")
    frames = frames.dropna()
    return add_target(frames)


def add_target(frames: pd.DataFrame) -> pd.DataFrame:
    """Heuristic forward-looking label (no ground-truth score exists in the data).

    d_future(t) = min over frames [t, t+HORIZON] of the nearest-defender-to-QB
    distance (within the pocket phase).  Linearly mapped:
        d_future <= D_COLLAPSED -> 0,  d_future >= D_CLEAN -> 100.
    """
    frames = frames.sort_values(C.FRAME_KEYS).copy()
    d_now = frames["def1_dist"]
    rev = d_now.iloc[::-1].groupby([frames["gameId"].iloc[::-1], frames["playId"].iloc[::-1]])
    d_future = rev.transform(lambda s: s.rolling(C.HORIZON + 1, min_periods=1).min()).iloc[::-1]
    frames["d_future"] = d_future
    score = (d_future - C.D_COLLAPSED) / (C.D_CLEAN - C.D_COLLAPSED)
    frames["safety_score"] = 100.0 * score.clip(0.0, 1.0)
    return frames


class DataNotFoundError(FileNotFoundError):
    pass


def check_data_dir() -> list[int]:
    """Validate the dataset folder and return the available game ids."""
    tracking = sorted(C.TRACKING_DIR.glob("tracking_*.csv"))
    missing = [f for f in ("plays.csv", "pffScoutingData.csv") if not (C.DATA_DIR / f).exists()]
    if not tracking or missing:
        searched = "\n".join(f"  - {p}" for p in C.DATA_CANDIDATES)
        problem = (f"no tracking_*.csv files in {C.TRACKING_DIR}" if not tracking
                   else f"missing {', '.join(missing)} in {C.DATA_DIR}")
        raise DataNotFoundError(
            f"Dataset not found: {problem}.\n"
            f"Searched (unless {C.DATA_ENV_VAR} / --data-dir is set):\n{searched}\n"
            "Fix: download the data (see README) or point to it explicitly, e.g.\n"
            "  python run.py --data-dir /path/to/nfl-big-data-bowl-regional-event-data/data")
    return sorted(int(p.stem.split("_")[1]) for p in tracking)


def build_all_frames(force: bool = False, workers: int | None = None) -> pd.DataFrame:
    if C.CACHE_PATH.exists() and not force:
        return pd.read_parquet(C.CACHE_PATH)
    game_ids = check_data_dir()
    workers = workers or min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = [p for p in ex.map(build_game_frames, game_ids) if len(p)]
    if not parts:
        raise ValueError(f"Read {len(game_ids)} tracking files from {C.TRACKING_DIR} but produced no "
                         "pocket-phase frames; check the files are the BDB 2023 tracking CSVs.")
    frames = pd.concat(parts, ignore_index=True)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(C.CACHE_PATH, index=False)
    return frames


def feature_columns(frames: pd.DataFrame) -> list[str]:
    """FEATURES list: everything except identifiers / target / helpers."""
    exclude = set(C.ID_COLS) | {"safety_score", "d_future"}
    return [c for c in frames.columns if c not in exclude]
