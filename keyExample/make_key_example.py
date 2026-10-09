"""Key example: animate one play's player movement next to its Pocket Safety Score.

Usage (from the repo root):
    python keyExample/make_key_example.py                      # default play
    python keyExample/make_key_example.py --game 2021092611 --play 3118 --fps 5
    python keyExample/make_key_example.py --data-dir /path/to/dataset

Writes into keyExample/:
    pocket_safety_example.gif   animation: field (left) + score/distance timelines (right)
    key_frames.png              static: 4 field snapshots above the score timeline
    play_frames.csv             per-frame predicted/target score, nearest-defender distance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from pocket_safety import config as C  # noqa: E402

# Murray -> Hopkins, 15 yd completion, held-out game, no PFF pressure, pocket stays clean.
DEFAULT_GAME, DEFAULT_PLAY = 2021091909, 476

TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D", "CAR": "#0085CA",
    "CHI": "#C83803", "CIN": "#FB4F14", "CLE": "#FF3C00", "DAL": "#003594", "DEN": "#FB4F14",
    "DET": "#0076B6", "GB": "#203731", "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778",
    "KC": "#E31837", "LA": "#003594", "LAC": "#0080C6", "LV": "#A5ACAF", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265", "NYJ": "#125740",
    "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#002244", "SF": "#AA0000", "TB": "#D50A0A",
    "TEN": "#4B92DB", "WAS": "#5A1414",
}
CMAP = plt.get_cmap("RdYlGn")
NORM = Normalize(0, 100)
FIELD_GREEN = "#3a7d44"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_play(game_id: int, play_id: int):
    from pocket_safety import data

    data.check_data_dir()
    trk = data.load_tracking(game_id)
    trk = trk[trk["playId"] == play_id].copy()
    if trk.empty:
        raise SystemExit(f"play {play_id} not found in game {game_id}")

    frames = data.build_game_frames(game_id)
    frames = frames[frames["playId"] == play_id].copy()
    frames["pred"] = _predictions(frames, game_id, play_id)
    qb_id = int(frames["nflId"].iloc[0])

    # Nearest defender to the QB in every pocket frame (identity, for highlighting).
    qb = trk.loc[trk["nflId"] == qb_id, ["frameId", "x", "y"]].rename(columns={"x": "qx", "y": "qy"})
    d = trk[~trk["is_offense"] & ~trk["is_ball"]].merge(qb, on="frameId")
    d["dist"] = np.hypot(d["x"] - d["qx"], d["y"] - d["qy"])
    nearest = d.loc[d.groupby("frameId")["dist"].idxmin(), ["frameId", "nflId", "jerseyNumber"]]
    nearest = nearest.rename(columns={"nflId": "nearest_def_nflId", "jerseyNumber": "nearest_def_jersey"})
    frames = frames.merge(nearest, on="frameId", how="left")

    plays = pd.read_csv(C.DATA_DIR / "plays.csv")
    info = plays[(plays["gameId"] == game_id) & (plays["playId"] == play_id)].iloc[0]
    games = pd.read_csv(C.DATA_DIR / "games.csv")
    ginfo = games[games["gameId"] == game_id].iloc[0]
    return trk, frames, qb_id, info, ginfo


def _predictions(frames: pd.DataFrame, game_id: int, play_id: int) -> np.ndarray:
    """Held-out predictions from outputs/ (version-independent); fall back to the saved model."""
    tp = C.OUT_DIR / "test_predictions.parquet"
    if tp.exists():
        p = pd.read_parquet(tp)
        p = p[(p["gameId"] == game_id) & (p["playId"] == play_id)]
        if len(p):
            m = frames[["frameId"]].merge(p[["frameId", "pred"]], on="frameId", how="left")
            if m["pred"].notna().all():
                print("predictions: outputs/test_predictions.parquet (held-out game)")
                return m["pred"].to_numpy()
    import joblib

    bundle = joblib.load(C.OUT_DIR / "model.joblib")
    print(f"predictions: outputs/model.joblib ({bundle.get('name', 'model')})")
    return np.clip(bundle["model"].predict(frames[bundle["features"]]), 0, 100)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw_field(ax, los_x, first_down_x, xlim):
    ax.set_facecolor(FIELD_GREEN)
    for x in range(10, 111, 5):
        if xlim[0] - 1 <= x <= xlim[1] + 1:
            ax.axvline(x, color="white", lw=1.2 if x % 10 == 0 else 0.6, alpha=0.55, zorder=0)
            if x % 10 == 0 and 10 < x < 110:
                yard = x - 10 if x <= 60 else 110 - x
                for y in (5, C.FIELD_WIDTH - 5):
                    ax.text(x, y, str(yard), color="white", alpha=0.6, ha="center", va="center",
                            fontsize=11, fontweight="bold", zorder=0)
    for xe in (0, 110):
        if xlim[0] < xe + 10 and xe < xlim[1]:
            ax.add_patch(Rectangle((xe, 0), 10, C.FIELD_WIDTH, color="#2c5f34", zorder=0))
    ax.axvline(los_x, color="#3fa9f5", lw=2, zorder=1)
    ax.axvline(first_down_x, color="#ffd400", lw=2, zorder=1)
    ax.set(xlim=xlim, ylim=(0, C.FIELD_WIDTH), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_frame(ax, trk, fid, qb_id, row, los_x, first_down_x, xlim, colors, phase):
    ax.clear()
    draw_field(ax, los_x, first_down_x, xlim)
    fr = trk[trk["frameId"] == fid]
    qb = fr[fr["nflId"] == qb_id]

    # Pocket ring: D_CLEAN-radius circle around the QB coloured by predicted score.
    if row is not None and len(qb):
        qx, qy = qb[["x", "y"]].iloc[0]
        col = CMAP(NORM(row["pred"]))
        ax.add_patch(Circle((qx, qy), C.D_CLEAN, facecolor=col, alpha=0.30, edgecolor="none", zorder=2))
        ax.add_patch(Circle((qx, qy), C.D_CLEAN, fill=False, edgecolor="black", lw=3.5, zorder=2))
        ax.add_patch(Circle((qx, qy), C.D_CLEAN, fill=False, edgecolor=col, lw=2, zorder=2))
        ax.add_patch(Circle((qx, qy), C.D_COLLAPSED, fill=False, edgecolor="white", ls=":", lw=1, zorder=2))
        nd = fr[fr["nflId"] == row["nearest_def_nflId"]]
        if len(nd):
            nx, ny = nd[["x", "y"]].iloc[0]
            ax.plot([qx, nx], [qy, ny], color="white", ls="--", lw=1.3, zorder=3)
            ax.text(qx - 1.0, qy + 1.6, f"{row['def1_dist']:.1f} yd", color="black", fontsize=8,
                    ha="right", fontweight="bold", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    for _, r in fr.iterrows():
        if r["is_ball"]:
            continue
        is_qb = r["nflId"] == qb_id
        is_nd = row is not None and r["nflId"] == row["nearest_def_nflId"]
        ax.scatter(r["x"], r["y"], s=230, c=colors[r["team"]], zorder=4,
                   edgecolors="gold" if is_qb else ("white" if is_nd else "black"),
                   linewidths=3 if (is_qb or is_nd) else 0.6)
        ax.text(r["x"], r["y"], f"{int(r['jerseyNumber'])}", color="white", fontsize=7,
                ha="center", va="center", fontweight="bold", zorder=5)
        ax.arrow(r["x"], r["y"], r["vx"] * 0.35, r["vy"] * 0.35, head_width=0.35, color="white",
                 alpha=0.7, lw=0.8, zorder=3, length_includes_head=True)
    ball = fr[fr["is_ball"]]
    if len(ball):
        ax.scatter(ball["x"], ball["y"], s=55, c="#8b4513", edgecolors="white", linewidths=0.8, zorder=7,
                   marker="o")

    t = (fid - row_snap(trk)) / 10
    label = {"pre": "pre-snap", "pocket": f"{t:.1f} s after snap", "post": "ball in the air"}[phase]
    score = f"   |   pocket safety {row['pred']:.0f}" if row is not None else ""
    ax.set_title(f"frame {fid}  ·  {label}{score}", fontsize=11)


def row_snap(trk) -> int:
    return int(trk.attrs["snap_frame"])


def timeline_axes(ax_s, ax_d, frames, snap, throw):
    t = (frames["frameId"] - snap) / 10
    t_end = (throw - snap) / 10
    for y0, y1 in ((0, 33), (33, 67), (67, 100)):
        ax_s.axhspan(y0, y1, color=CMAP(NORM((y0 + y1) / 2)), alpha=0.12, zorder=0)
    ax_s.set(xlim=(-0.05, t_end + 0.05), ylim=(-2, 102), ylabel="pocket safety score")
    ax_s.grid(alpha=0.3)
    ax_d.axhline(C.D_CLEAN, color="seagreen", ls=":", lw=1)
    ax_d.axhline(C.D_COLLAPSED, color="firebrick", ls=":", lw=1)
    ax_d.set(xlim=ax_s.get_xlim(), ylim=(0, max(8.0, frames["def1_dist"].max() + 0.5)),
             xlabel="seconds after snap", ylabel="nearest defender (yd)")
    ax_d.grid(alpha=0.3)
    for ax in (ax_s, ax_d):
        ax.axvline(0, color="#3fa9f5", lw=1)
        ax.axvline(t_end, color="0.2", lw=1)
    ax_s.text(0.02, 3, "snap", color="#3fa9f5", fontsize=8)
    ax_s.text(t_end - 0.02, 3, "throw", fontsize=8, ha="right")
    return t


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", type=int, default=DEFAULT_GAME)
    ap.add_argument("--play", type=int, default=DEFAULT_PLAY)
    ap.add_argument("--fps", type=int, default=5, help="5 = half speed (tracking is 10 Hz)")
    ap.add_argument("--data-dir")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    C.set_data_dir(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trk, frames, qb_id, info, ginfo = load_play(args.game, args.play)
    snap, throw = int(frames["frameId"].min()), int(frames["frameId"].max())
    trk.attrs["snap_frame"] = snap
    by_frame = frames.set_index("frameId")

    los_x = float(trk.loc[trk["is_ball"] & (trk["frameId"] == snap), "x"].iloc[0])
    first_down_x = los_x + float(info["yardsToGo"])
    xmax = max(trk.loc[~trk["is_ball"], "x"].max(), trk.loc[trk["is_ball"], "x"].max())
    xlim = (los_x - 12, min(120, max(los_x + 25, xmax + 3)))
    off, de = info["possessionTeam"], info["defensiveTeam"]
    colors = {off: TEAM_COLORS.get(off, "#d62728"), de: TEAM_COLORS.get(de, "#1f77b4"), "football": "#8b4513"}
    if colors[off] == colors[de]:
        colors[de] = "#1f77b4"

    title = (f"{ginfo['visitorTeamAbbr']} @ {ginfo['homeTeamAbbr']}, week {ginfo['week']} {ginfo['season']}  ·  "
             f"Q{info['quarter']} {info['gameClock']}  ·  {info['down']}&{info['yardsToGo']}\n"
             f"{info['playDescription']}")
    legend = [plt.Line2D([], [], marker="o", ls="", color=colors[off], ms=10, label=f"{off} (offense)"),
              plt.Line2D([], [], marker="o", ls="", color=colors[de], ms=10, label=f"{de} (defense)"),
              plt.Line2D([], [], marker="o", ls="", mfc="none", mec="gold", mew=2.5, ms=10, label="QB"),
              plt.Line2D([], [], marker="o", ls="", mfc="none", mec="white", mew=2.5, ms=10,
                         label="nearest defender"),
              plt.Line2D([], [], color="#3fa9f5", lw=2, label="line of scrimmage"),
              plt.Line2D([], [], color="#ffd400", lw=2, label="first down")]

    def phase(fid):
        return "pre" if fid < snap else ("pocket" if fid <= throw else "post")

    # ---- per-frame CSV --------------------------------------------------------
    frames.assign(seconds_after_snap=(frames["frameId"] - snap) / 10)[
        ["gameId", "playId", "frameId", "seconds_after_snap", "nflId", "pred", "safety_score",
         "def1_dist", "def1_closing_speed", "nearest_def_nflId", "nearest_def_jersey"]
    ].rename(columns={"nflId": "qb_nflId", "pred": "predicted_score", "safety_score": "target_score",
                      "def1_dist": "nearest_def_dist_yd", "def1_closing_speed": "nearest_def_closing_speed"}
             ).round(3).to_csv(out / "play_frames.csv", index=False)

    # ---- animation ------------------------------------------------------------
    fig = plt.figure(figsize=(15, 7.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], height_ratios=[1, 1, 0.75],
                          hspace=0.35, wspace=0.12, left=0.02, right=0.97, top=0.86, bottom=0.08)
    ax_f = fig.add_subplot(gs[:, 0])
    ax_s = fig.add_subplot(gs[0:2, 1])
    ax_d = fig.add_subplot(gs[2, 1], sharex=ax_s)
    fig.suptitle(title, fontsize=11)
    t_all = timeline_axes(ax_s, ax_d, frames, snap, throw)
    (tgt_line,) = ax_s.plot([], [], color="0.35", ls="--", lw=1.3, label="target (next 1 s)")
    (pred_line,) = ax_s.plot([], [], color="tab:blue", lw=2.5, label="predicted (PCA + Ridge)")
    (pred_dot,) = ax_s.plot([], [], "o", ms=9, mec="black", zorder=5)
    (dist_line,) = ax_d.plot([], [], color="tab:purple", lw=2)
    cursor_s, cursor_d = ax_s.axvline(np.nan, color="black", lw=1), ax_d.axvline(np.nan, color="black", lw=1)
    ax_s.legend(loc="upper right", fontsize=9)
    readout = ax_s.text(0.02, 0.96, "", transform=ax_s.transAxes, va="top", fontsize=10,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    seq = sorted(trk["frameId"].unique())
    seq = seq + [seq[-1]] * args.fps * 2          # hold the last frame for 2 s

    def update(fid):
        ph = phase(fid)
        row = by_frame.loc[fid].to_dict() if ph == "pocket" else None
        draw_frame(ax_f, trk, fid, qb_id, row, los_x, first_down_x, xlim, colors, ph)
        ax_f.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.85, ncol=2)
        k = int(np.searchsorted(frames["frameId"].to_numpy(), min(max(fid, snap), throw), side="right"))
        if ph == "pre":
            k = 0
        tgt_line.set_data(t_all[:k], frames["safety_score"].iloc[:k])
        pred_line.set_data(t_all[:k], frames["pred"].iloc[:k])
        dist_line.set_data(t_all[:k], frames["def1_dist"].iloc[:k])
        if k:
            r = frames.iloc[k - 1]
            tt = t_all.iloc[k - 1]
            pred_dot.set_data([tt], [r["pred"]])
            pred_dot.set_color(CMAP(NORM(r["pred"])))
            pred_dot.set_markeredgecolor("black")
            for c in (cursor_s, cursor_d):
                c.set_xdata([tt, tt])
            readout.set_text(f"predicted {r['pred']:5.1f}   target {r['safety_score']:5.1f}\n"
                             f"nearest defender #{int(r['nearest_def_jersey'])}: {r['def1_dist']:.1f} yd, "
                             f"closing {r['def1_closing_speed']:+.1f} yd/s")
        else:
            pred_dot.set_data([], [])
            readout.set_text("waiting for the snap…")
        return []

    anim = FuncAnimation(fig, update, frames=seq, interval=1000 / args.fps, blit=False)
    gif = out / "pocket_safety_example.gif"
    anim.save(gif, writer=PillowWriter(fps=args.fps), dpi=80)
    plt.close(fig)
    print(f"wrote {gif}")

    # ---- static key frames ----------------------------------------------------
    picks = [snap, snap + (throw - snap) // 3, snap + 2 * (throw - snap) // 3, throw]
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1], hspace=0.28, wspace=0.05,
                          left=0.04, right=0.98, top=0.88, bottom=0.07)
    zoom = (los_x - 12, los_x + 14)
    for i, fid in enumerate(picks):
        ax = fig.add_subplot(gs[0, i])
        draw_frame(ax, trk, fid, qb_id, by_frame.loc[fid].to_dict(), los_x, first_down_x, zoom, colors, "pocket")
        qy = float(trk.loc[(trk["frameId"] == fid) & (trk["nflId"] == qb_id), "y"].iloc[0])
        ax.set_ylim(qy - 15, qy + 15)
        ax.set_title(f"t = {(fid - snap) / 10:.1f} s  ·  score {by_frame.loc[fid, 'pred']:.0f}", fontsize=11)
        if i == 0:
            ax.legend(handles=legend, loc="lower left", fontsize=7, framealpha=0.85)
    ax_t = fig.add_subplot(gs[1, :])
    for y0, y1 in ((0, 33), (33, 67), (67, 100)):
        ax_t.axhspan(y0, y1, color=CMAP(NORM((y0 + y1) / 2)), alpha=0.12)
    ax_t.plot(t_all, frames["safety_score"], color="0.35", ls="--", lw=1.3, label="target (next 1 s)")
    ax_t.plot(t_all, frames["pred"], color="tab:blue", lw=2.5, label="predicted (PCA + Ridge)")
    for fid in picks:
        tt = (fid - snap) / 10
        ax_t.axvline(tt, color="black", lw=0.8, ls=":")
        ax_t.scatter([tt], [by_frame.loc[fid, "pred"]], s=90, c=[CMAP(NORM(by_frame.loc[fid, "pred"]))],
                     edgecolors="black", zorder=5)
    ax_t.set(xlabel="seconds after snap", ylabel="pocket safety score", ylim=(-2, 102))
    ax_t.grid(alpha=0.3)
    ax_t.legend(loc="lower left")
    fig.suptitle(title, fontsize=12)
    fig.savefig(out / "key_frames.png", dpi=110)
    plt.close(fig)
    print(f"wrote {out / 'key_frames.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
