"""What makes a pocket safe, and does a safe pocket make a play successful?

Builds one row per dropback (8.5k plays, 122 games) with
  * structure:   rushers vs blockers, formation, dropback type, play action, coverage, box count
  * mechanics:   measured 1.0 s after the snap - unblocked rushers, tackle-to-tackle width,
                 OL depth, QB-to-OL cushion; QB set depth and drift over the pocket
  * pocket:      observed pocket safety (target score), closest defender, time to throw
  * outcomes:    PFF pressure (hit/hurry/sack), sack, completion, INT, yards
and writes tables + figures to judge/analysis/output/.

Usage (repo root):  python judge/analysis/pocket_drivers.py [--data-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pocket_safety import config as C  # noqa: E402

OUT = HERE / "output"
T_MECH = 10          # frames after snap at which mechanics are measured (1.0 s)
ENGAGED_YD = 1.5     # rusher counts as blocked if a pass blocker is within this distance
OL = ["LT", "LG", "C", "RG", "RT"]


# --------------------------------------------------------------------------- #
# Per-play table
# --------------------------------------------------------------------------- #
def _game_mechanics(args) -> pd.DataFrame:
    game_id, data_dir = args
    C.set_data_dir(data_dir)
    from pocket_safety import data

    frames = data.build_game_frames(game_id)
    trk = data.load_tracking(game_id)
    pff = pd.read_csv(C.DATA_DIR / "pffScoutingData.csv",
                      usecols=["gameId", "playId", "nflId", "pff_role", "pff_positionLinedUp"])
    trk = trk.merge(pff[pff["gameId"] == game_id], on=["gameId", "playId", "nflId"], how="left")

    rows = []
    for pid, fr in frames.groupby("playId"):
        fr = fr.sort_values("frameId")
        snap, end = int(fr["frameId"].iloc[0]), int(fr["frameId"].iloc[-1])
        qb_id = int(fr["nflId"].iloc[0])
        p = trk[trk["playId"] == pid]
        los = float(p.loc[p["is_ball"] & (p["frameId"] == snap), "x"].iloc[0])
        qb = p[p["nflId"] == qb_id].set_index("frameId").sort_index()
        qbw = qb.loc[snap:end]
        r = {"gameId": game_id, "playId": pid,
             "ttt": (end - snap) / 10,
             "mean_score": fr["safety_score"].mean(),
             "min_score": fr["safety_score"].min(),
             "score_at_throw": fr["safety_score"].iloc[-1],
             "min_def_dist": fr["def1_dist"].min(),
             "qb_set_depth": los - qbw["x"].max() if len(qbw) else np.nan,    # deepest point reached
             "qb_snap_depth": los - qbw["x"].iloc[0] if len(qbw) else np.nan,
             "qb_drift": qbw["dis"].iloc[1:].sum() if len(qbw) else np.nan,
             "qb_lateral": abs(qbw["y"].iloc[-1] - qbw["y"].iloc[0]) if len(qbw) else np.nan}

        f1 = snap + T_MECH
        if end >= f1:
            a = p[p["frameId"] == f1]
            q = a[a["nflId"] == qb_id][["x", "y"]].to_numpy()[0]
            rush = a[a["pff_role"] == "Pass Rush"][["x", "y"]].to_numpy()
            blk = a[a["pff_role"] == "Pass Block"][["x", "y"]].to_numpy()
            ol = a[a["pff_positionLinedUp"].isin(OL) & (a["pff_role"] == "Pass Block")].set_index(
                "pff_positionLinedUp")
            if len(rush) and len(blk):
                d_rb = np.hypot(rush[:, None, 0] - blk[None, :, 0], rush[:, None, 1] - blk[None, :, 1]).min(1)
                d_rq = np.hypot(rush[:, 0] - q[0], rush[:, 1] - q[1])
                r["unblocked_1s"] = int((d_rb > ENGAGED_YD).sum())
                # unblocked rushers already inside 4 yd of the QB = immediate leak
                r["leak_1s"] = int(((d_rb > ENGAGED_YD) & (d_rq < 4.0)).sum())
            if {"LT", "RT"} <= set(ol.index):
                r["tackle_width_1s"] = abs(ol.loc["LT", "y"] - ol.loc["RT", "y"])
            if len(ol):
                ol_depth = los - ol["x"].mean()
                r["ol_depth_1s"] = ol_depth                       # + = OL pushed back behind LOS
                r["qb_cushion_1s"] = (los - q[0]) - ol_depth      # yards between QB and his OL
        rows.append(r)
    return pd.DataFrame(rows)


def build_play_table(data_dir: Path) -> pd.DataFrame:
    from pocket_safety import data

    game_ids = data.check_data_dir()
    with ProcessPoolExecutor(max_workers=8) as ex:
        mech = pd.concat(ex.map(_game_mechanics, [(g, str(data_dir)) for g in game_ids]), ignore_index=True)

    plays = pd.read_csv(C.DATA_DIR / "plays.csv")
    pff = pd.read_csv(C.DATA_DIR / "pffScoutingData.csv")
    pff["press"] = pff[["pff_hit", "pff_hurry", "pff_sack"]].fillna(0).sum(axis=1) > 0
    roles = pff.groupby(["gameId", "playId"]).agg(
        n_rush=("pff_role", lambda s: (s == "Pass Rush").sum()),
        n_block=("pff_role", lambda s: (s == "Pass Block").sum()),
        pressure=("press", "any")).reset_index()
    df = mech.merge(plays, on=["gameId", "playId"]).merge(roles, on=["gameId", "playId"])
    df["surplus"] = df["n_block"] - df["n_rush"]
    df["sack"] = df["passResult"].eq("S")
    df["scramble"] = df["passResult"].eq("R")
    att = df["passResult"].isin(["C", "I", "IN"])
    df["attempt"] = att
    df["complete"] = np.where(att, df["passResult"].eq("C"), np.nan)
    df["interception"] = np.where(att, df["passResult"].eq("IN"), np.nan)
    df["yards"] = df["playResult"]
    df["play_action"] = df["pff_playAction"].eq(1)
    return df


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def outcome_summary(g) -> pd.Series:
    return pd.Series({"plays": len(g),
                      "pressure_rate": g["pressure"].mean(),
                      "mean_pocket_score": g["mean_score"].mean(),
                      "sack_rate": g["sack"].mean(),
                      "completion_pct": g["complete"].mean(),
                      "int_rate": g["interception"].mean(),
                      "yards_per_dropback": g["yards"].mean()})


def _md(t: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    fmt = {"plays": "{:.0f}", "mean_pocket_score": "{:.1f}", "yards_per_dropback": "{:.2f}"}
    cols = [t.index.name or ""] + list(t.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in t.iterrows():
        vals = [fmt.get(c, "{:.1%}").format(v) if pd.notna(v) else "" for c, v in row.items()]
        lines.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join(lines)


def by(df, col, bins=None, labels=None, min_n=60) -> pd.DataFrame:
    key = pd.cut(df[col], bins, labels=labels, right=False) if bins is not None else df[col].copy()
    t = df.groupby(key.rename("_bin"), observed=True).apply(outcome_summary)
    t = t[t["plays"] >= min_n]
    t.index.name = col
    return t


def pressure_logit(df) -> tuple[pd.DataFrame, float]:
    """Multivariate: which factors predict PFF pressure, holding the others fixed?"""
    d = df[df["ttt"] >= T_MECH / 10].dropna(subset=["unblocked_1s", "tackle_width_1s", "ol_depth_1s"]).copy()
    num = ["surplus", "leak_1s", "unblocked_1s", "tackle_width_1s", "ol_depth_1s",
           "qb_snap_depth", "qb_drift", "ttt", "defendersInBox"]
    d["defendersInBox"] = d["defendersInBox"].fillna(d["defendersInBox"].median())
    cats = pd.get_dummies(d[["offenseFormation", "pff_passCoverageType"]].fillna("NA"), drop_first=False)
    cats = cats.drop(columns=[c for c in ("offenseFormation_SHOTGUN", "pff_passCoverageType_Zone") if c in cats])
    cats = cats.loc[:, cats.sum() >= 100].astype(float)
    flags = pd.DataFrame({"play_action": d["play_action"].astype(float),
                          "designed_rollout": d["dropBackType"].str.startswith("DESIGNED_ROLLOUT", na=False)
                          .astype(float)})
    X = pd.concat([d[num], flags, cats], axis=1)
    y = d["pressure"].astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    p = cross_val_predict(model, X, y, groups=d["gameId"], cv=GroupKFold(5), method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    model.fit(X, y)
    coef = model[-1].coef_[0]
    t = pd.DataFrame({"feature": X.columns, "odds_ratio_per_sd": np.exp(coef),
                      "sd": X.std().to_numpy(), "mean": X.mean().to_numpy()}).sort_values("odds_ratio_per_sd")
    return t, auc


def model_group_importance() -> pd.DataFrame:
    """Grouped permutation importance of the PCA+Ridge model on the held-out games."""
    import joblib

    from pocket_safety.data import build_all_frames
    from pocket_safety.model import adjusted_r2, n_predictors, predict_score, split_by_game

    bundle = joblib.load(C.OUT_DIR / "model.joblib")
    m, feats = bundle["model"], bundle["features"]
    _, test = split_by_game(build_all_frames())
    X, y = test[feats].copy(), test["safety_score"].to_numpy()
    p = n_predictors(m)
    base = adjusted_r2(y, predict_score(m, X), p)

    def g(pred):
        return [f for f in feats if pred(f)]
    groups = {
        "nearest 2 defenders: distance": g(lambda f: f in ("def1_dist", "def2_dist")),
        "other defenders: distance": g(lambda f: f.startswith("def") and f.endswith("_dist") and f[3] not in "12"),
        "defenders: closing speed": g(lambda f: f.startswith("def") and f.endswith("closing_speed")),
        "defenders: position rel. QB": g(lambda f: f.startswith("def") and f.split("_", 1)[1] in ("x", "y")),
        "defenders: speed/accel/angles": g(lambda f: f.startswith("def") and f.split("_", 1)[1] in
                                           ("s", "a", "dis", "o_sin", "o_cos", "dir_sin", "dir_cos")),
        "blockers: distance": g(lambda f: f.startswith("off") and f.endswith("_dist")),
        "blockers: position rel. QB": g(lambda f: f.startswith("off") and f.split("_", 1)[1] in ("x", "y")),
        "blockers: closing/speed/angles": g(lambda f: f.startswith("off") and f.split("_", 1)[1] in
                                            ("closing_speed", "s", "a", "dis", "o_sin", "o_cos", "dir_sin",
                                             "dir_cos")),
        "QB: depth & lateral position": g(lambda f: f in ("qb_x", "qb_y")),
        "QB: movement & facing": g(lambda f: f.startswith("qb_") and f not in ("qb_x", "qb_y")),
        "time since snap": g(lambda f: f in ("frameId", "frames_since_snap")),
    }
    rng = np.random.default_rng(C.RANDOM_STATE)
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(3):
            Xp = X.copy()
            Xp[cols] = Xp[cols].to_numpy()[rng.permutation(len(Xp))]
            drops.append(base - adjusted_r2(y, predict_score(m, Xp), p))
        rows.append({"group": name, "n_features": len(cols), "adj_r2_drop": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("adj_r2_drop", ascending=False), base


def pocket_clock(data_dir) -> pd.DataFrame:
    from pocket_safety.data import build_all_frames

    f = build_all_frames()[["gameId", "playId", "frames_since_snap", "safety_score", "def1_dist"]]
    f = f.assign(t=f["frames_since_snap"] / 10)
    q = f.groupby("t")["safety_score"].describe(percentiles=[0.25, 0.5, 0.75])
    q["alive_plays"] = f.groupby("t").size()
    # cumulative share of plays in which a defender has already been within 2 yd of the QB
    first = f[f["def1_dist"] < 2.0].groupby(["gameId", "playId"])["t"].min()
    n = f.groupby(["gameId", "playId"]).ngroups
    q["share_breached_2yd"] = [(first <= t).sum() / n for t in q.index]
    return q[q["alive_plays"] >= 200]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_success(t_dist, t_score):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for row, (t, xl) in enumerate(((t_dist, "closest any defender got to the QB (yd)"),
                                   (t_score, "mean pocket safety score during the play"))):
        x = [str(i) for i in t.index]
        for ax, (c, lab, fmt) in zip(axes[row], [("yards_per_dropback", "yards per dropback", "{:.1f}"),
                                                 ("completion_pct", "completion % (attempts)", "{:.0%}"),
                                                 ("sack_rate", "sack rate", "{:.0%}"),
                                                 ("int_rate", "interception rate", "{:.1%}")]):
            ax.bar(x, t[c], color=plt.get_cmap("RdYlGn")(np.linspace(0.1, 0.9, len(t))))
            for i, v in enumerate(t[c]):
                ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
            ax.set(title=lab, xlabel=xl if c == "yards_per_dropback" else "")
            ax.tick_params(axis="x", labelsize=8, rotation=30)
    fig.suptitle("Does a safe pocket make a successful play?  (all 8.5k dropbacks)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "success_by_pocket.png", dpi=120)
    plt.close(fig)


def fig_clock(q):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.fill_between(q.index, q["25%"], q["75%"], alpha=0.25, label="IQR")
    ax.plot(q.index, q["50%"], lw=2.5, label="median pocket safety score")
    ax.set(xlabel="seconds after snap", ylabel="pocket safety score", ylim=(0, 100))
    ax2 = ax.twinx()
    ax2.plot(q.index, 100 * q["share_breached_2yd"], color="firebrick", ls="--",
             label="% of plays with a defender already within 2 yd")
    ax2.set(ylabel="% plays breached", ylim=(0, 100))
    for t in (2.5, 3.0):
        if t in q.index:
            ax.axvline(t, color="0.5", ls=":", lw=1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9)
    ax.set_title("The pocket clock: safety decays with time after the snap")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "pocket_clock.png", dpi=120)
    plt.close(fig)


def fig_drivers(tables):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (title, t) in zip(axes.flat, tables.items()):
        x = [str(i) for i in t.index]
        ax.bar(x, 100 * t["pressure_rate"], color="firebrick", alpha=0.8, label="PFF pressure %")
        ax.set_ylim(0, 100)
        ax2 = ax.twinx()
        ax2.plot(x, t["mean_pocket_score"], "o-", color="seagreen", label="mean pocket score")
        ax2.set_ylim(0, 100)
        for i, n in enumerate(t["plays"]):
            ax.text(i, 2, f"n={int(n)}", ha="center", fontsize=7, color="white")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=25)
    axes.flat[0].set_ylabel("PFF pressure %")
    h1, l1 = axes.flat[0].get_legend_handles_labels()
    h2, l2 = axes.flat[0].get_figure().axes[-1].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.suptitle("What goes with a safe pocket?  (bars: pressure rate, dots: mean pocket score)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "pocket_drivers.png", dpi=120)
    plt.close(fig)


def fig_logit(t, auc):
    t = t.copy()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    col = np.where(t["odds_ratio_per_sd"] < 1, "seagreen", "firebrick")
    ax.barh(t["feature"], t["odds_ratio_per_sd"] - 1, left=1, color=col)
    ax.axvline(1, color="black", lw=1)
    ax.set(xscale="log", xlabel="odds ratio of PFF pressure per +1 SD (log scale)",
           title=f"Pressure drivers, all factors together (logistic, CV AUC={auc:.3f})")
    ax.set_xticks([0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5])
    ax.set_xticklabels(["0.7", "0.8", "0.9", "1", "1.25", "1.5", "2", "2.5"])
    ax.minorticks_off()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "pressure_odds_ratios.png", dpi=120)
    plt.close(fig)


def fig_importance(imp, base):
    fig, ax = plt.subplots(figsize=(8, 5))
    t = imp[::-1]
    ax.barh(t["group"], t["adj_r2_drop"], color="steelblue")
    ax.set(xlabel=f"drop in test adjusted R² when shuffled (baseline {base:.3f})",
           title="What the PCA + Ridge model relies on")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "model_group_importance.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir")
    args = ap.parse_args()
    data_dir = C.set_data_dir(args.data_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    df = build_play_table(data_dir)
    df.to_csv(OUT / "play_table.csv", index=False)
    print(f"{len(df)} dropbacks")

    tables = {
        "closest_defender": by(df, "min_def_dist", [0, 1, 2, 3, 4, 5, 99],
                               ["<1 yd", "1-2", "2-3", "3-4", "4-5", "5+ yd"]),
        "mean_pocket_score": by(df, "mean_score", [0, 20, 40, 60, 80, 101],
                                ["0-20", "20-40", "40-60", "60-80", "80-100"]),
        "pff_pressure": by(df, "pressure"),
        "time_to_throw": by(df, "ttt", [0, 2, 2.5, 3, 3.5, 4, 99], ["<2.0 s", "2.0-2.5", "2.5-3.0", "3.0-3.5",
                                                                   "3.5-4.0", "4.0+ s"]),
        "blocker_surplus": by(df, "surplus", [-9, 1, 2, 3, 99], ["<=0", "+1", "+2", "+3 or more"]),
        "n_rushers": by(df, "n_rush", [0, 4, 5, 6, 99], ["<=3", "4", "5", "6+"]),
        "unblocked_rushers_1s": by(df, "unblocked_1s", [0, 1, 2, 3, 99], ["0", "1", "2", "3+"]),
        "leaking_rushers_1s": by(df, "leak_1s", [0, 1, 2, 99], ["0", "1", "2+"]),
        "qb_cushion_1s": by(df, "qb_cushion_1s", [-99, 2.5, 3, 3.5, 4, 99],
                            ["<2.5 yd", "2.5-3", "3-3.5", "3.5-4", "4+ yd"]),
        "tackle_width_1s": by(df, "tackle_width_1s", [0, 6, 6.5, 7, 7.5, 8, 99],
                              ["<6 yd", "6-6.5", "6.5-7", "7-7.5", "7.5-8", "8+ yd"]),
        "ol_depth_1s": by(df, "ol_depth_1s", [-99, 0.5, 1.0, 1.5, 2.0, 99],
                          ["<0.5 yd", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0+ yd"]),
        "qb_snap_depth": by(df, "qb_snap_depth", [-99, 2.5, 4, 4.5, 5, 99],
                            ["<2.5 yd (under C)", "2.5-4", "4-4.5", "4.5-5", "5+ yd"]),
        "qb_drift": by(df, "qb_drift", [0, 3, 5, 7, 10, 999], ["<3 yd", "3-5", "5-7", "7-10", "10+ yd"]),
        "formation": by(df, "offenseFormation"),
        "dropback_type": by(df, "dropBackType"),
        "play_action": by(df, "play_action"),
        "coverage": by(df, "pff_passCoverageType"),
        "defenders_in_box": by(df, "defendersInBox", [0, 6, 7, 8, 99], ["<=5", "6", "7", "8+"]),
    }
    with open(OUT / "tables.md", "w") as fh:
        for name, t in tables.items():
            fh.write(f"### {name}\n\n{_md(t)}\n\n")
        ttt_bin = pd.cut(df["ttt"], [0, 2, 2.5, 3, 3.5, 4, 99], right=False,
                         labels=["<2.0 s", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0", "4.0+ s"])
        for metric, f in (("yards_per_dropback", "{:.2f}"), ("completion_pct", "{:.1%}"), ("plays", "{:.0f}")):
            src = {"yards_per_dropback": "yards", "completion_pct": "complete", "plays": "yards"}[metric]
            agg = "size" if metric == "plays" else "mean"
            x = df.groupby([ttt_bin.rename("ttt"), df["pressure"].rename("pressure")], observed=True)[src].agg(agg)
            x = x.unstack().rename(columns={False: "clean (no PFF pressure)", True: "pressured"})
            x.to_csv(OUT / f"ttt_x_pressure_{metric}.csv")
            fh.write(f"### time_to_throw x pressure: {metric}\n\n| ttt | " + " | ".join(x.columns) + " |\n|"
                     + "---|" * (len(x.columns) + 1) + "\n")
            for i, r in x.iterrows():
                fh.write(f"| {i} | " + " | ".join(f.format(v) for v in r) + " |\n")
            fh.write("\n")

    logit, auc = pressure_logit(df)
    logit.round(4).to_csv(OUT / "pressure_odds_ratios.csv", index=False)
    imp, base = model_group_importance()
    imp.round(4).to_csv(OUT / "model_group_importance.csv", index=False)
    clock = pocket_clock(data_dir)
    clock.round(3).to_csv(OUT / "pocket_clock.csv")

    fig_success(tables["closest_defender"], tables["mean_pocket_score"])
    fig_clock(clock)
    fig_drivers({k: tables[k] for k in ("time_to_throw", "n_rushers", "blocker_surplus", "unblocked_rushers_1s",
                                        "ol_depth_1s", "qb_drift", "qb_snap_depth", "dropback_type")})
    fig_logit(logit, auc)
    fig_importance(imp, base)

    summary = {"n_plays": len(df), "pressure_logit_cv_auc": auc, "model_test_adj_r2": base}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
