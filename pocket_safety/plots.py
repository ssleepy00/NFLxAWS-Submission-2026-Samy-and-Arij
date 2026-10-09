"""matplotlib figures for the Pocket Safety model."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .data import load_tracking


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(C.OUT_DIR / name, dpi=130)
    plt.close(fig)


def pca_variance(model):
    pca = model.named_steps["pca"]
    cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, marker=".")
    ax.set(xlabel="# principal components", ylabel="cumulative explained variance",
           title=f"PCA: {pca.n_components_} components kept ({cum[-1]:.1%} variance)")
    ax.grid(alpha=0.3)
    _save(fig, "pca_variance.png")


def pred_vs_actual(test):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    hb = axes[0].hexbin(test["safety_score"], test["pred"], gridsize=50, bins="log", cmap="viridis")
    axes[0].plot([0, 100], [0, 100], "r--", lw=1)
    axes[0].set(xlabel="target safety score", ylabel="predicted safety score", title="Test set: predicted vs target")
    fig.colorbar(hb, ax=axes[0], label="log10(frames)")
    axes[1].hist(test["pred"] - test["safety_score"], bins=80, color="steelblue")
    axes[1].set(xlabel="residual (pred - target)", ylabel="frames", title="Residuals")
    _save(fig, "pred_vs_actual.png")


def feature_importance(model, features, top=20):
    """Ridge coefficients mapped back through PCA + scaler to original features."""
    pca, reg = model.named_steps["pca"], model.named_steps["reg"]
    coef = (pca.components_.T @ reg.coef_)            # per standardised feature
    s = pd.Series(coef, index=features)
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(s.index, s.values, color=np.where(s.values > 0, "seagreen", "firebrick"))
    ax.set(xlabel="effect on score per +1 SD of feature", title="Top effective linear weights")
    _save(fig, "feature_importance.png")


def pff_validation(per_play, auc):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    groups = [per_play.loc[~per_play["pressure"], "min_pred"], per_play.loc[per_play["pressure"], "min_pred"]]
    ax.boxplot(groups, tick_labels=["no pressure", "PFF hit/hurry/sack"])
    ax.set(ylabel="min predicted safety score in play", title=f"External check vs PFF charting (AUC={auc:.3f})")
    _save(fig, "pff_validation.png")


def _pick_plays(test):
    plays = pd.read_csv(C.DATA_DIR / "plays.csv", usecols=["gameId", "playId", "passResult", "playDescription"])
    agg = test.groupby(["gameId", "playId"]).agg(n=("frameId", "size"), mn=("pred", "min"), mean=("pred", "mean"))
    agg = agg.reset_index().merge(plays, on=["gameId", "playId"])
    agg = agg[agg["n"].between(25, 45)]
    sack = agg[agg["passResult"] == "S"].sort_values("mn").iloc[0]
    clean = agg[agg["passResult"] == "C"].sort_values("mean", ascending=False).iloc[0]
    return sack, clean


def play_timelines(test):
    sack, clean = _pick_plays(test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, row, label in zip(axes, (sack, clean), ("Sack", "Clean completion")):
        p = test[(test["gameId"] == row["gameId"]) & (test["playId"] == row["playId"])].sort_values("frameId")
        t = (p["frameId"] - p["frameId"].min()) / 10
        ax.plot(t, p["safety_score"], "k--", label="target")
        ax.plot(t, p["pred"], color="tab:blue", lw=2, label="predicted")
        ax.set(title=f"{label}: game {row['gameId']} play {row['playId']}", xlabel="seconds since snap", ylim=(-3, 103))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("pocket safety score")
    axes[0].legend()
    _save(fig, "play_timelines.png")
    return sack


def field_snapshot(test, row):
    """Two frames of one play: players coloured by team, jersey numbers, QB ringed."""
    gid, pid = int(row["gameId"]), int(row["playId"])
    trk = load_tracking(gid)
    trk = trk[trk["playId"] == pid]
    p = test[(test["gameId"] == gid) & (test["playId"] == pid)].sort_values("frameId")
    qb_id = int(p["nflId"].iloc[0])
    f_early = int(p["frameId"].iloc[min(10, len(p) - 1)])
    f_low = int(p.loc[p["pred"].idxmin(), "frameId"])
    teams = [t for t in trk["team"].unique() if t != "football"]
    colors = dict(zip(teams, ["tab:blue", "tab:red"]))
    colors["football"] = "saddlebrown"

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
    for ax, fid in zip(axes, (f_early, f_low)):
        fr = trk[trk["frameId"] == fid]
        qb = fr[fr["nflId"] == qb_id].iloc[0]
        for _, r in fr.iterrows():
            ax.scatter(r["x"], r["y"], s=60 if r["is_ball"] else 260, c=colors[r["team"]],
                       edgecolors="gold" if r["nflId"] == qb_id else "k", linewidths=3 if r["nflId"] == qb_id else 0.5,
                       zorder=3)
            if not r["is_ball"]:
                ax.text(r["x"], r["y"], int(r["jerseyNumber"]), color="white", fontsize=7,
                        ha="center", va="center", zorder=4)
                ax.arrow(r["x"], r["y"], r["vx"] * 0.4, r["vy"] * 0.4, head_width=0.3, color="gray", zorder=2)
        ax.set(xlim=(qb["x"] - 6, qb["x"] + 14), ylim=(qb["y"] - 12, qb["y"] + 12), aspect="equal",
               xlabel="x (normalised, offense -> right)", ylabel="y")
        sc = p.loc[p["frameId"] == fid].iloc[0]
        ax.set_title(f"frame {fid}: predicted {sc['pred']:.0f} / target {sc['safety_score']:.0f}")
        ax.grid(alpha=0.2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[t], label=t, markersize=10) for t in teams]
    axes[0].legend(handles=handles, loc="upper left")
    fig.suptitle(f"Game {gid}, play {pid} (QB ringed in gold)")
    _save(fig, "field_snapshot.png")


def ridge_vs_lasso(cv: pd.DataFrame, comparison: dict):
    """CV adjusted R² across alpha for each PCA setting, Ridge vs Lasso."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, name in zip(axes, ("ridge", "lasso")):
        d = cv[cv["model"] == name]
        for ncomp, g in d.groupby("param_pca__n_components"):
            g = g.sort_values("param_reg__alpha")
            ax.errorbar(g["param_reg__alpha"].astype(float), g["mean_test_score"], yerr=g["std_test_score"],
                        marker="o", ms=3, capsize=2, label=f"PCA {float(ncomp):.0%} var")
        c = comparison[name]
        ax.set(xscale="log", xlabel="alpha", title=f"{name.title()}: best CV adj R²={c['cv_adj_r2']:.4f} "
               f"(test {c['test']['adj_r2']:.4f})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("3-fold GroupKFold adjusted R² (by game)")
    axes[0].set_ylim(bottom=max(0.0, cv["mean_test_score"].min() - 0.02))
    axes[1].legend(loc="lower left")
    _save(fig, "ridge_vs_lasso.png")


def lasso_sparsity(sp: pd.DataFrame, ridge_cv: float):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(sp["n_nonzero"], sp["cv_adj_r2"], yerr=sp["cv_adj_r2_std"], marker="o", capsize=3, label="Lasso (CV)")
    for _, r in sp.drop_duplicates("n_nonzero").iterrows():
        ax.annotate(f"α={r['alpha']:.3g}", (r["n_nonzero"], r["cv_adj_r2"]), fontsize=7,
                    xytext=(4, -10), textcoords="offset points")
    ax.axhline(ridge_cv, color="tab:orange", ls="--", label=f"Ridge best CV ({ridge_cv:.4f})")
    ax.set(xlabel="non-zero PCA components kept by Lasso", ylabel="CV adjusted R²",
           title="Lasso sparsity vs fit")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    _save(fig, "lasso_sparsity.png")


def make_all(out):
    model, features, test = out["model"], out["features"], out["test"]
    ridge_vs_lasso(out["cv"], out["results"]["comparison"])
    lasso_sparsity(out["sparsity"], out["results"]["comparison"]["ridge"]["cv_adj_r2"])
    pca_variance(model)
    pred_vs_actual(test)
    feature_importance(model, features)
    pff_validation(out["per_play"], out["auc"])
    sack = play_timelines(test)
    field_snapshot(test, sack)
