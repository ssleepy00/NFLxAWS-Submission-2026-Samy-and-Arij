"""PCA + {Ridge, Lasso}: leak-free split by gameId, CV tuning on adjusted R², evaluation."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .data import build_all_frames, feature_columns

# Trimmed grid: 0.80/0.90 PCA never won in a full 5-fold sweep (CV adj R² 0.706/0.721
# vs 0.765), and Ridge was flat to 4 d.p. for alpha 0.01..100.
PCA_GRID = [0.95, 0.99]
CV_FOLDS = 3
MODELS = {
    "ridge": (lambda: Ridge(), [0.01, 1.0, 100.0, 10000.0]),
    "lasso": (lambda: Lasso(max_iter=5000, tol=1e-3, random_state=C.RANDOM_STATE),
              [0.001, 0.01, 0.1, 0.3, 1.0]),
}
SPARSITY_ALPHAS = [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def make_pipeline(regressor) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(svd_solver="full", random_state=C.RANDOM_STATE)),
        ("reg", regressor),
    ])


# --------------------------------------------------------------------------- #
# Adjusted R²
# --------------------------------------------------------------------------- #
def n_predictors(model: Pipeline) -> int:
    """p for adjusted R²: predictors actually used by the regressor.

    Ridge keeps every PCA component -> p = n_components.
    Lasso can zero components out -> p = number of non-zero coefficients.
    """
    return int(np.count_nonzero(model.named_steps["reg"].coef_))


def adjusted_r2(y, pred, p: int) -> float:
    n = len(y)
    return float(1.0 - (1.0 - r2_score(y, pred)) * (n - 1) / (n - p - 1))


def predict_score(model: Pipeline, X) -> np.ndarray:
    return np.clip(model.predict(X), 0.0, 100.0)


def adj_r2_scorer(estimator: Pipeline, X, y) -> float:
    """GridSearchCV scorer (higher is better)."""
    return adjusted_r2(y, predict_score(estimator, X), n_predictors(estimator))


def _metrics(y, p, k: int) -> dict:
    return {"adj_r2": adjusted_r2(y, p, k),
            "rmse": float(root_mean_squared_error(y, p)),
            "mae": float(mean_absolute_error(y, p)),
            "p": int(k)}


# --------------------------------------------------------------------------- #
def split_by_game(frames: pd.DataFrame):
    gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)
    tr, te = next(gss.split(frames, groups=frames["gameId"]))
    train, test = frames.iloc[tr].reset_index(drop=True), frames.iloc[te].reset_index(drop=True)
    assert not set(train["gameId"]) & set(test["gameId"]), "game leakage between splits"
    return train, test


def pff_pressure_labels() -> pd.DataFrame:
    """Per-play flag: did PFF credit any defender with a hit, hurry or sack?"""
    s = pd.read_csv(C.DATA_DIR / "pffScoutingData.csv",
                    usecols=["gameId", "playId", "pff_hit", "pff_hurry", "pff_sack"])
    s["pressure"] = s[["pff_hit", "pff_hurry", "pff_sack"]].fillna(0).sum(axis=1) > 0
    return s.groupby(["gameId", "playId"])["pressure"].any().reset_index()


def _pff_auc(test: pd.DataFrame, pred: np.ndarray):
    per_play = (test.assign(pred=pred).groupby(["gameId", "playId"])["pred"].min()
                .rename("min_pred").reset_index()
                .merge(pff_pressure_labels(), on=["gameId", "playId"], how="inner"))
    return float(roc_auc_score(per_play["pressure"], -per_play["min_pred"])), per_play


def fit_model(name: str, X_tr, y_tr, groups) -> GridSearchCV:
    factory, alphas = MODELS[name]
    grid = GridSearchCV(
        make_pipeline(factory()),
        param_grid={"pca__n_components": PCA_GRID, "reg__alpha": alphas},
        cv=GroupKFold(n_splits=CV_FOLDS),
        scoring=adj_r2_scorer,
        n_jobs=-1,
    )
    return grid.fit(X_tr, y_tr, groups=groups)


def lasso_sparsity_path(n_comp: float, X_tr, y_tr, X_te, y_te, groups) -> pd.DataFrame:
    """Lasso trade-off at the chosen PCA level: alpha -> surviving components vs adj R².

    Scaler + PCA are fit once per fold and Lasso is warm-started from large to
    small alpha, so the whole path costs about one pipeline fit per fold.
    """
    def path(Xa, ya, Xb, yb):
        prep = Pipeline(make_pipeline(None).steps[:2]).set_params(pca__n_components=n_comp).fit(Xa)
        Za, Zb = prep.transform(Xa), prep.transform(Xb)
        reg = Lasso(max_iter=5000, tol=1e-3, warm_start=True, random_state=C.RANDOM_STATE)
        out = {}
        for alpha in sorted(SPARSITY_ALPHAS, reverse=True):
            reg.set_params(alpha=alpha).fit(Za, ya)
            p = int(np.count_nonzero(reg.coef_))
            out[alpha] = (p, adjusted_r2(yb, np.clip(reg.predict(Zb), 0, 100), p), Za.shape[1])
        return out

    folds = [path(X_tr.iloc[a], y_tr.iloc[a], X_tr.iloc[b], y_tr.iloc[b])
             for a, b in GroupKFold(n_splits=CV_FOLDS).split(X_tr, y_tr, groups)]
    full = path(X_tr, y_tr, X_te, y_te)
    rows = []
    for alpha in sorted(SPARSITY_ALPHAS):
        cv_scores = [f[alpha][1] for f in folds]
        rows.append({"alpha": alpha, "n_nonzero": full[alpha][0], "n_components": full[alpha][2],
                     "cv_adj_r2": float(np.mean(cv_scores)), "cv_adj_r2_std": float(np.std(cv_scores)),
                     "test_adj_r2": full[alpha][1]})
    return pd.DataFrame(rows)


def train_and_evaluate() -> dict:
    frames = build_all_frames()
    FEATURES = feature_columns(frames)          # ids excluded only here
    train, test = split_by_game(frames)
    X_tr, y_tr = train[FEATURES], train["safety_score"]
    X_te, y_te = test[FEATURES], test["safety_score"]

    comparison, fitted, cv_tables = {}, {}, []
    for name in MODELS:
        grid = fit_model(name, X_tr, y_tr, train["gameId"])
        model = grid.best_estimator_
        pred = predict_score(model, X_te)
        auc, _ = _pff_auc(test, pred)
        pca: PCA = model.named_steps["pca"]
        comparison[name] = {
            "best_params": {k: float(v) for k, v in grid.best_params_.items()},
            "n_pca_components": int(pca.n_components_),
            "n_nonzero_coefs": n_predictors(model),
            "cv_adj_r2": float(grid.best_score_),
            "test": _metrics(y_te, pred, n_predictors(model)),
            "pff_pressure_auc": auc,
        }
        fitted[name] = (model, pred)
        cv = pd.DataFrame(grid.cv_results_)
        cv.insert(0, "model", name)
        cv_tables.append(cv)

    cv_all = pd.concat(cv_tables, ignore_index=True)
    sparsity = lasso_sparsity_path(comparison["lasso"]["best_params"]["pca__n_components"],
                                   X_tr, y_tr, X_te, y_te, train["gameId"])
    comparison["lasso"]["sparsity_path"] = sparsity.to_dict(orient="records")

    # Select by cross-validated adjusted R² (never by test score).
    best = max(comparison, key=lambda k: comparison[k]["cv_adj_r2"])
    model, pred = fitted[best]
    test = test.copy()
    test["pred"] = pred
    auc, per_play = _pff_auc(test, pred)

    base_mean = np.full(len(y_te), y_tr.mean())
    d_now = test["def1_dist"]
    base_now = 100 * ((d_now - C.D_COLLAPSED) / (C.D_CLEAN - C.D_COLLAPSED)).clip(0, 1)

    results = {
        "n_frames": {"train": len(train), "test": len(test)},
        "n_games": {"train": int(train["gameId"].nunique()), "test": int(test["gameId"].nunique())},
        "n_features": len(FEATURES),
        "selected_model": best,
        "comparison": comparison,
        "baseline_train_mean": _metrics(y_te, base_mean, 0),
        "baseline_current_distance": _metrics(y_te, base_now, 1),
        "pff_pressure_rate_test": float(per_play["pressure"].mean()),
    }

    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "name": best, "features": FEATURES}, C.OUT_DIR / "model.joblib")
    test[C.FRAME_KEYS + ["nflId", "safety_score", "pred"]].to_parquet(
        C.OUT_DIR / "test_predictions.parquet", index=False)
    cv_all.drop(columns=[c for c in cv_all.columns if c == "params"]).to_csv(C.OUT_DIR / "cv_results.csv", index=False)
    (C.OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    return {"results": results, "model": model, "name": best, "features": FEATURES,
            "fitted": fitted, "cv": cv_all, "sparsity": sparsity, "train": train, "test": test,
            "per_play": per_play, "auc": auc}
