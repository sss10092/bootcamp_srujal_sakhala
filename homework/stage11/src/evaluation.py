
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@dataclass
class FitResult:
    model: object
    y_true: np.ndarray
    y_pred: np.ndarray
    rmse: float
    mae: float
    r2: float
    config: dict

def build_pipeline(
    numeric: List[str],
    categorical: List[str],
    impute_strategy: str = "mean",
    poly_degree: int = 1,
    model_type: str = "linear",
    alpha: float = 1.0,
) -> Pipeline:
    """
    Build a sklearn pipeline with imputation (mean/median), optional polynomial,
    and linear or ridge regression. Standardize numeric features after imputation.
    """
    if impute_strategy not in {"mean", "median"}:
        raise ValueError("impute_strategy must be 'mean' or 'median'")
    if model_type not in {"linear", "ridge"}:
        raise ValueError("model_type must be 'linear' or 'ridge'")

    num_steps = []
    num_steps.append(("impute", SimpleImputer(strategy=impute_strategy)))
    if poly_degree > 1:
        num_steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
    num_steps.append(("scale", StandardScaler(with_mean=True, with_std=True)))

    num_transformer = Pipeline(num_steps)

    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    pre = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric),
            ("cat", cat_transformer, categorical),
        ]
    )

    if model_type == "linear":
        model = LinearRegression()
    else:
        model = Ridge(alpha=alpha)

    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe

def fit_and_eval(
    df: pd.DataFrame,
    features: List[str],
    target: str = "y",
    impute_strategy: str = "mean",
    poly_degree: int = 1,
    model_type: str = "linear",
    alpha: float = 1.0,
) -> FitResult:
    """Fit pipeline and compute metrics in-sample (for simplicity in this homework)."""
    numeric = [c for c in features if df[c].dtype.kind in "fcbi"]
    categorical = [c for c in features if df[c].dtype.kind not in "fcbi"]

    pipe = build_pipeline(numeric, categorical, impute_strategy, poly_degree, model_type, alpha)

    X = df[features]
    y = df[target].values
    pipe.fit(X, y)
    y_pred = pipe.predict(X)

    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))
    r2 = float(r2_score(y, y_pred))

    return FitResult(pipe, y, y_pred, rmse, mae, r2, {
        "impute": impute_strategy,
        "poly_degree": poly_degree,
        "model_type": model_type,
        "alpha": alpha,
    })

def bootstrap_metric(
    df: pd.DataFrame,
    features: List[str],
    target: str = "y",
    impute_strategy: str = "mean",
    poly_degree: int = 1,
    model_type: str = "linear",
    alpha: float = 1.0,
    n_boot: int = 1000,
    random_state: int = 0,
) -> dict:
    """Bootstrap RMSE and MAE by resampling rows with replacement."""
    rng = np.random.default_rng(random_state)
    n = len(df)
    rmses, maes = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        dfi = df.iloc[idx].reset_index(drop=True)
        res = fit_and_eval(
            dfi, features, target, impute_strategy, poly_degree, model_type, alpha
        )
        rmses.append(res.rmse)
        maes.append(res.mae)
    def ci(arr, level=0.95):
        lo = np.percentile(arr, (1 - level) / 2 * 100)
        hi = np.percentile(arr, (1 + level) / 2 * 100)
        return float(lo), float(hi)
    out = {
        "rmse": {"mean": float(np.mean(rmses)), "ci95": ci(rmses)},
        "mae": {"mean": float(np.mean(maes)), "ci95": ci(maes)},
        "all_rmse": np.array(rmses),
        "all_mae": np.array(maes),
    }
    return out

def subgroup_metrics(df: pd.DataFrame, segment_col: str, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Compute RMSE/MAE/R2 by subgroup given in df[segment_col]."""
    out = []
    residuals = y_true - y_pred
    data = pd.DataFrame({segment_col: df[segment_col].values, "y_true": y_true, "y_pred": y_pred, "res": residuals})
    for seg, g in data.groupby(segment_col):
        rmse = float(np.sqrt(((g["res"]) ** 2).mean()))
        mae = float(np.abs(g["res"]).mean())
        # R2 per subgroup: 1 - SS_res/SS_tot within group
        ss_res = float((g["res"] ** 2).sum())
        ss_tot = float(((g["y_true"] - g["y_true"].mean()) ** 2).sum())
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        out.append({"segment": seg, "RMSE": rmse, "MAE": mae, "R2": r2, "n": len(g)})
    return pd.DataFrame(out).sort_values("segment")
