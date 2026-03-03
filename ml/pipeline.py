import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not installed — using sklearn fallback")
    from sklearn.ensemble import GradientBoostingRegressor

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

MODEL_PATH = "models/price_predictor.joblib"
MIN_TRAINING_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", 140))


def _build_features_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature frame with a mix of numeric and categorical columns.
    Categorical columns are kept as strings so they can be OneHotEncoded safely.
    """
    now_year = datetime.utcnow().year
    out = pd.DataFrame(index=df.index)

    def _norm_str(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
            .str.lower()
            .str.strip()
            .replace({"nan": "", "none": "", "null": ""})
        )

    # --- numeric ---
    out["year"] = pd.to_numeric(df["year"], errors="coerce")
    out["vehicle_age"] = (now_year - out["year"]).clip(0, 50)  # guardrails

    out["mileage"] = pd.to_numeric(df["mileage"], errors="coerce").clip(0, 600_000)
    out["miles_per_year"] = (out["mileage"] / (out["vehicle_age"].clip(1))).clip(0, 60_000)

    # Nonlinear transforms (helps luxury depreciation a LOT)
    out["log_mileage"] = np.log1p(out["mileage"])
    out["log_age"] = np.log1p(out["vehicle_age"])
    out["age_sq"] = (out["vehicle_age"] ** 2).clip(0, 2500)

    out["accident_count"] = pd.to_numeric(df.get("accident_count", 0), errors="coerce").fillna(0)
    out["owner_count"] = pd.to_numeric(df.get("owner_count", 1), errors="coerce").fillna(1)

    # --- categoricals (strings) ---
    out["make"] = df["make"].astype(str).str.lower().str.strip()
    out["model"] = df["model"].astype(str).str.lower().str.strip()
    out["state"] = (
        df.get("location_state", pd.Series([""] * len(df), index=df.index))
        .astype(str).str.upper().str.strip()
    )

    # --- simple flags ---
    luxury_makes = {
        "bmw", "mercedes-benz", "mercedes", "lexus", "audi", "acura", "infiniti", "cadillac",
        "lincoln", "porsche", "jaguar", "land rover", "volvo", "genesis",
        "tesla", "rivian", "lucid",
        "bentley", "rolls-royce", "lamborghini", "ferrari", "mclaren", "maserati",
    }
    truck_keywords = {"f-150", "f-250", "f-350", "silverado", "sierra", "ram", "tundra", "tacoma", "ranger", "colorado"}
    sports_keywords = {"corvette", "mustang", "camaro", "challenger", "charger", "86", "brz", "miata", "supra", "911", "cayman"}

    make_s = out["make"]
    model_s = out["model"]

    out["is_luxury"] = make_s.isin(luxury_makes).astype(float)
    out["is_truck"] = model_s.apply(lambda m: float(any(t in m for t in truck_keywords)))
    out["is_sports"] = model_s.apply(lambda m: float(any(s in m for s in sports_keywords)))

    # Luxury interaction features (forces model to learn steeper depreciation curve)
    out["lux_age"] = out["is_luxury"] * out["vehicle_age"]
    out["lux_mpy"] = out["is_luxury"] * out["miles_per_year"]

    # --- extra categoricals (safe fallback to "")
    out["trim"]         = _norm_str(df.get("trim", pd.Series([""] * len(df), index=df.index)))
    out["body_type"]    = _norm_str(df.get("body_type", pd.Series([""] * len(df), index=df.index)))
    out["drivetrain"]   = _norm_str(df.get("drivetrain", pd.Series([""] * len(df), index=df.index)))
    out["fuel_type"]    = _norm_str(df.get("fuel_type", pd.Series([""] * len(df), index=df.index)))
    out["transmission"] = _norm_str(df.get("transmission", pd.Series([""] * len(df), index=df.index)))

    out["zip3"] = (
        df.get("location_zip", pd.Series([""] * len(df), index=df.index))
        .astype(str)
        .str.strip()
        .str[:3]
        .replace({"": "000"})
    )

    return out


def train(df: pd.DataFrame) -> dict | None:
    logger.info("Starting model training run…")

    # -----------------------------
    # 1) Clean + filter
    # -----------------------------
    df = df.dropna(subset=["price", "mileage", "year", "make", "model"]).copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df = df.dropna(subset=["price", "mileage", "year"])
    df = df[df["price"].between(500, 100_000)]
    df = df[df["mileage"].between(0, 400_000)]

    if len(df) < MIN_TRAINING_SAMPLES:
        logger.warning(f"Not enough data to train ({len(df)} rows, need {MIN_TRAINING_SAMPLES})")
        return None

    # -----------------------------
    # 2) Build features + split
    # -----------------------------
    X_raw = _build_features_raw(df)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

    # log-space targets (this is what we train on)
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # -----------------------------
    # 3) Preprocess
    # -----------------------------
    numeric_features = [
        "year", "vehicle_age", "age_sq", "log_age",
        "mileage", "log_mileage", "miles_per_year",
        "accident_count", "owner_count",
        "is_luxury", "is_truck", "is_sports",
        "lux_age", "lux_mpy",
    ]
    categorical_features = [
        "make", "model", "state",
        "trim", "body_type", "drivetrain", "fuel_type", "transmission", "zip3",
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    # -----------------------------
    # 4) Candidates
    # -----------------------------
    candidates: dict[str, object] = {
        "linear": LinearRegression(),
        "rf": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }

    if HAS_XGB:
        candidates["xgb"] = XGBRegressor(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            objective="reg:squarederror",
        )

    # luxury weighting (LinearRegression + RF accept sample_weight; XGB accepts too)
    lux_w = 2.0
    sample_weight = np.where(X_train["is_luxury"].to_numpy() >= 0.5, lux_w, 1.0)

    def _eval(y_true_np: np.ndarray, y_pred_np: np.ndarray) -> tuple[float, float]:
        mae = float(mean_absolute_error(y_true_np, y_pred_np))
        rmse = float(np.sqrt(mean_squared_error(y_true_np, y_pred_np)))
        return mae, rmse

    fitted: dict[str, Pipeline] = {}
    metrics: dict[str, dict] = {}

    # -----------------------------
    # 5) Train + evaluate each candidate
    # -----------------------------
    for name, model in candidates.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocess),
            ("model", model),
        ])

        fit_kwargs: dict = {"model__sample_weight": sample_weight}

        # IMPORTANT: only pass eval_set/verbose to XGB
        if name == "xgb":
            fit_kwargs.update({
                "model__eval_set": [(pipe.named_steps["preprocess"].fit_transform(X_test), y_test_log)]
                # ^ NOTE: eval_set must be in the model's feature space. But since we're in a Pipeline,
                # we can't easily inject transformed X_test without fitting preprocess first.
                # So, we DO NOT pass eval_set via the pipeline. We'll compute training metrics manually below.
            })

        # Because passing eval_set through a sklearn Pipeline is awkward (needs transformed X),
        # keep it simple: just fit the pipeline normally.
        # (You still get proper test MAE/RMSE and train RMSE computed below.)
        pipe.fit(X_train, y_train_log, model__sample_weight=sample_weight)

        # ---- training RMSE (log-space) for ALL models (works consistently) ----
        pred_train_log = pipe.predict(X_train)
        train_rmse_log = float(np.sqrt(mean_squared_error(y_train_log.to_numpy(), pred_train_log)))

        # ---- test metrics in $ space ----
        pred_test = np.expm1(pipe.predict(X_test))
        y_true = y_test.to_numpy()
        mae, rmse = _eval(y_true, pred_test)

        # slice MAE: luxury vs non-lux
        test_is_lux = (X_test["is_luxury"].to_numpy() >= 0.5)
        lux_n = int(test_is_lux.sum())
        nonlux_n = int((~test_is_lux).sum())

        lux_mae = float(np.mean(np.abs(pred_test[test_is_lux] - y_true[test_is_lux]))) if lux_n > 0 else float("nan")
        nonlux_mae = float(np.mean(np.abs(pred_test[~test_is_lux] - y_true[~test_is_lux]))) if nonlux_n > 0 else float("nan")

        fitted[name] = pipe
        metrics[name] = {
            "mae": mae,
            "rmse": rmse,
            "train_rmse_log": train_rmse_log,
            "luxury_n": lux_n,
            "nonlux_n": nonlux_n,
            "luxury_mae": lux_mae,
            "nonlux_mae": nonlux_mae,
        }

        logger.info(
            f"Candidate {name} | MAE=${mae:,.0f} RMSE=${rmse:,.0f} | "
            f"trainRMSE(log)={train_rmse_log:.4f} | "
            f"luxury=${lux_mae:,.0f} (n={lux_n}) nonlux=${nonlux_mae:,.0f} (n={nonlux_n})"
        )

    # pick best by MAE
    best_name = min(metrics.keys(), key=lambda k: metrics[k]["mae"])
    pipe = fitted[best_name]
    best = metrics[best_name]

    logger.info(f"Selected model: {best_name} | MAE=${best['mae']:,.0f} RMSE=${best['rmse']:,.0f}")

    # -----------------------------
    # 6) Permutation importance on RAW features (24 cols)
    # -----------------------------
    try:
        pi = permutation_importance(
            pipe,
            X_test,                  # raw features
            y_test_log,              # log target (matches training)
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
            scoring="neg_mean_absolute_error",
        )

        imp = (
            pd.Series(pi.importances_mean, index=X_test.columns)
            .sort_values(ascending=False)
        )
        logger.info("Top permutation importances (raw features):\n" + imp.head(20).to_string())

    except Exception as e:
        logger.warning(f"Permutation importance failed: {e}")

    # -----------------------------
    # 7) Save
    # -----------------------------
    version = datetime.utcnow().strftime("%Y%m%d_%H%M")
    payload = {
        "model": pipe,
        "version": version,
        "selected_model": best_name,
        "metrics": metrics,
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(payload, MODEL_PATH)

    os.makedirs("models", exist_ok=True)
    joblib.dump(payload, MODEL_PATH)

    logger.info("\n" + "="*72)
    logger.info(f"MODEL TRAINING SUMMARY  |  Version: {version}")
    logger.info("="*72)

    for name, m in metrics.items():
        logger.info(
            f"{name.upper():<8} | "
            f"MAE=${m['mae']:>8,.0f}  "
            f"RMSE=${m['rmse']:>8,.0f}  "
            f"TrainRMSE(log)={m['train_rmse_log']:.4f}"
        )
        logger.info(
            f"          Luxury MAE=${m['luxury_mae']:>8,.0f} (n={m['luxury_n']})  "
            f"Non-Lux MAE=${m['nonlux_mae']:>8,.0f} (n={m['nonlux_n']})"
        )
        logger.info("-"*72)

    logger.info(f"SELECTED MODEL → {best_name.upper()}")
    logger.info(f"Rows after cleaning: {len(df)}")
    logger.info(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    logger.info(f"Train/Test split: {len(X_train)}/{len(X_test)}")
    logger.info("="*72 + "\n")

    return {
        "version": version,
        "selected_model": best_name,
        "metrics": metrics,
        "n": int(len(df)),
    }


def load() -> dict | None:
    if not os.path.exists(MODEL_PATH):
        return None
    payload = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model v{payload.get('version', '?')}")
    return payload


# ----------------------------
# Deal score: 0–100 (100 best)
# ----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _deal_score_from_prices(actual: float, predicted: float) -> float:
    """
    Map pricing gap to a stable 0–100 score.

    Let diff_pct = (predicted - actual) / predicted * 100
      - diff_pct > 0  => under market (good)
      - diff_pct < 0  => over market (bad)

    We map:
      diff_pct = +20%  -> 100
      diff_pct =   0%  -> 50
      diff_pct = -20%  -> 0
    and clamp beyond ±20%.
    """
    if predicted <= 0:
        return 50.0

    diff_pct = (predicted - actual) / predicted * 100.0  # + = good
    score = 50.0 + (diff_pct * 1.25)
    return _clamp(score, 0.0, 100.0)


def _deal_label(score: float) -> str:
    if score >= 90:
        return "Hidden Gem"
    if score >= 75:
        return "Great Deal"
    if score >= 60:
        return "Good Deal"
    if score >= 45:
        return "Fair Price"
    return "Overpriced"


def score_listings(df: pd.DataFrame) -> list[dict]:
    """
    Produces a 0–100 deal_score where:
      - 100 = best deal (most under market)
      - 50  = fair price
      - 0   = worst deal (most over market)
    """
    payload = load()
    if payload is None:
        logger.warning("No model found — cannot score")
        return []

    model = payload["model"]

    df = df.copy()
    df = df.dropna(subset=["price", "mileage", "year", "make", "model"])
    if df.empty:
        return []

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    if df.empty:
        return []

    logger.info(f"Scoring {len(df)} listings…")
    X = _build_features_raw(df)
    preds = np.expm1(model.predict(X))

    # Keep alignment with df rows
    df = df.reset_index(drop=True)

    results: list[dict] = []
    for i, row in df.iterrows():
        predicted = float(preds[i])
        actual = float(row["price"])
        if not np.isfinite(predicted) or predicted <= 0:
            continue
        if not np.isfinite(actual) or actual <= 0:
            continue

        score = _deal_score_from_prices(actual=actual, predicted=predicted)
        score = round(score, 1)

        results.append(
            {
                "listing_id": row.get("listing_id"),
                "predicted_price": round(predicted, 2),
                "deal_score": score,
                "deal_label": _deal_label(score),
                "model_version": payload.get("version", "unknown"),
            }
        )

    logger.info(f"Scored and saved {len(results)} predictions")
    return results


def update_popularity_snapshot(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []

    cohorts = (
        df.groupby(["year", "make", "model"])
        .agg(
            active_listings=("listing_id", "count"),
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            avg_mileage=("mileage", "mean"),
            avg_days_listed=("days_listed", "mean"),
        )
        .reset_index()
        .sort_values("active_listings", ascending=False)
    )

    cohorts["popularity_rank"] = range(1, len(cohorts) + 1)
    cohorts["sold_last_7d"] = 0

    result = cohorts.to_dict("records")
    logger.info(f"Popularity snapshot saved: {len(result)} cohorts")
    return result