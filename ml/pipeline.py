import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

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

    # If dealer_rating is totally missing, keep it but it won’t help; imputer will handle.
    out["dealer_rating"] = pd.to_numeric(df.get("dealer_rating", np.nan), errors="coerce")

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

    # Basic required fields
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

    X_raw = _build_features_raw(df)
    y = df["price"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

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

    if HAS_XGB:
        base_model = XGBRegressor(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            objective="reg:squarederror",
        )
    else:
        # sklearn fallback
        base_model = GradientBoostingRegressor(n_estimators=300, random_state=42)

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", base_model),
    ])

    # Train on log(price) for stability
    y_log_train = np.log1p(y_train)

    # Weight luxury cars more so the model stops regressing them to the mean
    lux_w = 2.0
    sample_weight = np.where(X_train["is_luxury"].to_numpy() == 1.0, lux_w, 1.0)

    if HAS_XGB:
        pipe.fit(X_train, y_log_train, model__sample_weight=sample_weight)
    else:
        # GradientBoostingRegressor supports sample_weight too
        pipe.fit(X_train, y_log_train, model__sample_weight=sample_weight)

    # Evaluate in original $ space
    # -----------------------------
    preds = None
    y_true = None

    try:
        preds = np.expm1(pipe.predict(X_test))
        y_true = y_test.to_numpy()

        mae = float(np.mean(np.abs(preds - y_true)))

        ss_res = float(np.sum((y_true - preds) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        logger.info(f"Price model trained | MAE=${mae:,.0f} R²={r2:.3f}")

        # -----------------------------
        # Slice diagnostics (luxury)
        # -----------------------------
        try:
            # X_test is a DataFrame because you split X_raw (a DataFrame)
            test_is_lux = (X_test["is_luxury"].to_numpy() >= 0.5)

            lux_n = int(test_is_lux.sum())
            nonlux_n = int((~test_is_lux).sum())

            lux_mae = float(np.mean(np.abs(preds[test_is_lux] - y_true[test_is_lux]))) if lux_n > 0 else float("nan")
            nonlux_mae = float(np.mean(np.abs(preds[~test_is_lux] - y_true[~test_is_lux]))) if nonlux_n > 0 else float("nan")

            logger.info(
                f"Slice MAE | luxury_n={lux_n} nonlux_n={nonlux_n} | "
                f"luxury=${lux_mae:,.0f} nonlux=${nonlux_mae:,.0f}"
            )
        except Exception as e:
            logger.warning(f"Luxury slice MAE calc failed: {e}")

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return None

    version = datetime.utcnow().strftime("%Y%m%d_%H%M")
    payload = {"model": pipe, "version": version}

    os.makedirs("models", exist_ok=True)
    joblib.dump(payload, MODEL_PATH)

    logger.info(f"Price model trained | MAE=${mae:,.0f} R²={r2:.3f}")
    logger.info(f"Saved model → {MODEL_PATH}")
    return {"mae": mae, "r2": r2, "version": version, "n": int(len(df))}


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