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

MODEL_PATH = "models/price_predictor.joblib"
MIN_TRAINING_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", 140))

MAKE_ENCODE = {}   # filled at train time
MODEL_ENCODE = {}


def _encode_col(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(mapping).fillna(0).astype(float)


def _build_features(df: pd.DataFrame, make_enc: dict, model_enc: dict) -> pd.DataFrame:
    now_year = datetime.utcnow().year
    feats = pd.DataFrame()

    feats["year"]           = pd.to_numeric(df["year"], errors="coerce").fillna(2018)
    feats["vehicle_age"]    = now_year - feats["year"]
    feats["mileage"]        = pd.to_numeric(df["mileage"], errors="coerce").fillna(50000)
    feats["miles_per_year"] = (feats["mileage"] / (feats["vehicle_age"].clip(1))).clip(0, 50000)
    feats["accident_count"] = pd.to_numeric(df.get("accident_count", 0), errors="coerce").fillna(0)
    feats["owner_count"]    = pd.to_numeric(df.get("owner_count", 1), errors="coerce").fillna(1)
    feats["dealer_rating"]  = pd.to_numeric(df.get("dealer_rating", 4.0), errors="coerce").fillna(4.0)

    feats["make_enc"]  = _encode_col(df["make"].astype(str).str.lower().str.strip(), make_enc)
    feats["model_enc"] = _encode_col(df["model"].astype(str).str.lower().str.strip(), model_enc)

    state_map = {"CA": 1, "NY": 2, "TX": 3, "FL": 4, "WA": 5}
    feats["state_enc"] = df.get("location_state", pd.Series(["CA"] * len(df))).map(state_map).fillna(0)

    luxury_makes = {
        "bmw", "mercedes-benz", "lexus", "audi", "acura", "infiniti", "cadillac",
        "rivian", "lincoln", "porsche", "lamborghini", "ferrari"
    }
    truck_makes_models = {"f-150", "f-250", "silverado", "sierra", "ram", "tundra", "tacoma", "ranger", "colorado"}
    sports_models = {"corvette", "mustang", "camaro", "challenger", "charger", "86", "brz", "miata", "supra"}

    make_s  = df["make"].astype(str).str.lower().str.strip()
    model_s = df["model"].astype(str).str.lower().str.strip()

    feats["is_luxury"] = make_s.isin(luxury_makes).astype(float)
    feats["is_truck"]  = model_s.apply(lambda m: float(any(t in m for t in truck_makes_models)))
    feats["is_sports"] = model_s.apply(lambda m: float(any(s in m for s in sports_models)))

    return feats


def train(df: pd.DataFrame) -> dict | None:
    logger.info("Starting model training run…")

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

    global MAKE_ENCODE, MODEL_ENCODE
    makes  = df["make"].astype(str).str.lower().str.strip().unique()
    models = df["model"].astype(str).str.lower().str.strip().unique()
    MAKE_ENCODE  = {m: i + 1 for i, m in enumerate(sorted(makes))}
    MODEL_ENCODE = {m: i + 1 for i, m in enumerate(sorted(models))}

    X = _build_features(df, MAKE_ENCODE, MODEL_ENCODE)
    y = df["price"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if HAS_XGB:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    else:
        model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)

    y_log_train = np.log1p(y_train)
    model.fit(X_train, y_log_train)

    preds = np.expm1(model.predict(X_test))
    y_true = y_test.to_numpy()

    mae = float(np.mean(np.abs(preds - y_true)))
    ss_res = float(np.sum((y_true - preds) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    version = datetime.utcnow().strftime("%Y%m%d_%H%M")
    payload = {"model": model, "make_enc": MAKE_ENCODE, "model_enc": MODEL_ENCODE, "version": version}

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
    # Align labels with your frontend thresholds (good -> low color risk)
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
    make_enc = payload["make_enc"]
    model_enc = payload["model_enc"]

    df = df.copy()
    df = df.dropna(subset=["price", "mileage", "year", "make", "model"])
    if df.empty:
        return []

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    if df.empty:
        return []

    logger.info(f"Scoring {len(df)} listings…")
    X = _build_features(df, make_enc, model_enc)
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