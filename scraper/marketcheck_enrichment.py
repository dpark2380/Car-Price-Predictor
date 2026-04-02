"""
scraper/marketcheck_enrichment.py — Marketcheck enrichment APIs
"""

import os
import json
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger

from scraper.api_usage import increment_call, ApiCallEvent

load_dotenv()

API_KEY   = os.getenv("MARKETCHECK_API_KEY", "")
BASE      = "https://mc-api.marketcheck.com/v2"   # search / recents endpoints
BASE_ANALYTICS = "https://api.marketcheck.com/v2"  # sales / market analytics endpoints
ZIP     = os.getenv("MARKETCHECK_ZIP", "90001")
RADIUS  = int(os.getenv("MARKETCHECK_RADIUS", 100))

CA_ZIPS = [
    "90001",  # Los Angeles
    "92101",  # San Diego
    "94102",  # San Francisco
    "95112",  # San Jose
    "95814",  # Sacramento
    "92626",  # Costa Mesa
    "93301",  # Bakersfield
    "93721",  # Fresno
    "93101",  # Santa Barbara
    "94538",  # Fremont
]

ZIP_CURSOR_PATH = Path("data/zip_cursor.json")


def _load_zip_cursor() -> int:
    try:
        if ZIP_CURSOR_PATH.exists():
            return int(json.loads(ZIP_CURSOR_PATH.read_text()).get("idx", 0))
    except Exception:
        pass
    return 0


def _save_zip_cursor(idx: int) -> None:
    ZIP_CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    ZIP_CURSOR_PATH.write_text(json.dumps({"idx": idx}))


class MarketCheckEnrichment:

    def predict_price(
        self,
        year: int,
        make: str,
        model: str,
        miles: int = 50000,
        trim: str | None = None,
    ) -> dict | None:
        try:
            params = {
                "api_key":  API_KEY,
                "car_type": "used",
                "year":     year,
                "make":     make,
                "model":    model,
                "miles":    miles,
            }
            if trim:
                params["trim"] = trim

            endpoint = "/predict/car/price"
            increment_call(ApiCallEvent(provider="marketcheck", endpoint=endpoint), n=1)

            r = requests.get(f"{BASE}{endpoint}", params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            return {
                "mc_predicted_price": data.get("price"),
                "mc_price_low":       data.get("price_range", {}).get("low"),
                "mc_price_high":      data.get("price_range", {}).get("high"),
                "mc_confidence":      data.get("confidence"),
            }
        except Exception as e:
            logger.warning(f"predict_price failed: {e}")
            return None

    def get_popular_cars(self, state: str = "CA", limit: int = 20) -> list:
        try:
            endpoint = "/search/car/popular"
            increment_call(ApiCallEvent(provider="marketcheck", endpoint=endpoint), n=1)

            r = requests.get(
                f"{BASE}{endpoint}",
                params={"api_key": API_KEY, "state": state, "rows": limit},
                timeout=15,
            )
            r.raise_for_status()
            return r.json().get("popular_cars", [])
        except Exception as e:
            logger.warning(f"get_popular_cars failed: {e}")
            return []

    def get_sales_stats(self, make: str, model: str, year: int | None = None) -> dict | None:
        try:
            params = {"api_key": API_KEY, "make": make, "model": model}
            if year:
                params["year"] = year

            endpoint = "/sales/stats/car"
            increment_call(ApiCallEvent(provider="marketcheck", endpoint=endpoint), n=1)

            r = requests.get(f"{BASE}{endpoint}", params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            return {
                "total_sales":   data.get("total"),
                "cpo_sales":     data.get("cpo"),
                "non_cpo_sales": data.get("non_cpo"),
                "dom_median":    data.get("dom_median"),
                "dom_mean":      data.get("dom_mean"),
                "dom_min":       data.get("dom_min"),
            }
        except Exception as e:
            logger.warning(f"get_sales_stats failed: {e}")
            return None

    def get_recent_listings(
        self,
        rows: int = 50,
        make: str | None = None,
        model: str | None = None,
        zip_code: str | None = None,
        rotate_ca_zips: bool = True,
    ) -> list:
        """
        If zip_code is provided -> use it.
        Else if rotate_ca_zips -> pick next ZIP from CA_ZIPS each call and persist cursor.
        Else -> fallback to ENV ZIP.
        """
        try:
            chosen_zip = zip_code

            if chosen_zip is None and rotate_ca_zips and CA_ZIPS:
                idx = _load_zip_cursor()
                chosen_zip = CA_ZIPS[idx % len(CA_ZIPS)]
                _save_zip_cursor((idx + 1) % len(CA_ZIPS))
            elif chosen_zip is None:
                chosen_zip = ZIP

            params = {
                "api_key": API_KEY,
                "zip": chosen_zip,
                "radius": RADIUS,
                "rows": rows,
            }
            if make:
                params["make"] = make
            if model:
                params["model"] = model

            endpoint = "/search/car/recents"
            increment_call(ApiCallEvent(provider="marketcheck", endpoint=endpoint), n=1)

            r = requests.get(f"{BASE}{endpoint}", params=params, timeout=15)
            r.raise_for_status()
            listings = r.json().get("listings", [])
            logger.info(f"Fetched {len(listings)} recent listings (zip={chosen_zip}, radius={RADIUS})")
            return listings

        except Exception as e:
            logger.warning(f"get_recent_listings failed: {e}")
            return []


MIN_LISTINGS_FOR_STATS = 5  # only fetch market stats for combos with this many listings


def enrich_with_sales_stats(df, cache_repo, fetch_missing: bool = True) -> "pd.DataFrame":
    """
    Joins national market-level sales stats onto a listings DataFrame at (make, model) level.

    Only fetches stats for combos with >= MIN_LISTINGS_FOR_STATS listings — rare combos
    get NaN which the pipeline imputer handles gracefully.

    Set fetch_missing=False to use only cached data without making any API calls.

    Adds columns: market_median_price, market_dom_median, price_to_market
    """
    import time
    import pandas as pd

    if df is None or df.empty:
        return df

    df = df.copy()

    df["_make_key"]  = df["make"].astype(str).str.lower().str.strip()
    df["_model_key"] = df["model"].astype(str).str.lower().str.strip()

    # Only consider combos with enough listings to be worth a dedicated API call
    combo_counts = df.groupby(["_make_key", "_model_key"]).size()
    eligible = combo_counts[combo_counts >= MIN_LISTINGS_FOR_STATS].index.tolist()

    # Build lookup from cache — keyed as (make, model), state stored as "NATIONAL"
    lookup: dict = {
        (make, model): stats
        for (make, model), stats in cache_repo.get_all_as_dict().items()
        if model != "NATIONAL"  # exclude old make-only entries
    }

    missing = [(m, mo) for m, mo in eligible if (m, mo) not in lookup]

    if not fetch_missing:
        if missing:
            logger.info(f"Skipping API fetch for {len(missing)} uncached (make, model) combos — using cache only")
        missing = []
    elif missing:
        logger.info(f"Fetching national sales stats for {len(missing)} new (make, model) combos")

    for make, model in missing:
        try:
            time.sleep(0.3)
            from scraper.api_usage import increment_call, ApiCallEvent
            increment_call(ApiCallEvent(provider="marketcheck", endpoint="/sales/car"), n=1)

            r = requests.get(
                f"{BASE_ANALYTICS}/sales/car",
                params={"api_key": API_KEY, "make": make, "model": model},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()

            price_stats = data.get("price_stats") or {}
            dom_stats   = data.get("dom_stats") or {}

            entry = {
                "median_price":       price_stats.get("median"),
                "trimmed_mean_price": price_stats.get("trimmed_mean"),
                "median_dom":         dom_stats.get("median"),
                "sample_count":       data.get("count"),
            }
            # Store with model as the "state" field to reuse existing cache schema
            cache_repo.upsert(make, model, entry)
            lookup[(make, model)] = {
                "market_median_price":       entry["median_price"],
                "market_trimmed_mean_price": entry["trimmed_mean_price"],
                "market_dom_median":         entry["median_dom"],
                "market_sample_count":       entry["sample_count"],
            }
            logger.info(f"  Cached sales stats: {make} {model} — median=${entry['median_price']:,}")

        except Exception as e:
            logger.warning(f"  Sales stats fetch failed for {make} {model}: {e}")

    # Join onto df by (make, model)
    stats_rows = [
        lookup.get((row["_make_key"], row["_model_key"]), {})
        for _, row in df[["_make_key", "_model_key"]].iterrows()
    ]
    stats_df = pd.DataFrame(stats_rows, index=df.index)
    for col in ["market_median_price", "market_trimmed_mean_price", "market_dom_median"]:
        df[col] = pd.to_numeric(stats_df.get(col, pd.Series([None] * len(df), index=df.index)), errors="coerce")

    df.drop(columns=["_make_key", "_model_key"], inplace=True)
    enriched = df["market_median_price"].notna().sum()
    logger.info(f"Sales stats enrichment: {enriched}/{len(df)} listings matched")
    return df