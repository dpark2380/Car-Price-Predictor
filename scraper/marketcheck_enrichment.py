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

API_KEY = os.getenv("MARKETCHECK_API_KEY", "")
BASE    = "https://mc-api.marketcheck.com/v2"
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