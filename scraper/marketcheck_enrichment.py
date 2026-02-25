"""
scraper/marketcheck_enrichment.py — Marketcheck enrichment APIs
"""

import os
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

API_KEY = os.getenv("MARKETCHECK_API_KEY", "")
BASE    = "https://mc-api.marketcheck.com/v2"
ZIP     = os.getenv("MARKETCHECK_ZIP", "90001")
RADIUS  = int(os.getenv("MARKETCHECK_RADIUS", 100))


class MarketCheckEnrichment:

    def predict_price(self, year: int, make: str, model: str,
                      miles: int = 50000, trim: str = None) -> dict | None:
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
            r = requests.get(f"{BASE}/predict/car/price", params=params, timeout=15)
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
            r = requests.get(
                f"{BASE}/search/car/popular",
                params={"api_key": API_KEY, "state": state, "rows": limit},
                timeout=15,
            )
            r.raise_for_status()
            return r.json().get("popular_cars", [])
        except Exception as e:
            logger.warning(f"get_popular_cars failed: {e}")
            return []

    def get_sales_stats(self, make: str, model: str, year: int = None) -> dict | None:
        try:
            params = {"api_key": API_KEY, "make": make, "model": model}
            if year:
                params["year"] = year
            r = requests.get(f"{BASE}/sales/stats/car", params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            return {
                "total_sales":  data.get("total"),
                "cpo_sales":    data.get("cpo"),
                "non_cpo_sales":data.get("non_cpo"),
                "dom_median":   data.get("dom_median"),
                "dom_mean":     data.get("dom_mean"),
                "dom_min":      data.get("dom_min"),
            }
        except Exception as e:
            logger.warning(f"get_sales_stats failed: {e}")
            return None

    def get_recent_listings(self, rows: int = 50, make: str = None,
                             model: str = None) -> list:
        try:
            params = {
                "api_key": API_KEY,
                "zip":     ZIP,
                "radius":  RADIUS,
                "rows":    rows,
            }
            if make:  params["make"]  = make
            if model: params["model"] = model
            r = requests.get(f"{BASE}/search/car/recents", params=params, timeout=15)
            r.raise_for_status()
            listings = r.json().get("listings", [])
            logger.info(f"Fetched {len(listings)} recent listings")
            return listings
        except Exception as e:
            logger.warning(f"get_recent_listings failed: {e}")
            return []
