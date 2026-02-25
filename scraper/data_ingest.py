"""
scraper/data_ingest.py — Marketcheck API data ingestion
"""

import os
import time
import random
import hashlib
from datetime import datetime

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

API_KEY = os.getenv("MARKETCHECK_API_KEY", "")
ZIP     = os.getenv("MARKETCHECK_ZIP", "93440")   # Change this to switch markets
RADIUS  = int(os.getenv("MARKETCHECK_RADIUS", 100))
BASE    = "https://mc-api.marketcheck.com/v2"


def _make_listing_id(listing: dict) -> str:
    vin = listing.get("vin", "")
    lid = listing.get("id", "")
    raw = f"{vin}-{lid}" if vin else lid
    if not raw:
        raw = hashlib.md5(str(listing).encode()).hexdigest()
    return raw


def map_listing(raw: dict, source: str = "marketcheck") -> dict | None:
    try:
        build  = raw.get("build") or {}
        dealer = raw.get("dealer") or {}
        extra  = raw.get("extra") or {}

        price   = raw.get("price") or raw.get("dp_price")
        mileage = raw.get("miles")
        year    = build.get("year") or raw.get("year")
        make    = build.get("make") or raw.get("make")
        model   = build.get("model") or raw.get("model")

        if not all([price, mileage, year, make, model]):
            return None
        if float(price) > 100_000:
            return None

        listing_id = _make_listing_id(raw)

        return {
            "listing_id":    listing_id,
            "vin":           raw.get("vin"),
            "url":           raw.get("vdp_url") or raw.get("dealer_vdp_url"),
            "scrape_source": source,
            "year":          int(year),
            "make":          str(make),
            "model":         str(model),
            "trim":          build.get("trim"),
            "body_style":    build.get("body_type"),
            "condition":     raw.get("car_type", "used"),
            "exterior_color":raw.get("exterior_color"),
            "price":         float(price),
            "mileage":       int(mileage),
            "accident_count":extra.get("accident_count", 0) or 0,
            "owner_count":   extra.get("owner_count", 1) or 1,
            "dealer_name":   dealer.get("name"),
            "dealer_rating": dealer.get("rating"),
            "location_city": dealer.get("city"),
            "location_state":dealer.get("state"),
            "location_zip":  dealer.get("zip"),
            "days_listed":   raw.get("dom", 0) or 0,
        }
    except Exception as e:
        logger.warning(f"map_listing failed: {e}")
        return None


class MarketCheckClient:
    def fetch_listings(self, rows_per_call=50, max_calls=3, **params) -> list:
        all_listings = []
        for i in range(max_calls):
            start = i * rows_per_call
            logger.info(f"  Marketcheck call {i+1}/{max_calls} (start={start})")
            try:
                r = requests.get(
                    f"{BASE}/search/car/active",
                    params={
                        "api_key":  API_KEY,
                        "zip":      ZIP,
                        "radius":   RADIUS,
                        "rows":     rows_per_call,
                        "start":    start,
                        "car_type": params.pop("car_type", "used"),
                        **params,
                    },
                    timeout=15,
                )
                r.raise_for_status()
                listings = r.json().get("listings", [])
                logger.info(f"  Got {len(listings)} listings (total: {len(all_listings) + len(listings)})")
                if not listings:
                    break
                all_listings.extend(listings)
                time.sleep(random.uniform(0.5, 1.2))
            except Exception as e:
                logger.warning(f"  Marketcheck call failed: {e}")
                break
        return all_listings


# Map target names to API params
TARGET_PARAMS = {
    "used_cars": {},  # general — returns whatever Marketcheck prioritises
}


class DataIngestor:
    CALLS_PER_TARGET = 1
    ROWS_PER_CALL    = 50

    def __init__(self):
        self.client = MarketCheckClient()

    def scrape_search(self, search_url: str = "", max_pages: int = 1,
                      search_target: str = "used_cars") -> list:
        logger.info(f"Fetching from Marketcheck: {search_target} (zip={ZIP}, radius={RADIUS}mi)")
        params = TARGET_PARAMS.get(search_target, {})
        raw_listings = self.client.fetch_listings(
            rows_per_call=self.ROWS_PER_CALL,
            max_calls=self.CALLS_PER_TARGET,
            car_type="used",
            **params,
        )
        mapped  = [map_listing(r, search_target) for r in raw_listings]
        cleaned = [l for l in mapped if l is not None]
        logger.info(f"Mapped {len(cleaned)}/{len(raw_listings)} listings successfully")
        return cleaned
