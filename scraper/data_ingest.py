"""
scraper/data_ingest.py — Marketcheck API data ingestion
"""

import os
import time
import random
import hashlib
import json
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger

from scraper.api_usage import increment_call, ApiCallEvent

load_dotenv()

API_KEY = os.getenv("MARKETCHECK_API_KEY", "")
ZIP     = os.getenv("MARKETCHECK_ZIP", "94119")   # Change this to switch markets
RADIUS  = int(os.getenv("MARKETCHECK_RADIUS", 100))
BASE    = "https://mc-api.marketcheck.com/v2"

PAGE_CURSOR_PATH = Path("data/page_cursor.json")


def _make_listing_id(listing: dict) -> str:
    vin = listing.get("vin", "")
    lid = listing.get("id", "")
    raw = f"{vin}-{lid}" if vin else lid
    if not raw:
        raw = hashlib.md5(str(listing).encode()).hexdigest()
    return raw


def extract_body_type(raw: dict) -> str | None:
    """
    Extract body type from Marketcheck-like payloads.
    Tries multiple common keys and nested locations.
    """
    build = raw.get("build") or {}

    candidate = (
        build.get("body_type")
        or build.get("body_style")
        or build.get("body_subtype")
        or raw.get("body_type")
        or raw.get("body_style")
        or raw.get("vehicle_type")
        or raw.get("type")
    )

    if not candidate:
        return None

    s = str(candidate).strip()
    return s or None


def normalize_body_type(bt: str | None) -> str | None:
    """
    Normalize common variations so filtering is consistent.
    """
    if not bt:
        return None

    s = bt.strip().lower()

    aliases = {
        # SUVs
        "sport utility": "SUV",
        "sport utility vehicle": "SUV",
        "suv": "SUV",
        "crossover": "SUV",

        # Passenger cars
        "sedan": "Sedan",
        "coupe": "Coupe",
        "hatchback": "Hatchback",
        "wagon": "Wagon",
        "convertible": "Convertible",

        # Trucks/Vans
        "pickup": "Truck",
        "pickup truck": "Truck",
        "truck": "Truck",
        "van": "Van",
        "minivan": "Van",
    }

    return aliases.get(s, bt.strip())


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

        body_type = normalize_body_type(extract_body_type(raw))

        return {
            "listing_id":     listing_id,
            "vin":            raw.get("vin"),
            "url":            raw.get("vdp_url") or raw.get("dealer_vdp_url"),
            "scrape_source":  source,
            "year":           int(year),
            "make":           str(make),
            "model":          str(model),
            "trim":           build.get("trim"),

            # NEW (preferred)
            "body_type":      body_type,
            # Keep existing field for backwards compatibility with your DB/schema
            "body_style":     body_type,

            "condition":      raw.get("car_type", "used"),
            "exterior_color": raw.get("exterior_color"),
            "price":          float(price),
            "mileage":        int(mileage),
            "accident_count": extra.get("accident_count", 0) or 0,
            "owner_count":    extra.get("owner_count", 1) or 1,
            "dealer_name":    dealer.get("name"),
            "dealer_rating":  dealer.get("rating"),
            "location_city":  dealer.get("city"),
            "location_state": dealer.get("state"),
            "location_zip":   dealer.get("zip"),
            "days_listed":    raw.get("dom", 0) or 0,
        }
    except Exception as e:
        logger.warning(f"map_listing failed: {e}")
        return None


def _load_page_cursor() -> int:
    try:
        if PAGE_CURSOR_PATH.exists():
            return int(json.loads(PAGE_CURSOR_PATH.read_text()).get("page", 0))
    except Exception:
        pass
    return 0


def _save_page_cursor(page: int) -> None:
    PAGE_CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGE_CURSOR_PATH.write_text(json.dumps({"page": int(page)}))


class MarketCheckClient:
    def fetch_listings(
        self,
        rows_per_call: int = 50,
        max_calls: int = 1,
        rotate_pages: bool = True,
        pages_in_rotation: int = 10,
        **params,
    ) -> list:
        """
        Rotates through pages across runs by persisting a cursor.

        - start_offset = (page_cursor % pages_in_rotation) * rows_per_call
        - start = start_offset + i*rows_per_call

        Cursor update (auto-reset):
          - if any call returns 0 listings -> reset cursor to 0
          - if any call returns < rows_per_call -> treat as last page, reset to 0
          - else advance cursor by max_calls (wrap by pages_in_rotation)
        """
        all_listings: list = []

        if pages_in_rotation < 1:
            pages_in_rotation = 1

        page_cursor = _load_page_cursor() if rotate_pages else 0
        start_offset = (page_cursor % pages_in_rotation) * rows_per_call

        reset_to_zero = False

        for i in range(max_calls):
            start = start_offset + i * rows_per_call
            logger.info(f"  Marketcheck call {i+1}/{max_calls} (start={start})")

            try:
                # IMPORTANT: don't mutate params across iterations
                _params = dict(params)

                increment_call(ApiCallEvent(provider="marketcheck", endpoint="/search/car/active"), n=1)
                r = requests.get(
                    f"{BASE}/search/car/active",
                    params={
                        "api_key":  API_KEY,
                        "zip":      ZIP,
                        "radius":   RADIUS,
                        "rows":     rows_per_call,
                        "start":    start,
                        "car_type": _params.pop("car_type", "used"),
                        **_params,
                    },
                    timeout=15,
                )
                r.raise_for_status()

                listings = r.json().get("listings", [])
                got = len(listings)
                logger.info(f"  Got {got} listings (total: {len(all_listings) + got})")

                if got == 0:
                    # Past the end; reset next run
                    reset_to_zero = True
                    break

                all_listings.extend(listings)

                if got < rows_per_call:
                    # Partial page -> likely last page; reset next run
                    reset_to_zero = True
                    break

                time.sleep(random.uniform(0.5, 1.2))

            except Exception as e:
                logger.warning(f"  Marketcheck call failed: {e}")
                # On error, don't advance cursor (safer)
                reset_to_zero = False
                break

        # Cursor update for next run
        if rotate_pages:
            if reset_to_zero:
                _save_page_cursor(0)
            else:
                next_page = (page_cursor + max_calls) % pages_in_rotation
                _save_page_cursor(next_page)

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

    def scrape_search(
        self,
        search_url: str = "",
        max_pages: int = 1,
        search_target: str = "used_cars",
    ) -> list:
        logger.info(f"Fetching from Marketcheck: {search_target} (zip={ZIP}, radius={RADIUS}mi)")
        params = TARGET_PARAMS.get(search_target, {})

        raw_listings = self.client.fetch_listings(
            rows_per_call=self.ROWS_PER_CALL,
            max_calls=self.CALLS_PER_TARGET,     # keep at 1 to avoid extra API calls
            rotate_pages=True,                   # rotate pages across runs
            pages_in_rotation=10,                # rotate through first 10 pages (0..9)
            car_type="used",
            **params,
        )

        mapped  = [map_listing(r, search_target) for r in raw_listings]
        cleaned = [l for l in mapped if l is not None]
        logger.info(f"Mapped {len(cleaned)}/{len(raw_listings)} listings successfully")
        return cleaned