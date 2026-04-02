"""
scripts/test_marketcheck_endpoints.py

Quick test script to explore:
  1. /v2/sales/car           — inferred sales stats
  2. /v2/predict/car/price   — Marketcheck fair retail price prediction
  3. /v2/predict/car/us/marketcheck_price — Marketcheck used car price

Usage:
    PYTHONPATH=. python3 scripts/test_marketcheck_endpoints.py
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MARKETCHECK_API_KEY", "")
BASE    = "https://api.marketcheck.com/v2"

# ── Test vehicle ──────────────────────────────────────────────────────────────
MAKE  = "Toyota"
MODEL = "Camry"
YEAR  = 2020
STATE = "CA"
ZIP   = "90001"
# Sample VIN for price prediction endpoints (2020 Toyota Camry)
VIN   = "4T1C11AK0LU928843"


def pretty(label: str, data: dict | list) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2))


# ── 1. Sales Stats ────────────────────────────────────────────────────────────
def test_sales():
    r = requests.get(
        f"{BASE}/sales/car",
        params={
            "api_key": API_KEY,
            "make":    MAKE,
            "model":   MODEL,
            "year":    YEAR,
            "state":   STATE,
        },
        timeout=15,
    )
    print(f"\n[sales/car] Status: {r.status_code}")
    if r.ok:
        pretty("Sales Stats — Toyota Camry 2020 (CA)", r.json())
    else:
        print(f"  Error: {r.text[:500]}")


# ── 2. Fair Retail Price Prediction ──────────────────────────────────────────
def test_price_prediction():
    r = requests.get(
        f"{BASE}/predict/car/price",
        params={
            "api_key":  API_KEY,
            "vin":      VIN,
            "zip":      ZIP,
            "miles":    60000,
            "car_type": "used",
        },
        timeout=15,
    )
    print(f"\n[predict/car/price] Status: {r.status_code}")
    if r.ok:
        pretty("Fair Retail Price Prediction — 2020 Toyota Camry", r.json())
    else:
        print(f"  Error: {r.text[:500]}")


# ── 3. Marketcheck Used Car Price ─────────────────────────────────────────────
def test_marketcheck_price():
    r = requests.get(
        f"{BASE}/predict/car/us/marketcheck_price",
        params={
            "api_key":     API_KEY,
            "vin":         VIN,
            "zip":         ZIP,
            "miles":       60000,
            "dealer_type": "franchise",
        },
        timeout=15,
    )
    print(f"\n[predict/car/us/marketcheck_price] Status: {r.status_code}")
    if r.ok:
        pretty("Marketcheck Used Car Price — 2020 Toyota Camry", r.json())
    else:
        print(f"  Error: {r.text[:500]}")


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: MARKETCHECK_API_KEY not set in .env")
        exit(1)

    print(f"Testing with: {YEAR} {MAKE} {MODEL} | VIN: {VIN} | State: {STATE}")
    test_sales()
    test_price_prediction()
    test_marketcheck_price()
