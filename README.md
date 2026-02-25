# Car Intel

Used car market intelligence tool powered by Marketcheck API + XGBoost.

## Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — add your MARKETCHECK_API_KEY and set MARKETCHECK_ZIP

# 4. Initial data scrape + train
PYTHONPATH=. python3 scheduler/runner.py --once

# 5. Start API (Terminal 1)
PYTHONPATH=. python3 api.py

# 6. Start frontend (Terminal 2)
cd frontend
npm install
npm start
```

## Changing market / zip code

Edit `scraper/data_ingest.py` and change the `ZIP` variable at the top:
```python
ZIP = "90001"   # LA
ZIP = "94102"   # SF
ZIP = "92101"   # San Diego
ZIP = "10001"   # NYC
```

Then re-scrape: `PYTHONPATH=. python3 scheduler/runner.py --scrape-only`

## Commands

```bash
# Scrape new listings
PYTHONPATH=. python3 scheduler/runner.py --scrape-only

# Retrain model only
PYTHONPATH=. python3 scheduler/runner.py --train-only

# Full run (scrape + score + train)
PYTHONPATH=. python3 scheduler/runner.py --once
```

## API Endpoints

| Endpoint | Description |
|---|---|
| GET /api/deals | Top deals sorted by score |
| GET /api/popular | Popular models ranking |
| GET /api/stats | Summary statistics |
| GET /api/trends | Price trends by make |
| GET /api/listings | All listings for scatter plot |
| GET /api/benchmark | Marketcheck price prediction |
| GET /api/sales-stats | 90-day sales velocity |
