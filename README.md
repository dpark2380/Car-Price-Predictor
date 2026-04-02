# Car Intel

A full-stack used car market intelligence tool that scrapes live listings from the Marketcheck API, predicts fair market prices using XGBoost, scores each listing as a deal (0–100), and displays everything in a React dashboard.

---

## How It Works

```
Marketcheck API → SQLite DB → XGBoost Model → Deal Scores → Flask API → React Dashboard
```

1. **Scraper** — Pulls active listings from Marketcheck, rotates through paginated results and multiple zip codes, normalizes fields, and upserts into SQLite.
2. **ML Pipeline** — Trains an XGBoost regressor on log-transformed prices with engineered features (vehicle age, miles/year, luxury flags, interaction terms). Compares XGBoost, Random Forest, and Linear Regression; picks the best by MAE.
3. **Deal Scoring** — Compares each listing's actual price to the predicted fair value and assigns a 0–100 score.
4. **Flask API** — Serves deal data, popularity rankings, market stats, price trends, and scatter plot data.
5. **React Dashboard** — Dark-themed UI with a rotating ticker of top deals, filterable deal table, popularity rankings, price trend charts, and a price vs. mileage scatter plot.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python, Flask |
| ML | XGBoost, scikit-learn, Pandas, NumPy |
| Database | SQLite via SQLAlchemy |
| Frontend | React 19, Recharts |
| Data Source | Marketcheck API v2 |

---

## Project Structure

```
Car-Price-Predictor/
├── api.py                        # Flask REST API (port 5001)
├── requirements.txt
├── config/
│   └── search_targets.json       # Search configuration
├── scraper/
│   ├── data_ingest.py            # Marketcheck API client
│   ├── marketcheck_enrichment.py # Enrichment endpoints
│   └── api_usage.py              # API quota tracking
├── db/
│   ├── models.py                 # SQLAlchemy models (SQLite)
│   └── repository.py             # CRUD operations
├── ml/
│   └── pipeline.py               # Model training, scoring, popularity snapshots
├── scheduler/
│   └── runner.py                 # Job orchestration (scrape, train, score)
├── frontend/
│   └── src/
│       ├── car-intel-dashboard.jsx
│       ├── components/
│       └── utils/
└── data/                         # Runtime state (cursors, API usage)
```

---

## Deal Scoring

Each listing is scored by comparing its asking price to the model's predicted fair market value:

```
score = 50 + ((predicted - actual) / predicted × 100 × 1.25)
score = clipped to [0, 100]
```

| Score | Label |
|---|---|
| 90–100 | Hidden Gem |
| 75–89 | Great Deal |
| 60–74 | Good Deal |
| 45–59 | Fair Price |
| 0–44 | Overpriced |

---

## ML Model Details

- **Algorithm:** XGBoost (`n_estimators=800, max_depth=5, learning_rate=0.05`)
- **Target:** `log(price)` — log-space regression for stability across price ranges
- **Features:** Vehicle age, mileage, miles/year, log transforms, polynomial terms, luxury/truck/sports flags, interaction features, plus one-hot encoded make, model, trim, body type, drivetrain, fuel type, transmission, state, zip prefix
- **Sample weighting:** Luxury brands (BMW, Mercedes, Lexus, Tesla, Porsche, etc.) weighted 2× during training
- **Train/test split:** 80/20, random seed 42
- **Model selection:** Lowest MAE on test set across XGBoost, Random Forest, and Linear Regression
- **Serialization:** `models/price_predictor.joblib`

---

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

---

## Commands

```bash
# Full run (scrape + train + score)
PYTHONPATH=. python3 scheduler/runner.py --once

# Scrape new listings only
PYTHONPATH=. python3 scheduler/runner.py --scrape-only

# Retrain model only
PYTHONPATH=. python3 scheduler/runner.py --train-only
```

---

## Changing Market / Zip Code

Edit `scraper/data_ingest.py` and change the `ZIP` variable at the top:

```python
ZIP = "90001"   # LA
ZIP = "94102"   # SF
ZIP = "92101"   # San Diego
ZIP = "10001"   # NYC
```

Then re-scrape:

```bash
PYTHONPATH=. python3 scheduler/runner.py --scrape-only
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/deals` | Top deals sorted by score. Query: `limit`, `min_score`, `make`, `model`, `body` |
| `GET /api/popular` | Most common year/make/model ranked by active listing count. Query: `limit` |
| `GET /api/stats` | Global market stats (avg price, min/max, active listing count) |
| `GET /api/trends` | Avg prices by month for top 5 makes (last 7 months) |
| `GET /api/listings` | All active listings (sampled to 800) with deal scores for scatter plot |
| `GET /api/market-popular` | Marketcheck popularity data (optional enrichment) |
| `GET /api/recent-listings` | Recent listings from Marketcheck (optional enrichment) |
