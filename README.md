# AIML Fraud Detection API

Minimal README for running and developing the AIML FastAPI service.

**Project layout**
- `api_connect.py` - FastAPI application exposing `/ping` and `/predict` endpoints
- `retrain_model.py` - retrain script that reads from DB and writes `model/models.joblib`
- `model/models.joblib` - trained model package (contains `model`, `enc1`, `enc2`)
- `db_init.py`, `db_handling.py` - database ORM and session helpers
- `requirements.txt` - pip dependencies
- `Dockerfile` - image definition to run the API

**Environment**
The app requires a `DB_URL` environment variable pointing to your Postgres database (SQLAlchemy URL).

Example `.env` (project root):
```
DB_URL=postgresql://user:password@host:5432/dbname
```

**Install (local)**
1. Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Make sure `model/models.joblib` exists (either retrained or copied in).

**Run API (local)**

```bash
uvicorn api_connect:app --host 127.0.0.1 --port 8000
```

Health check:
```
curl http://127.0.0.1:8000/ping
```

**Predict example**
POST to `/predict` with JSON body:

```json
{
  "acc_holder": "example_account_123",
  "features": [10.5,15.2,120.0,5.0,3.0,2.0,1000.0,500.0,300.0,1500.0,200.0,800.0,600.0,400.0,7.0,4.0,"ETH","USDT"]
}
```

Example curl:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"acc_holder":"a","features":[10.5,15.2,120.0,5.0,3.0,2.0,1000,500,300,1500,200,800,600,400,7,4,"ETH","USDT"]}'
```

**Retrain model**
The `retrain_model.py` script reads from the `transactions` table and saves the trained model package to `model/models.joblib` with keys `model`, `enc1`, `enc2` expected by `api_connect.py`.

Run:
```bash
python retrain_model.py
```

**Docker**
Build image:
```bash
docker build -t aiml-app:latest .
```
Run container (provide DB URL and mount model if needed):
```bash
docker run -p 8000:8000 -e DB_URL="postgresql://user:pass@host:5432/db" -v /path/to/local/model:/app/model aiml-app:latest
```

**Notes & Troubleshooting**
- Ensure `model/models.joblib` contains `model`, `enc1` and `enc2` keys. If the API raises a model load error, retrain and re-run.
- Feature ordering must match between training and inference; do not change the input order expected by `api_connect.py`.
- If XGBoost errors about dtypes, ensure numeric columns are numeric and categorical columns are encoded during retraining.

If you want, I can add a `docker-compose.yml`, a startup healthcheck, or an automated check in `api_connect.py` to validate the loaded pack—which would you prefer?