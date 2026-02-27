FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps needed by some Python packages (psycopg2, xgboost build, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Ensure model folder exists (if you will copy a prebuilt model into image)
RUN mkdir -p /app/model

EXPOSE 8000

# Default command to run the FastAPI app
CMD ["uvicorn", "api_connect:app", "--host", "0.0.0.0", "--port", "8000"]
