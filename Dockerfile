FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer — only re-runs when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Copy only the two model files the API actually needs at runtime
COPY models/xgboost_tuned.pkl ./models/xgboost_tuned.pkl
COPY models/scaler.pkl        ./models/scaler.pkl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
