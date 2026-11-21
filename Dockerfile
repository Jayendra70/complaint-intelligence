# API image for the shipped scikit-learn models (no torch, no GPU).
# /topics/trends needs data/processed/complaint_topics.parquet mounted at runtime;
# without it that endpoint returns 503 and everything else works.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/ ./src/
COPY models/ ./models/
COPY reports/weekly_brief_????-??-??.md ./reports/

# Code and models stay root-owned and read-only; the server runs unprivileged.
RUN useradd --uid 1000 --no-create-home app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
