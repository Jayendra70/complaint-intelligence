# Phase 6 - Docker image for the Complaint Intelligence API.
#
# It packages only what the SHIPPED baseline service needs: the light
# scikit-learn stack, the FastAPI app code, the two saved baseline models, the
# Phase 4 topics parquet, and the Phase 5 weekly briefs. No torch, no GPU.
#
# Build + run on their own:
#   docker build -t complaint-api .
#   docker run -p 8000:8000 complaint-api
# ...or use docker compose (see docker-compose.yml).
#
# NOTE: the models must exist before building. If models/ is empty, first run
#   python -m src.serving.train_baseline

FROM python:3.11-slim

WORKDIR /app

# 1. Install dependencies first, so this layer is cached when only code changes.
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# 2. Copy just the pieces the API loads at start-up.
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/complaint_topics.parquet ./data/processed/complaint_topics.parquet
COPY reports/weekly_brief_*.md ./reports/

# 3. The API listens on 8000.
EXPOSE 8000

# 4. Start the server. 0.0.0.0 so it is reachable from outside the container.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
