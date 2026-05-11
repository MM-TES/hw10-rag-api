FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 app
USER app
ENV PATH="/home/app/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app app/ ./app/
RUN pip install --user --no-cache-dir -e .

COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app data/ ./data/

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
