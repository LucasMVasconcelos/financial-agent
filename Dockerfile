# syntax=docker/dockerfile:1

# ---- Stage 1: build a wheel + resolve deps into a virtualenv ----
FROM python:3.12-slim AS builder

ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Copy only dependency manifests first to maximize layer cache reuse.
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root --no-directory

COPY src ./src
COPY README.md ./
RUN poetry install --only main

# ---- Stage 2: minimal runtime image ----
FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=2).raise_for_status()"

CMD ["uvicorn", "financial_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
