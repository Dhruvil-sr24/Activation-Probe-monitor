# ── Stage 1: build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user
RUN useradd --create-home --shell /bin/bash probe
WORKDIR /app

COPY --from=builder /install /usr/local
COPY --chown=probe:probe . .

USER probe
 
ENV MODEL_NAME="gpt2"
ENV DEVICE="cpu"
ENV TORCH_DTYPE="float32"
ENV MAX_LENGTH="128"
ENV PROBE_DIR="artifacts/probes"
ENV PROBE_THRESHOLD="0.5"
ENV API_HOST="0.0.0.0"
ENV API_PORT="8000"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
 
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
