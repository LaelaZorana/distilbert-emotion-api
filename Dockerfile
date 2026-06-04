# syntax=docker/dockerfile:1

# ---------- builder ----------
# Build wheels for the lean runtime deps in an isolated stage so the final image
# carries no build toolchain. Offline mode needs only these (no torch).
FROM python:3.11-slim AS builder

WORKDIR /build
RUN python -m pip install --no-cache-dir --upgrade pip wheel

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ---------- runtime ----------
FROM python:3.11-slim AS runtime

# WITH_MODEL=1 (default) installs the ML stack and serves the REAL fine-tuned model
# — this is what the public Hugging Face Space runs. Pass --build-arg WITH_MODEL=0 for
# the lean, torch-free image used by CI and local compose (which run the offline stub).
ARG WITH_MODEL=1

# Don't write .pyc, don't buffer stdout (so logs stream in real time).
# OFFLINE defaults to 0 (serve the real model); lean builds set OFFLINE=1 at run time
# (see docker-compose.yml and CI). The classifier falls back to the stub if a real
# load is requested but torch is absent, so a mismatched build still boots.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OFFLINE=0 \
    PORT=8000

# Non-root user — never run a network service as root.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

# Install the prebuilt lean wheels, then drop them.
COPY --from=builder /wheels /wheels
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Real-model stack (torch + transformers), pulled from the index. Skipped for the
# lean CI/local image (--build-arg WITH_MODEL=0).
RUN if [ "$WITH_MODEL" = "1" ]; then \
        pip install --no-cache-dir -r requirements-ml.txt ; \
    fi

# Application code (see .dockerignore for what is excluded).
COPY app/ ./app/
COPY demo/ ./demo/
COPY scripts/ ./scripts/

USER appuser
EXPOSE 8000

# Container-level liveness: the orchestrator restarts the container if this fails.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
url='http://127.0.0.1:%s/healthz' % os.getenv('PORT','8000'); \
sys.exit(0 if urllib.request.urlopen(url, timeout=2).status==200 else 1)"

# One uvicorn worker; the in-process micro-batcher handles concurrency. Scale
# horizontally (more replicas) rather than with multiple workers so the batcher
# stays effective per process.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
