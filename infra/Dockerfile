# CreditLens AI API image.
#
# Build it from the repository root, not from infra/, because the build needs src/:
#     docker build -f infra/Dockerfile -t creditlens-api .
#     docker run --rm -p 8000:8000 --env-file .env creditlens-api
#
# Two stages. The "builder" stage has the compilers and the pip cache and produces a
# virtual environment; the final stage copies only that environment. The result is a
# smaller image with no build tools in it, which is both faster to pull and a smaller
# attack surface.

# ---------------------------------------------------------------------------
# Stage 1: build the virtual environment
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is needed by a few wheels that have no prebuilt binary for slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Copy only what pip needs first. Docker caches this layer, so editing a Python file
# does not reinstall every dependency.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the extras the service actually runs with. Add more (agents, retrieval, ml)
# when those milestones are deployed; each one makes the image bigger. The SOAP mock in
# docker-compose.yml builds this same file with EXTRAS=integrations, because it needs spyne.
ARG EXTRAS="api,llm,security"
RUN pip install --upgrade pip && pip install ".[${EXTRAS}]"

# ---------------------------------------------------------------------------
# Stage 2: the runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLM_OFFLINE=0

# Run as a normal user. A container that runs as root and is broken into gives the
# attacker root inside the container, and sometimes a path out of it.
RUN useradd --create-home --uid 10001 creditlens

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=creditlens:creditlens src ./src
COPY --chown=creditlens:creditlens pyproject.toml README.md ./
# The database and generated data are mounted at run time, not baked in.
RUN mkdir -p /app/data && chown -R creditlens:creditlens /app

USER creditlens

EXPOSE 8000

# Docker checks this and marks the container unhealthy if /health stops answering.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# --factory is what lets create_app() read DATABASE_URL from the environment at start.
CMD ["uvicorn", "creditlens.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
