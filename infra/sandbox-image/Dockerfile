# syntax=docker/dockerfile:1.7
#
# Gateway code-execution sandbox image.
#
# Pinned to match Anthropic's documented code execution package set so that
# model-generated code behaves identically across providers. The image is
# intentionally heavy on data-science libraries (pandas/numpy/scipy/etc.) and
# pre-bakes them so user code never has to ``pip install``.
#
# Build:   docker build -t gateway-sandbox:dev .
# Run:     docker run --rm -p 8080:8080 gateway-sandbox:dev
# Health:  curl localhost:8080/health
#
# The image runs as a non-root user (uid 1000). One container hosts many
# isolated sessions; each session has its own /var/sandbox/sessions/<id>/
# directory tree with its own workspace and Python REPL subprocess.
# ``/var/sandbox`` is intended to be mounted as tmpfs in production so that
# session state stays ephemeral.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SANDBOX_SESSIONS_ROOT=/var/sandbox/sessions \
    PYTHONPATH=/opt/sandbox

# System packages: bash + the unix toolset Anthropic's sandbox advertises
# (grep, sed, gawk, coreutils sort, jq, curl). Tini supervises the exec
# server so SIGTERM from Kubernetes propagates correctly to uvicorn.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        coreutils \
        curl \
        file \
        gawk \
        grep \
        jq \
        sed \
        tini \
 && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# Python dependencies — pinned to match Anthropic's pre-installed set.
# Update these only when Anthropic bumps theirs; model code generation is
# sensitive to library version drift.
# --------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        # Server runtime
        "fastapi==0.135.*" \
        "uvicorn[standard]==0.44.*" \
        "pydantic==2.12.*" \
        "python-multipart==0.0.24" \
        # Data / scientific computing
        "numpy==2.2.*" \
        "pandas==2.2.*" \
        "scipy==1.14.*" \
        "scikit-learn==1.6.*" \
        "matplotlib==3.9.*" \
        "seaborn==0.13.*" \
        "pillow==11.*" \
        "sympy==1.13.*" \
        # I/O and document handling
        "requests==2.32.*" \
        "openpyxl==3.1.*" \
        "xlrd==2.0.*" \
        "xlsxwriter==3.2.*" \
        "python-dateutil==2.9.*" \
        "pytz==2025.*" \
        "beautifulsoup4==4.12.*" \
        "lxml==5.*" \
        "pypdf==5.*" \
        "python-docx==1.1.*" \
        "python-pptx==1.0.*"

# --------------------------------------------------------------------------
# Application code + per-session storage root
# --------------------------------------------------------------------------
RUN mkdir -p /opt/sandbox/sandbox /var/sandbox/sessions \
 && useradd --create-home --uid 1000 --shell /bin/bash sandbox \
 && chown -R sandbox:sandbox /opt/sandbox /var/sandbox

COPY --chown=sandbox:sandbox sandbox/ /opt/sandbox/sandbox/

USER sandbox
WORKDIR /opt/sandbox

EXPOSE 8080

# Health check uses the /health endpoint, which reports both server liveness
# and runner subprocess status.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

# tini handles PID 1 signal forwarding so SIGTERM cleanly stops uvicorn.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "sandbox.exec_server:app", "--host", "0.0.0.0", "--port", "8080"]
