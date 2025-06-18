# Australian Health Analytics Platform
# Modern Docker container with Python 3.11+ and UV package manager

FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV (modern Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Install dependencies with UV (5x faster than pip)
RUN uv sync --frozen

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p data/{raw,processed,outputs}

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command (can be overridden)
CMD ["uv", "run", "streamlit", "run", "src/web/streamlit/dashboard.py", "--server.address", "0.0.0.0"]