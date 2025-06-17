FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y     gcc     g++     gdal-bin     libgdal-dev     && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY pyproject.toml setup.py ./

# Set ownership and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3     CMD python -c "import src; print('Health check passed')" || exit 1

# Add metadata labels
LABEL maintainer="AHGD Analytics Team"
LABEL version="1.0.0"
LABEL description="Australian Health Data Analytics Platform"

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "src.cli"]