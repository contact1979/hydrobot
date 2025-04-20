# Use slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies and Poetry
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    htop \
    perf-tools-unstable \
    linux-perf \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="${PATH}:/root/.local/bin"

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies with performance optimizations
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi \
    && python -m pip install --no-cache-dir \
    orjson \
    ujson \
    cython

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 bot
USER bot

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENV=production
ENV PYTHONASYNCIODEBUG=0
ENV PYTHONOPTIMIZE=2

# Expose metrics port
EXPOSE 8000

# Health check with improved parameters
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/metrics || exit 1

# Command to run the bot with CPU affinity for better HFT performance
CMD ["taskset", "-c", "0-3", "poetry", "run", "python", "-m", "hft_scalping_bot.main"]