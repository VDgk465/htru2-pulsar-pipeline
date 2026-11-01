FROM python:3.11-slim

# Java is required by the optional Spark path (pyspark).
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir ".[spark]"

# Generated outputs are written here; mount them to keep them after the run.
VOLUME ["/app/reports", "/app/models", "/app/data"]

ENTRYPOINT ["python", "-m", "htru2_pipeline"]
CMD ["--config", "config/config.yaml"]
