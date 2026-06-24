FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY services /app/services
COPY workers /app/workers
COPY ml /app/ml

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
