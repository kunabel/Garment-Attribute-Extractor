# Base image for all microservices
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY common/requirements.txt /app/common/requirements.txt
RUN pip install --no-cache-dir -r /app/common/requirements.txt

# Copy shared utils package
COPY common /app/common

ENV PYTHONPATH=/app
