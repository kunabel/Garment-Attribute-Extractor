FROM attr-extractor-service-base:latest

WORKDIR /app

# Minimal native libs PaddleOCR/OpenCV expect
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installing service-specific requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "heuristic_service:app", \
     "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "300"]
