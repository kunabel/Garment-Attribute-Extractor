FROM attr-extractor-service-base:latest

WORKDIR /app

# Installing service-specific requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# Gunicorn used for extended timeout - vision service takes longer because of inference
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "vision_service:app", \
     "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "300"]
