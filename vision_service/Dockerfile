FROM attr-extractor-service-base:latest

WORKDIR /app

# Installing service-specific requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["uvicorn", "vision_service:app", "--host", "0.0.0.0", "--port", "8000"]
