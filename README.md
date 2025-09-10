# Garment-Attribute-Extractor

Garment-Attribute-Extractor is a modular system for **clothing attribute extraction**.  
It classifies garments from images (category, color, material, etc.), detects brand names from labels, and can enrich attributes via an LLM service.

---

## 📦 Services

The system is split into four microservices:

| Service         | Port | Description |
|-----------------|------|-------------|
| **Orchestrator** | 8000 | Public API. Receives image URLs, fans out to sub-services, merges results, logs to SQLite. |
| **Vision**       | 8001 | CLIP-based classifier (CPU-only). Predicts category, pattern, sleeve length, neckline, etc. |
| **Heuristic**    | 8002 | Heuristic-based extractor. Includes PaddleOCR for brand detection. |
| **LLM**          | 8003 | Simulated LLM service. Returns a pre-defined description of the garment simulating an OpenAI API call. |

Shared utilities live in **`common/utils`**. Each service has its own `Dockerfile` and `requirements.txt`.

---

## 🗂 Project Structure

```
.
├── base/                # Base Docker image (Python, shared deps)
├── common/
│   └── utils/           # Shared Python helpers
├── orchestrator/        # Main entrypoint service
│   ├── orchestrator.py
│   ├── db.py
│   └── requirements.txt
├── vision-service/      # Vision (CLIP) classifier
├── heuristic-service/   # OCR with fuzzy logic for brand detection
├── llm-service/         # LLM description of the garment
├── docker-compose.yaml  # Multi-service orchestration
├── AI_USAGE.md          # Document describing AI usage for the project
└── ARCHITECTURE.md      # Architecture diagram & flow description

```

---

## 🚀 Getting Started

### 1. Install Docker & Docker Compose
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac)  
- [Docker Engine + Compose plugin](https://docs.docker.com/compose/install/) (Linux)

Verify installation:
```bash
docker --version
docker compose version
```

### 2. Build and run services
```bash
docker compose build base --no-cache --progress=plain
docker compose up --build --force-recreate
```

This will start all four services. Logs are available via:
```bash
docker compose logs -f
```

### 3. Test endpoints

**Health check (orchestrator):**
```bash
curl http://localhost:8000/ping
```
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/ping"
```


**Full analysis request:**
```bash
curl -X POST http://localhost:8000/v1/items/analyze \
     -H "Content-Type: application/json" \
     -d '{"image_urls": [
          "https://AronRedivivum.github.io/Home-Assignment/Images/2025_08_2009_01_320011.JPG",
          "https://AronRedivivum.github.io/Home-Assignment/Images/2025_08_2009_01_490012.JPG",
          "https://AronRedivivum.github.io/Home-Assignment/Images/2025_08_2009_02_020040.JPG",
          "https://AronRedivivum.github.io/Home-Assignment/Images/2025_08_2009_02_150041.JPG"
     ]}'
```
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/v1/items/analyze" `
>>   -Method POST `
>>   -Headers @{ "Content-Type" = "application/json" } `
>>   -Body '{"image_urls":["https://AronRedivivum.github.io/Home-Assignment/Images/2025_08_2009_01_320011.JPG", "https://AronRedivivum.github.io/Home-Assignment/Images/2025_08_2009_01_490012.JPG", "https://AronRedivivum.github.io/Home-Assignment/Images/2025_08_2009_02_020040.JPG", "https://AronRedivivum.github.io/Home-Assignment/Images/2025_08_2009_02_150041.JPG"]}' `
>>   -OutFile "C:\response.json"
```
Note: This saves the response in C:\ root as well as a json.

---

## 📋 API Overview

### Orchestrator
- `POST /v1/items/analyze`  
  Input:  
  ```json
  {
    "image_urls": ["front.jpg", "back.jpg", "side.jpg", "label.jpg"]
  }
  ```
  Example output:
  ```json
  {
    "id": "uuid",
    "attributes": { "color": "red", "brand": "Nike", "category": "t-shirt" },
    "model_info": { "color": "heuristic_model", "brand": "heuristic-ocr", "category": "clip-model" },
    "processing": { "time_taken": 1.23, "status": "success" }
  }
  ```

### Vision Service
- `POST /classify` → returns fashion attributes from CLIP model.

### Heuristic Service
- `POST /extract` → returns brand of the garment.

### LLM Service
- `POST /generate` → returns a simulated description of the piece.

---

## 💾 Logging

- Each service logs to **stdout** (view with `docker compose logs`).  
- Orchestrator also logs structured request results to a local SQLite DB (`requests.db`).  
- For persistence volumes are mounted in `docker-compose.yaml` to persist DB on host:
  ```yaml
  volumes:
    - ./data/orchestrator:/app/data
  environment:
    - DB_FILE=/app/data/requests.db
  ```

  ### Spent time:
  ### Task A):
  ~10 hours. Trade-off on the 8-hour hard cap: There would have been no CLIP model implemented and most of the attributes would be simulated. 
  The body of the project would be still ready. I preferred a more full solution where I can experiment with available Vision models more.
  Experimenting with CLIP for vision and OCR for the heuristic model exposed shortcomings such as time-outs that would have been overlooked and would have decreased
  complexity and robustness of the submitted project.

  ### Task B):
  ~2hours - with the given answers, I could achieve the time target.
