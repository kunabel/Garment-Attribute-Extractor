flowchart LR
  Client[(Client)]
  subgraph ComposeHost[Docker Compose Host]
    Orchestrator[Orchestrator
    FastAPI :8000]
    Vision[Vision Service 
    FastAPI :8001
    CLIP on CPU]
    Heuristic[Heuristic Service
    FastAPI :8002
    PaddleOCR CPU]
    LLM[LLM Service
    FastAPI :8003]
    Common[CommonUtils
    shared lib]
    subgraph Vols[Volumes]
      ODB[(orchestrator/requests.db)]
      HDB[(heuristic/requests.db)]
      VDB[(vision/requests.db)]
      LDB[(llm/requests.db)]
      OCRCache[(ocr model cache)]
    end
  end

  Client -->|HTTP JSON| Orchestrator
  Orchestrator -->|/classify| Vision
  Orchestrator -->|/extract| Heuristic
  Orchestrator -->|/describe| LLM

  Vision -->|log_request| VDB
  Heuristic -->|log_request| HDB
  LLM -->|log_request| LDB
  Orchestrator -->|log_request| ODB

  Heuristic -.->|loads OCR models| OCRCache
  Vision -.->|preloads CLIP| Common
  Heuristic -.->|uses| Common
  LLM -.->|uses| Common
  Orchestrator -.->|uses| Common

sequenceDiagram
  participant C as Client
  participant O as Orchestrator
  participant V as Vision
  participant H as Heuristic
  participant L as LLM

  C->>O: POST /v1/items/analyze {image_urls[4]}
  par Fan-out
    O->>V: POST /classify {image_urls}
    O->>H: POST /extract {image_urls}
    O->>L: POST /describe {image_urls}
  end
  V-->>O: 200 {attributes, model_info} or error
  H-->>O: 200 {attributes, model_info, confidence} or error
  L-->>O: 200 {attributes, model_info} or error

  O->>O: merge partials, set status=success|degraded
  O->>ODB: INSERT request_log
  O-->>C: 200 {id, attributes, model_info, errors, processing}
