# orchestrator/db.py
import sqlite3, json

DB_FILE = "requests.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS request_logs (
        id TEXT PRIMARY KEY,
        image_urls TEXT NOT NULL,
        attributes TEXT NOT NULL,
        model_info TEXT NOT NULL,
        processing_time REAL NOT NULL,
        status TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def log_request(request_id, image_urls, attributes, model_info, processing_time, status):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO request_logs (id, image_urls, attributes, model_info, processing_time, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        request_id,
        json.dumps(image_urls),
        json.dumps(attributes),
        json.dumps(model_info),
        processing_time,
        status
    ))
    conn.commit()
    conn.close()
