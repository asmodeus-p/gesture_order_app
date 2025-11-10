# api/server.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from pathlib import Path
import sqlite3, json
from datetime import datetime

# Use the same DB file as your PyQt app
DB_PATH = Path(__file__).resolve().parent.parent / "orders.db"
KIOSK_PATH = Path(__file__).resolve().parent.parent / "kiosk"

api_app = FastAPI(title="Order API")

# ---------------- CORS ----------------
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Serve Kiosk Web App ----------------
api_app.mount("/assets", StaticFiles(directory=KIOSK_PATH / "assets"), name="assets")

@api_app.get("/")
def serve_kiosk():
    """Serve the kiosk index.html for the base URL"""
    return FileResponse(KIOSK_PATH / "index.html")


# --- Database setup ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_number TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL,
        items TEXT,
        total REAL DEFAULT 0.0
    )
    """)
    conn.commit()
    conn.close()

init_db()  # ensure schema exists

# --- Pydantic Models ---
class Item(BaseModel):
    name: str
    qty: int

class OrderRequest(BaseModel):
    order_number: str
    items: List[Item]
    total: float

# --- Helper to insert order ---
def add_order(order_number: str, items: List[dict], total: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()

    items_json = json.dumps(items, ensure_ascii=False)
    total_qty = sum(i.get("qty", 0) for i in items)

    c.execute("""
        INSERT INTO orders (order_number, status, created_at, items, total)
        VALUES (?, ?, ?, ?, ?)
    """, (order_number, "pending", now, items_json, total))
    conn.commit()
    conn.close()

# --- Routes ---
@api_app.post("/orders/add")
def create_order(order: OrderRequest):
    add_order(order.order_number, [i.dict() for i in order.items], order.total)
    return {
        "status": "ok",
        "order_number": order.order_number,
        "items_count": len(order.items)
    }
