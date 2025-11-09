# api/server.py
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
from pathlib import Path
from datetime import datetime

# Use the same DB file as your PyQt app
DB_PATH = Path(__file__).resolve().parent.parent / "orders.db"

api_app = FastAPI(title="Order API")

# Initialize database if needed
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_number TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL
    )
    """)
    # Add missing columns (safe migration)
    c.execute("PRAGMA table_info(orders);")
    cols = [r[1] for r in c.fetchall()]
    if "items" not in cols:
        c.execute("ALTER TABLE orders ADD COLUMN items TEXT;")
    if "qty" not in cols:
        c.execute("ALTER TABLE orders ADD COLUMN qty INTEGER DEFAULT 0;")
    if "total" not in cols:
        c.execute("ALTER TABLE orders ADD COLUMN total REAL DEFAULT 0.0;")
    conn.commit()
    conn.close()

init_db()  # ensure schema is ready

# --- API models ---
class OrderRequest(BaseModel):
    order_number: str
    items: list[str] = []
    quantities: list[int] = []

# --- Helper to insert order ---
def add_order(order_number, items, quantities):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    # Convert lists to text for storage
    items_str = ", ".join(items)
    qty = sum(quantities) if quantities else 0
    total = 0.0  # You can compute real total later
    c.execute(
        "INSERT INTO orders (order_number, status, created_at, items, qty, total) VALUES (?, ?, ?, ?, ?, ?)",
        (order_number, "pending", now, items_str, qty, total)
    )
    conn.commit()
    conn.close()

# --- Routes ---
@api_app.post("/orders/add")
def create_order(order: OrderRequest):
    add_order(order.order_number, order.items, order.quantities)
    return {"status": "ok", "order_number": order.order_number}
