# api/server.py
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("../orders.db")  # same DB as your PyQt app

api_app = FastAPI(title="Order API")

# Pydantic model
class OrderRequest(BaseModel):
    order_number: str
    items: list[str] = []
    quantities: list[int] = []

# Helper to add order
def add_order(order_number, items, quantities):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO orders (order_number, status, created_at) VALUES (?, ?, ?)",
        (order_number, "pending", now)
    )
    conn.commit()
    conn.close()

@api_app.post("/orders/add")
def create_order(order: OrderRequest):
    add_order(order.order_number, order.items, order.quantities)
    return {"status": "ok", "order_number": order.order_number}
