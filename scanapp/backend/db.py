"""
Local data model, following spec §4 exactly:

Product
ProductReferenceImage  (embedding_vector kept separate from image_path so the
                         ANN index (§3.4) can load vectors without touching
                         image blobs)
ScanLog
StockEvent (schema added in §7.1 — included here since it's the natural
            write-target for a confirmed scan; dashboard/analytics logic
            itself is out of scope for this pass)
"""

import sqlite3
import json
import uuid
import time
import threading
from contextlib import contextmanager

from . import config

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS product (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    brand TEXT,
    size_variant TEXT,
    price REAL,
    stock_qty INTEGER DEFAULT 0,
    qr_enabled INTEGER DEFAULT 0,
    last_image_update_at REAL
);

CREATE TABLE IF NOT EXISTS product_reference_image (
    id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES product(id) ON DELETE CASCADE,
    angle_label TEXT,                 -- front/back/left/right
    image_path TEXT,
    embedding_vector TEXT,            -- JSON list[float], quantized in production
    ocr_text_raw TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS scan_log (
    id TEXT PRIMARY KEY,
    timestamp REAL,
    matched_product_id TEXT,
    match_path TEXT,                  -- qr / auto_fused / candidate_pick / manual_new
    final_score REAL,
    resolved_by_user INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS stock_event (
    id TEXT PRIMARY KEY,
    product_id TEXT REFERENCES product(id),
    event_type TEXT,                  -- scan_in / scan_out / manual_adjustment / sale
    quantity_delta INTEGER,
    timestamp REAL,
    source TEXT                       -- scan / manual
);

CREATE INDEX IF NOT EXISTS idx_ref_image_product ON product_reference_image(product_id);
CREATE INDEX IF NOT EXISTS idx_scan_log_product ON scan_log(matched_product_id);
"""


def get_conn():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(config.DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA foreign_keys = ON")
    return _local.conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def cursor():
    conn = get_conn()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


# ---------- Product ----------

def add_product(name, brand="", size_variant="", price=0.0, qr_enabled=False):
    pid = str(uuid.uuid4())
    with cursor() as cur:
        cur.execute(
            "INSERT INTO product (id, name, brand, size_variant, price, stock_qty, qr_enabled, last_image_update_at) "
            "VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
            (pid, name, brand, size_variant, price, int(qr_enabled), time.time()),
        )
    return pid


def get_product(product_id):
    with cursor() as cur:
        cur.execute("SELECT * FROM product WHERE id = ?", (product_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def list_products():
    with cursor() as cur:
        cur.execute("SELECT * FROM product ORDER BY name")
        return [dict(r) for r in cur.fetchall()]


def adjust_stock(product_id, delta, event_type="scan_in", source="scan"):
    with cursor() as cur:
        cur.execute("UPDATE product SET stock_qty = stock_qty + ? WHERE id = ?", (delta, product_id))
        cur.execute(
            "INSERT INTO stock_event (id, product_id, event_type, quantity_delta, timestamp, source) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), product_id, event_type, delta, time.time(), source),
        )


# ---------- ProductReferenceImage ----------

def add_reference_image(product_id, angle_label, image_path, embedding_vector, ocr_text_raw=""):
    rid = str(uuid.uuid4())
    with cursor() as cur:
        cur.execute(
            "INSERT INTO product_reference_image "
            "(id, product_id, angle_label, image_path, embedding_vector, ocr_text_raw, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rid, product_id, angle_label, image_path, json.dumps(embedding_vector), ocr_text_raw, time.time()),
        )
        cur.execute("UPDATE product SET last_image_update_at = ? WHERE id = ?", (time.time(), product_id))
    return rid


def all_reference_vectors():
    """Returns list of (product_id, embedding_vector: list[float]) for the ANN/flat index (§3.4)."""
    with cursor() as cur:
        cur.execute("SELECT product_id, embedding_vector FROM product_reference_image")
        out = []
        for row in cur.fetchall():
            vec = json.loads(row["embedding_vector"]) if row["embedding_vector"] else []
            out.append((row["product_id"], vec))
        return out


def reference_ocr_texts():
    """Returns list of (product_id, ocr_text_raw) used for fuzzy-name matching (§3.2)."""
    with cursor() as cur:
        cur.execute(
            "SELECT DISTINCT p.id as product_id, p.name, p.brand "
            "FROM product p"
        )
        return [dict(r) for r in cur.fetchall()]


def find_product_by_qr_id(qr_payload):
    """QR payload is `shopid:sku_uuid` (§3.1) — here we just treat the UUID part as product id."""
    pid = qr_payload.split(":")[-1]
    return get_product(pid)


# ---------- ScanLog ----------

def log_scan(matched_product_id, match_path, final_score, resolved_by_user=False):
    sid = str(uuid.uuid4())
    with cursor() as cur:
        cur.execute(
            "INSERT INTO scan_log (id, timestamp, matched_product_id, match_path, final_score, resolved_by_user) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, time.time(), matched_product_id, match_path, final_score, int(resolved_by_user)),
        )
    return sid


def recent_scan_logs(limit=50):
    with cursor() as cur:
        cur.execute("SELECT * FROM scan_log ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]
