"""
Local data model, following spec §4, extended with a `category` column on
Product and a Sale / SaleItem pair for the consumer-GUI checkout flow.

Product
ProductReferenceImage  (embedding_vector kept separate from image_path so the
                         ANN index (§3.4) can load vectors without touching
                         image blobs)
ScanLog
StockEvent (§7.1 — write-target for confirmed scans and sales)
Sale / SaleItem — one checkout = one Sale + N SaleItem rows; finalizing a
                  checkout also writes matching StockEvent rows so both
                  views of "what happened" stay in sync.
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
    price REAL,                       -- selling price (SP)
    cost_price REAL,                  -- cost price (CP) — profit per unit = price - cost_price
    stock_qty INTEGER DEFAULT 0,
    qr_enabled INTEGER DEFAULT 0,
    category TEXT,
    barcode TEXT,                     -- real-world EAN-13/UPC-A/Code128/... value, distinct from the internal QR sticker payload
    last_image_update_at REAL
);

CREATE TABLE IF NOT EXISTS product_reference_image (
    id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES product(id) ON DELETE CASCADE,
    angle_label TEXT,                 -- front/back/left/right
    image_path TEXT,
    embedding_vector TEXT,            -- JSON list[float], quantized in production
    ocr_text_raw TEXT,                -- concatenated block text, used for fuzzy name matching
    ocr_blocks_json TEXT,             -- full OCR output for this image: JSON list[{text, conf, bbox}]
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

CREATE TABLE IF NOT EXISTS sale (
    id TEXT PRIMARY KEY,
    timestamp REAL,
    total REAL,
    item_count INTEGER,
    product_names TEXT,                -- comma-separated product names in this sale, for a quick glance without a join
    discount_amount REAL DEFAULT 0     -- rupees knocked off the pre-discount subtotal to reach `total`
);

CREATE TABLE IF NOT EXISTS sale_item (
    id TEXT PRIMARY KEY,
    sale_id TEXT NOT NULL REFERENCES sale(id) ON DELETE CASCADE,
    product_id TEXT REFERENCES product(id),
    name_snapshot TEXT,               -- copy of product name at sale time
    price_snapshot REAL,              -- copy of unit price at sale time
    quantity INTEGER
);

CREATE INDEX IF NOT EXISTS idx_ref_image_product ON product_reference_image(product_id);
CREATE INDEX IF NOT EXISTS idx_scan_log_product ON scan_log(matched_product_id);
CREATE INDEX IF NOT EXISTS idx_sale_item_sale ON sale_item(sale_id);
"""
# Note: no CREATE INDEX on product(barcode) here — on a DB that already
# had a product table before this feature was added, CREATE TABLE IF NOT
# EXISTS above is a no-op, so an index statement in the same script would
# run against the old table before _migrate() below gets a chance to add
# the column, and blow up with "no such column: barcode". _migrate()
# creates that index instead, after it's confirmed the column exists.


def get_conn():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(config.DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA foreign_keys = ON")
    return _local.conn


def _migrate(conn):
    """Add columns/tables introduced after a DB may already exist on disk."""
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(product)").fetchall()]
    if "category" not in cols:
        conn.execute("ALTER TABLE product ADD COLUMN category TEXT")
    if "barcode" not in cols:
        conn.execute("ALTER TABLE product ADD COLUMN barcode TEXT")
    if "cost_price" not in cols:
        conn.execute("ALTER TABLE product ADD COLUMN cost_price REAL")
    # Safe unconditionally: only reaches here once the column above is
    # guaranteed to exist, whether this DB was just created fresh or just
    # migrated from an older version.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_product_barcode ON product(barcode)")

    ref_cols = [r["name"] for r in conn.execute("PRAGMA table_info(product_reference_image)").fetchall()]
    if "ocr_blocks_json" not in ref_cols:
        conn.execute("ALTER TABLE product_reference_image ADD COLUMN ocr_blocks_json TEXT")

    sale_cols = [r["name"] for r in conn.execute("PRAGMA table_info(sale)").fetchall()]
    if "product_names" not in sale_cols:
        conn.execute("ALTER TABLE sale ADD COLUMN product_names TEXT")
    if "discount_amount" not in sale_cols:
        conn.execute("ALTER TABLE sale ADD COLUMN discount_amount REAL DEFAULT 0")


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    _migrate(conn)
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

def add_product(name, brand="", size_variant="", price=0.0, qr_enabled=False,
                 category="", initial_stock=0, barcode="", cost_price=0.0):
    pid = str(uuid.uuid4())
    with cursor() as cur:
        cur.execute(
            "INSERT INTO product (id, name, brand, size_variant, price, cost_price, stock_qty, qr_enabled, category, barcode, last_image_update_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, name, brand, size_variant, price, cost_price, int(initial_stock), int(qr_enabled), category, barcode or None, time.time()),
        )
    if initial_stock:
        with cursor() as cur:
            cur.execute(
                "INSERT INTO stock_event (id, product_id, event_type, quantity_delta, timestamp, source) "
                "VALUES (?, ?, 'manual_adjustment', ?, ?, 'manual')",
                (str(uuid.uuid4()), pid, int(initial_stock), time.time()),
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


def set_stock(product_id, quantity, source="manual"):
    """Set stock to an absolute value (rather than a relative delta), still logging
    the resulting change as a stock_event so history stays consistent."""
    product = get_product(product_id)
    if not product:
        return None
    delta = int(quantity) - int(product.get("stock_qty") or 0)
    with cursor() as cur:
        cur.execute("UPDATE product SET stock_qty = ? WHERE id = ?", (int(quantity), product_id))
        if delta:
            cur.execute(
                "INSERT INTO stock_event (id, product_id, event_type, quantity_delta, timestamp, source) "
                "VALUES (?, ?, 'manual_adjustment', ?, ?, ?)",
                (str(uuid.uuid4()), product_id, delta, time.time(), source),
            )
    return get_product(product_id)


_PRODUCT_EDITABLE_FIELDS = {"name", "brand", "size_variant", "price", "qr_enabled", "category", "barcode", "cost_price"}


def update_product(product_id, **fields):
    """Partial update of any editable product field. Ignores unknown/None fields.
    Does NOT touch stock_qty -- use adjust_stock/set_stock for that."""
    updates = {k: v for k, v in fields.items() if k in _PRODUCT_EDITABLE_FIELDS and v is not None}
    if not updates:
        return get_product(product_id)
    if "qr_enabled" in updates:
        updates["qr_enabled"] = int(bool(updates["qr_enabled"]))
    if "price" in updates:
        updates["price"] = float(updates["price"])
    if "cost_price" in updates:
        updates["cost_price"] = float(updates["cost_price"])
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with cursor() as cur:
        cur.execute(f"UPDATE product SET {set_clause} WHERE id = ?", (*updates.values(), product_id))
    return get_product(product_id)


def delete_product(product_id):
    """Deletes a product. stock_event and sale_item rows reference
    product(id) without ON DELETE CASCADE/SET NULL, and foreign_keys is
    enforced on this connection, so a straight DELETE on a product with
    any stock or sale history (i.e. almost any real product) previously
    failed with a foreign key constraint violation, which surfaced to
    the shopkeeper as "item cannot be deleted". Null out those
    historical rows' product_id first so the audit trail (what sold,
    what stock moved) survives, then the product itself, all in one
    transaction."""
    with cursor() as cur:
        cur.execute("UPDATE stock_event SET product_id = NULL WHERE product_id = ?", (product_id,))
        cur.execute("UPDATE sale_item SET product_id = NULL WHERE product_id = ?", (product_id,))
        cur.execute("UPDATE scan_log SET matched_product_id = NULL WHERE matched_product_id = ?", (product_id,))
        cur.execute("DELETE FROM product WHERE id = ?", (product_id,))
    return {"ok": True}


def find_products_by_name(query):
    """Loose name/brand search, used by the AI chat tool layer to resolve a
    product mentioned by name into an id before acting on it."""
    q = f"%{(query or '').lower()}%"
    with cursor() as cur:
        cur.execute(
            "SELECT * FROM product WHERE lower(name) LIKE ? OR lower(brand) LIKE ? ORDER BY name",
            (q, q),
        )
        return [dict(r) for r in cur.fetchall()]


# ---------- ProductReferenceImage ----------

def add_reference_image(product_id, angle_label, image_path, embedding_vector, ocr_text_raw="", ocr_blocks=None):
    """ocr_blocks: optional list[{text, conf, bbox}] — the raw per-block OCR
    output for this specific image (see ocr_extractor.extract_text_blocks),
    stored as JSON alongside ocr_text_raw (which is just those blocks'
    text concatenated, kept for fuzzy name matching)."""
    rid = str(uuid.uuid4())
    with cursor() as cur:
        cur.execute(
            "INSERT INTO product_reference_image "
            "(id, product_id, angle_label, image_path, embedding_vector, ocr_text_raw, ocr_blocks_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, product_id, angle_label, image_path, json.dumps(embedding_vector), ocr_text_raw,
             json.dumps(ocr_blocks or []), time.time()),
        )
        cur.execute("UPDATE product SET last_image_update_at = ? WHERE id = ?", (time.time(), product_id))
    return rid


def reference_images_for_product(product_id):
    """Returns each stored reference image's path plus its saved OCR results
    (both the concatenated text and the full per-block JSON) so a caller can
    inspect exactly what OCR read off that specific photo."""
    with cursor() as cur:
        cur.execute(
            "SELECT id, angle_label, image_path, ocr_text_raw, ocr_blocks_json, created_at "
            "FROM product_reference_image WHERE product_id = ? ORDER BY created_at",
            (product_id,),
        )
        out = []
        for row in cur.fetchall():
            d = dict(row)
            d["ocr_blocks"] = json.loads(d.pop("ocr_blocks_json") or "[]")
            out.append(d)
        return out


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
    """Returns product dicts (id, name, brand) used for fuzzy-name matching (§3.2)."""
    with cursor() as cur:
        cur.execute(
            "SELECT DISTINCT p.id as id, p.name, p.brand "
            "FROM product p"
        )
        return [dict(r) for r in cur.fetchall()]


def find_product_by_qr_id(qr_payload):
    """QR payload is `shopid:sku_uuid` (§3.1) — here we just treat the UUID part as product id."""
    pid = qr_payload.split(":")[-1]
    return get_product(pid)


def find_product_by_barcode(barcode_payload):
    """Looks up a product by a real-world barcode value (EAN-13/UPC-A/
    Code128/...), as opposed to find_product_by_qr_id which decodes our
    own internal `shopid:sku_uuid` QR sticker scheme. Used for both the
    checkout scanner and the "scan a barcode" step when adding a product."""
    if not barcode_payload:
        return None
    with cursor() as cur:
        cur.execute("SELECT * FROM product WHERE barcode = ?", (barcode_payload,))
        row = cur.fetchone()
        return dict(row) if row else None


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


# ---------- Sale / checkout ----------

def create_sale(cart_items, discount_amount=0.0):
    """
    cart_items: list of {"product_id", "quantity"} (quantity is however many
    units the shopkeeper scanned/added during this checkout).

    discount_amount: flat rupees knocked off the pre-discount subtotal
    (the frontend converts a percentage discount to a flat amount before
    calling this, since we only ever want one number of truth stored).

    Writes one Sale row (including a comma-separated snapshot of the
    product names in it, for a quick glance without a join, and the
    discount applied), one SaleItem row per line, decrements stock, and
    writes a matching StockEvent per line so ScanLog/StockEvent stay the
    single source of truth for "what happened" while Sale/SaleItem is the
    friendlier read-shape for an invoice.

    Returns the invoice dict: {id, timestamp, items, subtotal, discount_amount, total}.
    """
    sale_id = str(uuid.uuid4())
    ts = time.time()
    subtotal = 0.0
    line_items = []

    with cursor() as cur:
        cur.execute(
            "INSERT INTO sale (id, timestamp, total, item_count, product_names, discount_amount) VALUES (?, ?, 0, 0, '', ?)",
            (sale_id, ts, float(discount_amount or 0)),
        )
        for entry in cart_items:
            product = get_product(entry["product_id"])
            if not product:
                continue
            qty = int(entry.get("quantity", 1))
            if qty <= 0:
                continue
            price = float(product.get("price") or 0.0)
            line_total = price * qty
            subtotal += line_total

            cur.execute(
                "INSERT INTO sale_item (id, sale_id, product_id, name_snapshot, price_snapshot, quantity) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), sale_id, product["id"], product["name"], price, qty),
            )
            cur.execute("UPDATE product SET stock_qty = stock_qty - ? WHERE id = ?", (qty, product["id"]))
            cur.execute(
                "INSERT INTO stock_event (id, product_id, event_type, quantity_delta, timestamp, source) "
                "VALUES (?, ?, 'sale', ?, ?, 'scan')",
                (str(uuid.uuid4()), product["id"], -qty, ts),
            )
            line_items.append({
                "product_id": product["id"], "name": product["name"],
                "price": price, "quantity": qty, "line_total": line_total,
            })

        discount_amount = min(float(discount_amount or 0), subtotal)  # never let a discount push the total negative
        total = subtotal - discount_amount
        product_names = ", ".join(li["name"] for li in line_items)

        cur.execute(
            "UPDATE sale SET total = ?, item_count = ?, product_names = ?, discount_amount = ? WHERE id = ?",
            (total, sum(li["quantity"] for li in line_items), product_names, discount_amount, sale_id),
        )

    return {
        "id": sale_id, "timestamp": ts, "items": line_items,
        "subtotal": subtotal, "discount_amount": discount_amount, "total": total,
        "product_names": product_names,
    }


def today_sales_summary():
    """Revenue + units sold since local midnight, for the Inventory tab's sales strip."""
    midnight = time.time() - (time.time() % 86400)  # coarse UTC-day boundary; fine for a shop-hours summary
    with cursor() as cur:
        cur.execute(
            "SELECT COALESCE(SUM(total),0) as revenue, COALESCE(SUM(item_count),0) as units "
            "FROM sale WHERE timestamp >= ?",
            (midnight,),
        )
        row = cur.fetchone()
        return {"revenue": row["revenue"], "units": row["units"]}


def recent_sales(limit=20):
    with cursor() as cur:
        cur.execute("SELECT * FROM sale ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]
