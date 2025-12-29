from __future__ import annotations
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Annotated

import re
import requests  # type: ignore
from fastapi import Depends
from sqlalchemy import select, update, or_
from sqlalchemy.orm import Session

from src.common.logger import logger
from src.common.vector_store import vector_store
from src.db.database import Base, engine, get_db
from src.db.Models.product_models import Product

# ---------- служебные операции ----------

def create_db() -> str:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exp:
        if "already exists" in str(exp):
            logger.info("Database already exists")
            return "Database already exists"
        raise
    else:
        return "Database created successfully"

def drop_db() -> str:
    try:
        Base.metadata.drop_all(bind=engine)
    except Exception as exp:
        if "does not exist" in str(exp):
            logger.info("Database does not exist")
            return "Database does not exist"
        raise
    else:
        return "Database dropped successfully"

# ---------- загрузка/парсинг JSON ----------

def __get_json_from_url(
    address: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
) -> Any:
    resp = requests.get(address, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

def __extract_products_array(json_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not json_data or "Products" not in json_data:
        raise ValueError("Expected JSON object with key 'Products'")
    products = json_data["Products"]
    if not isinstance(products, list):
        raise ValueError("'Products' must be an array")
    return products

def __parse_flat_products(products: List[Dict[str, Any]]) -> List[Tuple[Optional[str], str, str]]:
    """
    -> [(external_id|None, name, price_str)] без преобразования цены.
    """
    out: List[Tuple[Optional[str], str, str]] = []
    for it in products:
        if not isinstance(it, dict):
            continue
        ext_id = (str(it.get("id", "")).strip() or None)
        name = str(it.get("name", "")).strip()
        price = str(it.get("price", "")).strip()
        if not name or price == "":
            continue
        out.append((ext_id, name, price))
    if not out:
        raise ValueError("No valid items in 'Products'")
    return out

# ---------- публичный импорт ----------

def update_db(
    db: Annotated[Session, Depends(get_db)],
    json_url: str = "",
    json_data: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Ожидает формат:
    { "Date": "...", "Products": [ { "id": "...", "name": "...", "price": "..." }, ... ] }
    Upsert по external_id (если есть) иначе по name. price сохраняется как строка.
    """
    data = json_data if json_data is not None else __get_json_from_url(json_url)
    items = __parse_flat_products(__extract_products_array(data))

    by_ext: Dict[str, Tuple[str, str]] = {}
    by_name: Dict[str, Tuple[str, str]] = {}
    for ext_id, name, price in items:
        if ext_id:
            by_ext[ext_id] = (name, price)
        else:
            by_name[name] = (name, price)

    inserted = updated = 0
    to_insert: List[Product] = []

    # existing by external_id
    if by_ext:
        ext_ids = list(by_ext.keys())
        rows_ext = db.execute(
            select(Product.id, Product.external_id).where(Product.external_id.in_(ext_ids))
        ).all()
        existing_by_ext = {row.external_id: row.id for row in rows_ext}  # type: ignore[attr-defined]
        for ext_id, (name, price) in by_ext.items():
            pid = existing_by_ext.get(ext_id)
            if pid:
                db.execute(update(Product).where(Product.id == pid).values(name=name, price=price))
                updated += 1
            else:
                to_insert.append(Product(external_id=ext_id, name=name, price=price))
                inserted += 1
    else:
        existing_by_ext = {}

    # existing by name (for items w/o external_id)
    if by_name:
        names = list(by_name.keys())
        rows_name = db.execute(
            select(Product.id, Product.name).where(Product.name.in_(names))
        ).all()
        existing_by_name = {row.name: row.id for row in rows_name}  # type: ignore[attr-defined]
        for name, (name_val, price) in by_name.items():
            pid = existing_by_name.get(name)
            if pid:
                db.execute(update(Product).where(Product.id == pid).values(price=price))
                updated += 1
            else:
                to_insert.append(Product(external_id=None, name=name_val, price=price))
                inserted += 1

    if to_insert:
        db.bulk_save_objects(to_insert)
    db.commit()

    total = inserted + updated
    logger.info("update_db: inserted=%s, updated=%s, total=%s", inserted, updated, total)

    # опционально — пересборка вектора по понедельникам 08–09
    now = datetime.now()
    if 1 == 1:
        try:
            names_all = get_all_products()
            msg = vector_store.rebuild_vector_store(products_names=names_all)
            logger.info("Vector store rebuilt: %s", msg)
        except Exception as e:
            logger.warning("Vector store rebuild failed: %s", e)

    return total

# ---------- утилиты ----------
MIN_TOKEN_LEN = 3

def _tokenize(q: str) -> List[str]:
    return [t for t in re.split(r"[^\wА-Яа-яЁё]+", q.lower()) if len(t) >= MIN_TOKEN_LEN]

def _best_match(query: str, candidates: List[Product]) -> Optional[Product]:
    if not candidates:
        return None
    scored = [(SequenceMatcher(None, query.lower(), p.name.lower()).ratio(), p) for p in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

def find_product_best(db: Session, query: str) -> Optional[Product]:
    p = db.scalar(select(Product).where(Product.name.ilike(f"%{query}%")))
    if p:
        return p
    tokens = _tokenize(query)
    if tokens:
        q = select(Product)
        for t in tokens:
            q = q.where(Product.name.ilike(f"%{t}%"))
        hits = db.scalars(q).all()
        if hits:
            return _best_match(query, hits)
        hits = db.scalars(select(Product).where(or_(*[Product.name.ilike(f"%{t}%") for t in tokens]))).all()
        if hits:
            return _best_match(query, hits)
    return None

def get_products_by_name(product_name: str) -> List[str]:
    db = next(get_db())
    rows = db.execute(
        select(Product.name).where(Product.name.ilike(f"%{product_name}%"))
    ).all()
    return [r[0] for r in rows]

def get_product_price_by_name(db: Session, product_name: str) -> Optional[dict]:
    p = find_product_best(db, product_name)
    if not p:
        return None
    return {"name": p.name, "external_id": p.external_id, "price": p.price}

def get_product_price(product_name: str) -> Optional[str]:
    db = next(get_db())
    row = db.execute(select(Product.price).where(Product.name == product_name)).first()
    return row[0] if row else None

def get_all_products() -> List[str]:
    db = next(get_db())
    rows = db.execute(select(Product.name)).all()
    return [r[0] for r in rows]
