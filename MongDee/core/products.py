"""Product catalog + simple FAQ matching for the AI Product Assistant.

Two kinds of entries live side by side, keyed by an arbitrary product_key:
  - demo entries whose key happens to match a YOLO11/COCO class name
    (bottle, cup, ...) — recognized instantly with zero setup;
  - custom entries trained from real product photos/video via the AI
    Trainer (see core/recognizer.py) — recognized by image-embedding
    matching instead of a fixed class name.
core/vision.py decides which detection path applies per product; this
module just stores the product info and answers FAQ questions either way.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent.parent / "products.json"


class ProductCatalog:
    def __init__(self, path: Path = DEFAULT_CATALOG_PATH):
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._comment = raw.pop("_comment", None)
        self.products = raw

    def class_names(self) -> list[str]:
        """All product keys in the catalog (kept for backward compatibility;
        prefer product_keys())."""
        return list(self.products.keys())

    def product_keys(self) -> list[str]:
        return list(self.products.keys())

    def get(self, class_name: str) -> dict | None:
        return self.products.get(class_name)

    def add_product(self, key: str, name: str, tagline: str = "", description: str = "",
                     faq: list[dict] | None = None, price: str = "") -> None:
        """Register a new (or update an existing) product and persist to disk —
        used by the AI Trainer when someone trains a brand-new product, and to
        edit an existing product's details afterwards (upsert by key)."""
        self.products[key] = {
            "name": name,
            "tagline": tagline,
            "price": price,
            "description": description,
            "faq": faq or [],
        }
        self.save()

    def remove_product(self, key: str) -> None:
        self.products.pop(key, None)
        self.save()

    def save(self) -> None:
        payload = {"_comment": self._comment, **self.products} if self._comment else self.products
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def answer_question(self, class_name: str, question: str) -> str:
        product = self.get(class_name)
        if not product:
            return "ขอโทษค่ะ ยังไม่มีข้อมูลสินค้านี้ในระบบ"

        faq = product.get("faq", [])
        if not question or not question.strip():
            return product.get("description", "")

        question_norm = question.strip().lower()
        best_score = 0.0
        best_answer = None
        for entry in faq:
            score = difflib.SequenceMatcher(None, question_norm, entry["q"].lower()).ratio()
            question_words = set(question_norm.split())
            faq_words = set(entry["q"].lower().split())
            if question_words and faq_words:
                overlap = len(question_words & faq_words) / len(question_words | faq_words)
                score = max(score, overlap)
            if score > best_score:
                best_score = score
                best_answer = entry["a"]

        if best_answer and best_score >= 0.3:
            return best_answer

        return (
            "ขอโทษค่ะ ทีมงานยังไม่ได้เตรียมคำตอบสำหรับคำถามนี้ไว้ "
            "กรุณาสอบถามเจ้าหน้าที่ประจำบูธเพิ่มเติมค่ะ"
        )
