import io
import json
import re
import uuid
import zipfile

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app import config, db, vision
from app.admin_auth import require_pin
from app.camera import imwrite_unicode
from app.schemas import GalleryOrder, ProductOut, ProductUpdate

router = APIRouter(prefix="/api/products", tags=["products"])

_COVER_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
_COVER_MAX_BYTES = 8 * 1024 * 1024
_COVER_MAX_DIM = 1600

GALLERY_MAX_IMAGES = 8  # matches the single thumbnail row in the popup

_VIDEO_EXTENSIONS = {"video/mp4": ".mp4", "video/webm": ".webm", "video/ogg": ".ogv"}
_VIDEO_MAX_BYTES = 200 * 1024 * 1024


@router.get("", response_model=list[ProductOut])
def list_products():
    stale = db.stale_product_ids()
    return [ProductOut(**db.product_out_fields(r), needs_reembed=r["id"] in stale) for r in db.list_products()]


@router.get("/{product_id}", response_model=ProductOut)
def get_product(product_id: int):
    row = db.get_product(product_id)
    if not row:
        raise HTTPException(404, "Product not found")
    return ProductOut(**db.product_out_fields(row))


def _safe_stem(name: str) -> str:
    """Filesystem-safe stem for a zip entry/download name, kept short and
    ASCII-only since Thai product names would otherwise need RFC 5987
    encoding to survive every browser's Content-Disposition parsing."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    return stem or "product"


# image_path on product_embeddings is only ever populated by future capture
# runs (see db.add_embedding) — enrollment today leaves it NULL, so a
# product's exportable image set is whatever of these three fields is set.
_EXPORT_IMAGE_FIELDS = ("thumbnail_path", "cover_image_path")
_EXPORT_VIDEO_FIELDS = ("video_path",)


def _add_product_to_zip(zf: zipfile.ZipFile, row, folder: str = "") -> None:
    product = dict(row)
    prefix = f"{folder}/" if folder else ""
    zf.writestr(f"{prefix}product.json", json.dumps(product, ensure_ascii=False, indent=2))
    for field in (*_EXPORT_IMAGE_FIELDS, *_EXPORT_VIDEO_FIELDS):
        rel_path = product.get(field)
        if not rel_path:
            continue
        disk_path = config.DATA_DIR / rel_path
        if not disk_path.is_file():
            continue
        arcname = f"{prefix}{field}{disk_path.suffix}"
        zf.write(disk_path, arcname)
    for number, image in enumerate(db.list_product_images(product["id"]), start=1):
        disk_path = config.DATA_DIR / image["path"]
        if disk_path.is_file():
            zf.write(disk_path, f"{prefix}gallery/{number:02d}{disk_path.suffix}")


@router.get("/export/all")
def export_all_products():
    """Same bundle as /{product_id}/export but for every product at once,
    each in its own subfolder so filenames never collide across products.
    Registered ahead of /{product_id}/export so "export" is never parsed
    as a product_id path parameter."""
    rows = db.list_products()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for row in rows:
            product = dict(row)
            folder = f"{_safe_stem(product['name'])}-{product['id']}"
            _add_product_to_zip(zf, row, folder=folder)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="mongdee-products-export.zip"'},
    )


@router.get("/{product_id}/export")
def export_product(product_id: int):
    """Bundles a product's metadata plus every image/video file it has
    (thumbnail, cover photo, video) into one downloadable zip — the export
    format the user actually wants is "whatever works", so one archive with
    a machine-readable JSON alongside the raw media covers json/png/jpg/
    video in a single click instead of forcing five separate downloads."""
    row = db.get_product(product_id)
    if not row:
        raise HTTPException(404, "Product not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        _add_product_to_zip(zf, row)
    buf.seek(0)

    filename = f"{_safe_stem(row['name'])}-{product_id}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.patch("/{product_id}", response_model=ProductOut)
def patch_product(product_id: int, body: ProductUpdate):
    if not db.get_product(product_id):
        raise HTTPException(404, "Product not found")
    fields = body.model_dump(exclude_unset=True)
    if fields:
        db.update_product(product_id, **fields)
    # Metadata edits don't touch embeddings, so the match index is unaffected.
    return ProductOut(**db.product_out_fields(db.get_product(product_id)))


@router.post("/{product_id}/cover", response_model=ProductOut)
async def upload_cover_image(product_id: int, file: UploadFile = File(...)):
    """A polished photo for the kiosk display — deliberately separate from
    the raw webcam training frames, which are picked for AI recognition
    quality (varied angles, plain background) rather than looking good on
    screen."""
    if not db.get_product(product_id):
        raise HTTPException(404, "Product not found")
    if file.content_type not in _COVER_CONTENT_TYPES:
        raise HTTPException(400, "รองรับเฉพาะไฟล์ภาพ JPEG/PNG/WEBP")

    raw = await file.read()
    if len(raw) > _COVER_MAX_BYTES:
        raise HTTPException(400, "ไฟล์ใหญ่เกินไป (จำกัด 8MB)")

    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "ไฟล์ภาพไม่ถูกต้องหรือเปิดไม่ได้")

    h, w = img.shape[:2]
    if max(h, w) > _COVER_MAX_DIM:
        scale = _COVER_MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    covers_dir = config.CAPTURES_DIR / "covers"
    covers_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.jpg"
    if not imwrite_unicode(covers_dir / filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise HTTPException(500, "บันทึกภาพปกไม่สำเร็จ กรุณาลองใหม่")

    db.set_cover_image(product_id, f"captures/covers/{filename}")
    return ProductOut(**db.product_out_fields(db.get_product(product_id)))


def _decode_photo(raw: bytes, content_type: str):
    if content_type not in _COVER_CONTENT_TYPES:
        raise HTTPException(400, "รองรับเฉพาะไฟล์ภาพ JPEG/PNG/WEBP")
    if len(raw) > _COVER_MAX_BYTES:
        raise HTTPException(400, "ไฟล์ใหญ่เกินไป (จำกัด 8MB ต่อรูป)")
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "ไฟล์ภาพไม่ถูกต้องหรือเปิดไม่ได้")
    h, w = img.shape[:2]
    if max(h, w) > _COVER_MAX_DIM:
        scale = _COVER_MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


@router.post("/{product_id}/images", response_model=ProductOut)
async def add_gallery_images(product_id: int, files: list[UploadFile] = File(...)):
    """Adds hand-picked photos to the popup gallery. Every file is validated
    before anything is saved, so a bad file never leaves a half-added batch."""
    if not db.get_product(product_id):
        raise HTTPException(404, "Product not found")
    existing = len(db.list_product_images(product_id))
    if existing + len(files) > GALLERY_MAX_IMAGES:
        raise HTTPException(400, f"เพิ่มรูปได้สูงสุด {GALLERY_MAX_IMAGES} รูปต่อสินค้า (ตอนนี้มี {existing} รูป)")

    decoded = [_decode_photo(await f.read(), f.content_type) for f in files]

    gallery_dir = config.CAPTURES_DIR / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for img in decoded:
        filename = f"{uuid.uuid4().hex}.jpg"
        if not imwrite_unicode(gallery_dir / filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
            raise HTTPException(500, "บันทึกภาพไม่สำเร็จ กรุณาลองใหม่")
        db.add_product_image(product_id, f"captures/gallery/{filename}")
    return ProductOut(**db.product_out_fields(db.get_product(product_id)))


@router.put("/{product_id}/images/order", response_model=ProductOut)
def reorder_gallery_images(product_id: int, body: GalleryOrder):
    if not db.get_product(product_id):
        raise HTTPException(404, "Product not found")
    current = [row["id"] for row in db.list_product_images(product_id)]
    if sorted(body.ids) != sorted(current):
        raise HTTPException(400, "รายการรูปไม่ตรงกับที่มีอยู่ กรุณารีเฟรชหน้าแล้วลองใหม่")
    db.reorder_product_images(product_id, body.ids)
    return ProductOut(**db.product_out_fields(db.get_product(product_id)))


@router.delete("/{product_id}/images/{image_id}", response_model=ProductOut, dependencies=[Depends(require_pin)])
def delete_gallery_image(product_id: int, image_id: int):
    image = db.get_product_image(image_id)
    if not image or image["product_id"] != product_id:
        raise HTTPException(404, "Image not found")
    db.delete_product_image(image_id)
    (config.DATA_DIR / image["path"]).unlink(missing_ok=True)
    return ProductOut(**db.product_out_fields(db.get_product(product_id)))


@router.post("/{product_id}/video", response_model=ProductOut)
async def upload_product_video(product_id: int, file: UploadFile = File(...)):
    """A short product video for the kiosk display — shown autoplaying in
    place of the cover photo whenever one's been uploaded (see kiosk.js)."""
    if not db.get_product(product_id):
        raise HTTPException(404, "Product not found")
    ext = _VIDEO_EXTENSIONS.get(file.content_type)
    if ext is None:
        raise HTTPException(400, "รองรับเฉพาะไฟล์วิดีโอ MP4/WEBM/OGG")

    raw = await file.read()
    if len(raw) > _VIDEO_MAX_BYTES:
        raise HTTPException(400, "ไฟล์ใหญ่เกินไป (จำกัด 200MB)")

    videos_dir = config.CAPTURES_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}{ext}"
    (videos_dir / filename).write_bytes(raw)

    db.set_product_video(product_id, f"captures/videos/{filename}")
    return ProductOut(**db.product_out_fields(db.get_product(product_id)))


@router.delete("/{product_id}/video", response_model=ProductOut, dependencies=[Depends(require_pin)])
def remove_product_video(product_id: int):
    """Reverts the kiosk display back to the cover photo/thumbnail."""
    if not db.get_product(product_id):
        raise HTTPException(404, "Product not found")
    db.set_product_video(product_id, None)
    return ProductOut(**db.product_out_fields(db.get_product(product_id)))


@router.delete("/{product_id}", dependencies=[Depends(require_pin)])
def delete_product(product_id: int):
    row = db.get_product(product_id)
    if not row:
        raise HTTPException(404, "Product not found")
    for image in db.list_product_images(product_id):
        (config.DATA_DIR / image["path"]).unlink(missing_ok=True)
    db.delete_product(product_id)
    vision.refresh_match_index()
    return {"ok": True}
