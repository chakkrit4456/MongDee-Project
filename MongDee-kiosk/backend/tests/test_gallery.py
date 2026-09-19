import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from app import config, db
from app.routers import products
from app.schemas import GalleryOrder


def photo(content_type="image/jpeg", size=(60, 80), name="a.jpg"):
    img = np.random.default_rng(size[0]).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return UploadFile(file=io.BytesIO(buf.tobytes()), filename=name, headers=Headers({"content-type": content_type}))


def fake(content_type="image/jpeg", data=b"not an image"):
    return UploadFile(file=io.BytesIO(data), filename="x.jpg", headers=Headers({"content-type": content_type}))


class GalleryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.patch = patch.multiple(config, DATA_DIR=base, CAPTURES_DIR=base / "captures", DB_PATH=base / "test.db")
        self.patch.start()
        db.init_db()
        self.pid = db.create_product("สินค้าทดสอบ", None, None, None, None, None, None, None)

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    async def test_add_lists_in_order_and_saves_files(self):
        out = await products.add_gallery_images(self.pid, [photo(size=(60, 80)), photo(size=(70, 90))])
        self.assertEqual(len(out.gallery), 2)
        for image in out.gallery:
            self.assertTrue((config.DATA_DIR / image.path).is_file())
        out = await products.add_gallery_images(self.pid, [photo(size=(50, 50))])
        self.assertEqual(len(out.gallery), 3)

    async def test_limit_of_eight_is_enforced_without_partial_saves(self):
        await products.add_gallery_images(self.pid, [photo(size=(40 + i, 40)) for i in range(6)])
        with self.assertRaises(HTTPException) as ctx:
            await products.add_gallery_images(self.pid, [photo(size=(60 + i, 40)) for i in range(3)])
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(len(db.list_product_images(self.pid)), 6)

    async def test_one_bad_file_rejects_the_whole_batch(self):
        for bad in (fake(), fake(content_type="application/pdf")):
            with self.assertRaises(HTTPException) as ctx:
                await products.add_gallery_images(self.pid, [photo(), bad])
            self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(db.list_product_images(self.pid), [])
        gallery_dir = config.CAPTURES_DIR / "gallery"
        self.assertFalse(gallery_dir.exists() and any(gallery_dir.iterdir()), "nothing written for a rejected batch")

    async def test_reorder_and_delete(self):
        out = await products.add_gallery_images(self.pid, [photo(size=(40 + i, 40)) for i in range(3)])
        ids = [g.id for g in out.gallery]
        out = products.reorder_gallery_images(self.pid, GalleryOrder(ids=[ids[2], ids[0], ids[1]]))
        self.assertEqual([g.id for g in out.gallery], [ids[2], ids[0], ids[1]])
        with self.assertRaises(HTTPException):
            products.reorder_gallery_images(self.pid, GalleryOrder(ids=[ids[0]]))

        gone = out.gallery[0]
        out = products.delete_gallery_image(self.pid, gone.id)
        self.assertEqual(len(out.gallery), 2)
        self.assertFalse((config.DATA_DIR / gone.path).exists(), "file is removed with its row")
        with self.assertRaises(HTTPException) as ctx:
            products.delete_gallery_image(self.pid, gone.id)
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_cannot_delete_another_products_image(self):
        other = db.create_product("อีกชิ้น", None, None, None, None, None, None, None)
        out = await products.add_gallery_images(other, [photo()])
        with self.assertRaises(HTTPException) as ctx:
            products.delete_gallery_image(self.pid, out.gallery[0].id)
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_deleting_a_product_removes_its_gallery_files(self):
        out = await products.add_gallery_images(self.pid, [photo()])
        path = config.DATA_DIR / out.gallery[0].path
        with patch.object(products.vision, "refresh_match_index"):
            products.delete_product(self.pid)
        self.assertFalse(path.exists())
        self.assertEqual(db.list_product_images(self.pid), [])


if __name__ == "__main__":
    unittest.main()
