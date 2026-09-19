import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

from app import config, db, enroll_sessions
from app.camera import CameraManager, CameraSnapshot, PresenceState
from app.matching import MatchIndex
from app.schemas import EnrollConfirmRequest


class EnrollmentFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_enrollment_requires_empty_platform_reference(self):
        # Enrollment now embeds with the booth background masked out, so it
        # needs the same empty-platform reference recognition uses.
        from app.routers import enrollment
        camera = CameraManager()
        frame = np.random.default_rng(2).integers(30, 220, (80, 100, 3), dtype=np.uint8)
        camera._snapshot = CameraSnapshot(frame, PresenceState.EMPTY, 0, time.time(), captured_at=time.monotonic())
        with patch.object(enrollment.camera_module, "camera_manager", camera), \
             patch.object(camera, "capture_burst_frames", return_value=[frame] * 3) as capture, \
             patch.object(enrollment.vision.feature_extractor, "embed_many", return_value=[np.ones(704)] * 3):
            self.assertFalse(camera.has_reference)
            with self.assertRaises(HTTPException) as ctx:
                await enrollment._capture_batch()
            self.assertEqual(ctx.exception.status_code, 400)
            capture.assert_not_called()

            camera._reference_frame = frame.copy()
            frames, embeddings = await enrollment._capture_batch()
            capture.assert_called_once_with(require_present=False)
            self.assertEqual(len(frames), 3)
            self.assertEqual(len(embeddings), 3)

    async def test_scan_confirm_appends_catalog_and_preserves_existing_product(self):
        from app.routers import enrollment
        frame = np.full((80, 100, 3), 120, dtype=np.uint8)
        vector = np.zeros(704, dtype=np.float32)
        vector[0] = 1
        with tempfile.TemporaryDirectory() as directory, patch.multiple(config, DATA_DIR=Path(directory), CAPTURES_DIR=Path(directory) / "captures", DB_PATH=Path(directory) / "test.db"), patch.object(enroll_sessions, "_sessions", {}), patch.object(enrollment.vision, "match_index", MatchIndex(704)), patch.object(enrollment, "_capture_batch", return_value=([frame] * 3, [vector] * 3)):
            db.init_db()
            existing = db.create_product("Existing", None, None, None, None, None, None, None)
            batch = await enrollment.start_enrollment()
            saved = enrollment.confirm_enrollment(batch.session_id, EnrollConfirmRequest(name="Scanned product", selected_frame_ids=[item.frame_id for item in batch.frames]))
            self.assertEqual(db.get_product(existing)["name"], "Existing")
            self.assertEqual(len(db.list_products()), 2)
            self.assertEqual(len(db.all_embeddings()), 3)
            self.assertEqual(enrollment.vision.match_index.match_burst([vector] * 5).ranked[0][0], saved.id)
            self.assertIsNone(enroll_sessions.get_session(batch.session_id))
            self.assertTrue((Path(directory) / saved.thumbnail_path).exists())
