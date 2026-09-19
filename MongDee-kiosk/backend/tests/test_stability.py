"""Regression cases for unstable recognition. No real camera/data writes."""
import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np

from app import config, db
from app.camera import CameraManager, CameraSnapshot, PresenceState, build_masked_crop, select_quality_frames
from app.matching import MatchIndex, MatchResult
from app.schemas import CalibrationSettings


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.index = MatchIndex(3)
        self.index.rebuild([(1, [1, 0, 0]), (2, [0, 1, 0])])

    def test_single_lucky_frame_cannot_win(self):
        result = self.index.match_burst([[1, 0, 0]] + [[0, 0.9, 0.3]] * 9)
        self.assertEqual(result.ranked[0][0], 2)
        self.assertEqual(result.status, "matched")

    def test_mixed_products_are_not_auto_matched(self):
        result = self.index.match_burst([[1, 0, 0]] * 5 + [[0, 1, 0]] * 5)
        self.assertNotEqual(result.status, "matched")

    def test_majority_without_enough_agreement_is_ambiguous(self):
        result = self.index.match_burst([[1, 0, 0]] * 6 + [[0, 1, 0]] * 4)
        self.assertNotEqual(result.status, "matched")

    def test_single_frame_is_never_auto_confirmed(self):
        self.assertEqual(self.index.match_burst([[1, 0, 0]]).status, "unstable")

    def test_near_tie_below_threshold_is_ambiguous(self):
        result = MatchResult([(1, 0.63), (2, 0.61)], 10, 1.0, 1.0)
        self.assertEqual(result.status, "ambiguous")

    def test_invalid_and_nonunit_embeddings(self):
        self.index.rebuild([(1, [9, 0, 0]), (2, [float("nan"), 0, 0]), (3, [0, 0, 0]), (4, [1])])
        result = self.index.match_burst([[4, 0, 0]] * 5)
        self.assertEqual(result.ranked, [(1, 1.0)])
        self.assertEqual(result.status, "matched")

    def test_no_catalog_is_unknown(self):
        self.index.rebuild([])
        self.assertEqual(self.index.match_burst([[1, 0, 0]] * 5).status, "unknown")

    def test_duplicate_templates_do_not_add_votes(self):
        self.index.rebuild([(1, [1, 0, 0])] * 100 + [(2, [0, 1, 0])])
        self.assertEqual(self.index.match_burst([[0, 1, 0]] * 5).ranked[0][0], 2)

    def test_invalid_live_vectors_cannot_inflate_frame_count(self):
        result = self.index.match_burst([[1, 0, 0]] + [[float("inf"), 0, 0]] * 5)
        self.assertEqual(result.frame_count, 1)
        self.assertEqual(result.status, "unstable")


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.roi_patch = patch.object(config, "ROI_FRACTION", (0.2, 0.15, 0.6, 0.75))
        self.roi_patch.start()
        self.addCleanup(self.roi_patch.stop)
        timing = patch.multiple(config, PRESENCE_ON_HOLD_S=0, PRESENCE_OFF_HOLD_S=0)
        timing.start()
        self.addCleanup(timing.stop)
        self.camera = CameraManager()
        self.background = np.full((120, 160, 3), 90, dtype=np.uint8)
        self.product = self.background.copy()
        self.product[35:90, 60:110] = 200
        self.camera._reference_frame = self.background.copy()

    def feed(self, frame, count=20):
        for _ in range(count):
            self.camera._process_frame(frame.copy())
        return self.camera.snapshot()

    def test_global_exposure_change_stays_empty(self):
        snap = self.feed(self.background + 35)
        self.assertEqual(snap.state, PresenceState.EMPTY)
        self.assertLess(snap.foreground_ratio, config.PRESENCE_OFF_RATIO)

    def test_exposure_compensation_does_not_erase_product(self):
        self.assertEqual(self.feed(self.product + 30).state, PresenceState.PRESENT)

    def test_stationary_product_is_never_learned_as_background(self):
        self.assertEqual(self.feed(self.product, 60).state, PresenceState.PRESENT)
        np.testing.assert_array_equal(self.camera._reference_frame, self.background)

    def test_brief_hand_does_not_trigger_presence(self):
        self.feed(self.background)
        self.feed(self.product, 1)
        self.assertEqual(self.feed(self.background).state, PresenceState.EMPTY)

    def test_high_frame_rate_still_requires_real_hold_time(self):
        with patch.object(config, "PRESENCE_ON_HOLD_S", 0.45), patch("app.camera.time.monotonic", return_value=100):
            self.assertEqual(self.feed(self.product, 30).state, PresenceState.EMPTY)
        with patch.object(config, "PRESENCE_ON_HOLD_S", 0.45), patch("app.camera.time.monotonic", return_value=100.5):
            self.assertEqual(self.feed(self.product, 1).state, PresenceState.PRESENT)

    def test_removal_changes_generation_and_returns_empty(self):
        before = self.feed(self.product)
        after = self.feed(self.background)
        self.assertEqual(after.state, PresenceState.EMPTY)
        self.assertGreater(after.generation, before.generation)

    def test_roi_move_same_size_clears_temporal_buffer(self):
        self.feed(self.product)
        generation = self.camera.snapshot().generation
        with patch.object(config, "ROI_FRACTION", (0.0, 0.0, 0.6, 0.75)):
            self.camera._process_frame(self.background)
        self.assertEqual(len(self.camera._frame_buffer), 1)
        self.assertGreater(self.camera.snapshot().generation, generation)

    def test_no_implicit_reference_from_occupied_first_frame(self):
        self.camera._reference_frame = None
        self.feed(self.product)
        self.assertFalse(self.camera.has_reference)
        self.assertEqual(self.camera.snapshot().state, PresenceState.EMPTY)

    def test_resolution_change_requires_new_reference(self):
        self.camera._process_frame(np.zeros((60, 80, 3), dtype=np.uint8))
        self.assertFalse(self.camera.has_reference)

    def test_reference_is_camera_specific_and_median_filtered(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "DATA_DIR", Path(directory)):
            self.feed(self.background)
            self.assertTrue(self.camera.capture_reference_now())
            other = CameraManager(1)
            other._load_reference_from_disk()
            self.assertFalse(other.has_reference)
            same = CameraManager(0)
            same._load_reference_from_disk()
            np.testing.assert_array_equal(same._reference_frame, self.background)

    def test_stale_frame_is_not_a_burst_or_stream(self):
        self.camera._snapshot = CameraSnapshot(self.product, PresenceState.PRESENT, 0.2, time.time(), captured_at=time.monotonic() - 10)
        self.assertEqual(self.camera.capture_burst_frames(), [])
        self.assertIsNone(self.camera.latest_jpeg())

    def test_frozen_camera_frame_not_duplicated(self):
        self.feed(self.product)
        with patch.multiple(config, BURST_DURATION_S=0.16, CAPTURE_SETTLE_S=0, BURST_SAMPLE_INTERVAL_S=0.01), patch("app.camera.select_quality_frames", side_effect=lambda f, roi: f):
            self.assertEqual(len(self.camera.capture_burst_frames()), 1)

    def test_generation_change_discards_whole_burst(self):
        self.feed(self.product)
        initial = self.camera.snapshot()
        removed = CameraSnapshot(self.background, PresenceState.EMPTY, 0, time.time(), initial.generation + 1, time.monotonic())
        with patch.object(self.camera, "snapshot", side_effect=[initial, initial, removed]):
            self.assertEqual(self.camera.capture_burst_frames(), [])

    def test_driver_repeating_identical_pixels_cannot_supply_multiple_votes(self):
        from dataclasses import replace
        self.feed(self.product)
        initial = self.camera.snapshot()
        with patch.object(self.camera, "snapshot", side_effect=lambda: replace(initial, captured_at=time.monotonic())), patch.multiple(config, BURST_DURATION_S=0.16, CAPTURE_SETTLE_S=0, BURST_SAMPLE_INTERVAL_S=0.01), patch("app.camera.select_quality_frames", side_effect=lambda frames, roi: frames):
            self.assertEqual(len(self.camera.capture_burst_frames()), 1)

    def test_manual_capture_accepts_presence_transition_without_reference(self):
        self.camera._reference_frame = None
        counter = 0
        def snapshot():
            nonlocal counter
            counter += 1
            return CameraSnapshot(self.background + counter, PresenceState.PRESENT if counter > 2 else PresenceState.EMPTY,
                                  0.2, time.time(), 2 if counter > 2 else 1, time.monotonic())
        with patch.object(self.camera, "snapshot", side_effect=snapshot), patch.multiple(config, BURST_DURATION_S=0.16, CAPTURE_SETTLE_S=0, BURST_SAMPLE_INTERVAL_S=0.01), patch("app.camera.select_quality_frames", side_effect=lambda frames, roi: frames):
            self.assertGreaterEqual(len(self.camera.capture_burst_frames(require_present=False)), 2)

    def test_masked_crop_suppresses_background_keeps_product(self):
        with patch.object(config, "ROI_FRACTION", (0.0, 0.0, 1.0, 1.0)):
            bg = np.random.default_rng(1).integers(40, 210, (120, 160, 3), dtype=np.uint8)
            frame = bg.copy()
            frame[40:90, 55:105] = 255  # bright product blob on the platform
            out = build_masked_crop(frame, bg, (0.0, 0.0, 1.0, 1.0))
            # Product pixels are preserved; a background corner is pushed away
            # from its original noisy value toward the blurred fill.
            self.assertGreater(float(out[60, 80].mean()), 230)
            self.assertGreater(
                float(np.mean(np.abs(out[:15, :15].astype(int) - frame[:15, :15].astype(int)))), 3)
            # No reference -> plain crop, unchanged.
            np.testing.assert_array_equal(build_masked_crop(frame, None, (0.0, 0.0, 1.0, 1.0)), frame)

    def test_blur_and_black_frames_filtered(self):
        rng = np.random.default_rng(5)
        sharp = rng.integers(30, 220, (120, 160, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(sharp, (31, 31), 10)
        frames = select_quality_frames([sharp, blurred, np.zeros_like(sharp)])
        self.assertEqual(len(frames), 1)
        self.assertIs(frames[0], sharp)


class CalibrationTests(unittest.TestCase):
    def test_invalid_roi_and_thresholds_rejected(self):
        valid = dict(roi=dict(x=0.2, y=0.2, w=0.5, h=0.5), presence_on_ratio=0.12, presence_off_ratio=0.04)
        for roi in [dict(x=0, y=0, w=0, h=1), dict(x=0.8, y=0, w=0.5, h=1), dict(x=float("nan"), y=0, w=1, h=1)]:
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                CalibrationSettings(**{**valid, "roi": roi})
        with self.assertRaises(ValueError):
            CalibrationSettings(**{**valid, "presence_off_ratio": 0.2})


class ScanTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from app import recognition_loop as loop
        # Scan code reads product rows; keep it off the project's real database.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        db_patch = patch.multiple(config, DATA_DIR=base, CAPTURES_DIR=base / "captures", DB_PATH=base / "test.db")
        db_patch.start()
        self.addCleanup(db_patch.stop)
        db.init_db()
        self.loop = loop
        self.camera = CameraManager()
        frame = np.random.default_rng(7).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        self.camera._reference_frame = frame
        self.camera._snapshot = CameraSnapshot(frame, PresenceState.PRESENT, 0.3, time.time(), 10, time.monotonic())
        self.frames = [frame] * 6
        self.state_patch = patch.object(loop, "recognition_state", loop.RecognitionState())
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)

    async def test_removal_during_inference_never_logs_result(self):
        def embed(frames):
            self.camera._snapshot = CameraSnapshot(None, PresenceState.EMPTY, 0, time.time(), 11)
            return [np.ones(self.loop.vision.feature_extractor.output_dim)] * 6
        with patch.object(self.camera, "capture_burst_frames", return_value=self.frames), patch.object(self.loop.vision.feature_extractor, "embed_many", side_effect=embed), patch.object(self.loop.db, "log_scan_event") as log:
            self.assertFalse(await self.loop._run_scan(self.camera))
            log.assert_not_called()

    async def test_bad_capture_never_logs_unknown_or_creates_enrollment(self):
        with patch.object(self.camera, "capture_burst_frames", return_value=[]), patch.object(self.loop.db, "log_scan_event") as log, patch.object(self.loop.enroll_sessions, "create_session") as session:
            self.assertFalse(await self.loop._run_scan(self.camera))
            log.assert_not_called()
            session.assert_not_called()
            self.assertEqual((await self.loop.recognition_state.get()).status, "unstable")

    async def test_catalog_changed_during_inference_discards_result(self):
        version = self.loop.vision.match_index.version
        with patch.object(self.camera, "capture_burst_frames", return_value=self.frames), patch.object(self.loop.vision.feature_extractor, "embed_many", return_value=[np.ones(self.loop.vision.feature_extractor.output_dim)] * 6), patch.object(self.loop.vision.match_index.__class__, "version", new_callable=unittest.mock.PropertyMock, side_effect=[version, version, version + 1]), patch.object(self.loop.db, "log_scan_event") as log:
            self.assertFalse(await self.loop._run_scan(self.camera))
            log.assert_not_called()

    async def test_retries_are_bounded_for_one_placement(self):
        calls = 0
        async def tick(_):
            nonlocal calls
            calls += 1
            if calls >= 12:
                raise asyncio.CancelledError
        with patch.object(config, "SCAN_RETRY_INTERVAL_S", 0), patch.object(self.loop, "_run_scan", new_callable=AsyncMock, return_value=False) as scan, patch.object(self.loop.asyncio, "sleep", side_effect=tick):
            with self.assertRaises(asyncio.CancelledError):
                await self.loop.recognition_loop(self.camera)
            self.assertEqual(scan.await_count, 1 + config.SCAN_MAX_RETRIES)

    async def test_current_api_hides_match_when_camera_goes_stale(self):
        from app.routers import recognition
        from app.schemas import RecognitionResult
        await self.loop.recognition_state.set(RecognitionResult(status="matched", camera_generation=10, updated_at=time.time()))
        self.camera._snapshot = CameraSnapshot(None, PresenceState.PRESENT, 0.3, time.time(), 10)
        with patch.object(recognition, "camera_manager", self.camera), patch.object(recognition, "recognition_state", self.loop.recognition_state):
            self.assertEqual((await recognition.current()).status, "camera_unavailable")

    async def test_manual_correction_updates_shared_state_without_fake_confidence(self):
        from app.routers import recognition
        from app.schemas import RecognitionResult, CorrectionRequest
        await self.loop.recognition_state.set(RecognitionResult(status="ambiguous", scan_event_id=7, camera_generation=10, updated_at=time.time()))
        product = dict(id=1, name="Test", created_at=0)
        with patch.object(recognition, "camera_manager", self.camera), patch.object(recognition, "recognition_state", self.loop.recognition_state), patch.object(recognition.db, "get_product", return_value=product), patch.object(recognition.db, "correct_scan_event") as correct:
            await recognition.correct(CorrectionRequest(scan_event_id=7, product_id=1))
            current = await self.loop.recognition_state.get()
            self.assertTrue(current.manually_confirmed)
            self.assertEqual(current.product.id, 1)
            self.assertIsNone(current.confidence)
            correct.assert_called_once_with(7, 1)

    async def test_shown_match_holds_until_a_second_scan_confirms_the_switch(self):
        from app.schemas import ProductOut, RecognitionResult
        dim = self.loop.vision.feature_extractor.output_dim
        await self.loop.recognition_state.set(RecognitionResult(
            status="matched", product=ProductOut(id=1, name="A", created_at=0),
            camera_generation=10, updated_at=time.time()))
        self.loop._clear_pending_switch()
        with patch.object(self.camera, "capture_burst_frames", return_value=self.frames), \
             patch.object(self.loop.vision.feature_extractor, "embed_many", return_value=[np.ones(dim)] * 6), \
             patch.object(self.loop.vision.match_index, "match_burst", return_value=MatchResult([(2, 0.9)], 6, 1.0, 1.0)), \
             patch.object(self.loop.db, "get_product", return_value=dict(id=2, name="B", created_at=0)), \
             patch.object(self.loop.db, "log_scan_event", return_value=1) as log:
            self.assertFalse(await self.loop._run_scan(self.camera))
            log.assert_not_called()
            self.assertEqual((await self.loop.recognition_state.get()).product.id, 1)
            self.assertTrue(await self.loop._run_scan(self.camera))
            self.assertEqual((await self.loop.recognition_state.get()).product.id, 2)
        self.loop._clear_pending_switch()

    async def test_batch_embeddings_remain_compatible_with_existing_extractor(self):
        extractor = self.loop.vision.feature_extractor
        frame = self.frames[0]
        single = await asyncio.to_thread(extractor.embed, frame)
        batch = await asyncio.to_thread(extractor.embed_many, [frame, frame])
        np.testing.assert_allclose(batch[0], single, atol=1e-5)
        np.testing.assert_allclose(batch[1], single, atol=1e-5)


class PersistenceTests(unittest.TestCase):
    def test_only_current_embedding_version_is_matchable(self):
        from app import db
        with tempfile.TemporaryDirectory() as directory, patch.multiple(
            config, DATA_DIR=Path(directory), CAPTURES_DIR=Path(directory) / "captures", DB_PATH=Path(directory) / "test.db"
        ):
            db.init_db()
            old = db.create_product("old", *[None] * 7, embeddings=[[1, 0]], embedding_version=1)
            new = db.create_product("new", *[None] * 7, embeddings=[[0, 1]], embedding_version=config.EMBED_VERSION)
            matchable = {row["product_id"] for row in db.all_embeddings()}
            self.assertEqual(matchable, {new})
            self.assertEqual(db.stale_product_ids(), {old})
            db.add_embedding(old, [0.5, 0.5], None)  # a fresh re-scan clears the flag
            self.assertEqual(db.stale_product_ids(), set())

    def test_failed_embedding_write_rolls_back_whole_product(self):
        from app import db
        with tempfile.TemporaryDirectory() as directory, patch.multiple(config, DATA_DIR=Path(directory), CAPTURES_DIR=Path(directory) / "captures", DB_PATH=Path(directory) / "test.db"):
            db.init_db()
            with self.assertRaises(TypeError):
                db.create_product("test", None, None, None, None, None, None, None, embeddings=[[1, 0], [object()]])
            self.assertEqual(db.list_products(), [])
            self.assertEqual(db.all_embeddings(), [])
            pid = db.create_product("test", None, None, None, None, None, None, None, embeddings=[[1, 0], [0, 1]])
            self.assertEqual(len(db.all_embeddings()), 2)
            self.assertEqual(db.get_product(pid)["name"], "test")


if __name__ == "__main__":
    unittest.main()
