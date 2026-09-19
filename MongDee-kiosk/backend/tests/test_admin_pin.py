import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app import admin_auth, config, db


def code_of(ctx):
    return ctx.exception.detail["code"]


class AdminPinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.patch = patch.multiple(config, DATA_DIR=base, CAPTURES_DIR=base / "captures", DB_PATH=base / "test.db")
        self.patch.start()
        db.init_db()
        admin_auth.reset_lockout()

    def tearDown(self):
        admin_auth.reset_lockout()
        self.patch.stop()
        self.tmp.cleanup()

    def test_no_pin_means_deleting_is_blocked_until_one_is_set(self):
        self.assertFalse(admin_auth.pin_is_set())
        with self.assertRaises(HTTPException) as ctx:
            admin_auth.require_pin("1234")
        self.assertEqual((ctx.exception.status_code, code_of(ctx)), (403, "pin_not_set"))

    def test_pin_is_stored_hashed_and_checked(self):
        admin_auth.set_pin("4821")
        stored = db.get_setting(admin_auth.PIN_KEY)
        self.assertTrue(stored.startswith("scrypt$"))
        self.assertNotIn("4821", stored, "the PIN itself is never stored")
        admin_auth.require_pin("4821")  # right PIN passes silently
        with self.assertRaises(HTTPException) as ctx:
            admin_auth.require_pin("0000")
        self.assertEqual((ctx.exception.status_code, code_of(ctx)), (401, "pin_invalid"))
        for missing in (None, ""):
            with self.assertRaises(HTTPException):
                admin_auth.require_pin(missing)

    def test_pin_format_is_four_to_twelve_digits(self):
        for bad in ("123", "1234567890123", "12ab", "12 34", ""):
            with self.assertRaises(HTTPException) as ctx:
                admin_auth.set_pin(bad)
            self.assertEqual(code_of(ctx), "pin_format", bad)
        for good in ("1234", "123456789012"):
            admin_auth.set_pin(good)

    def test_repeated_wrong_guesses_lock_the_check_for_a_minute(self):
        admin_auth.set_pin("4821")
        for i in range(admin_auth.MAX_FAILURES):
            with self.assertRaises(HTTPException):
                admin_auth.check_pin("0000", now=100.0 + i)
        with self.assertRaises(HTTPException) as ctx:
            admin_auth.check_pin("4821", now=110.0)  # even the right PIN is refused while locked
        self.assertEqual((ctx.exception.status_code, code_of(ctx)), (429, "pin_locked"))
        admin_auth.check_pin("4821", now=104.0 + admin_auth.LOCK_S + 1)  # lock has expired

    def test_a_correct_pin_clears_earlier_mistakes(self):
        admin_auth.set_pin("4821")
        for i in range(admin_auth.MAX_FAILURES - 1):
            with self.assertRaises(HTTPException):
                admin_auth.check_pin("0000", now=10.0 + i)
        admin_auth.check_pin("4821", now=20.0)
        for i in range(admin_auth.MAX_FAILURES - 1):
            with self.assertRaises(HTTPException) as ctx:
                admin_auth.check_pin("0000", now=30.0 + i)
            self.assertEqual(ctx.exception.status_code, 401, "not locked: the earlier mistakes were forgiven")

    def test_first_pin_can_be_set_freely_but_changing_needs_the_current_one(self):
        self.assertEqual(admin_auth.pin_status(), {"is_set": False})
        admin_auth.change_pin(admin_auth.PinBody(new_pin="1111"))
        self.assertEqual(admin_auth.pin_status(), {"is_set": True})
        with self.assertRaises(HTTPException) as ctx:
            admin_auth.change_pin(admin_auth.PinBody(new_pin="2222"))
        self.assertEqual(ctx.exception.status_code, 401)
        with self.assertRaises(HTTPException):
            admin_auth.change_pin(admin_auth.PinBody(new_pin="2222", current_pin="9999"))
        admin_auth.change_pin(admin_auth.PinBody(new_pin="2222", current_pin="1111"))
        admin_auth.require_pin("2222")
        with self.assertRaises(HTTPException):
            admin_auth.require_pin("1111")

    def test_clearing_the_pin_allows_a_new_one_to_be_chosen(self):
        admin_auth.set_pin("4821")
        admin_auth.clear_pin()
        self.assertFalse(admin_auth.pin_is_set())
        admin_auth.change_pin(admin_auth.PinBody(new_pin="7777"))
        admin_auth.require_pin("7777")


class GuardCoverageTests(unittest.TestCase):
    """Fails if someone removes the PIN requirement from a route that deletes data."""

    def test_every_data_deleting_route_requires_the_pin_header(self):
        from app.main import app
        paths = app.openapi()["paths"]
        must_be_guarded = [
            ("delete", "/api/products/{product_id}"),
            ("delete", "/api/products/{product_id}/video"),
            ("delete", "/api/products/{product_id}/images/{image_id}"),
        ]
        for method, path in must_be_guarded:
            names = [p["name"] for p in paths[path][method].get("parameters", [])]
            self.assertIn("x-admin-pin", names, f"{method.upper()} {path} must require the admin PIN")


if __name__ == "__main__":
    unittest.main()
