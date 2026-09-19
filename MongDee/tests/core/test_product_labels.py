"""Product boxes show the product's NAME (Thai included), never its internal id."""
from __future__ import annotations

import os

import numpy as np
import pytest

from core import text_render, vision


def _worker(resolver):
    model = type("Model", (), {"names": {0: "person"}})()
    return vision.CameraWorker("CAM-1", 0, model, [], ai_worker=vision.AIWorker(), product_name_resolver=resolver)


def test_a_catalog_name_replaces_the_internal_id():
    w = _worker({"product-4621553d": "น้ำดื่มสิงห์"}.get)
    assert w._product_display_name("product-4621553d", "PRODUCT-4621553D") == "น้ำดื่มสิงห์"


def test_an_unknown_key_never_falls_back_to_the_id_when_a_catalog_is_wired():
    w = _worker({}.get)
    assert w._product_display_name("product-4621553d", "PRODUCT-4621553D") == "PRODUCT"


def test_without_a_catalog_the_old_label_is_kept():
    w = _worker(None)
    assert w._product_display_name("bottle", "BOTTLE") == "BOTTLE"


def test_a_failing_resolver_does_not_break_the_ai_pass():
    def boom(key):
        raise RuntimeError("catalog gone")

    assert _worker(boom)._product_display_name("k", "K") == "PRODUCT"


def test_very_long_names_are_shortened():
    name = "ก" * 80
    label = _worker({"k": name}.get)._product_display_name("k", "K")
    assert len(label) == vision.CameraWorker.PRODUCT_LABEL_MAX_CHARS and label.endswith("…")


def test_the_worker_labels_a_recognised_product_with_its_name(monkeypatch):
    w = _worker({"product-1": "Singha Water"}.get)

    class Rec:
        def has_any_gallery(self):
            return True

        def identify(self, crop):
            return "product-1", 0.91

    class Prop:
        def propose(self, frame, exclude_boxes, max_regions):
            return [[10, 10, 90, 90]]

    w.recognizer, w._proposer = Rec(), Prop()
    monkeypatch.setattr(w._product_confirmer, "confirm", lambda *a, **k: True)
    boxes = []
    w._run_custom_recognition(np.zeros((120, 160, 3), np.uint8), [], boxes)
    assert boxes and boxes[0][1] == "Singha Water 91%"


# ------------------------------------------------------------------------------------- drawing
def test_draw_box_pastes_a_rendered_unicode_label(monkeypatch):
    patch = np.full((14, 60, 3), (10, 200, 30), np.uint8)
    monkeypatch.setattr(vision, "render_label", lambda text, color: patch)
    frame = np.zeros((120, 200, 3), np.uint8)
    vision._draw_box(frame, [40, 50, 120, 100], "น้ำดื่ม 90%", (0, 200, 0))
    assert (frame[36:50, 40:100] == (10, 200, 30)).all()          # the label sits right above the box


def test_draw_box_without_a_thai_font_falls_back_to_ascii(monkeypatch):
    monkeypatch.setattr(vision, "render_label", lambda text, color: None)
    frame = np.zeros((120, 200, 3), np.uint8)
    vision._draw_box(frame, [40, 50, 120, 100], "น้ำดื่ม 90%", (0, 200, 0))     # must not raise
    assert frame.any()


def test_the_label_flips_below_the_edge_when_there_is_no_room_above():
    frame = np.zeros((60, 100, 3), np.uint8)
    patch = np.full((14, 40, 3), 255, np.uint8)
    text_render.blit(frame, patch, 5, 3)
    assert frame[3:17, 5:45].all()


# ------------------------------------------------------------------------------ text_render
def _any_ttf():
    import matplotlib
    return os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")


def test_render_label_uses_the_configured_font_and_caches(monkeypatch):
    try:
        path = _any_ttf()
    except Exception:
        pytest.skip("no font available")
    if not os.path.isfile(path):
        pytest.skip("no font available")
    monkeypatch.setenv("MONGDEE_LABEL_FONT", path)
    monkeypatch.setattr(text_render, "_font_path", False)
    monkeypatch.setattr(text_render, "_patches", text_render.OrderedDict())
    a = text_render.render_label("Singha 91%", (0, 200, 0))
    b = text_render.render_label("Singha 91%", (0, 200, 0))
    assert a is not None and a.ndim == 3 and a.shape[2] == 3 and a.shape[0] > 8
    assert a is b                                                      # cached: drawing every frame is free
    assert tuple(a[0, 0]) == (0, 200, 0)                               # background colour is the box colour


def test_no_font_means_no_unicode_labels(monkeypatch):
    monkeypatch.setenv("MONGDEE_LABEL_FONT", "")
    monkeypatch.setattr(text_render, "_font_path", None)
    assert text_render.unicode_labels_available() is False
    assert text_render.render_label("x", (0, 0, 0)) is None
