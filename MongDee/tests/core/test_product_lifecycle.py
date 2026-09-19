"""Ghost products: a product deleted from the catalog must stop being matched, must not be
resurrected by a running import, and stale galleries on disk must be detectable/removable.

The state below is taken from the real project: products.json had 3 products with no gallery
while data/gallery still held product-4621553d.npy (30 samples) + manifest entry."""
import json
import threading
import time

import numpy as np
import pytest

from core.product_lifecycle import ProductDeleted, load_product_keys, reconcile
from core.recognizer import ProductRecognizer


def _catalog(path, keys):
    path.write_text(json.dumps({"_comment": "demo", **{k: {"name": k} for k in keys}}))


def _real_project_state(tmp_path):
    prod = tmp_path / "products.json"
    _catalog(prod, ["product-4b11ff33", "product-227cc848", "product-15bda15f"])
    gal = tmp_path / "gallery"; gal.mkdir()
    (gal / "manifest.json").write_text(json.dumps({"product-4621553d": {"count": 30, "updated_at": 1.0}}))
    np.save(gal / "product-4621553d.npy", np.ones((30, 4)))
    np.save(gal / "_calibration_mean.npy", np.ones(4))
    return prod, gal


# ------------------------------------------------------------------ offline audit
def test_loader_ignores_metadata_keys_and_accepts_other_shapes(tmp_path):
    p = tmp_path / "p.json"
    _catalog(p, ["a", "b"]); assert load_product_keys(str(p)) == {"a", "b"}
    p.write_text(json.dumps([{"key": "a"}, {"id": "b"}, "c"])); assert load_product_keys(str(p)) == {"a", "b", "c"}
    assert load_product_keys(str(tmp_path / "nope.json")) == set()


def test_reconcile_finds_the_real_ghost_and_dry_run_changes_nothing(tmp_path):
    prod, gal = _real_project_state(tmp_path)
    r = reconcile(str(prod), str(gal))
    assert r["orphans"] == ["product-4621553d"]
    assert sorted(r["no_gallery"]) == ["product-15bda15f", "product-227cc848", "product-4b11ff33"]
    assert (gal / "product-4621553d.npy").exists() and r["quarantined"] == []


def test_reconcile_apply_quarantines_never_deletes_and_is_idempotent(tmp_path):
    prod, gal = _real_project_state(tmp_path)
    r = reconcile(str(prod), str(gal), apply=True)
    assert json.loads((gal / "manifest.json").read_text()) == {}
    assert not (gal / "product-4621553d.npy").exists()
    assert len(r["quarantined"]) == 1 and r["quarantined"][0].endswith("product-4621553d.npy")
    assert (gal / "_calibration_mean.npy").exists()                     # calibration file untouched
    assert reconcile(str(prod), str(gal))["orphans"] == []


def test_reconcile_detects_npy_without_manifest_entry(tmp_path):
    prod, gal = _real_project_state(tmp_path)
    (gal / "manifest.json").write_text("{}")
    assert reconcile(str(prod), str(gal))["orphans"] == ["product-4621553d"]


# ------------------------------------------------------------------ runtime guard
def _recognizer(tmp_path, monkeypatch, live):
    rec = ProductRecognizer(gallery_dir=tmp_path, device="cpu")
    monkeypatch.setattr(rec, "_centered", lambda v: v)
    monkeypatch.setattr(rec, "embed", lambda img: np.array([1.0, 0.0]))
    rec.set_active_provider(lambda: set(live))
    return rec


def _img():
    return np.zeros((10, 10, 3), np.uint8)


def test_ghost_gallery_is_never_matched_even_if_it_is_still_on_disk(tmp_path, monkeypatch):
    live = {"real-product"}
    rec = _recognizer(tmp_path, monkeypatch, live)
    rec._gallery = {"ghost": np.array([[1.0, 0.0]]), "real-product": np.array([[0.0, 1.0]])}
    assert rec.identify(_img()) == (None, pytest.approx(0.0))          # query matches the ghost only
    live.add("ghost")
    assert rec.identify(_img())[0] == "ghost"                           # sanity: matches once it is live


def test_has_any_gallery_and_sample_counts_ignore_inactive(tmp_path, monkeypatch):
    rec = _recognizer(tmp_path, monkeypatch, {"a"})
    rec._gallery = {"ghost": np.ones((3, 2))}; rec._manifest = {"ghost": {"count": 3}}
    assert rec.has_any_gallery() is False and rec.sample_counts() == {}


def test_prune_inactive_quarantines_files_and_updates_manifest(tmp_path, monkeypatch):
    _, gal = _real_project_state(tmp_path)
    rec = _recognizer(gal, monkeypatch, {"product-4b11ff33"})
    assert "product-4621553d" in rec._gallery                           # it really was loaded from disk
    assert rec.prune_inactive() == ["product-4621553d"]
    assert "product-4621553d" not in rec._gallery
    assert json.loads((gal / "manifest.json").read_text()) == {}
    assert not (gal / "product-4621553d.npy").exists()
    assert any(gal.glob("_quarantine/*/product-4621553d.npy"))
    assert rec.prune_inactive() == []


def test_add_sample_refused_for_product_not_in_catalog(tmp_path, monkeypatch):
    rec = _recognizer(tmp_path, monkeypatch, {"real"})
    with pytest.raises(ProductDeleted):
        rec.add_sample("deleted-one", _img())
    assert "deleted-one" not in rec._gallery and not list(tmp_path.glob("deleted-one*"))
    assert rec.add_sample("real", _img()) == 1


def test_delete_during_running_import_does_not_resurrect_gallery(tmp_path, monkeypatch):
    """Mirrors api_delete_product (catalog removal FIRST, then clear_product) racing an import."""
    live = {"p1"}
    rec = _recognizer(tmp_path, monkeypatch, live)
    started = threading.Event(); stopped = threading.Event(); added = []

    def slow_embed(img):
        time.sleep(0.002)
        return np.array([1.0, 0.0])
    monkeypatch.setattr(rec, "embed", slow_embed)

    def importer():
        try:
            for i in range(5000):
                rec.add_sample("p1", _img()); added.append(i)
                if i == 5: started.set()
        except ProductDeleted:
            pass
        finally:
            stopped.set()

    t = threading.Thread(target=importer); t.start()
    assert started.wait(10)
    live.discard("p1")                     # catalog.remove_product
    rec.clear_product("p1")                # recognizer.clear_product
    assert stopped.wait(10); t.join()
    assert "p1" not in rec._gallery and "p1" not in rec._manifest and not list(tmp_path.glob("p1*.npy"))
    assert len(added) < 5000


def test_concurrent_delete_and_adds_stress(tmp_path, monkeypatch):
    for round_ in range(15):
        sub = tmp_path / f"r{round_}"; sub.mkdir()
        live = {"p"}
        rec = _recognizer(sub, monkeypatch, live)
        def hammer():
            for _ in range(100):
                try: rec.add_sample("p", _img())
                except ProductDeleted: return
        ts = [threading.Thread(target=hammer) for _ in range(3)]
        [t.start() for t in ts]; time.sleep(0.003)
        live.discard("p"); rec.clear_product("p")
        [t.join() for t in ts]
        assert "p" not in rec._gallery and not list(sub.glob("p.npy"))


def test_without_provider_behaviour_is_unchanged(tmp_path, monkeypatch):
    rec = ProductRecognizer(gallery_dir=tmp_path, device="cpu")
    monkeypatch.setattr(rec, "embed", lambda img: np.array([1.0, 0.0]))
    assert rec.add_sample("anything", _img()) == 1 and rec.has_any_gallery()
