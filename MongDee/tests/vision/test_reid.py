import numpy as np
import pytest

from vision.reid.backends import ColorHistogramBackend, build_backend
from vision.reid.config import ReIDConfig
from vision.reid.extractor import ReIDExtractor, TrackEmbeddingAggregator, cosine_similarity, l2_normalize
from vision.reid.quality import assess_crop


def _person(h=240, w=90, bgr=(180, 60, 40)):
    img = np.zeros((h + 40, w + 40, 3), np.uint8)
    img[20:20 + h, 20:20 + w] = bgr
    # add a little texture so it isn't perfectly flat (blur check)
    img[20:20 + h:4, 20:20 + w] = (bgr[0] // 2, bgr[1] // 2, bgr[2] // 2)
    return img, [20, 20, 20 + w, 20 + h]


def test_l2_normalize_and_cosine():
    v = np.array([3.0, 4.0])
    assert np.linalg.norm(l2_normalize(v)) == pytest.approx(1.0)
    assert cosine_similarity(v, v) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_similarity(None, v) == 0.0


def test_color_backend_same_crop_matches_itself():
    backend = ColorHistogramBackend()
    img, bbox = _person(bgr=(200, 50, 50))
    crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    e1 = l2_normalize(backend.embed(crop))
    e2 = l2_normalize(backend.embed(crop))
    assert cosine_similarity(e1, e2) == pytest.approx(1.0)


def test_color_backend_distinguishes_colors():
    backend = ColorHistogramBackend()
    blue_crop, b = _person(bgr=(200, 40, 40))
    red_crop, r = _person(bgr=(40, 40, 200))
    eb = backend.embed(blue_crop[b[1]:b[3], b[0]:b[2]])
    er = backend.embed(red_crop[r[1]:r[3], r[0]:r[2]])
    assert cosine_similarity(eb, er) < 0.5


def test_assess_crop_rejects_tiny():
    img = np.zeros((200, 200, 3), np.uint8)
    q = assess_crop(img, [10, 10, 40, 50], min_height_px=80, min_blur_var=15, min_aspect=0.15, max_aspect=0.9)
    assert not q.usable and q.reason == "too small"


def test_assess_crop_rejects_flat_blurry():
    img = np.full((300, 200, 3), 120, np.uint8)  # perfectly flat -> zero laplacian variance
    q = assess_crop(img, [20, 20, 110, 260], min_height_px=80, min_blur_var=15, min_aspect=0.15, max_aspect=0.9)
    assert not q.usable and "blurry" in q.reason


def test_assess_crop_accepts_reasonable():
    img, bbox = _person()
    q = assess_crop(img, bbox, min_height_px=80, min_blur_var=15, min_aspect=0.15, max_aspect=0.9)
    assert q.usable and q.score > 0.0


def test_extractor_color_backend_embed_and_gate():
    ex = ReIDExtractor(ReIDConfig(backend="color"))
    img, bbox = _person()
    vec, quality = ex.embed(img, bbox)
    assert vec is not None
    assert np.linalg.norm(vec) == pytest.approx(1.0)
    assert quality.usable

    tiny_vec, tiny_q = ex.embed(np.zeros((200, 200, 3), np.uint8), [10, 10, 40, 50])
    assert tiny_vec is None and not tiny_q.usable


def test_build_backend_color():
    assert isinstance(build_backend(ReIDConfig(backend="color")), ColorHistogramBackend)


def test_track_embedding_aggregator_keeps_best_and_averages():
    agg = TrackEmbeddingAggregator(max_embeddings=2)
    agg.add(np.array([1.0, 0.0, 0.0]), quality_score=0.9)
    agg.add(np.array([0.0, 1.0, 0.0]), quality_score=0.8)
    agg.add(np.array([0.0, 0.0, 1.0]), quality_score=0.1)  # too low -> dropped
    assert agg.sample_count == 2
    rep = agg.representative()
    assert np.linalg.norm(rep) == pytest.approx(1.0)
    assert rep[2] == pytest.approx(0.0)  # the low-quality one never contributed


def test_aggregator_empty_representative_is_none():
    assert TrackEmbeddingAggregator().representative() is None


def test_osnet_architecture_forward_shape_and_determinism():
    torch = pytest.importorskip("torch")
    from vision.reid.osnet import osnet_x0_25

    net = osnet_x0_25().eval()
    x = torch.randn(2, 3, 256, 128)
    with torch.no_grad():
        y1 = net(x)
        y2 = net(x)
    assert tuple(y1.shape) == (2, 512)
    assert torch.allclose(y1, y2)


def test_osnet_backend_requires_weights():
    from vision.reid.backends import OSNetBackend

    with pytest.raises(FileNotFoundError, match="weights_path"):
        OSNetBackend("", "osnet_x1_0")


@pytest.mark.slow
def test_torchvision_backend_real():
    pytest.importorskip("torchvision")
    from vision.reid.backends import TorchvisionBackend

    backend = TorchvisionBackend("resnet18", "cpu")
    assert backend.dim == 512
    img, bbox = _person()
    crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    e1 = l2_normalize(backend.embed(crop))
    e2 = l2_normalize(backend.embed(crop))
    assert cosine_similarity(e1, e2) == pytest.approx(1.0, abs=1e-4)
    assert e1.shape == (512,)
