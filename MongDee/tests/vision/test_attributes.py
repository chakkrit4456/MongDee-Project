import numpy as np
import pytest

from vision.attributes.aggregator import TrackAttributeAggregator
from vision.attributes.color import dominant_color
from vision.attributes.config import AttributeConfig
from vision.attributes.extractor import AttributeExtractor, AttributeValue, GenderAgeBackend, PersonAttributes


def _solid(h, w, bgr):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = bgr
    return img


def test_dominant_color_solid():
    name, rgb, conf = dominant_color(_solid(50, 50, (30, 30, 200)))  # BGR red
    assert name == "red"
    assert conf > 0.9


def test_dominant_color_mixed_is_low_confidence():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    _name, _rgb, conf = dominant_color(noise)
    assert conf < 0.6


def test_dominant_color_empty():
    assert dominant_color(np.zeros((0, 0, 3), np.uint8)) == ("unknown", (0, 0, 0), 0.0)


def _person_crop():
    """480px tall: blue shirt band over black trousers band on gray bg."""
    img = _solid(480, 200, (128, 128, 128))
    img[72:240] = (190, 70, 40)  # BGR-ish blue in the upper-body slice
    img[240:432] = (20, 20, 20)  # black lower-body slice
    return img


def test_extractor_reads_shirt_and_pants_color():
    ex = AttributeExtractor()
    attrs = ex.extract(_person_crop(), [0, 0, 200, 480])
    assert attrs.shirt_color.value == "blue"
    assert attrs.shirt_color.confidence > 0.5
    assert attrs.pants_color.value == "black"
    assert attrs.aspect_ratio == pytest.approx(200 / 480, abs=1e-2)
    assert attrs.build.value in ("slim", "average", "broad")


def test_extractor_tiny_crop_returns_unknowns():
    ex = AttributeExtractor()
    attrs = ex.extract(np.zeros((400, 400, 3), np.uint8), [10, 10, 40, 50])  # 30px tall
    assert attrs.shirt_color.value == "unknown"
    assert attrs.pants_color.value == "unknown"
    assert attrs.gender.value == "unknown"


def test_extractor_uses_gender_age_backend_when_present():
    class FakeGA(GenderAgeBackend):
        def predict_gender(self, crop):
            return "female", 0.82

        def predict_age_group(self, crop):
            return "adult", 0.7

    ex = AttributeExtractor(gender_age_backend=FakeGA())
    attrs = ex.extract(_person_crop(), [0, 0, 200, 480])
    assert attrs.gender.value == "female" and attrs.gender.confidence == 0.82
    assert attrs.age_group.value == "adult"


def test_attributes_to_dict_and_known_items():
    a = PersonAttributes(shirt_color=AttributeValue("blue", 0.8))
    d = a.to_dict()
    assert d["shirt_color"] == {"value": "blue", "confidence": 0.8}
    assert d["gender"] == {"value": "unknown", "confidence": 0.0}
    assert set(a.known_items()) == {"shirt_color"}


def test_config_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown field"):
        AttributeConfig.from_dict({"horizontal_inset": 0.2, "bogus": 1})


def test_aggregator_majority_vote_and_mean():
    agg = TrackAttributeAggregator()
    agg.add(PersonAttributes(shirt_color=AttributeValue("blue", 0.9), shirt_rgb=(0, 0, 200), aspect_ratio=0.4))
    agg.add(PersonAttributes(shirt_color=AttributeValue("blue", 0.8), shirt_rgb=(0, 0, 220), aspect_ratio=0.42))
    agg.add(PersonAttributes(shirt_color=AttributeValue("black", 0.5), shirt_rgb=(10, 10, 10), aspect_ratio=0.38))
    result = agg.result()
    assert result.shirt_color.value == "blue"
    assert result.shirt_color.confidence > 0.6
    assert result.shirt_rgb == (3, 3, 143)
    assert result.aspect_ratio == pytest.approx(0.4, abs=0.02)
    assert agg.sample_count == 3
