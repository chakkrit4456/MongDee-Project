"""Shared fakes for the vision package tests — a stand-in for an
ultralytics YOLO model that mimics just the slice of its API that
PersonDetector uses, so the detection logic can be tested without loading
real weights or running real inference.
"""

from __future__ import annotations


class _Scalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _Row:
    def __init__(self, coords):
        self._coords = list(coords)

    def tolist(self):
        return list(self._coords)


class _XYXY:
    def __init__(self, coords):
        self._coords = coords

    def __getitem__(self, idx):
        assert idx == 0
        return _Row(self._coords)


class FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy):
        self.cls = _Scalar(cls_id)
        self.conf = _Scalar(conf)
        self.xyxy = _XYXY(xyxy)


class _Boxes:
    def __init__(self, boxes):
        self._boxes = boxes

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)


class _Results:
    def __init__(self, boxes):
        self.boxes = _Boxes(boxes)


class FakeYOLO:
    """Returns a fixed list of FakeBox objects from predict(), after
    applying the same `classes=` / `conf=` filtering real ultralytics
    would, so tests can check PersonDetector both relies on and re-checks
    that filtering."""

    def __init__(self, boxes=None, names=None):
        self._boxes = list(boxes or [])
        self.names = names if names is not None else {0: "person", 1: "bicycle", 2: "car"}
        self.predict_calls: list[dict] = []

    def predict(self, source=None, **kwargs):
        self.predict_calls.append({"source_shape": getattr(source, "shape", None), **kwargs})
        classes = kwargs.get("classes")
        conf = kwargs.get("conf", 0.0)
        selected = [
            b
            for b in self._boxes
            if (classes is None or int(b.cls.item()) in classes) and float(b.conf.item()) >= conf
        ]
        return [_Results(selected)]
