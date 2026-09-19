"""Person vision pipeline — Phase 2+ of the MongDee multi-camera person
pipeline (see MongDee_Master_Prompt.md).

Distinct from core/vision.py, which is the booth's *product* recogniser.
This package is about detecting / tracking / re-identifying *people* across
the cameras managed by the `camera` package. Public surface:

    from vision import (
        PersonDetector, DetectionConfig, FrameProcessor,
        MultiCameraTracker, TrackingConfig, Track,
        DetectionPipeline, PipelineResult, Detection,
    )
"""

from vision.attributes import AttributeConfig, AttributeExtractor, PersonAttributes
from vision.booth_analytics import BoothAnalytics, InterestEvent
from vision.config import DetectionConfig
from vision.detection.detector import Detection, PersonDetector, resolve_device
from vision.events import PersonEvent
from vision.frame import FrameProcessor, ProcessedFrame
from vision.identity import (
    CameraTopology,
    GlobalIdentityManager,
    GlobalPerson,
    IdentityConfig,
    MatchResult,
    TrackSummary,
)
from vision.interest import (
    AssociationEngine,
    InterestConfig,
    InterestSession,
)
from vision.pipeline import CameraDetectionStats, DetectionPipeline, PipelineResult
from vision.product import (
    GIProductDatabase,
    ProductClassifier,
    ProductConfig,
    ProductVisionModule,
)
from vision.reid import ReIDConfig, ReIDExtractor
from vision.spatial import BoothLayout, BoothObject, CalibrationStore, CameraCalibration, WorldMapper
from vision.track_features import FeatureStoreConfig, TrackFeatureStore
from vision.tracking import ByteTracker, MultiCameraTracker, Track, TrackingConfig, TrackState

__all__ = [
    "DetectionConfig",
    "Detection",
    "PersonDetector",
    "resolve_device",
    "FrameProcessor",
    "ProcessedFrame",
    "DetectionPipeline",
    "PipelineResult",
    "CameraDetectionStats",
    "ByteTracker",
    "MultiCameraTracker",
    "Track",
    "TrackState",
    "TrackingConfig",
    "AttributeConfig",
    "AttributeExtractor",
    "PersonAttributes",
    "ReIDConfig",
    "ReIDExtractor",
    "TrackFeatureStore",
    "FeatureStoreConfig",
    "IdentityConfig",
    "GlobalIdentityManager",
    "GlobalPerson",
    "CameraTopology",
    "TrackSummary",
    "MatchResult",
    "PersonEvent",
    "BoothAnalytics",
    "InterestEvent",
    "AssociationEngine",
    "InterestConfig",
    "InterestSession",
    "ProductConfig",
    "ProductClassifier",
    "ProductVisionModule",
    "GIProductDatabase",
    "CameraCalibration",
    "CalibrationStore",
    "BoothLayout",
    "BoothObject",
    "WorldMapper",
]
