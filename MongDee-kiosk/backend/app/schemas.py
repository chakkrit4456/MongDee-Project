from typing import Optional

from pydantic import BaseModel, Field, model_validator


class GalleryImage(BaseModel):
    id: int
    path: str


class GalleryOrder(BaseModel):
    ids: list[int]


class ProductOut(BaseModel):
    id: int
    name: str
    category: Optional[str] = None
    origin: Optional[str] = None
    material: Optional[str] = None
    process: Optional[str] = None
    story: Optional[str] = None
    video_url: Optional[str] = None
    video_link: Optional[str] = None
    thumbnail_path: Optional[str] = None
    cover_image_path: Optional[str] = None
    video_path: Optional[str] = None
    price: Optional[float] = None
    production_date: Optional[str] = None
    expiry_date: Optional[str] = None
    created_at: float
    # True when the product has no embedding at the current version and must
    # be re-scanned before the kiosk can recognize it. Only the list views
    # populate this; elsewhere it stays False.
    needs_reembed: bool = False
    # Saved training-frame photos, in capture order, for the popup's 360
    # spin viewer. Empty for products enrolled before frame images were
    # kept (see db.get_product_frames) — the frontend falls back to a
    # static cover/thumbnail image in that case.
    spin_frames: list[str] = []
    # Photos added by hand in the manage page. When present the popup shows
    # these as its thumbnail row instead of the auto-captured angle frames.
    gallery: list[GalleryImage] = []


class ProductUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    origin: Optional[str] = None
    material: Optional[str] = None
    process: Optional[str] = None
    story: Optional[str] = None
    video_url: Optional[str] = None
    video_link: Optional[str] = None
    price: Optional[float] = None
    production_date: Optional[str] = None
    expiry_date: Optional[str] = None


class EnrollConfirmRequest(BaseModel):
    name: str
    category: Optional[str] = None
    origin: Optional[str] = None
    material: Optional[str] = None
    process: Optional[str] = None
    story: Optional[str] = None
    video_url: Optional[str] = None
    video_link: Optional[str] = None
    price: Optional[float] = None
    production_date: Optional[str] = None
    expiry_date: Optional[str] = None
    selected_frame_ids: list[str]


class EnrollFrameOut(BaseModel):
    frame_id: str
    thumbnail_data_url: str


class EnrollBatchResponse(BaseModel):
    session_id: str
    frames: list[EnrollFrameOut]


class RecognitionCandidate(BaseModel):
    product_id: int
    name: str
    score: float


class RecognitionResult(BaseModel):
    status: str  # "idle" | "scanning" | "matched" | "ambiguous" | "unknown"
    message: Optional[str] = None
    manually_confirmed: bool = False
    camera_generation: Optional[int] = None
    scan_event_id: Optional[int] = None
    product: Optional[ProductOut] = None
    confidence: Optional[float] = None
    candidates: list[RecognitionCandidate] = []
    # Set only on "unknown": an enrollment session already pre-loaded with
    # the frames just captured, so adding this as a new product doesn't
    # require recapturing it.
    pending_enroll_session_id: Optional[str] = None
    updated_at: float


class CorrectionRequest(BaseModel):
    scan_event_id: int
    product_id: int


class CameraDeviceOut(BaseModel):
    index: int
    active: bool
    width: Optional[int] = None
    height: Optional[int] = None


class CameraDevicesResponse(BaseModel):
    devices: list[CameraDeviceOut]
    current_index: int


class CameraSelectRequest(BaseModel):
    index: int = Field(ge=0, le=20)


class ROIFraction(BaseModel):
    x: float = Field(ge=0, lt=1, allow_inf_nan=False)
    y: float = Field(ge=0, lt=1, allow_inf_nan=False)
    w: float = Field(ge=0.01, le=1, allow_inf_nan=False)
    h: float = Field(ge=0.01, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def within_frame(self):
        if self.x + self.w > 1.0000001 or self.y + self.h > 1.0000001:
            raise ValueError("กรอบตรวจจับต้องอยู่ภายในภาพ")
        return self


class CalibrationSettings(BaseModel):
    roi: ROIFraction
    presence_on_ratio: float = Field(gt=0, lt=1, allow_inf_nan=False)
    presence_off_ratio: float = Field(ge=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def hysteresis(self):
        if self.presence_on_ratio <= self.presence_off_ratio:
            raise ValueError("เกณฑ์ตรวจพบต้องมากกว่าเกณฑ์ตรวจไม่พบ")
        return self


class AnalyticsSummaryRow(BaseModel):
    product_id: int
    name: str
    scan_count: int
    avg_confidence: Optional[float]
    last_scan: Optional[float]


class AnalyticsHourlyRow(BaseModel):
    hour: int
    count: int


class AnalyticsDailyRow(BaseModel):
    day: str
    count: int


class AnalyticsOverview(BaseModel):
    successful: int
    unmatched: int
    corrected: int
    # Matches whose product was deleted later; excluded from every chart.
    orphaned: int
    # Start of the "today" period, and whether staff moved it after midnight.
    day_start: float
    day_reset: bool
    success_rate: Optional[float]
    products_total: int
    products_scanned: int
