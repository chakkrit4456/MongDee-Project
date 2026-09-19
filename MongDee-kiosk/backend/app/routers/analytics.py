from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Query
from fastapi.responses import Response

from app import db, report_export
from app.schemas import AnalyticsDailyRow, AnalyticsHourlyRow, AnalyticsOverview, AnalyticsSummaryRow

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

Period = Literal["today", "7d", "30d", "all"]
XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@router.get("/overview", response_model=AnalyticsOverview)
def overview(period: Period = Query("all", alias="range")):
    return AnalyticsOverview(**db.analytics_overview(period))


@router.get("/summary", response_model=list[AnalyticsSummaryRow])
def summary(period: Period = Query("all", alias="range")):
    return [AnalyticsSummaryRow(**dict(r)) for r in db.analytics_summary(period)]


@router.get("/hourly", response_model=list[AnalyticsHourlyRow])
def hourly(period: Period = Query("all", alias="range")):
    return [AnalyticsHourlyRow(**dict(r)) for r in db.analytics_hourly(period)]


@router.get("/daily", response_model=list[AnalyticsDailyRow])
def daily(period: Period = Query("all", alias="range")):
    return [AnalyticsDailyRow(**row) for row in db.analytics_daily(period)]


@router.get("/export")
def export(period: Period = Query("all", alias="range")):
    """The report as an Excel workbook (summary, products, daily, hourly, every scan)."""
    now = datetime.now()
    filename = f"mongdee-report-{period}-{now.strftime('%Y%m%d-%H%M')}.xlsx"
    return Response(
        report_export.build_report(period, now=now),
        media_type=XLSX_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/reset-day")
def reset_day():
    """Restarts today's counters from now. Nothing is deleted: earlier scans
    stay in every other period and in exported reports."""
    db.set_day_reset()
    return db.analytics_overview("today")


@router.delete("/reset-day")
def undo_reset_day():
    db.clear_day_reset()
    return db.analytics_overview("today")
