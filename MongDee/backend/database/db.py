"""Database — a thin SQLite wrapper for the vision subsystem.

One connection, guarded by a lock (write volume is low: track ends and
events, not per-frame rows unless detection logging is on). WAL mode so a
reader — e.g. the API server — never blocks the writer. Schema is created
and upgraded via ordered migrations; `schema_version` records how far
we've applied.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from backend.database.schema import MIGRATIONS

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "mongdee_vision.db"


class Database:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # -- lifecycle ---------------------------------------------------------
    def _migrate(self) -> None:
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = self._conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            current = row["v"] or 0
            for version, ddl in MIGRATIONS:
                if version > current:
                    self._conn.executescript(ddl)
                    self._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                    current = version
            self._conn.commit()

    @property
    def schema_version(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            return row["v"] or 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- generic helpers -------------------------------------------------
    def execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def executemany(self, sql: str, seq_of_params) -> None:
        with self._lock:
            self._conn.executemany(sql, seq_of_params)
            self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_one(self, sql: str, params: tuple = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # -- domain writes -------------------------------------------------
    def upsert_camera(self, camera: dict) -> None:
        self.execute(
            """
            INSERT INTO cameras (id, name, protocol, url_redacted, location, status, resolution, fps, enabled, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, protocol=excluded.protocol, url_redacted=excluded.url_redacted,
                location=excluded.location, status=excluded.status, resolution=excluded.resolution,
                fps=excluded.fps, enabled=excluded.enabled, updated_at=excluded.updated_at
            """,
            (
                camera["id"], camera.get("name"), camera.get("protocol"), camera.get("url_redacted"),
                camera.get("location"), camera.get("status"), camera.get("resolution"),
                camera.get("fps"), int(camera.get("enabled", 1)), camera.get("updated_at", time.time()),
            ),
        )

    def set_camera_status(self, camera_id: str, status: str) -> None:
        self.execute(
            "UPDATE cameras SET status=?, updated_at=? WHERE id=?",
            (status, time.time(), camera_id),
        )

    def upsert_global_person(self, person: dict) -> None:
        now = time.time()
        self.execute(
            """
            INSERT INTO global_persons
                (id, first_seen, last_seen, estimated_gender, gender_confidence, estimated_age_group,
                 age_confidence, shirt_color, pants_color, cameras_seen, track_count, uncertain_links,
                 created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                first_seen=MIN(global_persons.first_seen, excluded.first_seen),
                last_seen=MAX(global_persons.last_seen, excluded.last_seen),
                estimated_gender=excluded.estimated_gender, gender_confidence=excluded.gender_confidence,
                estimated_age_group=excluded.estimated_age_group, age_confidence=excluded.age_confidence,
                shirt_color=excluded.shirt_color, pants_color=excluded.pants_color,
                cameras_seen=excluded.cameras_seen, track_count=excluded.track_count,
                uncertain_links=excluded.uncertain_links, updated_at=excluded.updated_at
            """,
            (
                person["id"], person.get("first_seen", now), person.get("last_seen", now),
                person.get("estimated_gender", "unknown"), person.get("gender_confidence", 0.0),
                person.get("estimated_age_group", "unknown"), person.get("age_confidence", 0.0),
                person.get("shirt_color", "unknown"), person.get("pants_color", "unknown"),
                json.dumps(person.get("cameras_seen", [])), person.get("track_count", 0),
                json.dumps(person.get("uncertain_links", [])), now, now,
            ),
        )

    def insert_track(self, track: dict) -> None:
        self.execute(
            """
            INSERT INTO camera_tracks
                (camera_id, local_track_id, global_person_id, start_time, end_time,
                 embedding_samples, quality, match_status, match_score)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                track["camera_id"], track["local_track_id"], track.get("global_person_id"),
                track.get("start_time"), track.get("end_time"), track.get("embedding_samples", 0),
                track.get("quality", 0.0), track.get("match_status"), track.get("match_score", 0.0),
            ),
        )

    def insert_event(self, event: dict) -> None:
        self.execute(
            """
            INSERT INTO events (event_type, timestamp, camera_id, global_person_id, local_track_id, match_score, payload)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                event["event_type"], event.get("timestamp", time.time()), event.get("camera_id"),
                event.get("global_id") or event.get("global_person_id"), event.get("local_track_id"),
                event.get("match_score", 0.0), json.dumps(event.get("payload", {})),
            ),
        )

    def insert_transition(self, from_camera: str, to_camera: str, global_person_id: str, transit_seconds: float) -> None:
        self.execute(
            "INSERT INTO camera_transitions (from_camera, to_camera, global_person_id, transit_seconds, timestamp) VALUES (?,?,?,?,?)",
            (from_camera, to_camera, global_person_id, transit_seconds, time.time()),
        )

    def insert_embedding(self, global_person_id: str, camera_id: str, local_track_id: int, vector: bytes, dim: int, quality: float) -> None:
        self.execute(
            "INSERT INTO reid_embeddings (global_person_id, camera_id, local_track_id, dim, vector, quality, created_at) VALUES (?,?,?,?,?,?,?)",
            (global_person_id, camera_id, local_track_id, dim, vector, quality, time.time()),
        )

    def insert_detections(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.executemany(
            "INSERT INTO detections (camera_id, local_track_id, x1, y1, x2, y2, confidence, timestamp) VALUES (?,?,?,?,?,?,?,?)",
            [
                (r["camera_id"], r.get("local_track_id"), r["x1"], r["y1"], r["x2"], r["y2"], r["confidence"], r["timestamp"])
                for r in rows
            ],
        )

    def log(self, level: str, source: str, message: str) -> None:
        self.execute(
            "INSERT INTO system_logs (level, source, message, timestamp) VALUES (?,?,?,?)",
            (level, source, message, time.time()),
        )

    # -- spatial / product / interest (schema v2) ---------------------
    def upsert_booth(self, booth: dict) -> None:
        self.execute(
            """INSERT INTO booths (id, name, width, length, unit, created_at) VALUES (?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name, width=excluded.width,
                   length=excluded.length, unit=excluded.unit""",
            (booth["id"], booth.get("name"), booth["width"], booth["length"], booth.get("unit", "m"), time.time()),
        )

    def save_layout(self, booth_id: str, layout: dict, created_by: str = "") -> int:
        """Insert a layout version, make it the active one. Returns the version."""
        version = layout.get("version", 1)
        with self._lock:
            self._conn.execute("UPDATE layouts SET active=0 WHERE booth_id=?", (booth_id,))
            self._conn.execute(
                """INSERT INTO layouts (booth_id, version, width, length, unit, active, objects, created_at, created_by)
                   VALUES (?,?,?,?,?,1,?,?,?)
                   ON CONFLICT(booth_id, version) DO UPDATE SET
                       width=excluded.width, length=excluded.length, unit=excluded.unit,
                       active=1, objects=excluded.objects, created_at=excluded.created_at""",
                (
                    booth_id, version, layout["width"], layout["length"], layout.get("unit", "m"),
                    json.dumps(layout.get("objects", [])), time.time(), created_by,
                ),
            )
            self._conn.commit()
        return version

    def active_layout(self, booth_id: str) -> dict | None:
        row = self.query_one("SELECT * FROM layouts WHERE booth_id=? AND active=1", (booth_id,))
        if row:
            row["objects"] = json.loads(row.get("objects") or "[]")
        return row

    def layout_versions(self, booth_id: str) -> list[dict]:
        return self.query(
            "SELECT booth_id, version, active, created_at, created_by FROM layouts WHERE booth_id=? ORDER BY version DESC",
            (booth_id,),
        )

    def activate_layout(self, booth_id: str, version: int) -> bool:
        existing = self.query_one("SELECT 1 FROM layouts WHERE booth_id=? AND version=?", (booth_id, version))
        if not existing:
            return False
        with self._lock:
            self._conn.execute("UPDATE layouts SET active=0 WHERE booth_id=?", (booth_id,))
            self._conn.execute("UPDATE layouts SET active=1 WHERE booth_id=? AND version=?", (booth_id, version))
            self._conn.commit()
        return True

    def booths(self) -> list[dict]:
        return self.query("SELECT * FROM booths ORDER BY id")

    def upsert_product(self, product: dict) -> None:
        self.execute(
            """INSERT INTO products (id, name, class_name, gi_reference, province, producer, category, description, image_url, active)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name, class_name=excluded.class_name,
                   gi_reference=excluded.gi_reference, province=excluded.province, producer=excluded.producer,
                   category=excluded.category, description=excluded.description, image_url=excluded.image_url,
                   active=excluded.active""",
            (
                product["id"], product.get("name"), product.get("class_name", ""), product.get("gi_reference", ""),
                product.get("province", ""), product.get("producer", ""), product.get("category", ""),
                product.get("description", ""), product.get("image_url", ""), int(product.get("active", 1)),
            ),
        )

    def products(self) -> list[dict]:
        return self.query("SELECT * FROM products WHERE active=1 ORDER BY id")

    def product(self, product_id: str) -> dict | None:
        return self.query_one("SELECT * FROM products WHERE id=?", (product_id,))

    def insert_product_detection(self, d: dict) -> None:
        self.execute(
            """INSERT INTO product_detections (product_id, camera_id, local_track_id, status, zone_id, x1, y1, x2, y2, confidence, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                d.get("product_id"), d["camera_id"], d.get("local_track_id"), d.get("status"), d.get("zone_id"),
                d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d.get("confidence", 0.0),
                d.get("timestamp", time.time()),
            ),
        )

    def insert_position(self, p: dict) -> None:
        self.execute(
            "INSERT INTO positions (global_person_id, camera_id, x, y, zone_id, confidence, timestamp) VALUES (?,?,?,?,?,?,?)",
            (p["global_person_id"], p.get("camera_id"), p["x"], p["y"], p.get("zone_id"), p.get("confidence", 0.0), p.get("timestamp", time.time())),
        )

    def person_movement(self, person_id: str, limit: int = 2000) -> list[dict]:
        return self.query(
            "SELECT x, y, zone_id, camera_id, confidence, timestamp FROM positions WHERE global_person_id=? ORDER BY timestamp LIMIT ?",
            (person_id, limit),
        )

    def insert_interest_event(self, e: dict) -> None:
        self.execute(
            """INSERT INTO interest_events (event_type, global_person_id, target_id, status, interest_score,
                   look_duration, dwell_duration, signals, timestamp, layout_version)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                e["event_type"], e["global_id"], e["target_id"], e.get("status"), e.get("score", 0.0),
                e.get("look_duration", 0.0), e.get("dwell_duration", 0.0), json.dumps(e.get("signals", [])),
                e.get("timestamp", time.time()), e.get("layout_version"),
            ),
        )

    def person_interests(self, person_id: str, limit: int = 200) -> list[dict]:
        rows = self.query(
            "SELECT * FROM interest_events WHERE global_person_id=? ORDER BY timestamp DESC LIMIT ?",
            (person_id, limit),
        )
        for r in rows:
            r["signals"] = json.loads(r.get("signals") or "[]")
        return rows

    def zone_analytics(self, zone_id: str) -> dict:
        visits = self.query_one("SELECT COUNT(DISTINCT global_person_id) AS n FROM positions WHERE zone_id=?", (zone_id,))["n"]
        interest = self.query(
            "SELECT status, COUNT(*) AS n FROM interest_events WHERE target_id=? AND event_type='CUSTOMER_INTEREST_ENDED' GROUP BY status",
            (zone_id,),
        )
        return {
            "zone_id": zone_id,
            "unique_visitors": visits,
            "interest_by_status": {r["status"]: r["n"] for r in interest},
        }

    def product_analytics(self, product_id: str) -> dict:
        detections = self.query_one("SELECT COUNT(*) AS n FROM product_detections WHERE product_id=?", (product_id,))["n"]
        by_status = {
            r["status"]: r["n"]
            for r in self.query("SELECT status, COUNT(*) AS n FROM product_detections WHERE product_id=? GROUP BY status", (product_id,))
        }
        return {"product_id": product_id, "total_detections": detections, "by_status": by_status}

    # -- domain reads (for the API / dashboard) -----------------------
    def cameras(self) -> list[dict]:
        return self.query("SELECT * FROM cameras ORDER BY id")

    def global_persons(self, limit: int = 500, since: float | None = None) -> list[dict]:
        if since is not None:
            rows = self.query(
                "SELECT * FROM global_persons WHERE last_seen >= ? ORDER BY last_seen DESC LIMIT ?",
                (since, limit),
            )
        else:
            rows = self.query("SELECT * FROM global_persons ORDER BY last_seen DESC LIMIT ?", (limit,))
        for r in rows:
            r["cameras_seen"] = json.loads(r.get("cameras_seen") or "[]")
            r["uncertain_links"] = json.loads(r.get("uncertain_links") or "[]")
        return rows

    def global_person(self, person_id: str) -> dict | None:
        row = self.query_one("SELECT * FROM global_persons WHERE id=?", (person_id,))
        if row:
            row["cameras_seen"] = json.loads(row.get("cameras_seen") or "[]")
            row["uncertain_links"] = json.loads(row.get("uncertain_links") or "[]")
            row["tracks"] = self.query(
                "SELECT * FROM camera_tracks WHERE global_person_id=? ORDER BY start_time", (person_id,)
            )
        return row

    def events(self, limit: int = 200, event_type: str | None = None, since: float | None = None) -> list[dict]:
        clauses, params = [], []
        if event_type:
            clauses.append("event_type=?")
            params.append(event_type)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self.query(f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ?", tuple(params))
        for r in rows:
            r["payload"] = json.loads(r.get("payload") or "{}")
        return rows

    def stats(self) -> dict:
        unique = self.query_one("SELECT COUNT(*) AS n FROM global_persons")["n"]
        by_gender = {
            r["estimated_gender"]: r["n"]
            for r in self.query("SELECT estimated_gender, COUNT(*) AS n FROM global_persons GROUP BY estimated_gender")
        }
        by_age = {
            r["estimated_age_group"]: r["n"]
            for r in self.query("SELECT estimated_age_group, COUNT(*) AS n FROM global_persons GROUP BY estimated_age_group")
        }
        cams = self.query("SELECT status, COUNT(*) AS n FROM cameras GROUP BY status")
        camera_status = {r["status"] or "unknown": r["n"] for r in cams}
        return {
            "unique_persons": unique,
            "by_gender": by_gender,
            "by_age_group": by_age,
            "camera_status": camera_status,
            "total_events": self.query_one("SELECT COUNT(*) AS n FROM events")["n"],
        }

    def camera_person_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.query("SELECT cameras_seen FROM global_persons"):
            for cam in json.loads(row.get("cameras_seen") or "[]"):
                counts[cam] = counts.get(cam, 0) + 1
        return counts
