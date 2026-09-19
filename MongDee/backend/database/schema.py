"""MongDee vision database schema (MongDee_Master_Prompt.md section 22,
plus events from section 27 and camera transitions from section 14).

SQLite, in its own file (data/mongdee_vision.db) separate from the booth
product database (core/database.py -> data/mongdee.db) so the two
subsystems stay independent — a schema change or corruption in one never
touches the other (section 87: modular, isolated failure).

Migrations are applied by backend/database/migrations.py; MIGRATIONS is the
ordered list. Never edit an already-shipped migration — add a new one.
"""

from __future__ import annotations

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE cameras (
            id           TEXT PRIMARY KEY,
            name         TEXT,
            protocol     TEXT,
            url_redacted TEXT,
            location     TEXT,
            status       TEXT,
            resolution   TEXT,
            fps          REAL,
            enabled      INTEGER DEFAULT 1,
            updated_at   REAL
        );

        CREATE TABLE global_persons (
            id                   TEXT PRIMARY KEY,
            first_seen           REAL,
            last_seen            REAL,
            estimated_gender     TEXT DEFAULT 'unknown',
            gender_confidence    REAL DEFAULT 0,
            estimated_age_group  TEXT DEFAULT 'unknown',
            age_confidence       REAL DEFAULT 0,
            shirt_color          TEXT DEFAULT 'unknown',
            pants_color          TEXT DEFAULT 'unknown',
            cameras_seen         TEXT DEFAULT '[]',
            track_count          INTEGER DEFAULT 0,
            uncertain_links      TEXT DEFAULT '[]',
            created_at           REAL,
            updated_at           REAL
        );

        CREATE TABLE camera_tracks (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id         TEXT,
            local_track_id    INTEGER,
            global_person_id  TEXT,
            start_time        REAL,
            end_time          REAL,
            embedding_samples INTEGER DEFAULT 0,
            quality           REAL DEFAULT 0,
            match_status      TEXT,
            match_score       REAL DEFAULT 0
        );
        CREATE INDEX idx_tracks_person ON camera_tracks(global_person_id);
        CREATE INDEX idx_tracks_camera ON camera_tracks(camera_id);

        CREATE TABLE detections (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id      TEXT,
            local_track_id INTEGER,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL,
            confidence     REAL,
            timestamp      REAL
        );
        CREATE INDEX idx_detections_time ON detections(timestamp);

        CREATE TABLE reid_embeddings (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            global_person_id TEXT,
            camera_id        TEXT,
            local_track_id   INTEGER,
            dim              INTEGER,
            vector           BLOB,
            quality          REAL,
            created_at       REAL
        );
        CREATE INDEX idx_embeddings_person ON reid_embeddings(global_person_id);

        CREATE TABLE events (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type       TEXT,
            timestamp        REAL,
            camera_id        TEXT,
            global_person_id TEXT,
            local_track_id   INTEGER,
            match_score      REAL DEFAULT 0,
            payload          TEXT DEFAULT '{}'
        );
        CREATE INDEX idx_events_time ON events(timestamp);
        CREATE INDEX idx_events_type ON events(event_type);
        CREATE INDEX idx_events_person ON events(global_person_id);

        CREATE TABLE camera_transitions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            from_camera      TEXT,
            to_camera        TEXT,
            global_person_id TEXT,
            transit_seconds  REAL,
            timestamp        REAL
        );

        CREATE TABLE system_logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            level     TEXT,
            source    TEXT,
            message   TEXT,
            timestamp REAL
        );
        """,
    ),
    (
        2,
        """
        -- spatial / product / interest (MongDee_Master_Prompt.md sections 70, 80)

        CREATE TABLE booths (
            id       TEXT PRIMARY KEY,
            name     TEXT,
            width    REAL,
            length   REAL,
            unit     TEXT DEFAULT 'm',
            created_at REAL
        );

        CREATE TABLE layouts (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            booth_id       TEXT,
            version        INTEGER,
            width          REAL,
            length         REAL,
            unit           TEXT DEFAULT 'm',
            active         INTEGER DEFAULT 0,
            objects        TEXT DEFAULT '[]',
            created_at     REAL,
            created_by     TEXT,
            UNIQUE(booth_id, version)
        );
        CREATE INDEX idx_layouts_booth ON layouts(booth_id);

        CREATE TABLE camera_calibrations (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id          TEXT,
            layout_version     INTEGER,
            version            INTEGER,
            homography         TEXT,
            image_points       TEXT,
            world_points       TEXT,
            reprojection_error REAL,
            created_at         REAL
        );
        CREATE INDEX idx_calib_camera ON camera_calibrations(camera_id);

        CREATE TABLE products (
            id                 TEXT PRIMARY KEY,
            name               TEXT,
            class_name         TEXT,
            gi_reference       TEXT,
            province           TEXT,
            producer           TEXT,
            category           TEXT,
            description        TEXT,
            image_url          TEXT,
            active             INTEGER DEFAULT 1
        );

        CREATE TABLE product_detections (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id     TEXT,
            camera_id      TEXT,
            local_track_id INTEGER,
            status         TEXT,
            zone_id        TEXT,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL,
            confidence     REAL,
            timestamp      REAL
        );
        CREATE INDEX idx_proddet_product ON product_detections(product_id);
        CREATE INDEX idx_proddet_time ON product_detections(timestamp);

        CREATE TABLE positions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            global_person_id TEXT,
            camera_id        TEXT,
            x REAL, y REAL,
            zone_id          TEXT,
            confidence       REAL,
            timestamp        REAL
        );
        CREATE INDEX idx_positions_person ON positions(global_person_id);
        CREATE INDEX idx_positions_time ON positions(timestamp);

        CREATE TABLE interest_events (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type       TEXT,
            global_person_id TEXT,
            target_id        TEXT,
            status           TEXT,
            interest_score   REAL,
            look_duration    REAL,
            dwell_duration   REAL,
            signals          TEXT DEFAULT '[]',
            timestamp        REAL,
            layout_version   INTEGER
        );
        CREATE INDEX idx_interest_person ON interest_events(global_person_id);
        CREATE INDEX idx_interest_target ON interest_events(target_id);
        CREATE INDEX idx_interest_time ON interest_events(timestamp);
        """,
    ),
]
