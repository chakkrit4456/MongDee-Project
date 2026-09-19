"""Camera Agent CLI entrypoint.

    python -m camera_agent.main configs/camera_agent.example.json

Captures locally-attached webcams (or RTSP/ONVIF/HTTP/HLS sources — same
camera/base.py CameraConfig used everywhere else in this project) and
streams them to a Remote AI Server (backend/main.py) over HTTP. See
docs/camera-agent.md.
"""

from __future__ import annotations

import logging
import signal
import sys
import time

from camera_agent.client import CameraAgentClient
from camera_agent.config import AgentConfig

logger = logging.getLogger("mongdee.camera_agent.main")


def _on_status(camera_id: str, status, message: str) -> None:
    logger.info("[%s] %s: %s", camera_id, getattr(status, "value", status), message)


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if len(argv) != 2:
        print("usage: python -m camera_agent.main <camera_agent_config.json>", file=sys.stderr)
        return 2
    try:
        config = AgentConfig.load(argv[1])
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    logger.info("starting Camera Agent: %d camera(s) -> %s", len(config.cameras), config.server_url)
    client = CameraAgentClient(config, on_status=_on_status)
    client.start()

    stopping = {"done": False}

    def handle_signal(signum, frame):
        if not stopping["done"]:
            stopping["done"] = True
            client.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while not stopping["done"]:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if not stopping["done"]:
            client.stop()
    logger.info("Camera Agent stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
