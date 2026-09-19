"""Builds the static Vercel dashboard from the single source of truth
(backend/api/dashboard.html + designer.html) — see docs/vercel-deployment.md.

    python scripts/build_web_dashboard.py [output_dir]

Run by Vercel's buildCommand (see vercel.json); also runnable locally to
preview exactly what gets deployed, e.g.:

    python scripts/build_web_dashboard.py apps/web/dist
    python -m http.server 5500 --directory apps/web/dist

Reads MONGDEE_AI_SERVER_URL from the environment (a Vercel Environment
Variable, e.g. https://ai.example.com:8100) as the *default* AI Server URL
baked into config.js. Leaving it unset defaults to same-origin, which is
almost certainly wrong for a Vercel deployment (the AI Server does not run
on vercel.app) — viewers can still point the page at the right server at
runtime via its own "server" settings dialog (saved in their browser),
without needing a rebuild, but setting this saves everyone that step.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "backend" / "api"


def build(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(SOURCE_DIR / "dashboard.html", output_dir / "index.html")
    shutil.copyfile(SOURCE_DIR / "designer.html", output_dir / "designer.html")

    api_base = os.environ.get("MONGDEE_AI_SERVER_URL", "").rstrip("/")
    if not api_base:
        print(
            "[build_web_dashboard] WARNING: MONGDEE_AI_SERVER_URL is not set — "
            "this dashboard will default to same-origin, which is not where the "
            "AI Server runs on Vercel. Set it in the Vercel project's Environment "
            "Variables, or have every viewer set it via the dashboard's own "
            "'server' button.",
            file=sys.stderr,
        )
    (output_dir / "config.js").write_text(
        f'window.MONGDEE_API_BASE = "{api_base}";\n', encoding="utf-8",
    )

    print(f"[build_web_dashboard] wrote {output_dir}/index.html, designer.html, config.js "
          f"(default AI Server: {api_base or '(same-origin — likely wrong for Vercel)'})")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "apps" / "web" / "dist"
    build(target)
