# Performance

## Two independent Adaptive Performance Engines

This project has two, because it has two independent AI pipelines with
different knobs:

| | Local/self-hosted (`core/vision.py`) | AI Server (`backend/`) |
|---|---|---|
| Controller | `core/performance.py`'s `AdaptiveController` | `backend/adaptive.py`'s `PipelineAdaptiveController` |
| Scope | **per camera** — each `CameraWorker` has its own detection rate and YOLO input size | **process-wide** — one shared detector/tracker serves every camera (`vision/pipeline.py`'s design for a CPU-only box) |
| Config | `configs/performance.example.json` | the `"adaptive"` block in `configs/mongdee.example.json` |

Both follow the same priority order and the same restraint:

1. Reduce AI inference rate first.
2. Reduce AI input resolution (YOLO `imgsz`) once rate is already at its
   floor.
3. Never touch display/stream FPS automatically — a controller two levers
   short of exhausted has no basis for deciding a viewer's live video
   should get choppier; that call is left to whoever owns the display
   pipeline (in the AI Server's case, nothing currently auto-reduces
   `/stream`'s FPS — see `docs/troubleshooting.md` if you need to do this
   manually today).
4. Restore in the reverse order once load is back under threshold.

Both read real, measured signals — CPU% (`core/performance.py`, backed by
`psutil`) and the pipeline's own measured per-camera latency — never a
static "this hardware should handle N cameras" guess. Hardware detection
(`detect_hardware()`) only seeds an initial tier label for logging/
`/metrics`; it never gates what the controller is allowed to do.

## Measuring, not guessing

- **Local/self-hosted**: `python web_server.py --benchmark 30` runs
  headlessly for 30s and prints a report (cameras, resolution, display
  FPS, AI FPS, frame drop %, CPU/RAM/GPU) — see
  `core/performance.py:format_benchmark_report()`.
- **AI Server**: `GET /metrics` returns live hardware + per-camera
  `effective_fps` / `last_latency_ms` from the running pipeline
  (`vision/pipeline.py`'s `CameraDetectionStats`) at any time; there is no
  separate `--benchmark` CLI for `backend/main.py` yet (see
  `docs/ai-server.md`'s "Known limits").

## Tuning thresholds

Nothing is hard-coded per the project's own rule — copy the relevant
example config, adjust, and pass it explicitly:

```bash
# local/self-hosted
python web_server.py --performance-config configs/performance.json

# AI Server: edit the "adaptive" block directly in your mongdee.json
```

Fields and defaults are documented inline in
`configs/performance.example.json` and `configs/mongdee.example.json`.

## What "different hardware, different results" means here

Neither controller claims to equalize performance across machines — a
faster CPU/GPU genuinely gets more FPS at a given quality level, and a
slower one adapts down instead of falling over. `core/performance.py`'s
`HardwareProfile.tier` (LOW/MEDIUM/HIGH) is informational, not a promise.
