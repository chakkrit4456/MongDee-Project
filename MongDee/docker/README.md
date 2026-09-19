# MongDee Vision — install & deploy

## Local (Linux — Ubuntu / Arch — or Windows)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# CPU (works anywhere)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# OR NVIDIA GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install opencv-python ultralytics fastapi "uvicorn[standard]" numpy requests pytest httpx

python scripts/download_models.py    # pre-fetch YOLO + torchvision Re-ID weights
python -m pytest -q                  # 224 tests (+ `-m slow` runs real inference)
python -m backend.main configs/mongdee.example.json
# open http://127.0.0.1:8100/  (dashboard)  and  /designer  (booth editor)
```

### GPU verification

```bash
python -c "import torch; print('cuda', torch.cuda.is_available())"
```

`detection.device` / `reid.device` default to `auto` — GPU is used automatically when present.

### Re-ID model (recommended for production)

The default Re-ID backend (`auto` → torchvision ResNet features) is a *generic* appearance
embedding, not identity-trained — cross-camera merge accuracy will be low. For production, get
purpose-trained OSNet weights and point the config at them:

```bash
python scripts/download_models.py --osnet-url <URL-to-osnet_x1_0.pth> --osnet-out models/osnet_x1_0.pth
```

```json
"reid": { "backend": "osnet", "osnet_weights": "models/osnet_x1_0.pth", "osnet_variant": "osnet_x1_0" }
```

## Docker

```bash
export CAM02_PASSWORD=... MONGDEE_API_KEY=...
docker compose -f docker/docker-compose.yml up --build
```

CPU image is ~2 GB. For GPU: change the base image in `docker/Dockerfile` to a CUDA runtime,
install the `cu121` torch wheels, and uncomment the `deploy.resources.devices` block in
`docker-compose.yml` (needs `nvidia-container-toolkit`).

## Production checklist (master prompt sections 41, 89, 92)

- [ ] Set `api.api_key` or `api.keys` (role-based). The server refuses nothing without it but logs a warning.
- [ ] Put camera passwords in env vars (`${VAR}` in the config), never in the file.
- [ ] Calibrate every camera (`spatial.calibration_path`) with >= 4 known floor points; check
      `reprojection_error` is small (`CalibrationStore.unreliable()`).
- [ ] Tune `identity.match_threshold` / `uncertain_threshold` and the tracking/interest thresholds
      against labelled footage — run `python scripts/evaluate.py annotations.json`. Do **not** ship the defaults.
- [ ] Decide data retention (`identity.person_ttl_sec`, `database.log_detections`) per the venue's
      privacy policy and applicable law (section 35).
- [ ] Load-test: this pipeline is one CPU worker shared across cameras — measure real fps per camera
      count on the target hardware and set `*.target_fps_per_camera` accordingly.

## Known limitations

See the "Known limitations" section in the top-level `README.md` and each module's README —
in short: CPU-bound on this dev machine, no purpose-trained Re-ID / face / gender / age / gaze
models wired in (all return `UNKNOWN` honestly, with hooks to add them), ONVIF/HLS verified only
against mock servers.
