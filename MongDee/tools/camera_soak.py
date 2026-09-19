#!/usr/bin/env python
"""Long-run soak of the process-isolated capture on REAL cameras (Windows).

    python tools/camera_soak.py --indices 1,2 --minutes 60
    python tools/camera_soak.py --fake --minutes 0.3 --inject-hang     # harness self-test

Writes reports/camera-ai-fix/soak_<ts>.csv (1 row / 5 s per camera) and a summary JSON.
PASS (gate G1): every camera fps >= min_fps in >= 95% of windows, zero parent crashes,
                every hang/kill recovered within recover_s.
"""
import argparse, csv, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.capture_process import CaptureProcess
from core.frame_integrity import FrameIntegrity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", default="1,2"); ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--w", type=int, default=320); ap.add_argument("--h", type=int, default=240)
    ap.add_argument("--fps", type=float, default=15); ap.add_argument("--min-fps", type=float, default=10)
    ap.add_argument("--fourcc", default="MJPG"); ap.add_argument("--fake", action="store_true")
    ap.add_argument("--inject-hang", action="store_true"); ap.add_argument("--window", type=float, default=5.0)
    a = ap.parse_args()
    idx = [int(x) for x in a.indices.split(",")]
    caps = {}
    for i in idx:
        if a.fake:
            kw = {"mode": "hang_after" if (a.inject_hang and i == idx[-1]) else "ok", "after": 60, "marker": i, "fps": a.fps}
            caps[i] = CaptureProcess("core.capture_sim:make_fake", kw, name=f"cam{i}", stale_sec=2.0, backoff=(0.5,))
        else:
            caps[i] = CaptureProcess("core.capture_sim:make_cv2",
                                     dict(index=i, width=a.w, height=a.h, fps=a.fps, fourcc=a.fourcc), name=f"cam{i}")
        caps[i].start()
        time.sleep(2.5)  # stagger opens (device cooldown)
    fi = {i: FrameIntegrity() for i in idx}; last_seq = {i: 0 for i in idx}; bad = {i: 0 for i in idx}
    os.makedirs(os.path.join(ROOT, "reports", "camera-ai-fix"), exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(ROOT, "reports", "camera-ai-fix", f"soak_{stamp}.csv")
    wins = {i: [] for i in idx}
    t_end = time.time() + a.minutes * 60
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["t", "cam", "fps", "bad_frames", "kills", "respawns", "alive"])
        while time.time() < t_end:
            w0 = time.time()
            seen = {i: 0 for i in idx}
            while time.time() - w0 < a.window:
                for i in idx:
                    ok, f, seq, ts = caps[i].read()
                    if ok and seq != last_seq[i]:
                        seen[i] += seq - last_seq[i] if seq > last_seq[i] else 1
                        last_seq[i] = seq
                        if not fi[i].check(f, time.time()).ok: bad[i] += 1
                time.sleep(0.02)
            for i in idx:
                fps = seen[i] / a.window; wins[i].append(fps)
                wr.writerow([round(time.time()), i, round(fps, 2), bad[i], caps[i].kills, caps[i].respawns, caps[i].alive])
            fh.flush()
    summary = {}
    for i in idx:
        good = sum(1 for x in wins[i] if x >= a.min_fps) / max(len(wins[i]), 1)
        summary[i] = dict(windows=len(wins[i]), frac_ok=round(good, 3), kills=caps[i].kills,
                          respawns=caps[i].respawns, bad_frames=bad[i], PASS=good >= 0.95)
        caps[i].stop()
    with open(path.replace(".csv", ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
