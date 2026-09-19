#!/usr/bin/env python
"""Hardware experiments E1-E7 for the USB camera problem (run ON THE WINDOWS MACHINE).

    python tools/camera_experiments.py --list
    python tools/camera_experiments.py --run E3 --indices 1,2 --secs 60
    python tools/camera_experiments.py --run all --indices 1,2 --secs 60
    python tools/camera_experiments.py --run E3 --fake          # dry-run without cameras (harness self-test)

Every experiment runs each camera in a CHILD process with a hard kill timeout, so a hung
DirectShow/MSMF read can never freeze this script. Results -> reports/camera-ai-fix/E*.json.
Nothing here modifies system settings (E6 is read-only PowerShell queries).
"""
from __future__ import annotations
import argparse, gc, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "camera-ai-fix")


# ------------------------------------------------------------------ worker (child) ---
def worker(a):
    """Open one or more cameras IN THIS PROCESS, read for --secs, print one JSON line."""
    if a.msmf_hw is not None:
        os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = str(a.msmf_hw)
    import numpy as np
    from core.frame_integrity import FrameIntegrity
    idxs = [int(x) for x in a.indices.split(",")]
    caps = []
    for k, i in enumerate(idxs):
        if k and a.stagger:
            time.sleep(a.stagger)
        t0 = time.time()
        if a.fake:
            from core.capture_sim import FakeCapture
            cap = FakeCapture(mode="ok", marker=i, fps=a.fps)
            opened = True
        else:
            import cv2
            api = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}[a.backend]
            cap = cv2.VideoCapture(i, api)
            if a.fourcc != "none":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*a.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.h)
            cap.set(cv2.CAP_PROP_FPS, a.fps); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            opened = cap.isOpened()
        caps.append(dict(i=i, cap=cap, opened=opened, open_ms=int((time.time() - t0) * 1000),
                         n=0, bad=0, fails=0, last=None, max_gap=0.0, fi=FrameIntegrity(), reasons={}))
    t_end = time.time() + a.secs
    while time.time() < t_end:
        for c in caps:
            if not c["opened"]:
                continue
            ok, f = c["cap"].read()
            now = time.time()
            if not ok or f is None:
                c["fails"] += 1; time.sleep(0.01); continue
            if c["last"] is not None:
                c["max_gap"] = max(c["max_gap"], now - c["last"])
            c["last"] = now; c["n"] += 1
            v = c["fi"].check(f, now)
            if not v.ok:
                c["bad"] += 1
                for r in v.reasons: c["reasons"][r] = c["reasons"].get(r, 0) + 1
    res = [dict(index=c["i"], opened=c["opened"], open_ms=c["open_ms"], frames=c["n"],
                fps=round(c["n"] / a.secs, 2), read_fail=c["fails"], bad_frames=c["bad"],
                bad_reasons=c["reasons"], max_gap_ms=int(c["max_gap"] * 1000)) for c in caps]
    for c in caps:
        try: c["cap"].release()
        except Exception: pass
    print("RESULT_JSON " + json.dumps(res), flush=True)


# ------------------------------------------------------------------ orchestrator ---
def run_child(a, indices, backend, fourcc, secs, stagger=0.0, msmf_hw=None, kill_after=None):
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--indices", ",".join(map(str, indices)),
           "--backend", backend, "--fourcc", fourcc, "--w", str(a.w), "--h", str(a.h), "--fps", str(a.fps),
           "--secs", str(secs), "--stagger", str(stagger)]
    if msmf_hw is not None: cmd += ["--msmf-hw", str(msmf_hw)]
    if a.fake: cmd += ["--fake"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    limit = kill_after or (secs + 45)
    try:
        out, _ = p.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        p.kill(); out, _ = p.communicate()
        return dict(status="HUNG_KILLED", after_s=limit, log_tail=out[-800:])
    for line in out.splitlines():
        if line.startswith("RESULT_JSON "):
            return dict(status="ok", cams=json.loads(line[12:]), log_tail=out[-400:])
    return dict(status="NO_RESULT", exit=p.returncode, log_tail=out[-800:])


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[{name}] -> reports/camera-ai-fix/{name}.json")
    print(json.dumps(obj, ensure_ascii=False)[:1500])


def E1(a):
    res = {"pygrabber": None, "cv2_props": []}
    try:
        from pygrabber.dshow_graph import FilterGraph
        g = FilterGraph(); names = g.get_input_devices(); res["pygrabber"] = []
        for i, n in enumerate(names):
            g2 = FilterGraph(); g2.add_video_input_device(i)
            fmts = g2.get_input_device().get_formats()
            res["pygrabber"].append(dict(index=i, name=n, formats=fmts[:40]))
            del g2; gc.collect()          # a live FilterGraph keeps the camera open -> cv2 could not open it
    except Exception as e:
        res["pygrabber"] = f"unavailable: {e!r}"
    if not a.fake:
        import cv2
        for i in [int(x) for x in a.indices.split(",")]:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            fc = int(cap.get(cv2.CAP_PROP_FOURCC))
            res["cv2_props"].append(dict(index=i, opened=cap.isOpened(),
                fourcc="".join(chr((fc >> 8 * k) & 255) for k in range(4)) if fc else None,
                w=cap.get(cv2.CAP_PROP_FRAME_WIDTH), h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT), fps=cap.get(cv2.CAP_PROP_FPS)))
            cap.release(); time.sleep(3)
    save("E1_formats", res)


def E2(a):
    idx = [int(x) for x in a.indices.split(",")]; out = []
    for stagger in (0.0, 2.5, 10.0):
        for fourcc in ("MJPG", "YUY2"):
            out.append(dict(stagger=stagger, fourcc=fourcc, backend="dshow", same_process=True,
                            **run_child(a, idx, "dshow", fourcc, a.secs, stagger=stagger)))
            time.sleep(a.cooldown)
    save("E2_same_process", out)


def E3(a):
    idx = [int(x) for x in a.indices.split(",")]
    procs = []
    for i in idx:      # one OS process per camera, started together
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--indices", str(i), "--backend", "dshow",
               "--fourcc", a.fourcc, "--w", str(a.w), "--h", str(a.h), "--fps", str(a.fps), "--secs", str(a.secs)]
        if a.fake: cmd.append("--fake")
        procs.append((i, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)))
    out = []
    for i, p in procs:
        try:
            txt, _ = p.communicate(timeout=a.secs + 45)
            r = [l for l in txt.splitlines() if l.startswith("RESULT_JSON ")]
            out.append(dict(index=i, status="ok" if r else "NO_RESULT", cams=json.loads(r[0][12:]) if r else None, log_tail=txt[-400:]))
        except subprocess.TimeoutExpired:
            p.kill(); out.append(dict(index=i, status="HUNG_KILLED"))
    save("E3_separate_processes", out)


def E4(a):
    idx = [int(x) for x in a.indices.split(",")]; out = []
    for hw in (0, 1):
        out.append(dict(msmf_hw=hw, **run_child(a, idx[:1], "msmf", "none", min(a.secs, 30), msmf_hw=hw, kill_after=60)))
        time.sleep(a.cooldown)
    save("E4_msmf", out)


def E5(a):
    ff = shutil.which("ffmpeg")
    if not ff:
        return save("E5_ffmpeg", dict(status="ffmpeg not on PATH (winget install Gyan.FFmpeg)"))
    lst = subprocess.run([ff, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                         capture_output=True, text=True, timeout=30)
    res = dict(list_devices=(lst.stderr or "")[-2500:], runs=[])
    if a.camera_name:
        for codec in ("mjpeg", "rawvideo"):
            t0 = time.time()
            cmd = [ff, "-hide_banner", "-f", "dshow", "-video_size", f"{a.w}x{a.h}", "-framerate", str(a.fps),
                   "-vcodec", codec, "-i", f"video={a.camera_name}", "-t", "20", "-f", "null", "-"]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                res["runs"].append(dict(codec=codec, rc=r.returncode, secs=round(time.time() - t0, 1), tail=r.stderr[-600:]))
            except subprocess.TimeoutExpired:
                res["runs"].append(dict(codec=codec, status="HUNG_KILLED"))
            time.sleep(a.cooldown)
    save("E5_ffmpeg", res)


def E6(a):
    ps = ("Get-PnpDevice -Class Camera,Image -PresentOnly | ForEach-Object { $p=$_; "
          "[pscustomobject]@{Name=$p.FriendlyName;Id=$p.InstanceId;Status=$p.Status;"
          "Loc=(Get-PnpDeviceProperty -InstanceId $p.InstanceId -KeyName DEVPKEY_Device_LocationInfo -EA SilentlyContinue).Data;"
          "Parent=(Get-PnpDeviceProperty -InstanceId $p.InstanceId -KeyName DEVPKEY_Device_Parent -EA SilentlyContinue).Data} } | ConvertTo-Json")
    res = {}
    if os.name == "nt":
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
        res["devices"] = r.stdout.strip() or r.stderr[-500:]
        r = subprocess.run(["powercfg", "/q"], capture_output=True, text=True, timeout=60)
        res["usb_selective_suspend"] = [l.strip() for l in r.stdout.splitlines() if "USB selective" in l or "Setting Index" in l][:12]
    else:
        res["status"] = "not Windows"
    save("E6_usb_topology", res)


def E7(a):
    idx = int(a.indices.split(",")[0]); out = []
    for gap in (0.0, 1.0, 3.0, 8.0, 15.0, 30.0):
        r = run_child(a, [idx], "dshow", a.fourcc, 5, kill_after=40)
        out.append(dict(gap_before_reopen_s=gap, **r)); time.sleep(gap)
    save("E7_cooldown", out)


def E0(a):
    """SOLO test: every camera index ALONE, one at a time, two pixel formats. Run this first - it tells
    which physical camera works at all, without any second camera or the app competing for the bus."""
    idx = [int(x) for x in a.indices.split(",")]; out = []
    for i in idx:
        for fourcc in ("MJPG", "YUY2"):
            r = run_child(a, [i], "dshow", fourcc, 8, kill_after=60)
            cam = (r.get("cams") or [{}])[0]
            print(f"  index {i} {fourcc}: {r['status']} opened={cam.get('opened')} fps={cam.get('fps')} "
                  f"read_fail={cam.get('read_fail')} bad_frames={cam.get('bad_frames')}")
            out.append(dict(index=i, fourcc=fourcc, **r)); time.sleep(3)
    save("E0_solo", out)


EXPS = dict(E0=E0, E1=E1, E2=E2, E3=E3, E4=E4, E5=E5, E6=E6, E7=E7)


def other_camera_users():
    """Windows only: python processes that hold cameras (the app, a soak run, another experiment)."""
    if os.name != "nt":
        return []
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne %d -and $_.CommandLine -match "
          "'web_server|camera_soak|camera_experiments|launcher\\.py|app\\.py|MONGDEE-AI-Booth' } | "
          "ForEach-Object { \"$($_.ProcessId) $($_.CommandLine)\" }" % os.getpid())
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30)
        return [l.strip() for l in r.stdout.splitlines() if l.strip() and "--worker" not in l]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", default="all"); ap.add_argument("--indices", default="1,2")
    ap.add_argument("--backend", default="dshow"); ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--w", type=int, default=320); ap.add_argument("--h", type=int, default=240)
    ap.add_argument("--fps", type=float, default=15); ap.add_argument("--secs", type=float, default=60)
    ap.add_argument("--stagger", type=float, default=0.0); ap.add_argument("--msmf-hw", type=int, default=None)
    ap.add_argument("--cooldown", type=float, default=20.0); ap.add_argument("--camera-name", default="")
    ap.add_argument("--fake", action="store_true"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    if a.list:
        print("\n".join(f"{k}: {v.__doc__ or v.__name__}" for k, v in EXPS.items())); return
    busy = [] if (a.fake or a.force) else other_camera_users()
    if busy:
        print("STOP: another process is probably using the cameras, so every open would fail and the results")
        print("would be meaningless (this already happened once: a soak run and the experiments overlapped).")
        print("Close these first, or pass --force:")
        for line in busy:
            print("   ", line[:200])
        sys.exit(2)
    for k in (EXPS if a.run == "all" else a.run.split(",")):
        print(f"=== {k} ==="); EXPS[k](a)


if __name__ == "__main__":
    main()
