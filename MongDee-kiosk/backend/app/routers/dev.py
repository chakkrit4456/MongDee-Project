"""Scene switcher for the simulated camera. Only mounted when KIOSK_FAKE_CAMERA is set."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.fake_camera import scene_control

router = APIRouter()


class SceneRequest(BaseModel):
    scene: str | None = None
    auto: bool | None = None


@router.get("/api/dev/fake-camera")
def get_state():
    return scene_control.state()


@router.post("/api/dev/fake-camera")
def set_state(body: SceneRequest):
    if body.scene is not None and not scene_control.set_scene(body.scene):
        raise HTTPException(404, f"ไม่พบฉาก {body.scene}")
    if body.auto is not None:
        scene_control.set_auto(body.auto)
    return scene_control.state()


@router.post("/api/dev/fake-camera/reload")
def reload_scenes():
    scene_control.reload()
    return scene_control.state()


_PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fake camera</title>
<style>body{font-family:sans-serif;margin:16px;max-width:640px}button{margin:4px;padding:10px 14px;font-size:16px}
button.on{background:#2b7;color:#fff}img{width:100%;border:1px solid #888;margin-top:12px}</style>
<h2>กล้องจำลอง</h2><div id="btns"></div>
<p><button id="auto"></button><button id="reload">โหลดรูปใหม่</button></p>
<img src="/api/camera/stream" alt="live">
<script>
async function call(body){const r=await fetch('/api/dev/fake-camera'+(body&&body.reload?'/reload':''),body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body.reload?{}:body)}:{});return r.json()}
function draw(s){const b=document.getElementById('btns');b.innerHTML='';
s.scenes.forEach(n=>{const e=document.createElement('button');e.textContent=n==='empty'?'แท่นว่าง':n;if(n===s.scene)e.className='on';e.onclick=async()=>draw(await call({scene:n}));b.appendChild(e)});
const a=document.getElementById('auto');a.textContent='โหมดอัตโนมัติ: '+(s.auto?'เปิด':'ปิด');a.onclick=async()=>draw(await call({auto:!s.auto}))}
document.getElementById('reload').onclick=async()=>draw(await call({reload:1}));
async function poll(){draw(await call())}poll();setInterval(poll,3000);
</script>"""


@router.get("/dev/fake-camera", response_class=HTMLResponse)
def page():
    return _PAGE
