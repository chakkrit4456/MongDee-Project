"""MongDee Camera Agent — lightweight, network-connected webcam capture
client (MongDee_Cloud_Vercel_Remote_AI_Server_Master_Prompt.md sections 4,
57, 58). Captures locally via camera.gateway.CameraGateway (already
torch-free) and pushes frames to a Remote AI Server over HTTP.

Deliberately has NO dependency on torch/ultralytics/the vision/ package —
"Camera Client = Lightweight, AI Server = Heavy" (section 58) — so this can
run on any Windows machine with just opencv-python, numpy and requests
installed, without any GPU or multi-gigabyte AI framework.
"""
