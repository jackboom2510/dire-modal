"""Modal làm **backend** cho demo: endpoint GPU trả DIRE map thật trong vài giây.

Vì sao dùng Modal ở đây
-----------------------
Demo trên HF Spaces (CPU free) mất **101 s** cho một ảnh `S=5`; trên A10G là **0,92 s**.
Nút "phân tích ảnh" của một buổi nói không thể là 101 s, nên phần tính DIRE phải
chạy ở nơi có GPU. Modal đã có sẵn checkpoint trên Volume (`/zero/ckpt/adm_lsun.pt`,
2,5 GB) — không phải upload lại đâu cả, đó là lợi thế lớn nhất so với một dịch vụ khác.

Ba thứ Modal cho, và đúng ba thứ demo cần
-----------------------------------------
* `@modal.asgi_app` — một FastAPI app thật, có route, mã lỗi, header;
* `@modal.enter(snap=True)` + `enable_memory_snapshot` — nạp ADM **trước** khi chụp
  snapshot, để container ấm không phải nạp lại 2,5 GB;
* `scaledown_window` / `min_containers` — chọn giữa "trả tiền cho container ấm"
  và "chờ cold start". Số cụ thể ở `docs/plans/demos/demo-modal-plan.md` §3.

Lệnh::

    modal deploy -m dire_zero.demo_api                    # endpoint công khai
    modal serve  -m dire_zero.demo_api                    # URL tạm để thử

    curl -s https://<workspace>--dire-demo-api.modal.run/health | jq

    # một ảnh, S=5
    python - <<'PY'
    import base64, json, urllib.request, pathlib
    body = json.dumps({"image_b64": base64.b64encode(pathlib.Path("a.png").read_bytes()).decode(),
                       "steps": 5}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json",
                                             "X-Demo-Token": TOKEN})
    print(json.loads(urllib.request.urlopen(req).read())["line"])
    PY

Secret: `modal secret create dire-demo DEMO_TOKEN=… DEMO_ORIGINS=https://<space>.hf.space`

Chưa có token ⇒ endpoint trả **503**, không phải chạy không bảo vệ. Đó là mặc định
có chủ ý: URL công khai + GPU = hoá đơn do người khác bấm.
"""

from __future__ import annotations

import modal

from . import volume as V

APP_NAME = "dire-demo"
CKPT = V.CKPT

#: A10G: 1,10 USD/giờ, 0,0917 s/step (đo) ⇒ 3,67 s/ảnh ở `S=20`.
GPU = "A10G"

#: Giữ container ấm bao lâu sau request cuối. 10 phút = cả một buổi nói, và
#: ~0,19 USD nếu không ai gọi lại trong 10 phút đó.
SCALEDOWN_S = 600

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "numpy==1.26.4", "pillow==10.4.0", "fastapi[standard]")
    .add_local_python_source("dire_zero")
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(V.VOLUME_NAME, create_if_missing=True)
VOLUMES = {V.MOUNT: vol}

#: Nạp model **một lần cho mỗi container**, giữ trong closure của ASGI app.
#: `enable_memory_snapshot` chụp sau khi factory chạy xong, nên phần nạp 2,5 GB
#: nằm trong snapshot chứ không nằm trên đường request.
@app.function(
    gpu=GPU,
    volumes=VOLUMES,
    timeout=120,
    scaledown_window=SCALEDOWN_S,
    enable_memory_snapshot=True,
    # Để trống ⇒ scale về 0 khi không ai gọi (0 USD lúc rảnh). Đặt
    # `min_containers=1` trong lúc nói nếu cần độ trễ thấp — xem plan §3.
    min_containers=None,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def api():
    import os

    from .checkpoint import load_adm, sha256_file
    from .demo_service import TorchEngine, build_app, origins_from_env, token_from_env
    from .diffusion import ModelCounter

    if not os.path.exists(CKPT):
        raise FileNotFoundError(
            f"thiếu {CKPT}. Đẩy lên bằng:\n"
            f"  modal volume put {V.VOLUME_NAME} <256x256_diffusion_uncond.pt> ckpt/adm_lsun.pt"
        )
    model = load_adm(CKPT, device="cuda", dtype="float16", strict=True)
    model.eval()
    engine = TorchEngine(ModelCounter(model), digest=sha256_file(CKPT), device="cuda")
    return build_app(
        engine,
        token=token_from_env(),
        origins=origins_from_env(),
        source=f"Modal {GPU} · {os.environ.get('MODAL_TASK_ID', 'local')[:8]}",
    )
