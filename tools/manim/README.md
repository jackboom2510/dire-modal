# `tools/manim` — video giải thích DIRE

Đây là **tiện ích**, không phải quy trình thực nghiệm: `scripts/` giữ đúng 2 quy
trình (`01_repro.sh`, `02_improve.sh`), còn thứ chỉ để *xem* thì ở `tools/`.

## 1. Năm scene

| Scene | Nội dung | Số lấy từ |
|---|---|---|
| `DiffusionTheory` | forward → reverse DDIM → inversion → residual → trực giác manifold | không có số |
| `DirePipeline` | ảnh → DIRE map → ResNet-50 → **1 logit** → sigmoid → P(fake) | không có số |
| `PaperTable2` | 8 generator × ACC của paper | `src/dire_modal/paper_table2.tsv` |
| `DataPipeline` | Baidu 76.6 KiB/s → staging → volume put → verify → eval 15 cặp × 4 | manifest + số đo |
| `TrainMechanism` | đường loss **thật** của `demo_train` | `docs/media/demo_train_log.json` |

Ba quy tắc, có `tests/test_manim_scenes.py` canh:

1. **Không có số nào viết thẳng trong `scenes.py`.** Số đến từ `data.py`, và
   `data.py` đọc TSV / manifest / JSON. Sửa dữ liệu ⇒ hình đổi theo.
2. **Chưa có log thì không vẽ đường cong.** `TrainMechanism` hiện khung "chưa có
   log" kèm lệnh cần chạy. Một đường loss bịa trông y hệt đường thật.
3. **Mọi scene có số mang nhãn nguồn** trên khung hình ("số của paper",
   "đo 2026-10-25", "demo — KHÔNG phải số của paper").

## 2. Render ở máy

```bash
# Linux (Debian/Ubuntu) — cairo/pango PHẢI cài trước, không có wheel
sudo apt install -y ffmpeg libcairo2-dev libpango1.0-dev pkg-config build-essential
python3 -m pip install manim

# Windows: chỉ cần ffmpeg, manimpango/pycairo có wheel
winget install Gyan.FFmpeg
py -m pip install manim
# nếu pip vẫn đòi build cairo:  conda install -c conda-forge manim

bash tools/manim/render.sh --check                  # liệt kê scene
bash tools/manim/render.sh --quality l              # nháp 480p15, vài giây/scene
bash tools/manim/render.sh --quality h --scene TrainMechanism
bash tools/manim/render.sh --quality h --gif        # kèm GIF 640px cho README
```

Đầu ra: `docs/media/<Scene>.mp4` (+ `.gif` nếu `--gif`). Thư mục tạm của manim là
`docs/media/_manim/` (đã gitignore).

**Không dùng `MathTex`**: nó cần TeX Live 1–2 GB. Công thức trong scene viết bằng
`Text` + ký hiệu Unicode. Font mặc định `DejaVu Sans` (phủ đủ dấu tiếng Việt);
đổi bằng `DIRE_FONT`.

## 3. Render trên Modal (tái lập được)

Máy Windows không có cairo, CI không có ffmpeg, và người clone repo không nên phải
cài 1.5 GB TeX/cairo để xem một video. Cách tái lập: **image riêng**.

> ⚠ **Không** cài manim vào `cpu_image`/`gpu_image`: manim cần `numpy>=2.1`, còn
> hai image đó pin `numpy==1.26.4` cho torch. Thêm manim vào là hỏng eval/train.

```python
# image riêng, chỉ dùng cho việc render — chưa có trong repo (xem mục 5)
manim_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libcairo2-dev", "libpango1.0-dev",
                 "pkg-config", "build-essential")
    .pip_install("manim==0.22.1")
)
```

Chi phí: render 5 scene ở 1080p60 trên CPU container ≈ vài phút ≈ **$0.01**; build
image lần đầu ~5 phút (apt + build cairo), sau đó cache.

## 4. Dữ liệu cho `TrainMechanism`

```bash
# 1. train thật (xem docs/adr-0003-demo-training.md) — L4, ~1 phút, ~$0.013
modal run -m dire_modal.demo_train::main --mode run

# 2. kéo log về
modal volume get dire-ckpt demo_train/demo_train ./docs/media
mv docs/media/demo_train/loss_log.json docs/media/demo_train_log.json

# 3. render
bash tools/manim/render.sh --scene TrainMechanism --quality h
```

Chưa có log ⇒ scene vẫn render, nhưng hiện khung *"Chưa có log thật — scene này
KHÔNG vẽ đường cong giả"*. Đó là trạng thái đúng, không phải lỗi.

## 5. Còn thiếu (chưa có trong repo)

| Việc | Trạng thái |
|---|---|
| `src/dire_modal/media.py` — `manim_image` + `@app.function` render trên Modal | ❌ chưa viết |
| GIF nhúng README | ❌ cần render trước (máy bạn hoặc Modal) |
| Scene cho AEROBLADE / `compute_dire` | ❌ ngoài phạm vi 5 scene hiện tại |
| Test render thật (`manim -ql` một frame) | ⏭ tự `skip` khi máy không có manim |
