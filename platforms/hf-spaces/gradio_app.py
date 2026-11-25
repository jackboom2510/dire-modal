"""DIRE — demo thuyết trình bằng **Gradio** (thay cho bản Streamlit).

Vì sao có tệp này
-----------------
Bản trước là Streamlit. Gradio được dựng lại vì một lý do cụ thể, không phải vì
thẩm mỹ: SDK `gradio` của HF Spaces tự lo hàng đợi, trạng thái phiên và nút
*Restart*; ba thứ đó là ba chỗ hỏng của một demo đứng trước người xem.

**Cái KHÔNG được dựng lại** là tab cũ của bản Gradio đầu tiên: upload ảnh RGB →
`P(fake)`. Model ResNet-50 trong repo nhận **DIRE map**, không nhận ảnh RGB.
Đưa ảnh RGB vào là chấm một phân bố chưa từng được train; con số vẫn ra, vẫn
trông hợp lý, và không có nghĩa gì. Lý do đầy đủ: `docs/demo-web-plan.md` §2.

Vì thế, ở đây **chỉ có một chỗ nhận ảnh của người xem**: tab *Tính DIRE thật*.
Nó tính **map** rồi hiện map — và nói rõ là không chấm. Muốn có điểm phân loại
thì phải có DIRE map **đã tính** đem qua detector, tức là tab *Gallery*, nơi mỗi
mẫu có `.json` ghi lại map đến từ đâu.

Dùng chung `demolib/` với bản Streamlit
--------------------------------------
`toy2d` (số học), `provenance` (nguồn gốc mẫu), `charts` (SVG), `cost` (chi phí),
`remote` (gọi endpoint Modal) — không có dòng nào của giao diện quyết định một
con số. Đó là lý do hai bản UI dùng chung được mà không lệch nhau về số.

Chạy
----
::

    DIRE_GALLERY_DIR=gallery python gradio_app.py       # http://127.0.0.1:7860
    GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 python gradio_app.py
"""

from __future__ import annotations

import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr

from demolib import charts, cost, provenance, remote, toy2d, live, frame_cache, cached_illustration

#: Tắt SSR trước khi launcher của HF đọc biến này. SSR = một tiến trình Node 24
#: đứng trước Python; nó chỉ thêm một chỗ có thể chết (log của lần deploy hỏng
#: kết thúc bằng "Stopping Node.js server...") mà không thêm gì cho 5 tab tĩnh.
#: `setdefault` để vẫn bật lại được bằng biến môi trường của Space khi cần đo.
os.environ.setdefault("GRADIO_SSR_MODE", "False")

HERE = Path(__file__).resolve().parent
_gallery_env = os.environ.get("DIRE_GALLERY_DIR")
_repo_gallery = HERE.parents[1] / "modal-runs" / "gallery"
if _gallery_env:
    _raw_gallery = Path(_gallery_env)
    GALLERY_DIR = _raw_gallery if _raw_gallery.is_absolute() else HERE / _raw_gallery
else:
    # In the monorepo, use the audited multi-generator gallery; in a standalone
    # Space export, fall back to its colocated gallery directory.
    GALLERY_DIR = _repo_gallery if any(_repo_gallery.glob("*.json")) else HERE / "gallery"

#: Câu bắt buộc mỗi lần nhắc tới số của bài báo. Viết một lần, dùng nhiều chỗ —
#: và test chặn mọi câu tự nhận độ chính xác của bài (xem
#: `tests/test_hf_spaces_gradio.py`).
PAPER = "số của paper (ICCV 2023), không phải của demo này"

TABLE2 = {
    "generators": ["ADM", "DDPM", "iDDPM", "PNDM", "SD-v1", "SD-v2", "LDM", "VQDiff"],
    "rows": [
        ("CNNDet*", ["100/100", "87,3/99,6", "100/100", "77,8/99,1", "77,4/85,8", "83,4/98,2", "96,0/99,9", "70,3/96,1"]),
        ("Patchfor*", ["100/100", "72,9/100", "100/100", "96,6/100", "63,2/71,3", "97,2/100", "97,3/100", "100/100"]),
        ("F3Net*", ["94,3/99,6", "93,7/99,6", "94,8/99,9", "94,1/99,2", "86,1/95,3", "83,6/91,7", "92,5/97,8", "93,4/98,7"]),
        ("DIRE (ADM)", ["100/100", "100/100", "100/100", "99,7/100", "99,7/100", "100/100", "100/100", "100/100"]),
    ],
}

PIPELINE = [
    ("maps", "kiểm tồn kho map: ô nào đủ/thiếu, tiền còn phải trả, lệnh chạy tiếp",
     "modal run -m dire_zero.modal_app::maps --n 500"),
    ("dire", "tính DIRE map, `2S` NFE/ảnh, lưu float16 thang [0, 2]",
     "modal run -m dire_zero.modal_app::dire --splits train,test --generator adm --n 500 --skip-existing"),
    ("train", "ResNet-50 trên map; ngưỡng chọn trên **val**, test chấm một lần",
     "modal run -m dire_zero.modal_app::train --generator adm --epochs 40"),
    ("evaluate", "chấm điểm **từng ảnh** → `scores.json` cho gallery",
     "modal run -m dire_zero.modal_app::evaluate --generator adm"),
    ("report", "gom kết quả + nhãn mức tái lập + cảnh báo",
     "modal run -m dire_zero.modal_app::report"),
]

#: `::dire --steps 20` là cấu hình mặc định; mọi ước lượng tiền dùng số này.
S_DEFAULT = 20


# ---------------------------------------------------------------------------
# Dữ liệu của gallery — đọc một lần lúc dựng app
# ---------------------------------------------------------------------------
SAMPLES, PROBLEMS = provenance.discover(GALLERY_DIR)


def gallery_choices() -> list[str]:
    """Nhãn trong dropdown. Rỗng là trạng thái **hợp lệ**, không phải lỗi."""
    return [s.selector for s in SAMPLES]


def _by_selector(selector: str) -> provenance.Sample | None:
    for s in SAMPLES:
        if s.selector == selector:
            return s
    return None


def gallery_summary() -> tuple[int, int]:
    """`(tổng số mẫu, số mẫu được phép hiện điểm)` — hiện ở đầu trang."""
    return len(SAMPLES), sum(1 for s in SAMPLES if s.score()[0])


def show_sample(selector: str):
    """Một mẫu → (ảnh trong gallery, dòng bằng chứng, khối điểm, JSON nguồn gốc).

    Quy tắc hiện điểm nằm ở `provenance.Sample.score()`, **không** ở đây: giao
    diện không được là nơi quyết định một con số có đáng hiện hay không.
    """
    s = _by_selector(selector)
    if s is None:
        return [], "Chưa chọn mẫu nào.", "", None

    panels: list[tuple[str, str]] = [(str(s.orig), "ảnh gốc 256×256"), (str(s.diff), "|x − x̂₀| · DIRE map")]
    if s.recon:
        panels.append((str(s.recon), "ảnh tái tạo (ADM, DDIM)"))

    score, reason = s.score()
    if score:
        p = float(score["p_fake"])
        thr = float(score.get("threshold", 0.5))
        verdict = "FAKE" if p >= thr else "REAL"
        block = (
            f"### P(fake) = **{p:.3f}**\n\n"
            f"Nhãn theo ngưỡng **{thr:.2f}**: **{verdict}** — ngưỡng chọn trên "
            f"`{score.get('threshold_split')}` (nguồn: {score.get('source', '—')}).\n\n"
            "Detector nhận **DIRE map**, không nhận ảnh RGB."
        )
    else:
        block = f"### P(fake) = —\n\n⚠️ {reason}"

    meta = dict(s.meta)
    meta["_sha256_ngắn"] = {k: str(v)[:12] + "…" for k, v in (s.sha256 or {}).items()}
    return panels, provenance.evidence_line(s), block, meta


def gallery_header() -> str:
    total, scored = gallery_summary()
    if total == 0:
        return (
            "### Gallery đang rỗng\n\n"
            "Không có panel trống giả ở đây: một demo không có map thì không có gì để "
            "chấm, và ba khung xám sẽ khiến người xem tưởng app hỏng.\n\n"
            + provenance.GENERATE_HELP
        )
    line = f"**{total} mẫu** · **{scored}** mẫu được phép hiện `P(fake)` (qua cổng `gate = passed`)."
    if PROBLEMS:
        line += f"\n\n⚠️ {len(PROBLEMS)} tệp `.json` bị từ chối — xem cuối trang."
    return line


def gallery_map_overview(samples=None):
    """Show every validated DIRE map, grouped in generator order; real maps stay separate."""
    rows = list(SAMPLES if samples is None else samples)
    by_gen = {}
    for sample in rows:
        by_gen.setdefault(sample.generator, []).append(sample)
    labels = {"adm": "ADM", "ddpm": "DDPM", "iddpm": "iDDPM", "pndm": "PNDM",
              "dalle2": "DALL·E 2", "diff-projectedgan": "Projected GAN",
              "diff-stylegan": "StyleGAN khuếch tán", "if": "Imagen / IF"}
    fake, real = [], []
    for gen, group in sorted(by_gen.items()):
        caption = labels.get(gen, gen)
        group = sorted(group, key=lambda sample: (sample.split, sample.name))
        target = real if gen == "real" else fake
        kind = "Ảnh thật" if gen == "real" else "Ảnh sinh"
        target.extend((str(sample.diff), f"{kind} · {caption} · {sample.split} · {sample.steps} bước")
                      for sample in group)
    chart = map_coverage_svg({g: len(v) for g, v in sorted(by_gen.items())})
    return chart, fake, real


def map_coverage_svg(counts: dict[str, int]) -> str:
    """Coverage chart only: counts validated gallery records, never map scores."""
    import html
    if not counts:
        return "<p>Chưa có metadata mẫu để lập biểu đồ độ phủ.</p>"
    max_count = max(counts.values()) or 1
    names = {"adm": "ADM", "ddpm": "DDPM", "iddpm": "iDDPM", "pndm": "PNDM",
             "dalle2": "DALL·E 2", "diff-projectedgan": "Projected GAN",
             "diff-stylegan": "StyleGAN khuếch tán", "if": "Imagen / IF", "real": "Ảnh thật"}
    row_h, left, bar_w = 28, 140, 420
    height = max(58, 28 * len(counts) + 28)
    parts = [f'<svg viewBox="0 0 640 {height}" role="img" aria-label="Số mẫu gallery theo generator" xmlns="http://www.w3.org/2000/svg">']
    for i, (key, val) in enumerate(sorted(counts.items())):
        y = 8 + i * row_h
        name = html.escape(names.get(key, key))
        w = max(2, int(bar_w * val / max_count))
        parts += [f'<text x="{left-10}" y="{y+16}" text-anchor="end" font-size="12" fill="#183746">{name}</text>',
                  f'<rect x="{left}" y="{y+2}" width="{w}" height="18" rx="5" fill="#17848a"/>',
                  f'<text x="{left+w+8}" y="{y+16}" font-size="12" fill="#183746">{val}</text>']
    parts.append("</svg>")
    return "".join(parts)


def cached_illustration_frame(steps: int, index: int):
    """Show a pre-rendered oracle-noise illustration without invoking ADM."""
    try:
        return cached_illustration.frame(int(steps), int(index))
    except (OSError, ValueError, KeyError) as e:
        return [], f"Không đọc được cache minh họa: {e}"


def clear_local_frame_cache() -> str:
    count = frame_cache.clear()
    return f"Đã xóa {count} lượt cache DIRE cục bộ. Cache mô phỏng tĩnh không bị xóa."


# ---------------------------------------------------------------------------
# Tab 3 — tính DIRE thật qua endpoint Modal (không chấm ảnh)
# ---------------------------------------------------------------------------
def _remote_creds() -> tuple[str, str]:
    url = os.environ.get("DEMO_URL", "")
    tok = os.environ.get("DEMO_TOKEN", "")
    # Trên HF Spaces, secret nằm ở biến môi trường; `st.secrets` không dùng ở đây.
    return url.strip(), tok.strip()


def _materialize(png: bytes, name: str) -> str:
    """Ghi PNG ra tệp tạm để Gradio hiện được — endpoint trả bytes, không trả đường dẫn."""
    d = Path(tempfile.mkdtemp(prefix="dire-demo-"))
    p = d / name
    p.write_bytes(png)
    return str(p)


@lru_cache(maxsize=1)
def _load_local_model_cached(path_text: str, mtime_ns: int, size: int):
    # mtime/size invalidate the in-process model if an operator replaces weights.
    return live.load_checkpoint(Path(path_text))


def local_status_markdown() -> str:
    status = live.status()
    if status.available:
        return (
            f"**Backend mặc định: tính tại chỗ trên CPU** · ADM checkpoint `{Path(status.ckpt).name}` "
            f"· torch {status.torch}. Ảnh đầu vào không rời máy chủ này. "
            "Kết quả và từng frame được lưu cache cục bộ theo hash ảnh + checkpoint + S."
        )
    return (
        "**Backend mặc định: tại chỗ.** Mô phỏng DDIM tổng hợp đã có sẵn trong cache để xem từng bước. "
        f"Tính DIRE thật bằng ADM hiện chưa sẵn sàng: {status.reason}. "
        "Khi torch và checkpoint ADM 256×256 được gắn tại chỗ, cùng nút sẽ chạy local và cache từng ảnh x_t / x̂₀; "
        "không tự gọi Modal hay tải checkpoint."
    )


def dire_status_markdown() -> str:
    return local_status_markdown()


def _local_result(image_path: str, steps: int, progress=None):
    status = live.status()
    if not status.available:
        raise RuntimeError(status.reason)
    checkpoint = Path(status.ckpt)
    st = checkpoint.stat()
    model, digest = _load_local_model_cached(str(checkpoint), st.st_mtime_ns, st.st_size)
    raw = Path(image_path).read_bytes()
    key = frame_cache.make_key(raw, digest, int(steps))
    result = frame_cache.load(key)
    cache_hit = result is not None
    if result is None:
        n = 0
        total = 2 * int(steps)

        def on_frame(item):
            nonlocal n
            n += 1
            if progress is not None:
                progress(n / total, desc=f"{item['phase']} · t={item['t']} · {n}/{total} lượt mạng")

        result = live.run_steps(model, image_path, steps=int(steps), on_frame=on_frame)
        frame_cache.store(key, result)
        result = frame_cache.load(key)
        if result is None:
            raise RuntimeError("Không đọc lại được cache vừa ghi.")
    folder = frame_cache.cache_root() / key
    filmstrip = []
    for i, item in enumerate(result["frames"]):
        phase = "đảo DDIM" if item["phase"] == "invert" else "tái tạo"
        filmstrip.extend([
            (str(folder / item["x_t"]), f"{i+1:02d} · {phase} · x_t · t={item['t']}"),
            (str(folder / item["x0_hat"]), f"{i+1:02d} · {phase} · x̂₀ · t={item['t']}"),
        ])
    result["cache_hit"] = cache_hit
    return raw, result, filmstrip, key


def run_dire(image_path: str | None, steps: int, backend: str = "local", progress=gr.Progress()):
    """Local ADM is default; remote GPU requires explicit backend selection."""
    if not image_path:
        return None, None, None, [], "Hãy chọn ảnh đầu vào. Có thể dùng mô phỏng cache bên dưới nếu chưa có checkpoint."
    if not 1 <= int(steps) <= 20:
        return None, None, None, [], "S phải nằm trong [1, 20]."

    if backend == "local":
        try:
            raw, result, filmstrip, key = _local_result(image_path, int(steps), progress)
        except Exception as e:
            return None, None, None, [], f"⚠️ Không chạy được ADM tại chỗ: {type(e).__name__}: {e}"
        orig = _materialize(raw, "orig.png")
        recon = _materialize(result["recon_png"], "recon.png")
        dmap = _materialize(result["map_png"], "map.png")
        cache_note = "cache hit · không chạy U-Net" if result["cache_hit"] else "cache miss · frames đã lưu"
        line = (
            f"`DIRE: local ADM · S={result['steps']} · {result['nfe']} NFE · {result['seconds']:.1f} s · "
            f"{cache_note} · cache {key[:12]}`\n\n"
            f"DIRE trung bình **{result['map_values']['mean']:.4f}** · p99 **{result['map_values']['p99']:.4f}** · "
            f"lớn nhất **{result['map_values']['max']:.4f}**. Đây là map, không phải kết luận thật/giả."
        )
        return orig, recon, dmap, filmstrip, line

    if backend == "modal":
        url, tok = _remote_creds()
        if not remote.configured(url, tok):
            return None, None, None, [], "Modal chưa được chọn cấu hình: cần DEMO_URL/DEMO_TOKEN. Không có lời gọi mạng nào được thực hiện."
        raw = Path(image_path).read_bytes()
        try:
            res = remote.call(url, tok, raw, int(steps), timeout=180.0)
        except remote.RemoteError as e:
            return None, None, None, [], f"⚠️ {e}"
        orig = _materialize(raw, "orig.png")
        recon = _materialize(res.recon_png, "recon.png")
        dmap = _materialize(res.map_png, "map.png")
        line = (
            f"`{res.line}`\n\nDIRE trung bình **{res.stats['mean']:.4f}** · "
            f"p99 **{res.stats.get('p99', float('nan')):.4f}** · {res.nfe} NFE · {res.seconds:.2f} s GPU. "
            "Map, không phải kết luận thật/giả."
        )
        return orig, recon, dmap, [], line
    return None, None, None, [], "Backend không hợp lệ; chọn local (mặc định) hoặc Modal (chủ động)."


# ---------------------------------------------------------------------------
# Tab 4 — thí nghiệm 2 chiều (chạy thật, 61 ms)
# ---------------------------------------------------------------------------
def run_toy(steps: int, n: int, clip: bool, seed: int):
    res = toy2d.run(steps=int(steps), n=int(n), clip=bool(clip), seed=int(seed))
    h = res["hist"]
    hist_svg = charts.histogram(h["edges"], h["gen"], h["real"])
    scatter_svg = charts.scatter(res["pairs"])
    sweep_svg = charts.sweep([{"steps": s, "ratio": toy2d.run(steps=s, n=min(int(n), 200),
                                                             clip=bool(clip), seed=int(seed))["ratio"]}
                              for s in (5, 10, 20, 100, 200)])
    metrics = (
        f"| | DIRE trung bình | trung vị | p90 |\n|---|---|---|---|\n"
        f"| ảnh sinh | {res['gen']['mean']:.4f} | {res['gen']['median']:.4f} | {res['gen']['p90']:.4f} |\n"
        f"| ảnh thật | {res['real']['mean']:.4f} | {res['real']['median']:.4f} | {res['real']['p90']:.4f} |\n\n"
        f"**tỉ lệ thật/sinh {res['ratio']:.2f}×** · **{res['nfe']} NFE/ảnh** · S={res['steps']} · "
        f"n={res['n']} mỗi lớp · seed={res['seed']}"
    )
    line = (f"`DIRE: thí nghiệm 2 chiều, score chính xác · detector: không có (không chấm ảnh) "
            f"· nhãn: minh hoạ lý thuyết · S={res['steps']}`")
    return hist_svg, scatter_svg, sweep_svg, metrics, line


# ---------------------------------------------------------------------------
# Tab 5 — lịch nhiễu, NFE, tiền
# ---------------------------------------------------------------------------
def schedule_rows(steps: int):
    marks = toy2d.timesteps(toy2d.T_TRAIN, int(steps))
    return [[t, round(toy2d.A[t], 6), round(math.sqrt(toy2d.A[t]), 5),
             round(math.sqrt(max(0.0, 1 - toy2d.A[t])), 5)] for t in marks]


def cost_block(n_images: int, steps: int, sec: float, gpu: str) -> tuple[str, str]:
    e = cost.estimate(int(n_images), int(steps), float(sec), gpu)
    return cost.estimate_line(e), (
        f"**NFE** {e['nfe_total']:,} · **GPU-giờ** {e['gpu_h']:.2f} · **USD** {e['usd']:.2f}"
    )


# ---------------------------------------------------------------------------
# Tab 6 — bảng của bài báo
# ---------------------------------------------------------------------------
def paper_table() -> list[list[str]]:
    return [[name, *cells] for name, cells in TABLE2["rows"]]


def paper_header_row() -> list[str]:
    return ["", *TABLE2["generators"]]


def repo_gaps() -> list[list[str]]:
    """Ba điều kiện để so được Table 2 — và tình trạng **thật** của repo."""
    return [
        ["1 000 ảnh mỗi ô", "500 (`--n 500`)", "CI ±2,5 điểm thay vì ±1,8"],
        ["8 generator", "3/8 cột (`adm`, `ddpm`, `iddpm`)", "chưa có ô SD-v1/SD-v2/LDM/VQDiff"],
        ["detector ImageNet pretrained", "`init=\"scratch\"`", "thấp hơn bảng, không cùng cấu hình"],
    ]


# ---------------------------------------------------------------------------
# Dựng giao diện
# ---------------------------------------------------------------------------
INTRO = f"""# DIRE — sai số tái tạo, không phải hộp đen

**Demo này không nhận ảnh của bạn để chấm thật/giả.** Nó cho xem ba thứ: một phép
đo chạy thật ngay trên CPU của Space, nguồn gốc của từng con số, và phần nào của
bài báo mà repo này **chưa** tái lập được.

### Vì sao không có ô "upload ảnh → P(fake)"

Một ô upload kèm `P(fake)` là thứ dễ làm nhất và cũng dễ sai nhất. DIRE đo **sai
số tái tạo qua một mô hình khuếch tán**, nên detector phải nhận **DIRE map**,
không phải ảnh RGB. Đưa ảnh RGB vào một model học trên map là chấm một phân bố đầu
vào chưa từng được huấn luyện — con số vẫn ra, vẫn trông hợp lý, và không có nghĩa gì.

Table 5 của bài cho thấy đổi đầu vào là đổi hẳn kết quả: trên SD-v1, `RGB` được
77,4/85,8 còn `DIRE` được 99,7/100 — {PAPER}.

Tab **Tính DIRE thật** có nhận ảnh, nhưng nó trả **DIRE map**, không trả xác suất.
"""

CAVEATS = f"""### Điều không được quên khi trình bày

- `S` là số bước DDIM. Đổi `S` là đổi kết quả (Table 4: `S=5` 87,5 · `S=10` 98,2 ·
  `S=20` 99,7 — đều là {PAPER}).
- Ngưỡng chọn trên **val**, không bao giờ trên test.
- Repo này tái lập **from scratch**: `R1+R2` là map + detector của mình; `R0` là
  dùng map/checkpoint của tác giả để *chấm lại* — hai thứ khác nhau, và nhãn phải
  hiện cạnh số.
- CPU free, 0 ₫, không gọi mạng ngoài (trừ tab DIRE khi có secret), không lưu ảnh.
"""


# Gallery không ép chiều cao để tránh vùng cuộn riêng; ảnh output được giới hạn
# theo chiều cao màn hình và co vừa khung, không bị cắt méo.
IMAGE_FIT_CSS = """
.fit-image img {
    object-fit: contain !important;
    max-width: 100%;
    max-height: min(60vh, 560px);
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="DIRE — sai số tái tạo", analytics_enabled=False) as demo:
        gr.Markdown(INTRO)

        # -- 1 · Gallery ----------------------------------------------------
        with gr.Tab("1 · Gallery"):
            gr.Markdown(gallery_header())
            gr.Markdown(
                "### So sánh DIRE map theo generator\n"
                "Các ô dưới chỉ so **map với map**, cùng cấu hình hiển thị [0, 2] + gamma 0,5. "
                "Ảnh thật là nhóm riêng. Biểu đồ đếm số mẫu metadata hợp lệ, không phải điểm chất lượng."
            )
            coverage_chart = gr.HTML()
            fake_map_gallery = gr.Gallery(label="DIRE map · toàn bộ mẫu ảnh sinh, nhóm theo generator", columns=4,
                                          object_fit="contain", height=None)
            real_map_gallery = gr.Gallery(label="DIRE map · toàn bộ ảnh thật (nhóm riêng)", columns=4,
                                          object_fit="contain", height=None)
            demo.load(gallery_map_overview, outputs=[coverage_chart, fake_map_gallery, real_map_gallery])
            choices = gallery_choices()
            if choices:
                pick = gr.Dropdown(choices, value=choices[0], label="Chọn mẫu",
                                   info="generator · split")
                panels = gr.Gallery(label="ảnh gốc · DIRE map · (tái tạo nếu có)",
                                    columns=3, object_fit="contain", height=None)
                evidence = gr.Markdown()
                score_md = gr.Markdown()
                with gr.Accordion("Nguồn gốc đầy đủ (.json)", open=False):
                    meta_json = gr.JSON()
                pick.change(show_sample, pick, [panels, evidence, score_md, meta_json])
                demo.load(show_sample, pick, [panels, evidence, score_md, meta_json])
            else:
                gr.Markdown(
                    "Khi có map, dựng gallery rồi khởi động lại:\n\n"
                    "```bash\npython tools/make_gallery.py --maps maps/test --orig data/test \\\n"
                    "    --n 8 --steps 20 --label R1+R2 \\\n"
                    "    --map-source \"maps/test/*.npy · ::dire --steps 20\"\n```\n\n"
                    + provenance.SCHEMA_HELP
                )
            if PROBLEMS:
                with gr.Accordion(f"⚠ {len(PROBLEMS)} tệp `.json` bị từ chối", open=False):
                    gr.Markdown("\n".join(f"- `{p}`" for p in PROBLEMS))
            gr.Markdown("Ảnh trong gallery: DiffusionForensics (Wang et al., ICCV 2023), "
                        "chỉ dùng cho mục đích nghiên cứu.")

        # -- 2 · Tính DIRE tại chỗ + mô phỏng cache --------------------------
        with gr.Tab("2 · Mô phỏng & tính DIRE"):
            gr.Markdown(
                "Tính **local là backend mặc định** khi có ADM checkpoint. Lượt đầu chạy `2S` forward; "
                "mỗi ảnh x_t và x̂₀ được ghi vào cache cục bộ, lượt sau cùng ảnh/checkpoint/S đọc lại cache. "
                "Không gửi ảnh tới Modal. Nếu checkpoint chưa được gắn, dùng mô phỏng tổng hợp đã cache bên dưới."
            )
            gr.Markdown(dire_status_markdown())
            with gr.Accordion("Mô phỏng DDIM từng bước · đã lưu sẵn", open=True):
                gr.Markdown(
                    "Chuỗi tổng hợp có nhiễu Gauss cố định (oracle), dùng phương trình DDIM `η=0`. "
                    "Đây là minh họa toán học, **không** phải ADM inference / DIRE map / kết quả đo."
                )
                with gr.Row():
                    cache_steps = gr.Dropdown([5, 10, 20], value=5, label="S · số bước minh họa")
                    cache_frame = gr.Slider(0, 40, value=0, step=1, label="Frame trong chuỗi 2S")
                cache_gallery = gr.Gallery(label="x_t và x̂₀ tại frame đã chọn", columns=2,
                                           object_fit="contain", height=None)
                cache_caption = gr.Markdown()
                demo.load(cached_illustration_frame, [cache_steps, cache_frame], [cache_gallery, cache_caption])
                cache_steps.change(cached_illustration_frame, [cache_steps, cache_frame], [cache_gallery, cache_caption])
                cache_frame.change(cached_illustration_frame, [cache_steps, cache_frame], [cache_gallery, cache_caption])

            gr.Markdown("### Tính DIRE thật trên ảnh đầu vào")
            backend = gr.Radio(
                [("Tại chỗ · mặc định", "local"), ("Modal GPU · chọn chủ động", "modal")],
                value="local", label="Backend", info="Modal chỉ được gọi sau khi chọn rõ backend và bấm nút."
            )
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(type="filepath", label="Ảnh đầu vào (bất kỳ kích thước)",
                                        sources=["upload", "clipboard"])
                    steps_in = gr.Dropdown([5, 10, 20], value=5, label="S · số bước DDIM",
                                           info="mỗi DIRE map cần 2S lượt U-Net")
                    go = gr.Button("Chạy / đọc cache DIRE", variant="primary")
                with gr.Column():
                    out_line = gr.Markdown()
                    with gr.Row():
                        img_orig = gr.Image(label="Ảnh gốc", height="min(60vh, 560px)",
                                            elem_classes=["fit-image"])
                        img_recon = gr.Image(label="Tái tạo x̂₀", height="min(60vh, 560px)",
                                             elem_classes=["fit-image"])
                        img_map = gr.Image(label="DIRE map", height="min(60vh, 560px)",
                                           elem_classes=["fit-image"])
            live_frames = gr.Gallery(label="Từng frame đã cache · x_t và x̂₀ cho mỗi lượt mạng",
                                     columns=4, object_fit="contain", height=None)
            gr.Markdown(
                "Cache giữ tối đa 8 lượt; gồm các ảnh trung gian x_t/x̂₀ (có thể để lộ nội dung đầu vào), "
                "chỉ trên đĩa máy chủ cục bộ và không lưu bản gốc riêng. Có thể xóa cache bên dưới."
            )
            cache_notice = gr.Markdown()
            clear_cache_button = gr.Button("Xóa cache frame cục bộ", variant="secondary")
            clear_cache_button.click(clear_local_frame_cache, outputs=cache_notice)
            go.click(run_dire, [image_in, steps_in, backend],
                     [img_orig, img_recon, img_map, live_frames, out_line])

        # -- 3 · Thí nghiệm 2 chiều -----------------------------------------
        with gr.Tab("3 · Thí nghiệm 2 chiều"):
            gr.Markdown(
                "Dữ liệu là **hỗn hợp Gauss 2 chiều**, nên score có **công thức đóng** — "
                "\"mô hình\" là score đó (giả định học hoàn hảo). Mọi khác biệt giữa hai "
                "lớp đo *đúng* phần hình học mà DIRE dựa vào, không lẫn sai số huấn luyện."
            )
            with gr.Row():
                toy_steps = gr.Slider(1, 200, value=S_DEFAULT, step=1, label="S (bước DDIM)")
                toy_n = gr.Slider(50, 400, value=200, step=50, label="Số mẫu mỗi lớp")
                toy_clip = gr.Checkbox(value=True, label="Cắt x̂₀ về [−1, 1]")
                toy_seed = gr.Number(value=3407, precision=0, label="seed")
            toy_metrics = gr.Markdown()
            with gr.Row():
                toy_hist = gr.HTML(label="phân bố DIRE")
                toy_scatter = gr.HTML(label="x₀ → x̂₀")
            toy_sweep = gr.HTML(label="quét S")
            toy_line = gr.Markdown()
            toy_inputs = [toy_steps, toy_n, toy_clip, toy_seed]
            # `release` cho slider (chỉ tính khi thả tay), `change` cho checkbox/number
            # (không có thao tác "thả"). Dùng `change` cho slider sẽ chạy 7 lượt quét
            # ở mỗi pixel kéo — 61 ms × 5 × số sự kiện.
            toy_steps.release(run_toy, toy_inputs,
                              [toy_hist, toy_scatter, toy_sweep, toy_metrics, toy_line])
            for comp in (toy_n, toy_clip, toy_seed):
                comp.change(run_toy, toy_inputs,
                            [toy_hist, toy_scatter, toy_sweep, toy_metrics, toy_line])
            demo.load(run_toy, toy_inputs,
                      [toy_hist, toy_scatter, toy_sweep, toy_metrics, toy_line])
            gr.Markdown(
                "Khi `S` tăng, **cả hai** lớp đều giảm (tái tạo sát hơn) và tỉ lệ tăng rồi "
                "bão hoà quanh ~2 — tăng `S` không còn mua thêm khả năng phân biệt, đúng "
                "tinh thần Table 4. Kết quả này **không phải** Table 2: ở đây không có ảnh, "
                "không JPEG, không nhiễu cảm biến; tỉ lệ ~1,9× và ACC/AP ~100 là hai đại "
                "lượng khác nhau."
            )

        # -- 4 · Lịch & chi phí ---------------------------------------------
        with gr.Tab("4 · Lịch & chi phí"):
            gr.Markdown(
                "Hai con số hay bị lẫn: `--steps` lúc **sinh** ảnh (50) và `S` lúc **tính "
                "DIRE** (20). Mỗi ảnh DIRE tốn `2S` lượt mạng — một lượt đảo ngược, một "
                "lượt tái tạo."
            )
            sched_steps = gr.Slider(1, 200, value=S_DEFAULT, step=1, label="S")
            sched_table = gr.Dataframe(
                headers=["t", "ᾱ_t", "√ᾱ_t", "√(1−ᾱ_t)"],
                value=schedule_rows(S_DEFAULT), interactive=False, label="mốc thời gian",
            )
            sched_steps.release(schedule_rows, sched_steps, sched_table)
            gr.Markdown("`t` 0-based như ADM: `ᾱ₀ = 1 − β₀ = 0,9999`, không phải 1.")
            with gr.Row():
                n_images = gr.Number(value=800, precision=0, label="Số ảnh")
                sec_step = gr.Number(value=cost.SEC_PER_STEP_MEASURED, label="sec/step (đo được)")
                gpu_pick = gr.Dropdown(list(cost.PRICES), value=list(cost.PRICES)[0], label="GPU")
            cost_line_md = gr.Markdown()
            cost_metrics = gr.Markdown()
            cost_inputs = [n_images, sched_steps, sec_step, gpu_pick]
            for comp in cost_inputs:      # `change` hợp cho cả Slider/Number/Dropdown
                comp.change(cost_block, cost_inputs, [cost_line_md, cost_metrics])
            demo.load(cost_block, cost_inputs, [cost_line_md, cost_metrics])
            gr.Markdown(
                "Cùng công thức với `dire_zero.budget.Estimate` — test khoá hai bên khớp. "
                f"`{cost.SEC_PER_STEP_MEASURED}` s/step do `::bench` đo (ADM 256², A10G, fp16, "
                "batch 1). Kiểm chéo: 829 GFLOP mỗi lượt forward ÷ 0,0917 s = 9,0 TFLOPS = "
                "29 % đỉnh fp16 của A10G — bình thường cho batch 1."
            )
            gr.Markdown(
                "**Điểm dễ sai nhất**: mốc cuối là `round((S−1)·T/S) = 950` khi `S=20`, "
                "không phải `999`. Dùng trọn 1 000 bước cho ra map khác hẳn, trông vẫn hợp "
                "lý, và tốn 25 lần tiền."
            )

        # -- 5 · Bảng 2 & tái lập -------------------------------------------
        with gr.Tab("5 · Bảng 2 & tái lập"):
            gr.Markdown(f"### Table 2 của bài — toàn bộ bảng dưới là {PAPER}")
            gr.Dataframe(headers=paper_header_row(), value=paper_table(), interactive=False)
            gr.Markdown(
                "ACC/AP (%). `*` = baseline phải huấn luyện lại trên ảnh sinh của ADM; "
                "baseline **không** huấn luyện lại chỉ đạt 51–59 ACC, tức thất bại hoàn toàn "
                "trước ảnh khuếch tán."
            )
            gr.Markdown("### Ba điều kiện để so được với bảng đó — cả ba đang thiếu")
            gr.Dataframe(headers=["Điều kiện", "Repo hiện có", "Hệ quả"],
                         value=repo_gaps(), interactive=False)
            gr.Markdown("### Đường tái lập")
            for k, what, cmd in PIPELINE:
                gr.Markdown(f"**`::{k}`** — {what}\n\n```bash\n{cmd}\n```")
            gr.Markdown(CAVEATS)

        gr.Markdown(
            "---\n"
            "CPU free · 0 ₫ · không lưu ảnh người dùng · mã ở `platforms/hf-spaces/` · "
            "số của bài báo luôn kèm nhãn *“số của paper (ICCV 2023)”*."
        )
    return demo


#: Bắt buộc phải có, **ở cấp module**. HF Spaces (SDK gradio) không chạy
#: `python app_file.py` như một script: nó `import` tệp rồi tìm một object tên
#: `demo` để tự gọi `launch()`. Không có biến này thì log ra
#:
#:     GRADIO_HOT_RELOAD: Launching demo not found in __main__. Using 'demo'
#:     * Running on local URL: http://0.0.0.0:7860, with SSR
#:     Stopping Node.js server...
#:
#: — tức launcher rơi vào nhánh dự phòng, dựng SSR proxy, rồi không có app nào
#: phục vụ nên tiến trình Python kết thúc và proxy chết theo. Ứng dụng *không*
#: hỏng; nó chưa bao giờ được giao cho launcher.
demo = build_app()
# A single CPU inference at a time; concurrent uploads must not multiply RAM/runtime.
demo.queue(default_concurrency_limit=1, max_size=8)

#: Dòng "đã sẵn sàng" của chính app, để log của Space có một mốc rõ ràng giữa
#: hai loại dòng vô hại (`pip` cảnh báo root, `* To create a public link…`) và
#: lỗi thật (`Traceback`, `ModuleNotFoundError`, `Stopping Node.js server…`).
#: In ở cấp module vì đây là lúc HF import tệp — tức lúc nó chuẩn bị phục vụ.
print(
    f"[demo] {len(demo.blocks)} component · {len(SAMPLES)} mẫu gallery"
    + ("" if SAMPLES else " (gallery/ rỗng — hợp lệ, xem README)")
    + " · sẵn sàng cho launcher",
    flush=True,
)


if __name__ == "__main__":  # pragma: no cover - chỉ chạy khi phục vụ thật
    # `show_api` đã bị bỏ khỏi `launch()` ở Gradio 6; truyền vào sẽ TypeError ngay
    # lúc khởi động — tức Space build xong rồi chết ở giây đầu.
    #
    # `ssr_mode=False`: SSR là một tiến trình Node đứng trước Python. Trên CPU free
    # nó chỉ thêm một chỗ có thể chết (đúng như log "Stopping Node.js server...")
    # mà không thêm gì cho một app 5 tab. Chỉ truyền khi bản gradio đang cài có
    # tham số này — 5.x không có, và TypeError ở đây giết app ngay giây đầu.
    _kw = {}
    import inspect as _inspect  # noqa: PLC0415

    if "ssr_mode" in _inspect.signature(demo.launch).parameters:
        _kw["ssr_mode"] = False
    demo.launch(
        css=IMAGE_FIT_CSS,
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        **_kw,
    )
