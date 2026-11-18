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
from pathlib import Path

import gradio as gr

from demolib import charts, cost, provenance, remote, toy2d

#: Tắt SSR trước khi launcher của HF đọc biến này. SSR = một tiến trình Node 24
#: đứng trước Python; nó chỉ thêm một chỗ có thể chết (log của lần deploy hỏng
#: kết thúc bằng "Stopping Node.js server...") mà không thêm gì cho 5 tab tĩnh.
#: `setdefault` để vẫn bật lại được bằng biến môi trường của Space khi cần đo.
os.environ.setdefault("GRADIO_SSR_MODE", "False")

HERE = Path(__file__).resolve().parent
_raw_gallery = Path(os.environ.get("DIRE_GALLERY_DIR", "gallery"))
GALLERY_DIR = _raw_gallery if _raw_gallery.is_absolute() else HERE / _raw_gallery

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


def dire_status_markdown() -> str:
    url, tok = _remote_creds()
    if remote.configured(url, tok):
        host = url.split("//")[-1].split("/")[0]
        h = remote.health(url)
        extra = ""
        if h is not None:
            extra = (f"\n\nengine: `{h.engine}` · S ∈ [{h.s_range[0]}, {h.s_range[1]}] · "
                     f"tối đa {h.max_bytes / 1024 / 1024:.0f} MB/ảnh")
        return f"Đang dùng endpoint GPU: `{host}` — DIRE tính trên Modal (A10G), ~0,9 s cho `S=5`.{extra}"
    return (
        "**Chế độ này đang tắt.** Chưa có `DEMO_URL`/`DEMO_TOKEN`.\n\n"
        "Tính tại chỗ trên CPU của Space là **101 s/ảnh** với `S=5` (đo trên 2 vCPU), "
        "vượt ngưỡng 45 s đã chốt ở `docs/demo-web-plan.md` §3 — nên không bật mặc định.\n\n"
        "Đặt hai secret đó (Settings → Secrets) để dùng endpoint Modal: "
        "`docs/demo-modal-plan.md`. Token nằm ở server của Space, không xuống trình duyệt."
    )


def run_dire(image_path: str | None, steps: int):
    """Ảnh → (gốc, tái tạo, map, dòng bằng chứng). **Không** trả xác suất.

    Đây là điểm khác biệt so với bản Gradio đầu tiên của repo: đường này kết thúc
    ở **DIRE map**, không ở `P(fake)`. Một ảnh lẻ cũng không đủ để kết luận: cần
    phân bố của cả hai lớp, và cần detector nhận map.
    """
    url, tok = _remote_creds()
    if not remote.configured(url, tok):
        return None, None, None, "Chưa cấu hình `DEMO_URL`/`DEMO_TOKEN` — không có gì để chạy."
    if not image_path:
        return None, None, None, "Hãy chọn ảnh, hoặc chọn một mẫu trong gallery."

    raw = Path(image_path).read_bytes()
    try:
        res = remote.call(url, tok, raw, int(steps), timeout=180.0)
    except remote.RemoteError as e:
        return None, None, None, f"⚠️ {e}"

    orig = _materialize(raw, "orig.png")
    recon = _materialize(res.recon_png, "recon.png")
    dmap = _materialize(res.map_png, "map.png")
    line = (
        f"`{res.line}`\n\n"
        f"- DIRE trung bình **{res.stats['mean']:.4f}** · p99 **{res.stats.get('p99', float('nan')):.4f}** "
        f"· lớn nhất **{res.stats['max']:.4f}**\n"
        f"- {res.nfe} NFE · **{res.seconds:.2f} s** GPU"
        + ("\n- Ảnh vào không phải 256×256 nên đã được resize — ADM chỉ học ở 256²." if res.resized else "")
        + "\n\n**Đây là DIRE map, không phải kết luận thật/giả.** Endpoint chỉ tính map; "
        "muốn có điểm phân loại phải có detector nhận map, và phải biết ngưỡng chọn ở đâu."
    )
    return orig, recon, dmap, line


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
                              for s in (5, 10, 20, 50, 100, 200)])
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


def build_app() -> gr.Blocks:
    with gr.Blocks(title="DIRE — sai số tái tạo", analytics_enabled=False) as demo:
        gr.Markdown(INTRO)

        # -- 1 · Gallery ----------------------------------------------------
        with gr.Tab("1 · Gallery"):
            gr.Markdown(gallery_header())
            choices = gallery_choices()
            if choices:
                pick = gr.Dropdown(choices, value=choices[0], label="Chọn mẫu",
                                   info="generator · split")
                panels = gr.Gallery(label="ảnh gốc · DIRE map · (tái tạo nếu có)",
                                    columns=3, object_fit="contain", height=340)
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

        # -- 2 · Tính DIRE thật ---------------------------------------------
        with gr.Tab("2 · Tính DIRE thật"):
            gr.Markdown(
                "Tab này chạy **đúng hai vòng lặp** của DIRE (đảo ngược DDIM rồi tái tạo) "
                "trên ADM. Nó trả **map**, không trả `P(fake)`."
            )
            gr.Markdown(dire_status_markdown())
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(type="filepath", label="ảnh vào (bất kỳ kích thước)",
                                        sources=["upload", "clipboard"])
                    steps_in = gr.Slider(1, 20, value=5, step=1, label="S (số bước DDIM)",
                                         info="mỗi bước là 2 lượt forward")
                    go = gr.Button("Chạy DIRE", variant="primary")
                with gr.Column():
                    out_line = gr.Markdown()
                    with gr.Row():
                        img_orig = gr.Image(label="ảnh gốc", height=220)
                        img_recon = gr.Image(label="x̂₀", height=220)
                        img_map = gr.Image(label="DIRE map", height=220)
            go.click(run_dire, [image_in, steps_in], [img_orig, img_recon, img_map, out_line])

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
            # ở mỗi pixel kéo — 61 ms × 7 × số sự kiện.
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
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        **_kw,
    )
