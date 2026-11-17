"""DIRE — demo thuyết trình, chạy trên HF Spaces (Streamlit, CPU free).

Nguyên tắc của tệp này, ngắn hơn cả code
----------------------------------------
1. **Không** có đường nào nhận ảnh của người xem rồi trả `P(fake)`. Một model
   ResNet-50 học trên DIRE map không có nghĩa gì khi nhận ảnh RGB thô; app cũ
   (`platforms/hf-spaces/app.py` phiên bản Gradio) làm đúng điều đó, và đó là lý
   do nó bị thay. Xem `docs/demo-web-plan.md` §2.
2. Mọi con số hiện ra đều kèm **nguồn**: dòng
   `DIRE: … · detector: … · nhãn: … · S=…` nằm ngay dưới số, không nằm trong tab
   "Giới thiệu".
3. Không có chỗ nào biến độ chính xác của bài báo thành của demo. Số của bài
   báo, khi nhắc, phải kèm chữ **"số của paper (ICCV 2023)"** — và test chặn bằng
   chuỗi, nên đừng viết lại câu cấm đó ở dạng trích dẫn.
4. Trạng thái thiếu dữ liệu là **nội dung**: gallery rỗng thì hiện lệnh sinh
   gallery, không hiện ba khung ảnh trống.

Cái gì chạy thật ở đây
----------------------
`toy2d.py` chạy một mô hình khuếch tán 2 chiều có score **chính xác** (hỗn hợp
Gauss) rồi đo DIRE bằng đúng hai vòng lặp đảo ngược/tái tạo. Đó là phép đo thật,
chỉ không phải phép đo trên ảnh. Còn `charts.py` vẽ SVG, `provenance.py` kiểm
nguồn gốc mẫu — cả hai đều thuần Python, không tải gì lúc khởi động.

Cái gì nằm ở đâu
---------------
`demolib/` là thư viện của demo: `toy2d` (số học), `provenance` (nguồn gốc mẫu),
`charts` (SVG), `cost` (chi phí). Tệp này chỉ là giao diện. Lý do tách: mọi thứ
cần kiểm đều kiểm được bằng `pytest` mà không phải dựng một Streamlit app, và
không có dòng nào của giao diện quyết định một con số.
"""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path

import streamlit as st

from demolib import charts, cost, live, provenance, remote, toy2d

HERE = Path(__file__).resolve().parent
_raw_gallery = Path(os.environ.get("DIRE_GALLERY_DIR", "gallery"))
GALLERY_DIR = _raw_gallery if _raw_gallery.is_absolute() else HERE / _raw_gallery

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
    ("checkdata", "kiểm manifest, sha, cân bằng lớp **theo từng bài toán**",
     "modal run -m dire_zero.modal_app::checkdata --splits test"),
    ("bench", "đo `sec/step` — cổng bắt buộc trước mọi ước lượng tiền",
     "modal run -m dire_zero.modal_app::bench"),
    ("dire", "tính DIRE map, `2S` NFE/ảnh, lưu float16 thang [0, 2]",
     "modal run -m dire_zero.modal_app::dire --splits train,test --generator adm --n 200 --max-usd 2"),
    ("train", "ResNet-50 trên map; ngưỡng chọn trên **val**, test chấm một lần",
     "modal run -m dire_zero.modal_app::train --generator adm --epochs 40"),
    ("report", "gom kết quả + nhãn mức tái lập + cảnh báo",
     "modal run -m dire_zero.modal_app::report"),
]


# ---------------------------------------------------------------------------
# Trang
# ---------------------------------------------------------------------------
def page_intro(samples: list[provenance.Sample]) -> None:
    st.title("DIRE — sai số tái tạo, không phải hộp đen")
    st.markdown(
        "**Demo này không nhận ảnh của bạn.** Nó cho xem ba thứ: một phép đo chạy "
        "thật ngay trên CPU của Space, nguồn gốc của từng con số, và phần nào của "
        "bài báo mà repo này **chưa** tái lập được."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Mẫu trong gallery", len(samples))
    c2.metric("Mẫu có điểm detector", sum(1 for s in samples if s.score()[0]))
    c3.metric("Chi phí chạy demo", "0 ₫")

    st.markdown("### Vì sao không có ô upload ảnh")
    st.markdown(
        "Một ô upload kèm `P(fake)` là thứ dễ làm nhất và cũng dễ sai nhất. "
        "DIRE đo **sai số tái tạo qua một mô hình khuếch tán**, nên detector phải "
        "nhận **DIRE map**, không phải ảnh RGB. Đưa ảnh RGB vào một model học trên "
        "map là chấm một phân bố đầu vào chưa từng được huấn luyện — con số vẫn ra, "
        "vẫn trông hợp lý, và không có nghĩa gì.\n\n"
        "Table 5 của bài cho thấy đổi đầu vào là đổi hẳn kết quả: trên SD-v1, "
        f"`RGB` được 77,4/85,8 còn `DIRE` được 99,7/100 — {PAPER}."
    )

    st.markdown("### Ba thứ trong demo, và mỗi thứ đến từ đâu")
    st.table([
        {"Phần": "Thí nghiệm 2 chiều", "Chạy ở đâu": "ngay trên Space, thuần Python",
         "Nó là gì": "mô hình khuếch tán có score chính xác; đo DIRE thật",
         "Nhãn": "minh hoạ lý thuyết"},
        {"Phần": "Lịch nhiễu & NFE", "Chạy ở đâu": "ngay trên Space",
         "Nó là gì": "`ᾱ_t` từ `β_t` tuyến tính, khớp `dire_zero.schedule`",
         "Nhãn": "công thức"},
        {"Phần": "Gallery", "Chạy ở đâu": "ảnh tính trước, xem tại chỗ",
         "Nó là gì": "ảnh gốc · DIRE map · điểm detector, kèm `.json` nguồn gốc",
         "Nhãn": "R0 / R1+R2"},
    ])

    st.markdown("### Điều không được quên khi trình bày")
    st.markdown(
        "- `S` là số bước DDIM. Đổi `S` là đổi kết quả (Table 4: `S=5` 87,5 · "
        "`S=10` 98,2 · `S=20` 99,7 — đều là " + PAPER + ").\n"
        "- Ngưỡng chọn trên **val**, không bao giờ trên test.\n"
        "- Repo này tái lập **from scratch**: `R1+R2` là map + detector của mình; "
        "`R0` là dùng map/checkpoint của tác giả để *chấm lại* — hai thứ khác nhau, "
        "và nhãn phải hiện cạnh số."
    )
    st.info(
        "Demo chạy trên CPU free (2 vCPU, 16 GB). Không GPU, không gọi mạng ngoài, "
        "không lưu ảnh của ai. Toàn bộ mã ở `platforms/hf-spaces/`.",
        icon="ℹ️",
    )


def page_gallery(samples: list[provenance.Sample], problems: list[str]) -> None:
    st.title("Gallery — chế độ A")
    st.caption(
        "Ảnh tính trước trên GPU, xem tại chỗ. Đây là chế độ duy nhất không cần "
        "tính DIRE lúc chạy, nên nó là chế độ mặc định của một Space CPU."
    )

    if problems:
        with st.expander(f"⚠ {len(problems)} tệp `.json` bị từ chối (không hiện mẫu đó)", expanded=False):
            for p in problems:
                st.markdown(f"- `{p}`")

    if not samples:
        st.warning("**Gallery đang rỗng.** Không có panel trống giả ở đây.", icon="⚠️")
        st.markdown(provenance.GENERATE_HELP)
        st.markdown(provenance.SCHEMA_HELP)
        st.caption(
            "Trạng thái này là nội dung, không phải lỗi: một demo không có map thì "
            "không có gì để chấm, và hiện ba khung xám sẽ khiến người xem tưởng app hỏng."
        )
        return

    labels = [s.selector for s in samples]
    pick = st.selectbox("Chọn mẫu", labels, index=0)
    s = samples[labels.index(pick)]

    score, reason = s.score()

    cols = st.columns(3 if s.recon else 2)
    with cols[0]:
        st.image(str(s.orig), caption="ảnh gốc 256×256", width="stretch")
    with cols[1]:
        st.image(str(s.diff), caption="|x − x̂₀| · DIRE map", width="stretch")
    if s.recon:
        with cols[2]:
            st.image(str(s.recon), caption="ảnh tái tạo (ADM, DDIM)", width="stretch")
    else:
        st.caption(
            "Không có panel ảnh tái tạo: `::dire` không lưu ảnh trung gian, và một khung "
            "xám ở đây sẽ là hình giả. Muốn có, phải thêm cờ lưu recon vào `::dire`."
        )

    st.markdown(f"`{provenance.evidence_line(s)}`")

    d, e = st.columns([1, 2])
    with d:
        if score:
            p = float(score["p_fake"])
            st.metric("P(fake)", f"{p:.3f}",
                      help="detector nhận DIRE map, không nhận ảnh RGB")
        else:
            st.metric("P(fake)", "—", help="chưa đủ điều kiện để hiện số")
    with e:
        if score:
            thr = float(score.get("threshold", 0.5))
            st.markdown(
                f"Nhãn theo ngưỡng **{thr:.2f}**: **{'FAKE' if float(score['p_fake']) >= thr else 'REAL'}** "
                f"— ngưỡng chọn trên `{score.get('threshold_split')}` "
                f"(nguồn: {score.get('source', '—')})"
            )
        else:
            st.warning(reason, icon="⚠️")
    st.caption(
        f"generator **{s.generator}** · split **{s.split}** · "
        f"orig sha256 `{s.sha256.get('orig', '—')[:12]}…` · "
        f"map sha256 `{s.sha256.get('map', '—')[:12]}…`"
    )
    with st.expander("Nguồn gốc đầy đủ (`.json`)"):
        st.json(s.meta)
    st.caption(
        "Ảnh trong gallery: DiffusionForensics (Wang et al., ICCV 2023), "
        "chỉ dùng cho mục đích nghiên cứu."
    )


def page_toy() -> None:
    st.title("Thí nghiệm 2 chiều — phép đo chạy thật")
    st.markdown(
        "Dữ liệu là **hỗn hợp Gauss 2 chiều**, nên phân phối sau khi thêm nhiễu cũng "
        "là hỗn hợp Gauss và **score có công thức đóng**. \"Mô hình\" là score đó — "
        "giả định học hoàn hảo. Vì vậy mọi khác biệt giữa hai lớp đo *đúng* phần "
        "hình học mà DIRE dựa vào, không lẫn sai số huấn luyện."
    )

    with st.expander("Toán, và hai vòng lặp", expanded=False):
        st.markdown(
            "Với `p_g = Σ_k w_k 𝒩(μ_k, Σ_k)`, ở mức `ᾱ`:\n\n"
            "`p_t = Σ_k w_k 𝒩(√ᾱ·μ_k, ᾱΣ_k + (1−ᾱ)I)`\n\n"
            "`ε̂(x,t) = −√(1−ᾱ) · ∇log p_t(x)`, rồi `x̂₀ = (x − √(1−ᾱ)·ε̂)/√ᾱ`.\n\n"
            "**Đảo ngược** (tăng `t`): `x_{t'} = √ᾱ'·x̂₀ + √(1−ᾱ')·ε̂`. "
            "**Tái tạo** (giảm `t`): lặp lại cùng công thức, lấy `x̂₀` ở bước cuối.\n\n"
            "`DIRE(x₀) = |x₀ − x̂₀|`, mỗi ảnh tốn `2S` lượt \"mạng\"."
        )

    c1, c2, c3, c4 = st.columns(4)
    steps = c1.slider("S (bước DDIM)", 1, 200, 20, help="Table 4 của bài quét 5/10/20/50")
    n = c2.slider("Số mẫu mỗi lớp", 50, 400, 200, step=50)
    clip = c3.checkbox("Cắt x̂₀ về [−1, 1]", value=True,
                       help="tắt để xem phép cắt có phải cơ chế hay không")
    seed = c4.number_input("seed", 0, 10**6, 3407, step=1)

    res = _cached_run(int(steps), int(n), bool(clip), int(seed))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("DIRE · ảnh sinh (TB)", f"{res['gen']['mean']:.4f}")
    m2.metric("DIRE · ảnh thật (TB)", f"{res['real']['mean']:.4f}")
    m3.metric("tỉ lệ thật/sinh", f"{res['ratio']:.2f}×",
              help="1,0× = không tách được; càng lớn càng tách tốt")
    m4.metric("NFE / ảnh", res["nfe"])

    if res["steps"] <= 2:
        st.warning(
            f"Với `S = {res['steps']}`, **cả hai** lớp có sai số cỡ 1e-3, nên tỉ lệ "
            "rất nhạy với mẫu và không đọc được như một kết luận. Đây là giới hạn "
            "của phép đo, không phải của hình vẽ.",
            icon="⚠️",
        )

    st.markdown(
        f"`DIRE: thí nghiệm 2 chiều, score chính xác · detector: không có (không chấm ảnh) "
        f"· nhãn: không áp dụng — minh hoạ lý thuyết · S={res['steps']} · n={res['n']} mỗi lớp`"
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Phân bố DIRE**")
        h = res["hist"]
        st.markdown(charts.histogram(h["edges"], h["gen"], h["real"]), unsafe_allow_html=True)
        st.caption(
            f"Trung vị {res['gen']['median']:.4f} (sinh) vs {res['real']['median']:.4f} (thật) · "
            f"p90 {res['gen']['p90']:.4f} vs {res['real']['p90']:.4f}"
        )
    with right:
        st.markdown("**x₀ → x̂₀**")
        st.markdown(charts.scatter(res["pairs"]), unsafe_allow_html=True)
        st.caption("Mũi tên càng dài = tái tạo càng lệch. Ảnh sinh nằm gần mô hình hơn.")

    st.markdown("### Quét `S`")
    sweep = _cached_sweep(bool(clip), int(seed), min(int(n), 200))
    st.markdown(charts.sweep(sweep), unsafe_allow_html=True)
    st.dataframe(
        [{"S": p["steps"], "NFE/ảnh": p["nfe"], "DIRE sinh": p["gen"],
          "DIRE thật": p["real"], "tỉ lệ": p["ratio"]} for p in sweep],
        width="stretch", hide_index=True,
    )
    st.caption(
        "Đọc hình này cho đúng: khi `S` tăng, **cả hai** lớp đều giảm (tái tạo sát hơn), "
        "và tỉ lệ tăng rồi bão hoà quanh ~2. Tỉ lệ bão hoà nghĩa là tăng `S` không còn "
        "mua thêm khả năng phân biệt — đúng tinh thần Table 4, nơi `S=20` đã gần `S=50`."
    )

    with st.expander("Vì sao kết quả này KHÔNG phải Table 2"):
        st.markdown(
            "- `p_g` là một hỗn hợp Gauss *đã biết*; ADM là mô hình 552 M tham số học "
            "trên LSUN-Bedroom, và nó không hoàn hảo.\n"
            "- Không có ảnh, không có nén JPEG, không có nhiễu cảm biến, không có "
            "tần số cao — ba ứng viên giải thích vì sao DIRE tách được cả ảnh GAN.\n"
            "- Ở đây tỉ lệ ~1,9×; trên ảnh, Table 2 báo ACC/AP ~99–100 với ADM. Hai "
            "đại lượng khác nhau, không suy ra nhau.\n"
            "- Giá trị của thí nghiệm là **cơ chế**: nếu ngay cả trong trường hợp lý "
            "tưởng này DIRE không tách được, thì mọi giải thích trên ảnh đều sai."
        )

    st.download_button(
        "Tải kết quả (.json)",
        data=json.dumps(res, ensure_ascii=False, indent=2),
        file_name=f"dire-toy-S{res['steps']}-n{res['n']}.json",
        mime="application/json",
    )



def page_live(samples: list[provenance.Sample]) -> None:
    st.title("Tính DIRE thật — chậm, và nói rõ chậm bao nhiêu")
    st.markdown(
        "Trang này chạy **đúng hai vòng lặp** của DIRE (đảo ngược DDIM rồi tái tạo) "
        "trên ADM, ngay trong tiến trình này — không gọi mạng, không dùng map tính sẵn. "
        "Cái giá là thời gian, và nó được đo chứ không phỏng đoán:"
    )
    st.table([
        {"Việc": "1 lượt forward ADM 256²", "Trên 2 vCPU": f"{live.MEASURED_CPU_FORWARD_SEC:.1f} s",
         "Trên A10G": "0,09 s"},
        {"Việc": "DIRE với `S = 1`", "Trên 2 vCPU": f"{live.estimate_seconds(1):.0f} s", "Trên A10G": "0,2 s"},
        {"Việc": "DIRE với `S = 5`", "Trên 2 vCPU": f"{live.estimate_seconds(5):.0f} s", "Trên A10G": "0,9 s"},
        {"Việc": "DIRE với `S = 20` (cấu hình của bài)", "Trên 2 vCPU": f"{live.estimate_seconds(20):.0f} s",
         "Trên A10G": "3,7 s"},
    ])
    st.caption("Số 2 vCPU: đo tại chỗ, " + live.MEASURED_CPU_NOTE +
               ". Số A10G: từ `::bench` (0,0917 s/step) × `2S`.")

    r_url, r_tok = _remote_creds()
    if remote.configured(r_url, r_tok):
        _live_remote(samples, r_url, r_tok)
        return

    stt = live.status()
    if not stt.available:
        st.warning("**Chế độ này đang tắt.** " + stt.reason, icon="⚠️")
        st.markdown("Để bật (chỉ nên bật khi có người thật sự ngồi xem):")
        for h in stt.hints:
            st.markdown(f"- `{h}`")
        st.markdown(
            "Vì sao mặc định tắt: `docs/demo-web-plan.md` §3 chốt ngưỡng — p95 > 45 s "
            "thì tắt chế độ tự tính. Đo được **101 s** cho `S = 5`, tức vượt ngưỡng 2,2 lần. "
            "Bật lên thì mỗi người xem bấm một lần là 1,7 phút CPU của Space."
        )
        st.markdown(
            "Muốn thấy **quá trình** mà không chờ: sang trang *Thí nghiệm 2 chiều* — cùng "
            "hai vòng lặp, cùng phép đo, chạy 61 ms/lượt."
        )
        st.markdown("---")
        st.markdown("### Cách nhanh nhất: để GPU làm, Space chỉ hiện")
        st.markdown(
            "Đặt `DEMO_URL` và `DEMO_TOKEN` trong **Space → Settings → Secrets** để dùng "
            "endpoint trên Modal (A10G, ~0,9 s cho `S=5` thay vì 101 s). Token nằm ở "
            "server của Space, **không** xuống trình duyệt. Cách dựng endpoint, chi phí "
            "và giới hạn: `docs/demo-modal-plan.md`."
        )
        return

    st.success(f"Đã có torch {stt.torch} và checkpoint `{Path(stt.ckpt).name}`.", icon="✅")

    # Đầu vào: mẫu trong gallery nếu có, nếu không thì ảnh tổng hợp — và nói rõ.
    src_path, src_label = _pick_image(samples)

    c1, c2, c3 = st.columns(3)
    steps = c1.select_slider("S (số bước DDIM)", options=[1, 2, 5, 10, 20], value=5)
    long_ok = c2.checkbox(
        f"Tôi chấp nhận chờ ~{live.estimate_seconds(steps):.0f} s",
        value=steps <= 5,
        help="mỗi bước là 2 lượt forward; đây là thời gian CPU thật, không phải thanh chạy giả",
    )
    go = c3.button("Chạy DIRE thật", type="primary", disabled=not long_ok)

    if go:
        model = _live_model()
        img = src_path if src_path else _synthetic_image()
        bar = st.progress(0.0, text="đang chạy…")
        grid = st.empty()
        log = st.empty()
        frames: list[dict] = []
        total = 2 * steps

        def on_frame(d: dict) -> None:
            frames.append(d)
            k = len(frames)
            bar.progress(k / total, text=f"{d['phase']} · t={d['t']} · {k}/{total} bước")
            grid.image(d["x0_png"], caption=f"x̂₀ tại t={d['t']} ({d['phase']})", width=220)
            log.dataframe(
                [{"bước": i + 1, "pha": f["phase"], "t": f["t"], "giây": f["sec"],
                  "rms(ε̂)": f["eps_rms"], "|x₀−x̂₀| TB": f["map_mean"]}
                 for i, f in enumerate(frames)],
                hide_index=True, width="stretch",
            )

        try:
            out = live.run_steps(model, img, steps=steps, on_frame=on_frame)
        except Exception as e:  # pragma: no cover - chỉ khi checkpoint hỏng
            st.error(f"Không chạy được: {type(e).__name__}: {e}")
            return
        bar.progress(1.0, text="xong")
        st.session_state["live_out"] = out
        st.session_state["live_steps"] = steps
        st.session_state["live_src"] = src_label

    out = st.session_state.get("live_out")
    if not out:
        return

    steps_done = st.session_state.get("live_steps", steps)
    line = (f"DIRE: tính tại chỗ, ADM (256x256_diffusion_uncond) · detector: không có "
            f"(trang này chỉ tính map) · nhãn: không áp dụng · S={steps_done} · "
            f"{out['nfe']} NFE · {out['seconds']:.1f} s")
    st.markdown(f"`{line}`")
    a, b, c = st.columns(3)
    a.image(str(src_path) if src_path else _synthetic_image(), caption="ảnh gốc", width="stretch")
    b.image(out["recon_png"], caption="x̂₀ sau khi đảo ngược rồi tái tạo", width="stretch")
    c.image(out["map_png"], caption="|x₀ − x̂₀| (DIRE), thang [0, 2], gamma 0,5", width="stretch")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("DIRE trung bình", f"{out['map_values']['mean']:.4f}")
    m2.metric("DIRE p99", f"{out['map_values']['p99']:.4f}")
    m3.metric("DIRE lớn nhất", f"{out['map_values']['max']:.4f}")
    m4.metric("giây / forward", f"{sum(out['timings']) / max(1, len(out['timings'])):.2f}")

    st.caption(
        "Đọc kết quả cho đúng: đây là **một** ảnh, nên con số DIRE của nó không phải "
        "một ngưỡng phân loại. Muốn kết luận thật hay sinh cần phân bố của cả hai lớp "
        "(trang Gallery), và cần detector nhận map — không phải ảnh."
    )
    with st.expander("Từng bước của phép tính"):
        st.dataframe(
            [{"bước": i + 1, "pha": f["phase"], "t": f["t"], "giây": f["sec"],
              "rms(ε̂)": f["eps_rms"], "|x₀−x̂₀| TB": f["map_mean"]}
             for i, f in enumerate(out["frames"])],
            hide_index=True, width="stretch",
        )


def _remote_creds() -> tuple[str, str]:
    """`st.secrets` trước, biến môi trường sau — để chạy thử ở máy không cần Space."""
    url = tok = ""
    try:
        url = str(st.secrets.get("DEMO_URL", "") or "")
        tok = str(st.secrets.get("DEMO_TOKEN", "") or "")
    except Exception:
        pass
    return url or os.environ.get("DEMO_URL", ""), tok or os.environ.get("DEMO_TOKEN", "")


def _pick_image(samples: list[provenance.Sample]) -> tuple[Path | None, str]:
    if samples:
        labels = [s.selector for s in samples]
        pick = st.selectbox("Ảnh đầu vào (từ gallery)", labels, index=0)
        return samples[labels.index(pick)].orig, pick
    st.info(
        "Gallery rỗng nên dùng **ảnh tổng hợp** để minh hoạ phép tính. Số đo trên ảnh "
        "tổng hợp không nói gì về ảnh thật hay ảnh sinh — nó chỉ chứng minh đường tính chạy.",
        icon="ℹ️",
    )
    return None, "ảnh tổng hợp (gradient + nhiễu)"


def _synthetic_png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(_synthetic_image()).save(buf, format="PNG")
    return buf.getvalue()


def _live_remote(samples: list[provenance.Sample], url: str, token: str) -> None:
    """DIRE tính trên Modal — Space chỉ gửi ảnh và hiện kết quả."""
    host = url.split("//")[-1].split("/")[0]
    st.success(f"Đang dùng endpoint GPU: `{host}` — DIRE tính trên Modal (A10G).", icon="⚡")
    h = remote.health(url)
    if h is None:
        st.caption("Không đọc được `/health` (endpoint đang ngủ?) — vẫn thử gọi được.")
    else:
        st.caption(f"engine: `{h.engine}` · S ∈ [{h.s_range[0]}, {h.s_range[1]}] · "
                   f"tối đa {h.max_bytes / 1024 / 1024:.0f} MB/ảnh")

    src_path, src_label = _pick_image(samples)
    steps = st.select_slider("S (số bước DDIM)", options=[1, 2, 5, 10, 20], value=5,
                             key="remote_S")
    st.caption(
        f"Ước tính: **{0.0917 * 2 * steps:.1f} s** GPU (đo: 0,0917 s/step) + ~0,3 s mạng. "
        "Không cần tick chờ — đây là GPU, không phải CPU của Space."
    )
    if not st.button("Chạy DIRE trên Modal", type="primary"):
        return

    raw = src_path.read_bytes() if src_path else _synthetic_png()
    with st.spinner(f"đang tính S={steps} trên A10G…"):
        try:
            res = remote.call(url, token, raw, steps, timeout=120.0)
        except remote.RemoteError as e:
            st.error(str(e))
            return

    st.markdown(f"`{res.line}`")
    a, b, c = st.columns(3)
    a.image(raw, caption="ảnh gốc", width="stretch")
    b.image(res.recon_png, caption="x̂₀ sau khi đảo ngược rồi tái tạo", width="stretch")
    c.image(res.map_png, caption="|x₀ − x̂₀| (DIRE), thang [0, 2], gamma 0,5", width="stretch")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("DIRE trung bình", f"{res.stats['mean']:.4f}")
    m2.metric("DIRE p99", f"{res.stats.get('p99', float('nan')):.4f}")
    m3.metric("DIRE lớn nhất", f"{res.stats['max']:.4f}")
    m4.metric("giây (GPU)", f"{res.seconds:.2f}")
    if res.resized:
        st.caption("Ảnh vào không phải 256×256 nên đã được resize — ADM chỉ học ở 256².")
    st.caption(
        f"nguồn: `{src_label}` · {res.nfe} NFE · endpoint chỉ tính map, **không** chấm "
        "ảnh: muốn có điểm phân loại phải có detector nhận map."
    )


@st.cache_resource(show_spinner="đang nạp ADM (552,8 M tham số, ~2,2 GB)…")
def _live_model():
    model, digest = live.load_checkpoint()
    return model


def _synthetic_image():
    """Ảnh 256² tổng hợp, tất định — để minh hoạ phép tính khi gallery rỗng."""
    import numpy as np

    yy, xx = np.mgrid[0:256, 0:256]
    base = np.stack([xx, yy, (xx + yy) // 2], axis=-1).astype("float32") / 255.0
    noise = np.random.default_rng(3407).normal(0, 0.05, base.shape)
    return np.clip((base + noise) * 255, 0, 255).astype("uint8")


def page_schedule() -> None:
    st.title("Lịch nhiễu, NFE và tiền")
    st.markdown(
        "Hai con số hay bị lẫn: `--steps` lúc **sinh** ảnh (50) và `S` lúc **tính DIRE** "
        "(20). Mỗi ảnh DIRE tốn `2S` lượt mạng — một lượt đảo ngược, một lượt tái tạo."
    )

    steps = st.slider("S", 1, 200, 20, key="sched_S")
    rows, marks = [], toy2d.timesteps(toy2d.T_TRAIN, steps)
    for t in marks:
        a = toy2d.A[t]
        rows.append({"t": t, "ᾱ_t": round(a, 6), "√ᾱ_t": round(math.sqrt(a), 5),
                     "√(1−ᾱ_t)": round(math.sqrt(max(0.0, 1 - a)), 5)})
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(rows, height=280, width="stretch", hide_index=True)
        st.caption("`t` 0-based như ADM: `ᾱ₀ = 1 − β₀ = 0,9999`, không phải 1.")
    with c2:
        st.markdown("**NFE và chi phí**")
        n_images = st.number_input("Số ảnh", 1, 200_000, 800, step=100)
        sec = st.number_input("sec/step (đo được)", 0.0001, 5.0,
                              cost.SEC_PER_STEP_MEASURED, step=0.01, format="%.4f")
        gpu = st.selectbox("GPU", list(cost.PRICES), index=0)
        e = cost.estimate(int(n_images), int(steps), float(sec), gpu)
        st.code(cost.estimate_line(e), language="text")
        a, b, d = st.columns(3)
        a.metric("NFE", f"{e['nfe_total']:,}")
        b.metric("GPU-giờ", f"{e['gpu_h']:.2f}")
        d.metric("USD", f"{e['usd']:.2f}")
        st.caption(
            "Cùng công thức với `dire_zero.budget.Estimate` — test khoá hai bên khớp. "
            f"`{cost.SEC_PER_STEP_MEASURED}` s/step do `::bench` đo (ADM 256², A10G, fp16, "
            "batch 1). Kiểm chéo: 829 GFLOP mỗi lượt forward ÷ 0,0917 s = 9,0 TFLOPS = "
            "29 % đỉnh fp16 của A10G — mức bình thường cho batch 1, không phải số bất thường."
        )

    st.markdown("### Phép đảo ngược, viết ra đúng như code chạy")
    st.code(
        "# đảo ngược: tăng t\n"
        "x = x0_img\n"
        "for i, t in enumerate(ts):\n"
        "    x0_hat, eps = model(x, t)          # một lượt U-Net\n"
        "    t_next = ts[min(i + 1, len(ts) - 1)]\n"
        "    a_next = alphas_cumprod[t_next]\n"
        "    x = sqrt(a_next) * x0_hat + sqrt(1 - a_next) * eps\n"
        "\n"
        "# tái tạo: giảm t, lấy x0_hat ở bước cuối\n"
        "for i in reversed(range(len(ts))):\n"
        "    x0_hat, eps = model(x, ts[i])\n"
        "    if i == 0: return x0_hat\n"
        "    a_prev = alphas_cumprod[ts[i - 1]]\n"
        "    x = sqrt(a_prev) * x0_hat + sqrt(1 - a_prev) * eps\n"
        "\n"
        "dire = abs(x0_img - x0_hat)            # map, thang [0, 2]",
        language="python",
    )
    st.caption(
        "Điểm dễ sai nhất: mốc cuối là `round((S−1)·T/S) = 950` khi `S=20`, không phải "
        "999. Dùng trọn 1000 bước cho ra map khác hẳn, trông vẫn hợp lý, và tốn 25 lần tiền."
    )


def page_paper() -> None:
    st.title("Bảng 2 của bài — và đường tái lập của repo")
    st.caption(f"Toàn bộ bảng dưới là {PAPER}.")

    header = "| | " + " | ".join(TABLE2["generators"]) + " |"
    sep = "|" + "---|" * (len(TABLE2["generators"]) + 1)
    body = [header, sep]
    for name, cells in TABLE2["rows"]:
        body.append("| **" + name + "** | " + " | ".join(cells) + " |")
    st.markdown("\n".join(body))
    st.caption(
        "ACC/AP (%). `*` = baseline phải huấn luyện lại trên ảnh sinh của ADM; "
        "baseline **không** huấn luyện lại chỉ đạt 51–59 ACC, tức là thất bại hoàn toàn "
        "trước ảnh khuếch tán."
    )

    st.markdown("### Ba điều kiện để so được với bảng đó — cả ba đang thiếu")
    st.table([
        {"Điều kiện": "1 000 ảnh mỗi ô", "Repo hiện có": "200 (`--n 200`)", "Hệ quả": "khoảng tin cậy rộng hơn"},
        {"Điều kiện": "8 generator", "Repo hiện có": "1 (`adm`)", "Hệ quả": "chưa có ô nào so được"},
        {"Điều kiện": "detector ImageNet pretrained", "Repo hiện có": "`init=\"scratch\"`", "Hệ quả": "thấp hơn bảng, không cùng cấu hình"},
    ])

    st.markdown("### Đường tái lập")
    for k, what, cmd in PIPELINE:
        st.markdown(f"**`::{k}`** — {what}")
        st.code(cmd, language="bash")

    st.markdown("### Hạn chế phải nói ra")
    st.markdown(
        "- `:dire` chạy **batch 1** = 9,0 TFLOPS (29 % đỉnh A10G). Gộp 8 ảnh chỉ hứa "
        "**1,5–2×** (trần tuyệt đối là 31,2/9,0 = 3,5×, không đạt được ở batch nhỏ), "
        "không đổi kết quả số học.\n"
        "- Chưa có `--skip-existing`, nên mỗi generator mới phải tính lại cả ảnh `real`.\n"
        "- `:train` không có `--generator adm` thì gộp mọi generator ⇒ **không** so được Table 2.\n"
        "- Repo chưa có kết quả `R1+R2` nào trên 1 000 ảnh/ô; mọi con số trong bảng trên là của bài."
    )


# ---------------------------------------------------------------------------
# Cache — mỗi lần kéo slider không nên tính lại từ đầu
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=64)
def _cached_run(steps: int, n: int, clip: bool, seed: int) -> dict:
    return toy2d.run(steps=steps, n=n, clip=clip, seed=seed)


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_sweep(clip: bool, seed: int, n: int) -> list[dict]:
    out = []
    # Bắt đầu từ 5 như Table 4. `S ≤ 2` không đưa vào: sai số hai lớp cùng ~1e-3
    # nên tỉ lệ nhảy theo mẫu, không theo cơ chế — vẽ lên sẽ thành một điểm giả.
    for s in (5, 10, 20, 50, 100, 200):
        r = toy2d.run(steps=s, n=n, clip=clip, seed=seed)
        out.append({"steps": s, "nfe": r["nfe"], "gen": r["gen"]["mean"],
                    "real": r["real"]["mean"], "ratio": r["ratio"]})
    return out


def main() -> None:
    st.set_page_config(page_title="DIRE — sai số tái tạo", page_icon="🔍", layout="wide")
    samples, problems = provenance.discover(GALLERY_DIR)

    with st.sidebar:
        st.markdown("## DIRE")
        st.caption("ICCV 2023 · Wang et al. · bản viết lại from scratch")
        page = st.radio(
            "Trang",
            ["1 · Demo này là gì", "2 · Gallery (chế độ A)", "3 · Tính DIRE thật",
             "4 · Thí nghiệm 2 chiều", "5 · Lịch & chi phí", "6 · Bảng 2 & tái lập"],
            label_visibility="collapsed",
        )
        st.divider()
        st.metric("Mẫu trong gallery", len(samples))
        if problems:
            st.caption(f"⚠ {len(problems)} `.json` bị từ chối")
        st.caption("CPU free · 0 ₫ · không gọi mạng ngoài · không lưu ảnh người dùng")

    if page.startswith("1"):
        page_intro(samples)
    elif page.startswith("2"):
        page_gallery(samples, problems)
    elif page.startswith("3"):
        page_live(samples)
    elif page.startswith("4"):
        page_toy()
    elif page.startswith("5"):
        page_schedule()
    else:
        page_paper()


main()
