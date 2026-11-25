#!/usr/bin/env python3
"""Sinh figure cho báo cáo — một entry point, mỗi hình một lệnh.

Vì sao code vẽ nằm ở ``tools/`` chứ không ở ``src/dire_zero/``
--------------------------------------------------------------
``dire_zero`` tự khai nó chỉ phụ thuộc numpy/torch/PIL, và
``tests/test_zero_isolation.py`` canh đúng điều đó. Kéo matplotlib vào package
đó là phá một lời khai đã có test — và làm image Docker của ``modal_app`` nặng
thêm vài trăm MB cho một việc không bao giờ chạy trên GPU. Nên: package sinh
**số**, tools vẽ **hình**.

Ba luật của file này
--------------------
1. **Không có số nào được bịa.** Hàm nào cũng trả về một ``sidecar`` ghi rõ nó
   đã đọc file nào, bao nhiêu mẫu, seed nào. Hình không có sidecar là hình không
   kiểm được.
2. **Nhãn mức tái lập nằm trên khung**, không nằm trong chú thích. ``R0`` =
   map/detector của tác giả; ``R1+R2`` = map của mình, ảnh của tác giả;
   ``R1+R2+R3`` = tất cả của mình; ``schematic`` = hình vẽ, **không phải số đo**.
3. **Không import matplotlib ở tầng module.** ``--list`` phải chạy được trên máy
   chưa cài matplotlib, và test phải kiểm được cấu trúc mà không cần vẽ.

Chạy
----
    python tools/figures.py list
    python tools/figures.py f3-pipeline   --out figs
    python tools/figures.py a1-provenance --out figs
    python tools/figures.py f5-noise      --out figs --maps F:/dire-zero-r0/maps --label R0
    python tools/figures.py f1-teaser     --out figs --images DIR --maps DIR --label R1+R2

Kế hoạch đầy đủ (hình nào, đợt nào, vì sao): ``docs/plans/research/figures-plan.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_SEED = 3407

#: Ảnh/dữ liệu map lấy từ đâu ⇒ được nói mức nào. Không có giá trị thứ năm.
ALLOWED_LABELS = ("R0", "R1+R2", "R1+R2+R3", "schematic")

#: Kích thước cột theo chuẩn paper: cột đơn 3,5 in, cột đôi 7,2 in.
COL1, COL2 = 3.5, 7.2


# --------------------------------------------------------------------------- #
# Hạ tầng: matplotlib chỉ được import khi thật sự vẽ
# --------------------------------------------------------------------------- #

def _mpl():
    """Import matplotlib kiểu ``Agg`` (không cần display), kèm thông báo rõ ràng."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise SystemExit(
            "cần matplotlib để vẽ hình:\n  pip install matplotlib\n"
            f"(chi tiết: {exc})"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(plt):
    """Nét mảnh, chữ nhỏ, không viền thừa — để hình in ra không bị 'béo'."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,      # chữ nhúng dạng TrueType: sửa được, tìm được
    })


def _stamp(fig, plt, label: str, note: str = "") -> None:
    """Nhãn mức tái lập ở góc dưới trái — chỗ khó crop mất nhất."""
    if label not in ALLOWED_LABELS:
        raise ValueError(f"nhãn không hợp lệ: {label!r} (chỉ {ALLOWED_LABELS})")
    txt = label if label == "schematic" else f"nhãn: {label}"
    if note:
        txt += f" · {note}"
    fig.text(0.01, 0.01, txt, fontsize=6, color="0.25", ha="left", va="bottom")


def _save(fig, plt, out_dir: Path, name: str, sidecar: dict, also_png: bool = True) -> dict:
    """Ghi ``.pdf`` (+ ``.png``) và ``data/<name>.json`` rồi đóng hình."""
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    if also_png:
        fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    sidecar = dict(sidecar)
    sidecar.setdefault("figure", name.upper())
    sidecar["name"] = name
    sidecar["created"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sidecar["output"] = str(pdf)
    missing = [k for k in ("kind", "claim_label", "sources") if k not in sidecar]
    if missing:
        raise ValueError(f"sidecar của {name} thiếu {missing} — hình không kiểm được")
    (out_dir / "data" / f"{name}.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_index(out_dir)
    print(f"[xong] {pdf}  ·  nhãn {sidecar['claim_label'] or 'schematic'}")
    return sidecar


def write_index(out_dir: Path) -> Path:
    """Sinh ``figs/README.md`` từ chính các sidecar — bảng không thể lệch hình."""
    out_dir = Path(out_dir)
    rows = []
    for p in sorted((out_dir / "data").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        src = d.get("sources") or {}
        src_txt = ", ".join(f"`{Path(str(v)).name}`" for v in src.values() if v) or "—"
        rows.append(
            f"| {d.get('figure', '?')} | `{d.get('cmd', '')}` | "
            f"{d.get('claim_label') or 'schematic'} | {d.get('n') or '—'} | {src_txt} |"
        )
    body = [
        "# Figure — sinh lại bằng một lệnh",
        "",
        "File này do `tools/figures.py` sinh; đừng sửa tay.",
        "",
        "| Hình | Lệnh | Nhãn | n | Nguồn |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "Nhãn: `R0` map/detector của tác giả · `R1+R2` map của mình, ảnh của tác giả · "
        "`R1+R2+R3` tất cả của mình · `schematic` hình vẽ, **không phải số đo**.",
        "",
    ]
    p = out_dir / "README.md"
    p.write_text("\n".join(body), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# Đọc dữ liệu: map, ảnh, manifest
# --------------------------------------------------------------------------- #

def map_to_float(arr):
    """Map về ``float32`` trong thang ``[0, 2]`` bất kể file gốc lưu kiểu gì.

    Ba định dạng gặp trong thực tế, và **phải phân biệt** vì ``uint8`` chia 255
    còn ``float16`` thì không:

    * ``uint8`` (map tác giả) — giá trị 0…255, không phải 0…2;
    * ``float16/32`` trong ``[0, 2]`` — ``|x − x̂|`` với ``x ∈ [−1, 1]``;
    * ``float`` đã chuẩn hoá ``[0, 1]`` — nhận ra bằng ``max ≤ 1.001``.

    Đoán sai một bước ở đây làm mọi thống kê lệch **âm thầm** — đúng loại lỗi
    ``tools/triplet_view.py`` đã ghi lại với map uint8 của tác giả.
    """
    import numpy as np

    a = np.asarray(arr)
    if a.dtype.kind in "ui":
        return a.astype("float32") / 255.0 * 2.0
    a = a.astype("float32")
    return a * 2.0 if a.max(initial=0.0) <= 1.001 else a


def load_maps(maps_dir: Path, split: str = "test", generator: str = "adm",
              n: int = 4, seed: int = DEFAULT_SEED) -> list:
    """Đọc tối đa ``n`` map ``.npy`` của một generator, chọn bằng hash (tái lập được)."""
    import numpy as np

    d = Path(maps_dir) / split / generator
    files = sorted(d.glob("*.npy"))
    if not files:
        raise SystemExit(f"không có .npy trong {d} — kiểm lại --maps/--split/--generator")
    files.sort(key=lambda p: hashlib.sha256(f"{seed}/{p.name}".encode()).hexdigest())
    return [map_to_float(np.load(p)) for p in files[:n]]


def _map_dirs(maps_dir: Path, split: str) -> list[str]:
    d = Path(maps_dir) / split
    if not d.is_dir():
        raise SystemExit(f"không có {d}")
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def pair_tars(image_tar: Path, map_tar: Path) -> dict:
    """Hai tar có ghép được **theo thứ tự** không? Trả ``dict``, không ném.

    Vì sao phải kiểm: giả định ngầm của hình teaser là "member thứ *i* của
    ``images/<gen>.tar.gz`` ứng với member thứ *i* của ``dire/<gen>.tar.gz``".
    Chưa ai chứng minh. Nếu sai, teaser ghép map của ảnh A lên ảnh B — vẫn ra
    một hình rất thuyết phục và hoàn toàn sai. Đây là phép kiểm rẻ nhất (chỉ đọc
    header, không giải nén) để biến giả định đó thành dữ kiện.
    """
    def stems(path: Path, suffixes: tuple[str, ...]) -> list[str]:
        out = []
        with tarfile.open(path, "r:*") as tf:
            for m in tf:
                if m.isfile() and m.name.lower().endswith(suffixes):
                    out.append(Path(m.name).stem)
        return out

    img = stems(image_tar, (".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    mp = stems(map_tar, (".npy",))
    ok = img == mp and bool(img)
    return {
        "ok": ok,
        "image_tar": str(image_tar),
        "map_tar": str(map_tar),
        "n_images": len(img),
        "n_maps": len(mp),
        "names_sha256": hashlib.sha256("\n".join(img).encode()).hexdigest()[:16],
        "method": "order",
        "first_mismatch": next(
            ({"i": i, "image": a, "map": b} for i, (a, b) in enumerate(zip(img, mp)) if a != b), None
        ),
    }


def read_manifest(path: Path) -> list[dict]:
    """Đọc ``manifest.tsv`` 10 cột (không kéo package vào để tools chạy độc lập)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    cols = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        vals = line.split("\t")
        row = dict(zip(cols, vals))
        row["label"] = int(row["label"])
        rows.append(row)
    return rows


def manifest_sha(rows: list[dict]) -> str:
    """Sha của **nội dung** (sha256 ảnh + nhãn), không của đường dẫn."""
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r.get("sha256", "")):
        h.update(f"{r.get('sha256', '')}:{r['label']}\n".encode())
    return h.hexdigest()


def percentile_norm(arr, lo: float = 1.0, hi: float = 99.0):
    """Chuẩn hoá percentile — **theo lớp**, không theo từng ảnh.

    Chuẩn hoá từng ảnh làm mọi ảnh trông giống nhau: đó là cách biến một hình
    so sánh thành một dãy hình trang trí.
    """
    import numpy as np

    a = np.asarray(arr)
    p1, p99 = np.percentile(a, lo), np.percentile(a, hi)
    if p99 <= p1:
        return np.zeros_like(a)
    return np.clip((a - p1) / (p99 - p1), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Đợt 0 — hình vẽ, không cần dữ liệu, không GPU
# --------------------------------------------------------------------------- #

def f2_concept(args) -> dict:
    """F2 — sơ đồ khái niệm: vì sao ảnh sinh ra "quay về" còn ảnh thật thì không."""
    plt = _mpl()
    _style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.0))

    for ax, title, box in (
        (axes[0], "ảnh do diffusion sinh ra", "x_g′ ≈ x_g"),
        (axes[1], "ảnh thật", "x_r′ ≠ x_r"),
    ):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.add_patch(plt.Circle((2.5, 3.0), 1.5, fill=False, lw=1.0))
        ax.add_patch(plt.Circle((6.0, 3.0), 1.5, fill=False, lw=1.0))
        ax.text(2.5, 3.0, "$x$", ha="center", va="center")
        ax.text(6.0, 3.0, "$x'$", ha="center", va="center")
        ax.annotate("", xy=(4.4, 3.0), xytext=(4.1, 3.0),
                    arrowprops={"arrowstyle": "->", "lw": 0.8})
        ax.text(5.0, 4.7, box, ha="center", fontsize=6.5)
        ax.text(2.5, 0.9, "$p_g$" if "sinh" in title else "$p_r$", ha="center", fontsize=7)
        ax.text(6.0, 0.9, "$p_g$", ha="center", fontsize=7)
        ax.set_title(title, fontsize=7.5)

    fig.suptitle("DIRE: khoảng cách giữa ảnh và ảnh tái tạo", fontsize=8)
    _stamp(fig, plt, "schematic", "không phải số đo")
    return _save(fig, plt, Path(args.out), "f2-concept", {
        "figure": "F2",
        "kind": "schematic",
        "claim_label": None,
        "n": None,
        "seed": None,
        "sources": {},
        "cmd": "python tools/figures.py f2-concept --out figs",
        "warnings": ["hình vẽ minh hoạ, không chứa số liệu"],
    })


def f3_pipeline(args) -> dict:
    """F3 — pipeline DDIM inversion → reconstruction → DIRE → classifier.

    Trục thời gian lấy **mốc thật** từ ``dire_zero.schedule``, không gõ tay: nếu
    ai đổi ``S`` hay công thức mốc, hình đổi theo và không thể lệch với code.
    """
    from dire_zero.schedule import nfe_per_image, timesteps

    steps = int(args.steps)
    marks = timesteps(1000, steps)
    plt = _mpl()
    _style(plt)

    fig, ax = plt.subplots(figsize=(COL2, 1.9))
    ax.set_xlim(-0.5, len(marks) - 0.5)
    ax.set_ylim(0, 4)
    ax.axis("off")

    y = 2.0
    ax.plot([0, len(marks) - 1], [y, y], lw=0.8, color="0.6")
    for i, t in enumerate(marks):
        ax.plot([i], [y], marker="o", ms=2.2, color="0.35")
    ax.text(-0.3, y + 0.35, "$x_0$", fontsize=7, ha="left")
    ax.text(len(marks) - 0.7, y + 0.35, "$x_0'$", fontsize=7, ha="right")
    ax.text(len(marks) / 2, y + 0.95, "$x_T \\sim \\mathcal{N}(0, I)$",
            fontsize=7, ha="center")

    ax.annotate("", xy=(len(marks) / 2, y + 0.75), xytext=(0.4, y + 0.25),
                arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "C0"})
    ax.annotate("", xy=(len(marks) - 0.6, y + 0.25), xytext=(len(marks) / 2, y + 0.75),
                arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "C1"})
    ax.text(0.9, y + 0.62, "$I(\\cdot)$ — DDIM inversion (Eq. 8)", fontsize=6.5, color="C0")
    ax.text(len(marks) / 2 + 0.3, y + 0.62, "$R(\\cdot)$ — DDIM sample (Eq. 5)",
            fontsize=6.5, color="C1")

    ax.text(-0.3, 0.35, "DIRE $= |x_0 - x_0'|$", fontsize=7)
    ax.text(len(marks) - 0.3, 0.35,
            f"$S={steps}$ · NFE/ảnh $=2S={nfe_per_image(steps)}$",
            fontsize=6.5, ha="right", color="0.3")
    ax.set_title("Một ảnh đi qua DIRE: 40 lượt mạng cho $S=20$", fontsize=8)

    _stamp(fig, plt, "schematic", f"S={steps}")
    return _save(fig, plt, Path(args.out), "f3-pipeline", {
        "figure": "F3",
        "kind": "schematic",
        "claim_label": None,
        "n": None,
        "seed": None,
        "sources": {"schedule": "src/dire_zero/schedule.py"},
        "cmd": f"python tools/figures.py f3-pipeline --out figs --steps {steps}",
        "steps": steps,
        "timesteps": marks,
        "nfe_per_image": nfe_per_image(steps),
        "warnings": [
            "mốc thời gian đọc từ schedule.py, không gõ tay",
            "hình vẽ minh hoạ, không chứa số liệu đo",
        ],
    })


def a1_provenance(args) -> dict:
    """A1 — ô nào là của tác giả, ô nào của mình; R0/R1+R2/R3 nghĩa là gì.

    Hình này là **hàng rào** cho mọi hình khác: nó trả lời "con số này của ai"
    ngay trên khung, thay vì để người đọc tự suy từ chú thích.
    """
    plt = _mpl()
    _style(plt)

    cols = ["dữ liệu thật\n(LSUN)", "ảnh giả\n(images/)", "DIRE map", "detector"]
    rows = ["R0\nchấm lại", "R1+R2\nmap+detector", "R1+R2+R3\ntất cả của mình"]
    # 1 = của tác giả (xám), 2 = của mình (xanh)
    grid = [
        [1, 1, 1, 1],
        [1, 1, 2, 2],
        [2, 2, 2, 2],
    ]
    colors = {1: "0.82", 2: "C0"}
    texts = {1: "tác giả", 2: "của mình"}

    fig, ax = plt.subplots(figsize=(COL2 * 0.78, 1.75))
    for i, row in enumerate(grid):
        for j, kind in enumerate(row):
            ax.add_patch(plt.Rectangle((j, len(grid) - 1 - i), 0.94, 0.9,
                                       facecolor=colors[kind], edgecolor="white", lw=1.2))
            ax.text(j + 0.47, len(grid) - 1 - i + 0.45, texts[kind],
                    ha="center", va="center", fontsize=6)
    for j, c in enumerate(cols):
        ax.text(j + 0.47, len(grid) + 0.15, c, ha="center", va="bottom", fontsize=6.5)
    for i, r in enumerate(rows):
        ax.text(-0.15, len(grid) - 1 - i + 0.45, r, ha="right", va="center", fontsize=6.5)
    ax.set_xlim(-2.6, len(cols))
    ax.set_ylim(-0.15, len(grid) + 0.9)
    ax.axis("off")
    ax.set_title("Mức tái lập = ô nào do ai làm", fontsize=8)

    _stamp(fig, plt, "schematic", "bảng phân quyền, không phải kết quả")
    return _save(fig, plt, Path(args.out), "a1-provenance", {
        "figure": "A1",
        "kind": "schematic",
        "claim_label": None,
        "n": None,
        "seed": None,
        "sources": {"plan": "docs/plans/research/figures-plan.md", "levels": "docs/adr/adr-0006-reproduction-levels.md"},
        "cmd": "python tools/figures.py a1-provenance --out figs",
        "warnings": ["hình vẽ: quy ước nhãn, không chứa số liệu"],
    })


# --------------------------------------------------------------------------- #
# Đợt 1 — R0, chỉ đọc map/ảnh đã có, 0 đồng GPU
# --------------------------------------------------------------------------- #

def f5_noise(args) -> dict:
    """F5 — noise pattern + phổ tần số của DIRE (thật vs sinh) + phổ radial.

    Paper chỉ nói bằng mắt ("frequency bands are more abundant"). Hình này thêm
    đường radial trung bình ± CI, tức là định lượng hoá đúng câu đó.
    """
    import numpy as np

    plt = _mpl()
    _style(plt)
    split = args.split
    gens = [g for g in ("real", args.generator) if g in _map_dirs(Path(args.maps), split)]
    if len(gens) < 2:
        raise SystemExit(
            f"cần cả 'real' và '{args.generator}' trong {Path(args.maps) / split}"
        )

    data = {g: load_maps(Path(args.maps), split, g, args.n, args.seed) for g in gens}
    # chuẩn hoá **theo lớp**: một mốc chung cho mọi panel cùng generator, nên
    # "sáng hơn" trong hình là do dữ liệu, không do cách chuẩn hoá
    lo = {g: float(np.percentile(np.concatenate([m.ravel() for m in v]), 1)) for g, v in data.items()}
    hi = {g: float(np.percentile(np.concatenate([m.ravel() for m in v]), 99)) for g, v in data.items()}

    fig = plt.figure(figsize=(COL2, 2.9))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.95], hspace=0.55, wspace=0.25)

    for row, g in enumerate(gens):
        for col, m in enumerate(data[g][:2]):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(np.clip((m - lo[g]) / max(hi[g] - lo[g], 1e-6), 0, 1),
                      cmap="magma", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{g} · DIRE", fontsize=6.5)

        ax = fig.add_subplot(gs[row, 2:])
        # Bỏ thành phần DC (trung bình) trước khi FFT: nếu không, một đỉnh ở tần số 0
        # lớn hơn mọi thứ khác vài bậc ⇒ log-scale nén toàn bộ phổ thành một màu.
        # Đây là bẫy phổ biến nhất khi vẽ phổ ảnh, và nó làm hình trông "sạch" mà
        # không còn thông tin.
        spec = np.abs(np.fft.fftshift(np.fft.fft2(data[g][0] - data[g][0].mean())))
        ax.imshow(np.log1p(spec), cmap="viridis", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{g} · $|\\mathcal{{F}}(DIRE)|$ (log)", fontsize=6.5)

    # radial spectrum: định lượng câu "frequency bands are more abundant"
    ax = fig.add_subplot(gs[2, :])
    for g in gens:
        curves = []
        for m in data[g]:
            f = np.abs(np.fft.fftshift(np.fft.fft2(m - m.mean())))
            c = f.shape[0] // 2
            yy, xx = np.indices(f.shape)
            r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2).astype(int)
            prof = np.bincount(r.ravel(), f.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
            curves.append(prof[: c + 1])
        k = min(len(c) for c in curves)
        arr = np.stack([c[:k] for c in curves])
        mu, sd = arr.mean(0), arr.std(0) / max(np.sqrt(len(arr)), 1)
        x = np.arange(k)
        mu, sd, x = mu[1:], sd[1:], x[1:]        # bỏ r=0: vẫn là đỉnh DC
        ax.plot(x, mu, lw=0.9, label=g)
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.2, lw=0)
    ax.set_xlabel("bán kính tần số (px)")
    ax.set_ylabel("biên độ TB")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set_title("phổ radial — thật vs sinh", fontsize=7)

    _stamp(fig, plt, args.label, f"{split} · n={args.n}/lớp")
    return _save(fig, plt, Path(args.out), "f5-noise", {
        "figure": "F5",
        "kind": "measurement",
        "claim_label": args.label,
        "n": args.n * len(gens),
        "seed": args.seed,
        "sources": {"maps": str(Path(args.maps) / split), "generators": gens},
        "cmd": (f"python tools/figures.py f5-noise --out figs --maps {args.maps} "
                f"--split {split} --generator {args.generator} --n {args.n} "
                f"--label {args.label} --seed {args.seed}"),
        "warnings": [
            "chuẩn hoá percentile theo từng lớp, không theo từng ảnh",
            "phổ radial là phần thêm so với paper — n=nhỏ nên CI còn rộng",
        ],
    })


def f1_teaser(args) -> dict:
    """F1 — teaser: ảnh gốc trên, DIRE map dưới, cho lớp thật và lớp sinh.

    Chỉ nhận thư mục đã ``ingest`` (``<split>/<gen>/*.png``) + map đã giải nén
    (``<split>/<gen>/*.npy``) **cùng tên stem** — đó chính là phép kiểm ghép cặp.
    Không có cặp khớp thì dừng: ghép sai còn tệ hơn không có hình.
    """
    import numpy as np
    from PIL import Image

    plt = _mpl()
    _style(plt)
    split, img_root, map_root = args.split, Path(args.images), Path(args.maps)
    gens = ["real", args.generator]

    panels = []
    for g in gens:
        imgs = sorted((img_root / split / g).glob("*.png"))
        maps = sorted((map_root / split / g).glob("*.npy"))
        shared = sorted({p.stem for p in imgs} & {p.stem for p in maps})
        if not shared:
            raise SystemExit(
                f"{g}: không có cặp ảnh/map nào cùng tên stem.\n"
                f"  ảnh: {img_root / split / g} ({len(imgs)} file)\n"
                f"  map: {map_root / split / g} ({len(maps)} file)\n"
                "Ghép theo thứ tự chỉ hợp lệ khi đã kiểm bằng `pair_tars`; "
                "ở đây cần tên khớp. Xem docs/plans/research/figures-plan.md §2.1."
            )
        shared.sort(key=lambda x: hashlib.sha256(f"{args.seed}/{x}".encode()).hexdigest())
        panels.append((g, [(img_root / split / g / f"{x}.png",
                            map_root / split / g / f"{x}.npy") for x in shared[: args.n]]))

    nrow = 2 * len(panels)
    # `squeeze=False` để **luôn** nhận mảng 2 chiều. Không có nó, `args.n == 1` trả
    # một Axes và code index `axes[row, col]` sẽ nổ — với đúng một mẫu, tức là ca
    # người dùng hay chạy nhất khi thử.
    fig, axes = plt.subplots(nrow, args.n, figsize=(1.2 * args.n + 0.6, 1.05 * nrow),
                             squeeze=False)

    # chuẩn hoá **theo lớp** (giống F5): nếu chuẩn hoá từng ảnh thì mọi panel
    # đều "vừa sáng", và so sánh độ lớn DIRE giữa hai lớp trở thành vô nghĩa.
    scale: dict[str, tuple[float, float]] = {}
    for g, pairs in panels:
        flat = [np.load(mp).astype("float32").ravel() for _, mp in pairs]
        vals = np.concatenate(flat)
        scale[g] = (float(np.percentile(vals, 1)), float(np.percentile(vals, 99)))

    for pi, (g, pairs) in enumerate(panels):
        for col, (ip, mp) in enumerate(pairs):
            ax = axes[2 * pi, col]
            ax.imshow(np.asarray(Image.open(ip).convert("RGB")))
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{g}\nảnh", fontsize=6.5)

            ax = axes[2 * pi + 1, col]
            lo_g, hi_g = scale[g]
            span = max(hi_g - lo_g, 1e-6)
            ax.imshow(np.clip((np.load(mp).astype("float32") - lo_g) / span, 0, 1),
                      cmap="magma", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel("DIRE", fontsize=6.5)

    fig.suptitle("Ảnh gốc và DIRE map", fontsize=8)
    _stamp(fig, plt, args.label, f"{split} · n={args.n}/lớp")
    return _save(fig, plt, Path(args.out), "f1-teaser", {
        "figure": "F1",
        "kind": "measurement",
        "claim_label": args.label,
        "n": args.n * len(panels),
        "seed": args.seed,
        "sources": {"images": str(img_root / split), "maps": str(map_root / split),
                    "generators": [g for g, _ in panels]},
        "cmd": (f"python tools/figures.py f1-teaser --out figs --images {args.images} "
                f"--maps {args.maps} --split {split} --generator {args.generator} "
                f"--n {args.n} --label {args.label} --seed {args.seed}"),
        "pairing": "stem (sha16 sau ingest) — không ghép theo thứ tự",
        "warnings": [
            "ghép cặp theo tên stem, không theo thứ tự member trong tar",
            "chuẩn hoá percentile theo từng lớp (không theo từng ảnh)",
            "ảnh chọn bằng hash của seed, không chọn 'ảnh đẹp nhất'",
        ],
    })


def a6_audit(args) -> dict:
    """A6 — audit dữ liệu: số ảnh theo split × generator, tỉ lệ lớp, nguồn manifest."""
    import numpy as np

    plt = _mpl()
    _style(plt)
    rows = read_manifest(Path(args.manifest))
    if not rows:
        raise SystemExit(f"{args.manifest} rỗng")

    counts: dict[tuple[str, str], int] = {}
    for r in rows:
        counts[(r["split"], r["generator"])] = counts.get((r["split"], r["generator"]), 0) + 1
    splits = sorted({s for s, _ in counts})
    gens = sorted({g for _, g in counts}, key=lambda g: (g != "real", g))

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 0.55 + 0.28 * len(gens)),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    grid = np.zeros((len(gens), len(splits)))
    for (s, g), n in counts.items():
        grid[gens.index(g), splits.index(s)] = n
    im = ax.imshow(grid, cmap="Blues", aspect="auto")
    for i in range(len(gens)):
        for j in range(len(splits)):
            if grid[i, j]:
                ax.text(j, i, f"{int(grid[i, j])}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels(splits)
    ax.set_yticks(range(len(gens)))
    ax.set_yticklabels(gens)
    ax.set_title("số ảnh: split × generator", fontsize=7.5)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    ax = axes[1]
    n_real = sum(r["label"] == 0 for r in rows)
    n_fake = sum(r["label"] == 1 for r in rows)
    ax.barh([0], [n_real], color="0.7", label="thật")
    ax.barh([0], [n_fake], left=[n_real], color="C0", label="giả")
    ratio = max(n_real, n_fake) / max(min(n_real, n_fake), 1)
    boundary = 5 * min(n_real, n_fake)          # quá mốc này là mất cân bằng
    ax.axvline(boundary, color="C3", ls="--", lw=0.8)
    ax.text(boundary, 0.32, "mốc 5:1", color="C3", fontsize=6, ha="right")
    ax.set_yticks([])
    ax.set_xlabel("số ảnh")
    ax.legend(frameon=False, ncol=2, loc="lower center")
    ax.set_title(f"tỉ lệ {n_real}:{n_fake} = {ratio:.2f}:1", fontsize=7.5)

    _stamp(fig, plt, "schematic", "audit dữ liệu, không phải kết quả")
    return _save(fig, plt, Path(args.out), "a6-audit", {
        "figure": "A6",
        "kind": "measurement",
        "claim_label": args.label,
        "n": len(rows),
        "seed": None,
        "sources": {"manifest": str(args.manifest),
                    "card": str(args.card) if args.card else None,
                    "manifest_sha256": manifest_sha(rows)},
        "cmd": f"python tools/figures.py a6-audit --out figs --manifest {args.manifest}",
        "n_real": n_real,
        "n_fake": n_fake,
        "counts": {f"{s}/{g}": n for (s, g), n in sorted(counts.items())},
        "warnings": [
            "chỉ đếm từ manifest (ingest đếm thật), không suy từ dung lượng archive",
            "nhãn ở đây là nguồn dữ liệu, không phải mức tái lập của kết quả",
        ],
    })


# --------------------------------------------------------------------------- #
# Đợt 2 — cần scores (detector đã chấm)
# --------------------------------------------------------------------------- #

def _load_scores(args):
    import numpy as np

    scores = np.load(args.scores).astype("float64").ravel()
    labels = np.load(args.labels).astype("int64").ravel()
    if scores.shape != labels.shape:
        raise SystemExit(f"scores {scores.shape} ≠ labels {labels.shape}")
    if args.generators:
        gens = np.load(args.generators, allow_pickle=False).ravel()
    else:
        gens = np.array(["all"] * len(scores))
    if len(gens) != len(scores):
        raise SystemExit(f"generators {len(gens)} ≠ scores {len(scores)}")
    return scores, labels, gens


def a2_scores(args) -> dict:
    """A2 — phân bố điểm theo generator (ridge plot) + AUC từng dòng."""
    import numpy as np
    from dire_zero.metrics import roc_auc

    plt = _mpl()
    _style(plt)
    scores, labels, gens = _load_scores(args)
    order = [g for g in sorted(set(gens.tolist()), key=lambda g: (g != "real", g))]

    fig, ax = plt.subplots(figsize=(COL2 * 0.8, 0.42 * len(order) + 0.9))
    real = scores[labels == 0]
    aucs: dict[str, float | None] = {}
    for i, g in enumerate(order):
        sel = gens == g
        fake = scores[sel & (labels == 1)] if g != "real" else np.array([])
        if real.size:
            ax.scatter(real, np.full(real.shape, i) - 0.18, s=2, color="0.6", alpha=0.5)
        if fake.size:
            ax.scatter(fake, np.full(fake.shape, i) + 0.18, s=2, color="C0", alpha=0.5)
        # mỗi dòng = "thật vs generator này", không phải "thật vs phần còn lại":
        # lấy toàn bộ mẫu thật cộng mẫu của generator đang xét, nếu không thì
        # tập con toàn lớp 1 và AUC vô nghĩa (NaN).
        pair = sel | (labels == 0)
        auc = roc_auc(scores[pair].tolist(), labels[pair].tolist())
        aucs[g] = None if auc != auc else round(float(auc), 4)   # NaN ⇒ null
        txt = "—" if aucs[g] is None else f"AUC {auc:.3f}"
        ax.text(1.01, i, txt, transform=ax.get_yaxis_transform(),
                fontsize=6, va="center", ha="left", color="0.25")
    ax.axvline(0.5, color="C3", ls="--", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("điểm của detector (P(fake))")
    ax.set_title(f"phân bố điểm theo generator · n={len(scores)}", fontsize=8)
    fig.subplots_adjust(right=0.82)

    _stamp(fig, plt, args.label, "ngưỡng chọn trên val, không trên test")
    return _save(fig, plt, Path(args.out), "a2-scores", {
        "figure": "A2",
        "kind": "measurement",
        "claim_label": args.label,
        "n": int(len(scores)),
        "seed": None,
        "sources": {"scores": str(args.scores), "labels": str(args.labels),
                    "generators": str(args.generators) if args.generators else None},
        "cmd": (f"python tools/figures.py a2-scores --out figs --scores {args.scores} "
                f"--labels {args.labels}"
                + (f" --generators {args.generators}" if args.generators else "")),
        "auc": aucs,
        "warnings": ["mỗi dòng là một generator; đừng đọc hình gộp thay cho hình này"],
    })


def a3_roc(args) -> dict:
    """A3 — ROC/PR theo generator, kèm AP; n nhỏ thì phải thấy được độ rộng."""
    import numpy as np
    from dire_zero.metrics import average_precision, roc_auc

    plt = _mpl()
    _style(plt)
    scores, labels, gens = _load_scores(args)
    order = [g for g in sorted(set(gens.tolist())) if g != "real"][: args.max_generators]

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3))
    for g in order:
        # "thật + generator này" — cùng lý do như A2: tập con thuần lớp 1 thì
        # ROC không tồn tại (không có âm tính để tính FPR).
        sel = (gens == g) | (labels == 0)
        s, y = scores[sel], labels[sel]
        if len(set(y.tolist())) < 2:
            continue
        thr = np.unique(s)[::-1]
        tp = np.array([((s >= t) & (y == 1)).sum() for t in thr], dtype=float)
        fp = np.array([((s >= t) & (y == 0)).sum() for t in thr], dtype=float)
        tpr = tp / max(y.sum(), 1)
        fpr = fp / max((y == 0).sum(), 1)
        axes[0].plot(np.r_[0, fpr, 1], np.r_[0, tpr, 1], lw=0.9,
                     label=f"{g} (AUC {roc_auc(s.tolist(), y.tolist()):.3f})")
        prec = tp / np.maximum(tp + fp, 1)
        axes[1].plot(tpr, prec, lw=0.9, label=f"{g} (AP {average_precision(s.tolist(), y.tolist()):.3f})")

    axes[0].plot([0, 1], [0, 1], ls=":", lw=0.7, color="0.6")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC", fontsize=7.5)
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("PR", fontsize=7.5)
    for ax in axes:
        ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.tight_layout()

    _stamp(fig, plt, args.label, f"n={len(scores)} · CI bootstrap: xem eval.json")
    return _save(fig, plt, Path(args.out), "a3-roc", {
        "figure": "A3",
        "kind": "measurement",
        "claim_label": args.label,
        "n": int(len(scores)),
        "seed": None,
        "sources": {"scores": str(args.scores), "labels": str(args.labels),
                    "generators": str(args.generators) if args.generators else None},
        "cmd": (f"python tools/figures.py a3-roc --out figs --scores {args.scores} "
                f"--labels {args.labels}"
                + (f" --generators {args.generators}" if args.generators else "")),
        "warnings": [
            "đường cong KHÔNG kèm dải CI ở đây — CI nằm trong eval.json (bootstrap_ci)",
            "n nhỏ ⇒ một điểm khác biệt cũng làm đường cong đổi hình dạng",
        ],
    })


def f4_robustness(args) -> dict:
    """F4 — plot measured AP under the paper's blur/JPEG conditions from ledger JSON."""
    plt = _mpl()
    _style(plt)
    source = Path(args.results)
    payload = json.loads(source.read_text(encoding="utf-8"))
    conditions = payload.get("by_condition") or {}
    expected = {"clean", "blur_sigma1", "blur_sigma2", "blur_sigma3", "jpeg_q65", "jpeg_q30"}
    if set(conditions) != expected:
        raise SystemExit(f"Figure 4 cần đủ điều kiện {sorted(expected)}; có {sorted(conditions)}")
    models = ("DIRE", "RGB")
    generators = ("adm", "ddpm", "iddpm", "pndm", "sdv2")
    fig, ax = plt.subplots(figsize=(COL2, 2.8))
    names = list(conditions)
    x = list(range(len(names)))
    seen = 0
    for model in models:
        for generator in generators:
            vals = []
            for name in names:
                cell = (((conditions[name] or {}).get("by_model") or {}).get(model) or {}).get(generator)
                if not cell or cell.get("n", 0) <= 0 or cell.get("ap") is None:
                    raise SystemExit(f"thiếu AP: {name}/{model}/{generator}")
                value = float(cell["ap"])
                if not 0 <= value <= 1:
                    raise SystemExit(f"AP ngoài [0,1]: {name}/{model}/{generator}={value}")
                vals.append(value * 100)
            ax.plot(x, vals, marker="o", ms=2.5, lw=1.0 if model == "DIRE" else 0.75,
                    ls="-" if model == "DIRE" else "--",
                    label=f"{model} · {generator}")
            seen += 1
    ax.set_xticks(x)
    ax.set_xticklabels(("clean", "blur σ=1", "blur σ=2", "blur σ=3", "JPEG 65", "JPEG 30"), rotation=15)
    ax.set_ylim(0, 102)
    ax.set_ylabel("Average Precision (%)")
    ax.set_title("Robustness theo điều kiện nhiễu loạn", fontsize=8)
    ax.grid(axis="y", lw=0.35, alpha=0.4)
    ax.legend(frameon=False, ncol=2, fontsize=5.2, loc="lower left")
    fig.tight_layout()
    subset = payload.get("subset") or {}
    subset_status = str(subset.get("subset_status", "full-data"))
    subtitle = "DIRE tính lại sau perturbation; RGB là baseline ResNet-50 của dự án"
    if subset_status == "pilot":
        subtitle = f"PILOT SUBSET · {subset.get('per_cell')} / cell · " + subtitle
    _stamp(fig, plt, args.label, subtitle)
    return _save(fig, plt, Path(args.out), "f4-robustness", {
        "figure": "F4",
        "kind": "measurement",
        "claim_label": args.label,
        "n": sum(int(cell.get("n", 0)) for c in conditions.values()
                 for m in models for cell in (c.get("by_model", {}).get(m, {}).values())),
        "seed": None,
        "sources": {"results": str(source)},
        "cmd": f"python tools/figures.py f4-robustness --out figs --results {source}",
        "conditions": names,
        "series": seen,
        "subset": subset or None,
        "subset_status": subset_status,
        "warnings": (["Figure 4 uses a pilot subset; this is not a full-data paper result"]
                     if subset_status == "pilot" else []) + [
            "paper Figure 4 uses CNNDetection* and PatchForensics* baselines; this run plots project ResNet-50 RGB vs DIRE and is not an exact reproduction of those baselines",
            "paper text leaves unclear whether DIRE is recomputed after perturbation; this run follows the explicit recompute policy in its input ledger",
            str((payload.get("protocol") or {}).get("test_generator_scope_note") or
                "test generator scope is not recorded in the source ledger"),
        ],
    })


# --------------------------------------------------------------------------- #
# Registry + CLI
# --------------------------------------------------------------------------- #

FIGURES = {
    "f2-concept": (f2_concept, "schematic", "—", "sơ đồ khái niệm p_r vs p_g"),
    "f3-pipeline": (f3_pipeline, "schematic", "—", "pipeline inversion → DIRE → classifier"),
    "a1-provenance": (a1_provenance, "schematic", "—", "ô nào của tác giả ⇒ nhãn R0/R1+R2"),
    "f5-noise": (f5_noise, "R0 (từ map tác giả)", "--maps", "noise + phổ tần số + phổ radial"),
    "f1-teaser": (f1_teaser, "R0 hoặc R1+R2", "--images --maps", "ảnh gốc | DIRE"),
    "f4-robustness": (f4_robustness, "R1+R2", "--results", "AP theo blur/JPEG"),
    "a6-audit": (a6_audit, "schematic", "--manifest", "audit dữ liệu: split × generator"),
    "a2-scores": (a2_scores, "R1+R2", "--scores --labels", "phân bố điểm theo generator"),
    "a3-roc": (a3_roc, "R1+R2", "--scores --labels", "ROC/PR theo generator"),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="figures.py",
        description="Sinh figure cho báo cáo — mỗi hình một lệnh, có sidecar truy nguồn.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, label_default="schematic"):
        sp.add_argument("--out", default="figs", help="thư mục ra (mặc định figs)")
        sp.add_argument("--label", default=label_default, choices=ALLOWED_LABELS,
                        help="nhãn mức tái lập ghi trên hình")
        sp.add_argument("--seed", type=int, default=DEFAULT_SEED)

    sp = sub.add_parser("list", help="liệt kê hình + đầu vào cần có")
    sp.set_defaults(func=None)

    common(sub.add_parser("f2-concept", help="F2 sơ đồ khái niệm"))
    sp = sub.add_parser("f3-pipeline", help="F3 pipeline")
    sp.add_argument("--steps", type=int, default=20)
    common(sp)

    common(sub.add_parser("a1-provenance", help="A1 phân quyền nhãn"))

    sp = sub.add_parser("f5-noise", help="F5 noise + phổ")
    sp.add_argument("--maps", required=True)
    sp.add_argument("--split", default="test")
    sp.add_argument("--generator", default="adm")
    sp.add_argument("--n", type=int, default=3)
    common(sp, "R0")

    sp = sub.add_parser("f1-teaser", help="F1 teaser ảnh | DIRE")
    sp.add_argument("--images", required=True)
    sp.add_argument("--maps", required=True)
    sp.add_argument("--split", default="test")
    sp.add_argument("--generator", default="adm")
    sp.add_argument("--n", type=int, default=3)
    common(sp, "R1+R2")

    sp = sub.add_parser("f4-robustness", help="F4 AP dưới blur/JPEG từ Figure 4 ledger")
    sp.add_argument("--results", required=True, help="figure4.json do Modal runner sinh")
    common(sp, "R1+R2")

    sp = sub.add_parser("a6-audit", help="A6 audit dữ liệu")
    sp.add_argument("--manifest", required=True)
    sp.add_argument("--card", default=None)
    common(sp)

    for name, help_txt in (("a2-scores", "A2 phân bố điểm"), ("a3-roc", "A3 ROC/PR")):
        sp = sub.add_parser(name, help=help_txt)
        sp.add_argument("--scores", required=True)
        sp.add_argument("--labels", required=True)
        sp.add_argument("--generators", default=None)
        if name == "a3-roc":
            sp.add_argument("--max-generators", type=int, default=6)
        common(sp, "R1+R2")

    return p


def cmd_list() -> int:
    print(f"{'tên':<14} {'nhãn mặc định':<20} {'đầu vào':<18} mô tả")
    for name, (_fn, label, inputs, desc) in FIGURES.items():
        print(f"{name:<14} {label:<20} {inputs:<18} {desc}")
    print("\nChi tiết + thứ tự làm: docs/plans/research/figures-plan.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "list":
        return cmd_list()
    fn = FIGURES[args.cmd][0]
    fn(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
