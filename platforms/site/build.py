# platforms/site/build.py
"""Sinh `data.json` cho trang pitch deck — MỌI con số đều đọc từ repo.

Vì sao không viết số thẳng vào HTML
-----------------------------------
Một trang giới thiệu là nơi số liệu dễ trôi nhất: chép 99.9 vào một thẻ `<div>`,
rồi ba tháng sau code đổi mà trang vẫn "đẹp". Đúng loại lỗi mà repo này đã sửa
nhiều lần (xem `tests/test_paper_table.py`, `tests/test_manim_scenes.py`).

Nên kiến trúc là: **HTML không chứa số nào**. Trang chỉ có khung + `data-bind`
trỏ tới khoá trong `data.json`; `data.json` do file này sinh ra từ:

| Nguồn | Dùng cho |
|---|---|
| `src/dire_modal/paper_table2.tsv` | bảng Table 2, trung bình 4 detector |
| `src/dire_modal/manifest.py` (tier `e2`) | số cặp, số detector, số container |
| `src/dire_modal/demo_train.py` + `demo_train.yaml` | tham số và chi phí của demo |
| `docs/runbook-free.md` (số đo) | tốc độ Baidu, bẫy `ls -l` |

Sửa dữ liệu ⇒ chạy lại `build.py` ⇒ trang đổi theo. `--check` phát hiện
`data.json` bị cũ (dùng trong `tests/test_site.py`).

Chạy::

    python3 platforms/site/build.py            # ghi data.json
    python3 platforms/site/build.py --check    # exit 1 nếu data.json đã cũ
    python3 platforms/site/build.py --stdout   # in ra, không ghi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
OUT = HERE / "data.json"

# Số đo, KHÔNG phải suy ra: mỗi cái phải có ngày và nguồn trong tài liệu.
# Ghi ở đây (không ở HTML) để trang web không thể tự bịa số.
MEASURED = {
    "baidu_kib_s": {
        "value": 76.6,
        "text": "76.6 KiB/s",
        "source": "đo 2026-10-25 · docs/runbook-free.md §2",
        "note": "trần của TÀI KHOẢN free, không phải của container",
    },
    "ls_rounding": {
        "shown": "119.22MB",
        "reversed_bytes": 125_011_230,
        "real_bytes": 125_015_131,
        "delta_bytes": 3_901,
        "source": "log thật 2026-10-25 · docs/runbook-free.md §Sự cố 2026-10-25",
    },
    "fc_weight": {
        "paper_shape": "[1, 2048]",
        "old_code_shape": "[2, 2048]",
        "source": "sự cố 2026-10-28 · PR #18",
    },
}

# Trạng thái — PROSE, nhưng có `state` máy đọc được để trang tự đếm.
# `state`: done = code + đã kiểm bằng test · run = code xong, CHƯA chạy thật
#          todo = chưa có gì · out = ngoài phạm vi (có lý do)
STATUS = [
    {"name": "Tải + verify dữ liệu", "state": "run",
     "note": "code + test xong; chưa chạy trọn trong CI", "ref": "download.py"},
    {"name": "Eval bằng checkpoint tác giả", "state": "run",
     "note": "code + test xong; **số của mình chưa có**", "ref": "eval.py"},
    {"name": "Tái lập Table 2", "state": "run",
     "note": "quy trình `scripts/01_repro.sh` sẵn sàng, chờ chạy", "ref": "01_repro.sh"},
    {"name": "Mirror HF / Kaggle + sha256", "state": "todo",
     "note": "code có; `manifests/*_sha256.tsv` còn trống ⇒ CỔNG 3 chưa đóng",
     "ref": "archive.py"},
    {"name": "AEROBLADE (FR-3)", "state": "todo",
     "note": "`src/dire_modal/aero.py` chưa tồn tại", "ref": "spec.md FR-3"},
    {"name": "Fusion DIRE ⊕ AEROBLADE (FR-5)", "state": "todo",
     "note": "`fusion.py` chưa có; phụ thuộc FR-3", "ref": "spec.md FR-5"},
    {"name": "Demo HF Spaces", "state": "run",
     "note": "code + hướng dẫn; chờ checkpoint & gallery", "ref": "platforms/hf-spaces"},
    {"name": "Video giải thích (manim)", "state": "run",
     "note": "5 scene + kế hoạch 8 chương; **chưa render**", "ref": "tools/manim"},
    {"name": "Train lại detector", "state": "out",
     "note": "ngoài phạm vi: N1 — chỉ cải thiện, không train lại", "ref": "spec.md N1"},
    {"name": "Stage-2: compute DIRE toàn bộ", "state": "run",
     "note": "30–60 A100-h ⇒ ngoài ngân sách free tier", "ref": "compute_dire.py"},
]

# Việc còn thiếu, xếp theo thứ tự làm — dùng cho slide "Roadmap".
ROADMAP = [
    {"what": "Sửa `eval --tiers` để lọc thật", "why":
     "tham số đang bị bỏ qua (`eval.py:869` dùng `rglob` cả `/raw/dire/test`) ⇒ "
     "`--tiers e2` vẫn chấm 24 cặp: tốn tiền và bẩn số", "cost": "~30 phút"},
    {"what": "`aero.py` (FR-3)", "why":
     "chưa có ⇒ toàn bộ phần \"cải thiện\" của dự án không tồn tại", "cost": "1 PR"},
    {"what": "`fusion.py` (FR-5)", "why": "đóng góp riêng; cần FR-3 xong",
     "cost": "sau FR-3"},
    {"what": "`manifests/*_sha256.tsv`", "why":
     "`verify` đang in \"CHƯA đối chiếu sha256\" ⇒ CỔNG 3 chưa đóng", "cost": "1 lần chạy"},
]


#: Tên hiển thị của 4 detector. `lsun_iddpm` → `iDDPM` chứ không phải `IDDPM`:
#: paper viết hoa chữ cái đầu của "iDDPM", và tên sai làm người đọc đối chiếu
#: nhầm dòng trong bảng giấy.
DETECTOR_LABEL = {
    "lsun_adm": "ADM",
    "lsun_pndm": "PNDM",
    "lsun_iddpm": "iDDPM",
    "lsun_stylegan": "StyleGAN",
}


def _load_repo():
    """Import package thật (không chép logic).

    Một chi tiết gây nhiễu: cả `eval` và `demo_train` đều đăng ký hàm `main`
    trong cùng `modal.App`, nên import cả hai trong một tiến trình làm Modal in
    `function name 'main' collision!`. Ở đây chỉ cần hàm THUẦN của hai module;
    hạ mức log của client Modal trong lúc import để output build không có cảnh
    báo trông như lỗi. (Xem ADR-0003: import `eval` bên trong hàm là quy ước
    của `demo_train`.)
    """
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import logging  # noqa: PLC0415

    # `logging.disable` chứ không `logger.setLevel`: thông báo của Modal đi qua
    # logger nội bộ do nó tự tạo, nên đặt level trên một tên logger cụ thể
    # không chặn được (đã thử). Chỉ tắt trong lúc import rồi bật lại.
    logging.disable(logging.WARNING)
    try:
        from dire_modal import demo_train, eval as E, manifest  # noqa: PLC0415
    finally:
        logging.disable(logging.NOTSET)
    return demo_train, E, manifest


def _paper(E) -> dict:
    """Table 2 + trung bình từng detector — tính từ TSV, không chép."""
    table = E.load_paper_table()
    gens = list(E.load_paper_table()["lsun_adm"].keys())
    gens = [g for g in gens if g != "avg"]
    detectors = []
    for ckpt, gens_map in table.items():
        accs = [gens_map[g]["acc"] for g in gens]
        aps = [gens_map[g]["ap"] for g in gens]
        detectors.append({
            "id": ckpt,
            "label": DETECTOR_LABEL.get(ckpt, ckpt.replace("lsun_", "").upper()),
            "acc": {g: gens_map[g]["acc"] for g in gens},
            "ap": {g: gens_map[g]["ap"] for g in gens},
            "mean_acc": round(sum(accs) / len(accs), 2),
            "mean_ap": round(sum(aps) / len(aps), 2),
            "avg_row": {"acc": gens_map["avg"]["acc"], "ap": gens_map["avg"]["ap"]},
        })
    return {"generators": gens, "detectors": detectors,
            "source": "src/dire_modal/paper_table2.tsv · ICCV 2023, Table 2"}


def _pipeline(manifest) -> dict:
    """Đếm từ manifest: cặp = tar GIẢ (không tính `real`)."""
    tier = "e2"
    ents = manifest.resolve_many(tier)
    tars = [e for e in ents if e.remote.endswith(".tar.gz")]
    fakes = [e for e in tars if not e.remote.endswith("/real.tar.gz")]
    ckpts = [e for e in ents if e.remote.endswith((".pth", ".pt"))]
    return {
        "tier": tier,
        "items": len(ents),
        "tars": len(tars),
        "pairs": len(fakes),
        "checkpoints": len(ckpts),
        "source": f"manifest tier `{tier}` — đếm tar, KHÔNG đếm `real` là generator",
    }


def _demo(demo_train) -> dict:
    """Tham số + chi phí của demo_train, dùng CHÍNH hàm ước lượng của module."""
    cfg = demo_train.load_config()
    est = demo_train.estimate(
        images=int(cfg["images"]), epochs=int(cfg["epochs"]),
        batch=int(cfg["batch_size"]), sec_per_step=float(cfg["sec_per_step"]),
        usd_per_gpu_hour=float(cfg["usd_per_gpu_hour"]),
        val_images=int(cfg["val_images"]),
    )
    return {
        "labels": demo_train.DEMO_LABEL,
        "images_epochs": f"{cfg['images']} × {cfg['epochs']}",
        "seed": cfg["seed"], "arch": cfg["arch"], "n_out": cfg["n_out"],
        "loss": cfg["loss"], "batch_size": cfg["batch_size"],
        "lr": cfg["lr"], "epochs": cfg["epochs"], "images": cfg["images"],
        # Số của TÁC GIẢ (không phải của demo) — để slide so sánh quy mô mà
        # không phải gõ "400" vào HTML. Nguồn: ADR-0003, bảng so sánh.
        "paper_epochs": 400,
        "steps": est["train_steps"], "val_steps": est["val_steps"],
        "gpu_seconds": est["gpu_seconds"], "usd": est["usd"],
        "gpu": cfg["gpu"], "sec_per_step": est["sec_per_step"],
        "measured": False,   # sec_per_step là ước lượng — trang phải nói rõ
        "source": "src/dire_modal/demo_train.yaml · docs/adr-0003-demo-training.md",
    }


def build() -> dict:
    demo_train, E, manifest = _load_repo()
    paper = _paper(E)
    pipeline = _pipeline(manifest)
    pipeline["containers"] = pipeline["pairs"] * len(paper["detectors"])
    # Khoá suy ra: để HTML không phải tự nhân/chia (một phép tính trong template
    # là một chỗ nữa có thể sai, và nó không có test).
    pipeline["detectors_count"] = len(paper["detectors"])
    demo = _demo(demo_train)
    counts = {"done": 0, "run": 0, "todo": 0, "out": 0}
    for s in STATUS:
        counts[s["state"]] = counts.get(s["state"], 0) + 1
    return {
        "schema": 1,
        # Ba loại nhãn nguồn — mọi con số trên trang thuộc đúng MỘT loại.
        "labels": {
            "paper": "số của paper (ICCV 2023) — CHƯA phải số đo của mình",
            "measured": "số ĐO của repo · xem nguồn từng ô",
            "demo": demo["labels"],
        },
        # Dòng trạng thái trên bìa. Tính Ở ĐÂY chứ không trong JS: đây là câu
        # khẳng định về số liệu, phải có test, và test chỉ kiểm được Python.
        "honesty": {
            "pill": (f"code xong: {counts['run']} · chưa có: {counts['todo']}"
                     f" · số của dự án: chưa có"),
        },
        "generated_by": "platforms/site/build.py",
        "sources": {
            "paper": "src/dire_modal/paper_table2.tsv",
            "manifest": "src/dire_modal/manifest.py (tier e2)",
            "demo": "src/dire_modal/demo_train.yaml",
            "measured": "docs/runbook-free.md (số đo 2026-10-25/28)",
        },
        "paper": paper,
        "pipeline": pipeline,
        "demo": demo,
        "measured": MEASURED,
        "status": {"counts": counts, "items": STATUS},
        "roadmap": ROADMAP,
    }


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="không ghi; exit 1 nếu data.json đã cũ")
    ap.add_argument("--stdout", action="store_true", help="in ra stdout")
    args = ap.parse_args(argv)

    text = _dump(build())

    if args.stdout:
        sys.stdout.write(text)
        return 0

    if args.check:
        if not OUT.exists():
            print(f"[!] thiếu {OUT.relative_to(ROOT)} — chạy build.py")
            return 1
        if OUT.read_text(encoding="utf-8") != text:
            print(f"[!] {OUT.relative_to(ROOT)} đã CŨ so với dữ liệu trong repo.\n"
                  "    Chạy: python3 platforms/site/build.py")
            return 1
        print(f"[ok] {OUT.relative_to(ROOT)} khớp dữ liệu")
        return 0

    OUT.write_text(text, encoding="utf-8")
    payload = json.loads(text)
    print(f"[ok] {OUT.relative_to(ROOT)} "
          f"({len(payload['paper']['detectors'])} detector · "
          f"{payload['pipeline']['pairs']} cặp · "
          f"{payload['pipeline']['containers']} container · "
          f"{len(payload['roadmap'])} việc còn thiếu)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
