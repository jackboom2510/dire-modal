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
| `src/dire_modal/paper_table2.tsv` | các biến thể DIRE trong Table 2 của paper |
| `src/dire_modal/paper_detector_benchmark.tsv` | baseline trong Table 2 của paper |
| `src/dire_modal/paper_table5.tsv` | đối chiếu representation trong Table 5 của paper |

Sửa dữ liệu ⇒ chạy lại `build.py` ⇒ trang đổi theo. `--check` phát hiện
`data.json` bị cũ (dùng trong `tests/test_site.py`).

Chạy::

    python3 platforms/site/build.py            # ghi data.json
    python3 platforms/site/build.py --check    # exit 1 nếu data.json đã cũ
    python3 platforms/site/build.py --stdout   # in ra, không ghi
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
OUT = HERE / "data.json"
PAPER_BASELINES = SRC / "dire_modal/paper_detector_benchmark.tsv"
PAPER_TABLE5 = SRC / "dire_modal/paper_table5.tsv"

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
    {"name": "Local CLI + wrapper Modal (FR-9)", "state": "done",
     "note": "`dire detect/batch/serve` chạy không cần Modal; `dire_modal.api` là wrapper. "
             "Cùng lõi `dire_core` ⇒ điểm số khớp (cổng S9)", "ref": "adr-0005"},
    {"name": "Train lại detector (FR-8)", "state": "todo",
     "note": "chưa train; không có kết quả repo để báo cáo", "ref": "spec.md FR-8"},
    {"name": "Stage-2: compute DIRE toàn bộ", "state": "run",
     "note": "chưa chạy toàn bộ; chưa có số đo compute để báo cáo", "ref": "compute_dire.py"},
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
    """Import the paper-table reader from the repository package."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import logging  # noqa: PLC0415

    logging.disable(logging.WARNING)
    try:
        from dire_modal import eval as E  # noqa: PLC0415
    finally:
        logging.disable(logging.NOTSET)
    return E


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


def _paper_method_comparisons() -> dict:
    """Table 2 baselines and Table 5 input representations, sourced from TSVs."""
    def read_rows(path: Path) -> list[dict]:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream, delimiter="\t"))

    baselines = [
        {**row, "acc": float(row["acc"]), "ap": float(row["ap"])}
        for row in read_rows(PAPER_BASELINES)
    ]
    table5_rows = [
        {**row, "acc": float(row["acc"]), "ap": float(row["ap"])}
        for row in read_rows(PAPER_TABLE5)
    ]
    table5 = {}
    for row in table5_rows:
        table5.setdefault(row["representation"], {})[row["generator"]] = {
            "acc": row["acc"], "ap": row["ap"]}
    rgb, dire = table5["RGB"]["sdv1"], table5["DIRE"]["sdv1"]
    return {
        "detector_baselines": baselines,
        "table5": table5,
        "table5_rows": table5_rows,
        "sdv1_delta_pp": {
            "acc": round(dire["acc"] - rgb["acc"], 1),
            "ap": round(dire["ap"] - rgb["ap"], 1),
        },
        "baseline_source": "src/dire_modal/paper_detector_benchmark.tsv · ICCV 2023, Table 2",
        "table5_source": "src/dire_modal/paper_table5.tsv · ICCV 2023, Table 5",
    }


def build() -> dict:
    E = _load_repo()
    paper = _paper(E)
    paper["comparisons"] = _paper_method_comparisons()
    counts = {"done": 0, "run": 0, "todo": 0, "out": 0}
    for item in STATUS:
        counts[item["state"]] = counts.get(item["state"], 0) + 1
    return {
        "schema": 1,
        "labels": {
            "paper": "kết quả được công bố trong paper ICCV 2023; không phải số đo của repo",
            "repo": "trạng thái repo, không phải benchmark hiệu năng",
        },
        "honesty": {
            "pill": "chỉ hiển thị kết quả paper đã công bố · chưa có benchmark repo được xác minh",
        },
        "generated_by": "platforms/site/build.py",
        "sources": {
            "paper_table2": "src/dire_modal/paper_table2.tsv",
            "paper_baselines": "src/dire_modal/paper_detector_benchmark.tsv",
            "paper_table5": "src/dire_modal/paper_table5.tsv",
        },
        "paper": paper,
        "status": {"counts": counts, "items": STATUS},
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
          f"({len(payload['paper']['detectors'])} DIRE rows · "
          f"{len(payload['paper']['generators'])} paper generators · "
          f"{len(payload['status']['items'])} repo status items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
