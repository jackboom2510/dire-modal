# tools/manim/data.py
"""Số cho các scene manim — đọc từ nguồn trong repo, KHÔNG chép tay.

Vì sao file này tách khỏi `scenes.py`
-------------------------------------
Một video giải thích rất dễ trở thành nguồn số thứ hai, sai dần so với code. Ba
quy tắc ở đây giữ nó không trôi:

1. **Không có số của paper trong file này.** `paper_matrix()` đọc
   `src/dire_modal/paper_table2.tsv` — cùng file mà `eval::report` dùng để tính
   Δ, nên video và báo cáo không thể lệch nhau.
2. **Không có số của đường loss trong file này.** `loss_log()` đọc
   `docs/media/demo_train_log.json` do `demo_train --mode run` sinh ra; chưa có
   file thì trả ``None`` và scene phải nói "chưa có log", **không** vẽ đường cong
   giả (NFR-6).
3. **Số của pipeline được TÍNH, không gõ.** `pairs()` đếm từ manifest (tier
   `e2`), `containers()` = `pairs() × len(detectors)`. Sửa manifest thì video đổi
   theo; đó là toàn bộ mục đích.

Chỉ dùng thư viện chuẩn (cho `data.py`): scene có thể chạy trong venv manim
không có pyyaml/torch. Riêng `pairs()` cần `dire_modal.manifest`, nên nó import
theo yêu cầu và lỗi có thông điệp rõ nếu thiếu `src/` trên `PYTHONPATH`.

Kiểm: `tests/test_manim_scenes.py`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# tools/manim/data.py → repo root là parents[2].
ROOT = Path(__file__).resolve().parents[2]
PAPER_TABLE = ROOT / "src" / "dire_modal" / "paper_table2.tsv"
# Chỗ `demo_train --mode run` ghi ra (sau khi `modal volume get` về máy).
DEFAULT_LOSS_LOG = ROOT / "docs" / "media" / "demo_train_log.json"

# Thang của trục ACC. Là 100 vì đơn vị là PHẦN TRĂM — không phải một con số đo.
PERCENT_MAX = 100.0

# 8 cột của Table 2 — đúng thứ tự paper in.
GENERATORS = ["adm", "ddpm", "iddpm", "pndm", "sdv1", "sdv2", "ldm", "vqdiffusion"]


class DataMissing(SystemExit):
    """Thiếu dữ liệu để vẽ ⇒ dừng, KHÔNG vẽ số thay thế.

    ``SystemExit`` (không phải ``Exception``) để một scene không thể vô tình bắt
    rồi "vẫn render cho xong": render ra một khung hình thiếu số còn tệ hơn
    không render.
    """


# ---------------------------------------------------------------------------
# 1. Số của paper — đọc TSV, cùng nguồn với `eval::report`
# ---------------------------------------------------------------------------
def paper_rows(path: Path | str | None = None) -> list[dict]:
    """``[{ckpt, gen, acc, ap}]`` theo đúng thứ tự file (đã sắp xếp trong TSV)."""
    p = Path(path) if path else PAPER_TABLE
    if not p.exists():
        raise DataMissing(f"[!] Không thấy bảng paper: {p}\n    Chạy từ gốc repo.")
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("ckpt\t"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise DataMissing(f"[!] Dòng hỏng trong {p.name}: {line!r}")
        ckpt, gen, acc, ap = parts
        rows.append({"ckpt": ckpt, "gen": gen,
                     "acc": float(acc), "ap": float(ap)})
    if not rows:
        raise DataMissing(f"[!] {p} rỗng — không có số nào để vẽ.")
    return rows


def detectors(path: Path | str | None = None) -> list[str]:
    """4 detector của paper, theo thứ tự xuất hiện."""
    seen: list[str] = []
    for r in paper_rows(path):
        if r["ckpt"] not in seen:
            seen.append(r["ckpt"])
    return seen


def paper_matrix(ckpt: str | None = None, path: Path | str | None = None
                 ) -> dict[str, dict[str, float]]:
    """``{generator: {"acc": %, "ap": %}}`` cho **một** detector (mặc định hàng đầu).

    Không trả số của ``avg``: đó là trung bình paper in sẵn, scene tự tính từ 8 ô
    để mắt người xem thấy nó khớp.
    """
    rows = paper_rows(path)
    want = ckpt or rows[0]["ckpt"]
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["ckpt"] != want or r["gen"] == "avg":
            continue
        out[r["gen"]] = {"acc": r["acc"], "ap": r["ap"]}
    if not out:
        raise DataMissing(f"[!] Không có hàng nào cho detector {want!r}.")
    return out


def mean_acc(ckpt: str | None = None, path: Path | str | None = None) -> float:
    """Trung bình ACC của 8 generator — tính tại đây, không chép từ hàng `avg`."""
    vals = [v["acc"] for v in paper_matrix(ckpt, path).values()]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# 2. Số của pipeline — TÍNH từ manifest, không gõ
# ---------------------------------------------------------------------------
def _manifest():
    """Import `dire_modal.manifest` (cần `src/` trên `PYTHONPATH`)."""
    import sys

    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from dire_modal import manifest  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise DataMissing(
            f"[!] Không import được dire_modal.manifest: {exc}\n"
            "    Chạy từ gốc repo (render.sh đã export PYTHONPATH=src)."
        ) from exc
    return manifest


def pairs(tier: str = "e2") -> int:
    """Số cặp (fake vs real) mà tier trả về = số tar GIẢ, không tính `real`.

    Với `e2`: 17 mục = 1 checkpoint + 16 tar, trong đó 1 tar là `real` ⇒ **15**.
    Con số này từng bị ghi sai thành "16 generator" ở 5 chỗ trong tài liệu; tính
    từ manifest là cách duy nhất để nó không sai lại.
    """
    ents = _manifest().resolve_many(tier)
    fakes = [e for e in ents
             if e.remote.endswith(".tar.gz") and "/real.tar.gz" not in e.remote]
    if not fakes:
        raise DataMissing(f"[!] Tier {tier!r} không có tar giả nào — sai tier?")
    return len(fakes)


def containers(tier: str = "e2") -> int:
    """Số container eval = số cặp × số detector (4 detector × 15 cặp = 60)."""
    return pairs(tier) * len(detectors())


def pipeline_facts(tier: str = "e2") -> list[dict]:
    """Các con số của Scene 4, mỗi cái kèm ``source`` để người xem kiểm lại.

    Chỉ hai giá trị ở đây là hằng số **đo được** (tốc độ Baidu, dung lượng tier);
    phần còn lại đều suy ra từ chúng hoặc từ manifest.
    """
    return [
        # Con số DUY NHẤT ở đây là đo được, và nó là trần của tài khoản free —
        # không phải thông lượng kỳ vọng. Mọi số còn lại đều suy ra.
        {"key": "baidu_rate", "text": "76.6 KiB/s", "value": 76.6,
         "source": "đo 2026-10-25, docs/runbook-free.md §2"},
        {"key": "pairs", "text": str(pairs(tier)), "value": pairs(tier),
         "source": f"manifest tier {tier}: tar giả, KHÔNG tính real"},
        {"key": "detectors", "text": str(len(detectors())), "value": len(detectors()),
         "source": "paper_table2.tsv — 4 hàng"},
        {"key": "containers", "text": str(containers(tier)), "value": containers(tier),
         "source": "pairs × detectors (tính, không gõ)"},
    ]


# ---------------------------------------------------------------------------
# 3. Số của đường loss — đọc log THẬT, không có thì KHÔNG vẽ
# ---------------------------------------------------------------------------
def loss_log_path() -> Path:
    """`DIRE_LOSS_LOG` ghi đè được để render bản log khác mà không sửa code."""
    env = os.environ.get("DIRE_LOSS_LOG", "").strip()
    return Path(env) if env else DEFAULT_LOSS_LOG


def loss_log(path: Path | str | None = None) -> dict | None:
    """Đọc `loss_log.json` của `demo_train`; ``None`` nếu chưa chạy.

    Trả ``None`` thay vì một dict rỗng: scene phải phân biệt được "chưa có log"
    với "log có 0 step", và cả hai đều không phải "vẽ đường cong giả".
    """
    p = Path(path) if path else loss_log_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise DataMissing(f"[!] {p} không đọc được: {exc}") from exc
    if not isinstance(data, dict) or not data.get("steps"):
        return None
    return data


def loss_points(log: dict) -> list[tuple[int, float]]:
    """``[(step, loss)]`` — dữ liệu thô để vẽ polyline."""
    pts: list[tuple[int, float]] = []
    for s in log.get("steps") or []:
        pts.append((int(s["step"]), float(s["loss"])))
    if not pts:
        raise DataMissing("[!] Log không có step nào — không có gì để vẽ.")
    return pts


def val_points(log: dict) -> list[tuple[int, float]]:
    """``[(epoch, val_acc)]``, BỎ epoch chưa có val (ô trống ≠ 0)."""
    return [(int(e["epoch"]), float(e["val_acc"]))
            for e in (log.get("epochs") or []) if e.get("val_acc") is not None]


def demo_label(log: dict | None = None) -> str:
    """Nhãn bắt buộc trên khung hình của scene training.

    Lấy từ chính log để không thể quên: nếu log do `demo_train` sinh ra thì nhãn
    đi kèm dữ liệu, không phải một câu viết thêm trong scene.
    """
    if log and log.get("label"):
        return str(log["label"])
    return "demo — chưa có log (chạy demo_train --mode run)"
