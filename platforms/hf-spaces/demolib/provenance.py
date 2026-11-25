"""Đọc gallery và quyết định **khi nào** được phép hiện một điểm số.

Vì sao tách khỏi `app.py`
------------------------
Quy tắc của demo này gói trong đúng một câu: *một con số chỉ được hiện khi biết
nó đến từ đâu*. Câu đó phải kiểm được bằng test, nên nó nằm ở đây chứ không nằm
rải rác trong các `if` của giao diện.

Mỗi mẫu là bốn tệp ảnh cộng một `.json`:

* ``<tên>_orig.png`` — ảnh gốc 256×256 (**bắt buộc**);
* ``<tên>_diff.png`` — ``|x − x̂₀|`` vẽ theo thang cố định (**bắt buộc**);
* ``<tên>_recon.png`` — ảnh tái tạo qua ADM (**tuỳ chọn**: ``::dire`` không lưu
  ảnh trung gian này, nên phần lớn mẫu sẽ không có; thiếu thì app không hiện panel đó);
* ``<tên>.json`` — nguồn gốc, theo lược đồ dưới.

Lược đồ `.json` (khoá bắt buộc, không có khoá "tuỳ chọn" cho phần nguồn gốc)::

    {
      "name": "lsun_adm_test_0000",
      "generator": "adm",                  # generator sinh ảnh
      "split": "test",                     # train / val / test
      "label": "R0",                       # R0 | R1+R2 | R1+R2+R3
      "map_source": "maps/test/adm/0000.npy · ::dire (mình tính)",
      "sha256": {"orig": "...", "map": "..."},
      "S": 20,
      "detector": {                        # TUỲ CHỌN — thiếu thì không hiện điểm
        "name": "lsun_adm.pth (tác giả)",
        "gate": "passed",                  # passed | failed | missing
        "threshold": 0.5,
        "threshold_split": "val",          # phải là val, không được test
        "p_fake": 0.97,
        "source": "R0"
      }
    }

Ba điều kiện để `score()` trả về một con số, và cả ba đều là điều kiện *từ chối*:

1. ``detector.gate == "passed"`` — cổng 2 của `docs/plans/demos/demo-web-plan.md` §5 đã chạy;
2. ``detector.threshold_split == "val"`` — ngưỡng chọn trên val, không bao giờ trên test;
3. ``p_fake`` là số thực trong ``[0, 1]``.

Thiếu một điều kiện thì giao diện hiện **lý do**, không hiện số. Đây là điểm khác
biệt giữa "demo chưa có dữ liệu" và "demo bịa dữ liệu": cái thứ hai luôn kèm một
con số trông hợp lý.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "REQUIRED",
    "LABELS",
    "Sample",
    "GalleryError",
    "load_sample",
    "discover",
    "evidence_line",
    "SCHEMA_HELP",
    "GENERATE_HELP",
]

REQUIRED = ("name", "generator", "split", "label", "map_source", "sha256", "S")

#: Nhãn mức tái lập, dùng đúng chữ của `docs/dataset-own.md`. Không nhận nhãn
#: khác: một nhãn tự nghĩ ra sẽ khiến người xem tưởng có kết quả tái lập.
LABELS = ("R0", "R1+R2", "R1+R2+R3")

MIN_S, MAX_S = 1, 1000


class GalleryError(ValueError):
    """.json sai lược đồ — luôn nêu tên tệp và khoá cụ thể."""


@dataclass(frozen=True)
class Sample:
    name: str
    generator: str
    split: str
    label: str
    map_source: str
    sha256: dict[str, str]
    steps: int
    orig: Path
    diff: Path
    recon: Path | None
    meta: dict

    @property
    def selector(self) -> str:
        return f"{self.name} · {self.generator}/{self.split}"

    def score(self) -> tuple[dict | None, str]:
        """Trả `(điểm, lý do)`. Có số ⇒ lý do rỗng; không có ⇒ lý do để hiện."""
        det = self.meta.get("detector")
        if not isinstance(det, dict):
            return None, "chưa có điểm: mẫu này không kèm khối `detector`"
        gate = det.get("gate")
        if gate != "passed":
            return None, (
                "chưa có điểm: detector chưa qua **cổng 2** "
                f"(`gate = {gate!r}`) — xem `docs/plans/demos/demo-web-plan.md` §5"
            )
        if det.get("threshold_split") != "val":
            return None, (
                "chưa có điểm: ngưỡng không chọn trên `val` "
                f"(`threshold_split = {det.get('threshold_split')!r}`)"
            )
        p = det.get("p_fake")
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
            return None, f"chưa có điểm: `p_fake` không hợp lệ ({p!r})"
        return det, ""


def _hex64(v: object) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v.lower())


def load_sample(meta_path: Path) -> Sample:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise GalleryError(f"{meta_path.name}: JSON hỏng — {e}") from None
    except OSError as e:
        raise GalleryError(f"{meta_path.name}: không đọc được — {e}") from None
    if not isinstance(meta, dict):
        raise GalleryError(f"{meta_path.name}: phải là object JSON")

    missing = [k for k in REQUIRED if k not in meta]
    if missing:
        raise GalleryError(f"{meta_path.name}: thiếu khoá {missing}")

    label = meta["label"]
    if label not in LABELS:
        raise GalleryError(f"{meta_path.name}: nhãn {label!r} không thuộc {LABELS}")

    sha = meta["sha256"]
    if not isinstance(sha, dict) or not sha:
        raise GalleryError(f"{meta_path.name}: `sha256` phải là object {{tệp: hash}}")
    bad = [k for k, v in sha.items() if not _hex64(v)]
    if bad:
        raise GalleryError(f"{meta_path.name}: sha256 sai định dạng cho {bad}")

    try:
        steps = int(meta["S"])
    except (TypeError, ValueError):
        raise GalleryError(f"{meta_path.name}: `S` phải là số nguyên, nhận {meta['S']!r}") from None
    if not MIN_S <= steps <= MAX_S:
        raise GalleryError(f"{meta_path.name}: `S` ngoài [{MIN_S}, {MAX_S}]")

    stem = meta_path.with_suffix("")
    files = {}
    for key, suffix in (("orig", "_orig.png"), ("diff", "_diff.png")):
        p = Path(str(stem) + suffix)
        if not p.is_file():
            raise GalleryError(f"{meta_path.name}: thiếu tệp ảnh `{p.name}`")
        files[key] = p
    recon_path = Path(str(stem) + "_recon.png")
    files["recon"] = recon_path if recon_path.is_file() else None

    for key in ("name", "generator", "split", "map_source"):
        if not isinstance(meta[key], str) or not meta[key].strip():
            raise GalleryError(f"{meta_path.name}: `{key}` phải là chuỗi không rỗng")

    return Sample(
        name=meta["name"],
        generator=meta["generator"],
        split=meta["split"],
        label=label,
        map_source=meta["map_source"],
        sha256={k: v.lower() for k, v in sha.items()},
        steps=steps,
        orig=files["orig"],
        diff=files["diff"],
        recon=files["recon"],
        meta=meta,
    )


def discover(root: Path | str) -> tuple[list[Sample], list[str]]:
    """Trả `(mẫu hợp lệ, lỗi)`. Một `.json` hỏng **không** làm sập cả gallery."""
    root = Path(root)
    samples: list[Sample] = []
    problems: list[str] = []
    if not root.is_dir():
        return samples, [f"thư mục gallery không tồn tại: {root}"]
    for meta_path in sorted(root.glob("*.json")):
        try:
            samples.append(load_sample(meta_path))
        except GalleryError as e:
            problems.append(str(e))
    samples.sort(key=lambda s: (s.generator, s.split, s.name))
    return samples, problems


def evidence_line(s: Sample) -> str:
    """Dòng bắt buộc dưới mỗi kết quả: DIRE ở đâu, detector nào, nhãn gì, S bao nhiêu."""
    det = s.meta.get("detector")
    det_name = det.get("name") if isinstance(det, dict) and det.get("name") else "chưa có"
    return f"DIRE: {s.map_source} · detector: {det_name} · nhãn: {s.label} · S={s.steps}"


SCHEMA_HELP = """Khoá bắt buộc trong `<tên>.json`:

| Khoá | Ví dụ | Vì sao phải có |
|---|---|---|
| `name` | `lsun_adm_test_0000` | tên mẫu, trùng tiền tố ba tệp PNG |
| `generator` | `adm` | Table 2 tách theo generator, không gộp |
| `split` | `test` | ngưỡng chọn trên `val`; trộn split là tự lừa |
| `label` | `R0` | mức tái lập: `R0` / `R1+R2` / `R1+R2+R3` |
| `map_source` | `maps/test/adm/0000.npy · ::dire` | ai tính DIRE map |
| `sha256` | `{"orig": "…", "map": "…"}` | chứng minh ảnh hiện ra đúng là ảnh đã chấm |
| `sha256.map` | hash của tệp `.npy` gốc | ảnh PNG chỉ để xem; `.npy` mới là dữ liệu |
| `S` | `20` | số bước DDIM; đổi S là đổi kết quả (Table 4) |

Khối `detector` là **tuỳ chọn**, và chỉ khi có nó *và* `gate = "passed"` thì
giao diện mới hiện `p_fake`.
"""

GENERATE_HELP = """```bash
# 1. maps (đã có sau `::dire`): maps/test/{real,adm}/*.npy, float16, thang [0, 2]
# 2. dựng gallery — đường dẫn truyền tường minh, không đoán cây thư mục:
python tools/make_gallery.py \\
    --maps  <volume>/maps/test \\
    --orig  <volume>/images/test \\
    --out   platforms/hf-spaces/gallery \\
    --n 12 --steps 20
# 3. commit thư mục gallery/ rồi push Space
```

`make_gallery.py` từ chối chạy nếu thiếu ảnh gốc cho một map, nếu `.npy` không
phải `[256, 256]`, hoặc nếu `--steps` không khớp `S` ghi trong `maps/dire.json`.
"""
