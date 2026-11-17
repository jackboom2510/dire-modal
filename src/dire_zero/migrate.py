"""Chuyển dữ liệu từ Volume cũ sang bố cục `dire-zero` — **chỉ đọc cũ, chỉ ghi mới**.

Tình huống thật
---------------
Checkpoint ADM (2,5 GB) đã nằm trên Modal từ trước, ở một trong sáu đường dẫn
khác nhau mà repo này từng dùng (``volume.py`` liệt kê đủ). Tải lại 2,5 GB từ
Baidu ở 76 KiB/s là ~9 giờ — nên việc cần làm **không phải** là tải lại, mà là
tìm đúng file đã có và bê nó sang chỗ mới.

Bốn luật của module này
-----------------------
1. **Không xoá.** Không có hàm nào trong file này gọi ``rm``/``unlink``. Volume cũ
   giữ nguyên byte trên byte; dọn dẹp là việc làm tay, sau khi ``verify`` xanh.
2. **Không ghi đè nội dung khác.** Đích đã có file cùng tên nhưng sha khác ⇒
   dừng, báo ``conflict``, không ghi. Ghi đè im lặng là cách mất một checkpoint
   tốt mà không ai biết.
2b. **"Không có nguồn" không phải "xung đột".** Hai thứ cần hai cách xử lý: xung
   đột là có dữ liệu **khác** ở đích và cần người quyết định; thiếu nguồn chỉ là
   không có gì để chép. Gộp chúng lại (bản đầu đã gộp) làm `apply` ném
   "1 xung đột" trong khi sự thật là *đích đã có ADM, không có bản cũ để so* —
   người dùng đi tìm một cuộc đụng độ không tồn tại.
3. **Mặc định là ``plan``.** Muốn ghi thì phải gọi ``apply``; muốn ghi thật thì
   ``apply`` phải có ``confirm=True``. Cùng lý do như ``demo_train``: gõ thiếu
   một cờ không được phép tiêu tiền hay đổi dữ liệu.
4. **Không đoán theo tên.** Tên file không đủ để biết nó **là gì**: trên volume
   thật, ``checkpoints/lsun_adm.pth`` là **detector của tác giả** (283 MB,
   23,5 M tham số), không phải trọng số ADM (~2,2 GB, 552,8 M tham số). Cùng
   tên, khác hẳn thứ. Vì vậy ADM phải qua **sàn kích thước**, còn detector tác
   giả đi đường riêng sang ``ckpt/author/`` — không bao giờ vào
   ``ckpt/adm_lsun.pt``.
5. **Có biên lai.** Mỗi lần ``apply`` ghi ``_migration/receipt.json``: nguồn,
   đích, size, sha256 hai đầu, thời điểm, phiên bản bố cục. Không có biên lai thì
   lần chạy sau không biết cái gì đã chuyển.

Copy trong container, không qua máy bạn
---------------------------------------
Hàm ``apply`` chạy trên Modal với cả ba volume được mount. ``shutil.copyfileobj``
đọc/ghi trong cùng máy ⇒ **không có egress**, không tốn dung lượng mạng, không
phụ thuộc ổ F: của bạn. Đó là lý do migration là một app Modal chứ không phải
script local.

Lệnh::

    modal run -m dire_zero.migrate::survey      # cũ có gì, đang ở đâu (0 ghi)
    modal run -m dire_zero.migrate::plan        # sẽ chuyển gì, bao nhiêu GB (0 ghi)
    modal run -m dire_zero.migrate::apply --confirm          # ADM + detector tác giả
    modal run -m dire_zero.migrate::apply --confirm
    modal run -m dire_zero.migrate::verify      # đối chiếu biên lai với Volume
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import modal

from . import volume as V

APP_NAME = "dire-zero-migrate"

#: Image tối giản: chỉ cần stdlib. Không torch, không numpy — migration không
#: được phép kéo 2 GB image chỉ để copy file.
image = modal.Image.debian_slim(python_version="3.11").add_local_python_source("dire_zero")

app = modal.App(APP_NAME, image=image)

vol_new = modal.Volume.from_name(V.VOLUME_NAME, create_if_missing=True)
vol_old_raw = modal.Volume.from_name("dire-raw", create_if_missing=True)
vol_old_ckpt = modal.Volume.from_name("dire-ckpt", create_if_missing=True)

#: Volume mới ghi được; hai volume cũ chỉ **đọc** (code không ghi vào chúng).
VOLUMES = {
    V.MOUNT: vol_new,
    "/raw": vol_old_raw,
    "/ckpt": vol_old_ckpt,
}

#: Tên file checkpoint mà các đường cũ từng dùng. Khớp theo **tên**, không theo
#: đường dẫn — vì chính đường dẫn là thứ đang sai.
CKPT_NAMES = ("adm_lsun.pt", "lsun_adm.pth", "256x256_diffusion_uncond.pt")

#: Trọng số ADM 256² vô điều kiện: 552,8 M tham số × 4 B ≈ **2,2 GB**. Sàn này
#: không phải "cho chắc": trên volume thật có sẵn file *tên là* `lsun_adm.pth`
#: nhưng chỉ 282.534.013 B — đó là detector ResNet-50 của tác giả. Không có sàn,
#: migration sẽ bê nhầm nó vào `ckpt/adm_lsun.pt` và `selfcheck` nổ sau đó, khi
#: biên lai đã khai thành công.
ADM_MIN_BYTES = 1_500_000_000

#: Detector tác giả: 23,5 M tham số + trạng thái optimizer ⇒ ~283 MB. Khoảng này
#: đủ rộng cho cả `.pth` chỉ có trọng số, và đủ hẹp để không nuốt phải ADM.
AUTHOR_DETECTOR_MIN = 100 * 1024**2
AUTHOR_DETECTOR_MAX = 400 * 1024**2

#: Đuôi file không bao giờ chuyển: rác của quá trình tải/giải nén.
SKIP_SUFFIXES = (".downloading", ".part", ".tmp", ".partial")

#: Thư mục không chuyển: staging tạm và file bị giữ lại để kiểm tay.
SKIP_DIRS = (
    ".staging", "_held", "__pycache__", ".ipynb_checkpoints",
    ".bench",      # bản sao checkpoint để đo băng thông — không phải dữ liệu
    ".state",      # sổ trạng thái của công cụ tải
)

CHUNK = 4 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Phần thuần: không đụng đĩa, test được bằng tmp_path
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    """Một file ứng viên trên Volume cũ."""

    path: str                # tuyệt đối trong container
    size: int
    sha256: str = ""         # rỗng = chưa băm (survey không băm 2,5 GB)


@dataclass
class Action:
    """Một việc sẽ làm (hoặc đã làm) với một file."""

    kind: str                # copy | skip | conflict | missing
    source: str
    target: str
    size: int = 0
    sha256_source: str = ""
    sha256_target: str = ""
    reason: str = ""

    @property
    def moves_bytes(self) -> bool:
        return self.kind == "copy"


@dataclass
class Plan:
    actions: list[Action] = field(default_factory=list)

    @property
    def to_copy(self) -> list[Action]:
        return [a for a in self.actions if a.kind == "copy"]

    @property
    def conflicts(self) -> list[Action]:
        return [a for a in self.actions if a.kind == "conflict"]

    @property
    def missing(self) -> list[Action]:
        return [a for a in self.actions if a.kind == "missing"]

    @property
    def bytes_to_copy(self) -> int:
        return sum(a.size for a in self.to_copy)

    def summary(self) -> dict:
        return {
            "n_copy": len(self.to_copy),
            "n_skip": sum(1 for a in self.actions if a.kind == "skip"),
            "n_conflict": len(self.conflicts),
            "n_missing": len(self.missing),
            "bytes_to_copy": self.bytes_to_copy,
            "gb_to_copy": round(self.bytes_to_copy / 1e9, 2),
        }


def sha256_file(path: str | Path, chunk: int = CHUNK) -> str:
    """Băm theo luồng — không nạp 2,5 GB vào RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def is_skipped(path: str | Path) -> str:
    """Trả lý do bỏ qua (rỗng ⇒ không bỏ). Tách khỏi ``discover`` để test riêng."""
    p = Path(path)
    if p.suffix.lower() in SKIP_SUFFIXES:
        return f"đuôi tạm {p.suffix}"
    if any(part in SKIP_DIRS for part in p.parts):
        return "thư mục tạm/giữ lại"
    return ""


def classify_ckpt_size(size: int) -> str:
    """``adm`` | ``author-dector`` | ``too-small`` — thuần số, test không cần file lớn.

    Đây là chỗ duy nhất quyết định "283 MB là cái gì". Tách khỏi `discover` để
    kiểm được bằng vài con số thay vì tạo file 2 GB trong test.
    """
    if size >= ADM_MIN_BYTES:
        return "adm"
    if AUTHOR_DETECTOR_MIN <= size <= AUTHOR_DETECTOR_MAX:
        return "author-detector"
    return "too-small"


def discover_ckpts(roots: dict[str, str] | None = None, min_bytes: int = 0) -> list[Candidate]:
    """Tìm mọi checkpoint ứng viên trong các volume cũ (chỉ ``stat``, không băm).

    So khớp **theo tên file**, không theo đường dẫn: đây chính là điểm mà bố cục
    cũ sai, nên dùng lại đường dẫn cũ để tìm là lặp lại lỗi cũ.
    """
    roots = roots or V.OLD_VOLUMES
    out: list[Candidate] = []
    for _label, root in roots.items():
        base = Path(root)
        if not base.is_dir():
            continue
        for name in CKPT_NAMES:
            for p in sorted(base.rglob(name)):
                if not p.is_file() or is_skipped(p):
                    continue
                size = p.stat().st_size
                if size < min_bytes:
                    continue
                out.append(Candidate(path=str(p), size=size))
    out.sort(key=lambda c: (-c.size, c.path))
    return out


def discover_adm(roots: dict[str, str] | None = None) -> tuple[list[Candidate], list[dict]]:
    """ADM thật (qua **sàn** kích thước) + danh sách file cùng tên bị loại vì quá nhỏ."""
    roots = roots or V.OLD_VOLUMES
    found = discover_ckpts(roots)
    adm = [c for c in found if classify_ckpt_size(c.size) == "adm"]
    rejected = [
        {"path": c.path, "size": c.size,
         "reason": "quá nhỏ so với ADM 2,2 GB — nhiều khả năng là detector tác giả"}
        for c in found if classify_ckpt_size(c.size) != "adm"
    ]
    return adm, rejected


def discover_author_detectors(roots: dict[str, str] | None = None) -> list[Candidate]:
    """Detector của tác giả: ``<volume cũ>/checkpoints/*.pth`` trong khoảng 100–400 MB.

    Không khớp theo danh sách tên cố định: cây thật có sáu generator
    (``lsun_adm``, ``lsun_iddpm``, ``lsun_pndm``, ``lsun_stylegan``,
    ``imagenet_adm``, ``celebahq_sdv2``) và danh sách đó có thể còn dài thêm.
    Khoảng kích thước + vị trí ``checkpoints/`` là hai dấu hiệu đủ chắc.
    """
    roots = roots or V.OLD_VOLUMES
    out: list[Candidate] = []
    for _label, root in roots.items():
        base = Path(root) / "checkpoints"
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.pth")):
            if not p.is_file() or is_skipped(p):
                continue
            size = p.stat().st_size
            if AUTHOR_DETECTOR_MIN <= size <= AUTHOR_DETECTOR_MAX:
                out.append(Candidate(path=str(p), size=size))
    out.sort(key=lambda c: c.path)
    return out


def plan_author(cands: list[Candidate], target_dir: str | Path) -> Plan:
    """Mỗi detector tác giả → ``ckpt/author/<tên gốc>``. Giữ tên: tên = generator."""
    target_dir = Path(target_dir)
    plan = Plan()
    for c in cands:
        dst = target_dir / Path(c.path).name
        if not dst.exists():
            plan.actions.append(Action(kind="copy", source=c.path, target=str(dst),
                                       size=c.size, reason="chưa có ở đích"))
        elif dst.stat().st_size == c.size:
            plan.actions.append(Action(kind="skip", source=c.path, target=str(dst),
                                       size=c.size, reason="đã có, cùng kích thước"))
        else:
            plan.actions.append(Action(kind="conflict", source=c.path, target=str(dst),
                                       size=c.size,
                                       reason="đích có nhưng khác kích thước — không ghi đè"))
    return plan


def plan_copy(candidates: list[Candidate], target: str | Path, target_exists: bool) -> Plan:
    """Quyết định copy/skip/conflict cho **một** đích duy nhất."""

    target = str(target)
    if not candidates:
        # Đích đã có ⇒ không có gì phải làm, và cũng không có gì để so: bản cũ
        # không tồn tại nên "kiểm sha hai đầu" là bất khả. `verify` (biên lai) và
        # `selfcheck` (nạp thật) mới là hai cổng kiểm cho trường hợp này.
        if target_exists:
            return Plan([Action(
                kind="skip", source="", target=target,
                reason="đích đã có; không có bản nào ở volume cũ để so sha — "
                       "kiểm bằng `::verify` hoặc `::selfcheck`",
            )])
        return Plan([Action(
            kind="missing", source="", target=target,
            reason="không tìm thấy checkpoint ADM nào đủ lớn (≥ 1,5 GB) ở volume cũ; "
                   "đích cũng chưa có — phải đẩy 256x256_diffusion_uncond.pt bằng "
                   "`modal volume put`",
        )])

    # Ứng viên lớn nhất trước (đã sắp trong discover). Nhiều bản cùng tên ở nhiều
    # đường là chuyện thật; chọn bản lớn nhất là chọn bản **đầy đủ nhất**, và
    # mọi bản khác được ghi vào biên lai để người dùng kiểm.
    best = candidates[0]
    if target_exists:
        return Plan([
            Action(kind="conflict", source=best.path, target=target, size=best.size,
                   reason="đích đã tồn tại — KHÔNG ghi đè; chạy verify để so sha"),
            *[
                Action(kind="skip", source=c.path, target=target, size=c.size,
                       reason="bản sao khác của cùng checkpoint")
                for c in candidates[1:]
            ],
        ])
    return Plan([
        Action(kind="copy", source=best.path, target=target, size=best.size,
               reason="chưa có ở đích"),
        *[
            Action(kind="skip", source=c.path, target=target, size=c.size,
                   reason="bản sao khác của cùng checkpoint")
            for c in candidates[1:]
        ],
    ])


def plan_data(old_root: str | Path, new_root: str | Path) -> Plan:
    """Kế hoạch cho cây dữ liệu cũ (``zero/**`` hoặc ``data/**``) → bố cục mới.

    Chỉ chuyển những thư mục mà bố cục mới có chỗ: ``data/``, ``maps/``, ``gen/``,
    ``runs/`` (và ``ckpt/`` xử lý riêng ở trên). Mọi thứ khác được **báo**, không
    im lặng bỏ: người dùng cần biết cái gì sẽ không sang nhà mới.
    """
    old_root, new_root = Path(old_root), Path(new_root)
    plan = Plan()
    if not old_root.is_dir():
        return plan

    for src in sorted(old_root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(old_root)
        skip = is_skipped(src)
        if skip:
            plan.actions.append(Action(kind="skip", source=str(src), target="", size=src.stat().st_size,
                                       reason=skip))
            continue
        if rel.parts and rel.parts[0] == "ckpt":
            continue                     # checkpoint đi đường riêng, có kiểm sha
        dst = new_root / rel
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            plan.actions.append(Action(kind="skip", source=str(src), target=str(dst),
                                       size=src.stat().st_size, reason="đã có, cùng kích thước"))
            continue
        if dst.is_file():
            plan.actions.append(Action(kind="conflict", source=str(src), target=str(dst),
                                       size=src.stat().st_size,
                                       reason="đích có nhưng khác kích thước — không ghi đè"))
            continue
        plan.actions.append(Action(kind="copy", source=str(src), target=str(dst),
                                   size=src.stat().st_size, reason="chưa có ở đích"))
    return plan


# --------------------------------------------------------------------------- #
# Phần chạm đĩa: copy có kiểm, chỉ ghi vào volume mới
# --------------------------------------------------------------------------- #

def copy_verified(src: str | Path, dst: str | Path, expect_sha: str = "") -> dict:
    """Copy theo luồng, ghi file tạm rồi ``os.replace``; trả sha hai đầu.

    Ghi tạm rồi đổi tên để **không bao giờ** để lại một file đích cụt trông như
    file đầy: lần chạy sau sẽ thấy nó, thấy kích thước khác, và báo conflict.
    Đó là cách một lần Ctrl-C biến thành một lỗi khó hiểu ở lần chạy sau.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".migrating")

    h = hashlib.sha256()
    t0 = time.time()
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        for block in iter(lambda: fi.read(CHUNK), b""):
            h.update(block)
            fo.write(block)
    got = h.hexdigest()
    if expect_sha and got != expect_sha:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"sha nguồn lệch: {got[:12]} ≠ {expect_sha[:12]} ({src})")
    os.replace(tmp, dst)
    return {
        "source": str(src),
        "target": str(dst),
        "bytes": dst.stat().st_size,
        "sha256_source": got,
        "sha256_target": sha256_file(dst),
        "seconds": round(time.time() - t0, 2),
    }


def receipt_path(root: str | Path = V.MOUNT) -> Path:
    return Path(root) / "_migration" / "receipt.json"


def write_receipt(root: str | Path, payload: dict) -> Path:
    p = receipt_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def read_receipt(root: str | Path = V.MOUNT) -> dict:
    p = receipt_path(root)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def verify_receipt(root: str | Path = V.MOUNT) -> dict:
    """Đối chiếu biên lai với đĩa: file còn đó không, sha có khớp không."""
    rec = read_receipt(root)
    results = []
    for item in rec.get("copied", []):
        target = Path(item["target"])
        if not target.is_file():
            results.append({"target": str(target), "ok": False, "reason": "mất file"})
            continue
        got = sha256_file(target)
        ok = got == item.get("sha256_target", got)
        row = {"target": str(target), "ok": ok, "sha256": got[:16]}
        if not ok:
            row["reason"] = "sha lệch"     # phân biệt với "mất file": hai cách xử lý khác nhau
        results.append(row)
    bad = [r for r in results if not r["ok"]]
    return {
        "layout_version": rec.get("layout_version"),
        "created": rec.get("created"),
        "n_checked": len(results),
        "n_bad": len(bad),
        "bad": bad[:10],
        "ok": bool(results) and not bad,
    }


# --------------------------------------------------------------------------- #
# Entrypoint Modal
# --------------------------------------------------------------------------- #

def _tree_summary(root: Path, max_entries: int = 24) -> dict:
    """Thống kê cây: thư mục cấp 1–2, số file, tổng byte. Chỉ đọc."""
    if not root.is_dir():
        return {"exists": False}
    dirs: dict[str, dict] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        key = "/".join(rel.parts[:2]) if len(rel.parts) > 1 else "(gốc)"
        d = dirs.setdefault(key, {"files": 0, "bytes": 0})
        d["files"] += 1
        try:
            d["bytes"] += p.stat().st_size
        except OSError:
            pass
    items = sorted(dirs.items(), key=lambda kv: -kv[1]["bytes"])[:max_entries]
    return {
        "exists": True,
        "n_files": sum(d["files"] for d in dirs.values()),
        "gb": round(sum(d["bytes"] for d in dirs.values()) / 1e9, 2),
        "top": [{"path": k, **v, "gb": round(v["bytes"] / 1e9, 3)} for k, v in items],
    }


@app.function(volumes=VOLUMES, cpu=2, timeout=1800)
def survey() -> dict:
    """Volume cũ đang có gì, checkpoint nằm ở đâu. **Không ghi gì.**"""
    old = {label: _tree_summary(Path(root)) for label, root in V.OLD_VOLUMES.items()}
    cands = discover_ckpts()
    adm, rejected = discover_adm()
    authors = discover_author_detectors()
    new = _tree_summary(Path(V.MOUNT))
    out = {
        "new_volume": {"name": V.VOLUME_NAME, "mount": V.MOUNT, "layout_version": V.LAYOUT_VERSION},
        "old": old,
        "ckpt_candidates": [asdict(c) for c in cands],
        "adm_candidates": [asdict(c) for c in adm],
        "adm_rejected_by_size": rejected,
        "author_detectors": [asdict(c) for c in authors],
        "new": new,
        "layout": V.LAYOUT,
        "warnings": [],
    }
    if not cands:
        out["warnings"].append(
            "không thấy checkpoint nào trong volume cũ. Nếu bạn biết nó ở đường khác, "
            "thêm tên file vào CKPT_NAMES rồi chạy lại survey."
        )
    if rejected:
        out["warnings"].append(
            f"{len(rejected)} file *tên giống* checkpoint nhưng chỉ ≤ "
            f"{AUTHOR_DETECTOR_MAX // 1024**2} MB ⇒ không phải ADM (2,2 GB). "
            "Rất có thể là detector tác giả; chúng đi sang ckpt/author/, không vào ckpt/adm_lsun.pt."
        )
    if not adm:
        out["warnings"].append(
            "KHÔNG có ứng viên ADM nào đủ lớn (≥ 1,5 GB). Migration không thể tạo ra "
            "trọng số từ chỗ không có: cần đẩy 256x256_diffusion_uncond.pt (~2,2 GB) "
            "lên bằng `modal volume put`."
        )
    if len(cands) > 1:
        out["warnings"].append(
            f"{len(cands)} bản checkpoint cùng loại — plan sẽ chọn bản lớn nhất và ghi "
            "mọi bản khác vào biên lai; kiểm tay nếu dung lượng khác nhau đáng kể."
        )
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


@app.function(volumes=VOLUMES, cpu=2, timeout=1800)
def plan() -> dict:
    """Kế hoạch chuyển: copy gì, bỏ gì, xung đột gì. **Không ghi gì.**"""
    adm, rejected = discover_adm()
    target = V.CKPT
    p = plan_copy(adm, target, Path(target).exists())

    authors = discover_author_detectors()
    author_plan = plan_author(authors, V.CKPT_AUTHOR)

    # Dữ liệu cũ có thể nằm ở /raw/zero/{data,maps,gen} (đường clean-room) hoặc
    # ngay /raw/{data,maps}. Chọn **cái có thật**, không đoán.
    data_root = next(
        (r for r in (Path("/raw/zero"), Path("/raw"))
         if (r / "data").is_dir() or (r / "maps").is_dir()),
        None,
    )
    data_plan = plan_data(data_root, Path(V.MOUNT)) if data_root else Plan()

    out = {
        "ckpt": {
            "target": target,
            "candidates": [asdict(c) for c in adm],
            "rejected_by_size": rejected,
            "actions": [asdict(a) for a in p.actions],
            **p.summary(),
        },
        "author": {
            "target_dir": V.CKPT_AUTHOR,
            "n_candidates": len(authors),
            "actions": [asdict(a) for a in author_plan.actions],
            **author_plan.summary(),
        },
        "data": {
            "source_root": str(data_root) if data_root else "",
            "n_actions": len(data_plan.actions),
            "actions_preview": [asdict(a) for a in data_plan.actions[:50]],
            **data_plan.summary(),
        },
        "layout_version": V.LAYOUT_VERSION,
        "will_write": 0,
        "note": "plan không ghi gì. Chạy `apply --confirm` để thực hiện.",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


@app.function(volumes=VOLUMES, cpu=2, timeout=12 * 3600)
def apply(confirm: bool = False, data: bool = False, author: bool = True) -> dict:
    """Thực hiện chuyển. **Chỉ ghi vào volume mới**, không xoá gì ở volume cũ.

    ``data=False`` là mặc định có chủ ý: người ta gọi migration để **bê checkpoint
    2,5 GB**, không phải để tự động chép cả cây dữ liệu. Muốn bê dữ liệu thì nói
    rõ ``--data`` — và ``plan`` đã in sẵn ``n_copy`` của phần đó để quyết định.
    """
    if not confirm:
        raise RuntimeError(
            "từ chối ghi: cần `--confirm`. Chạy `plan` trước để xem sẽ chuyển gì."
        )

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    copied: list[dict] = []
    skipped: list[dict] = []
    conflicts: list[dict] = []
    missing: list[dict] = []

    adm, rejected = discover_adm()
    p = plan_copy(adm, V.CKPT, Path(V.CKPT).exists())
    for a in p.actions:
        if a.kind == "copy":
            copied.append(copy_verified(a.source, a.target, a.sha256_source))
        elif a.kind == "conflict":
            conflicts.append(asdict(a))
        elif a.kind == "missing":
            missing.append(asdict(a))
        else:
            skipped.append(asdict(a))

    # Trạng thái thật của đích ADM — thứ mà `missing` không nói được một mình.
    adm_size = Path(V.CKPT).stat().st_size if Path(V.CKPT).exists() else 0
    adm_state = {
        "target": V.CKPT,
        "exists": bool(adm_size),
        "size": adm_size,
        "classify": classify_ckpt_size(adm_size) if adm_size else "absent",
    }

    # Detector tác giả + số `repro_*` của họ: đây là đường R0 (chấm lại trên map
    # có sẵn, 0 đồng GPU). Mặc định bật vì thuần bổ sung, không ghi đè gì.
    author_copied: list[dict] = []
    if author:
        ap = plan_author(discover_author_detectors(), V.CKPT_AUTHOR)
        for a in ap.actions:
            if a.kind == "copy":
                author_copied.append(copy_verified(a.source, a.target))
            elif a.kind == "conflict":
                conflicts.append(asdict(a))
            else:
                skipped.append(asdict(a))
        results_root = Path("/ckpt/results")
        if results_root.is_dir():
            for src in sorted(results_root.rglob("*")):
                if src.is_file() and not is_skipped(src):
                    dst = Path(V.RUNS_AUTHOR) / src.relative_to(results_root)
                    if not dst.exists():
                        author_copied.append(copy_verified(src, dst))

    if data:
        for old_zero in (Path("/raw/zero"), Path("/raw")):
            if (old_zero / "data").is_dir() or (old_zero / "maps").is_dir():
                dp = plan_data(old_zero, Path(V.MOUNT))
                for a in dp.actions:
                    if a.kind == "copy":
                        copied.append(copy_verified(a.source, a.target))
                    elif a.kind == "conflict":
                        conflicts.append(asdict(a))
                break

    payload = {
        "layout_version": V.LAYOUT_VERSION,
        "created": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_volumes": V.OLD_VOLUMES,
        "target_volume": V.VOLUME_NAME,
        "copied": copied,
        "copied_author": author_copied,
        "adm_rejected_by_size": rejected,
        "skipped": skipped[:500],
        "n_skipped": len(skipped),
        "conflicts": conflicts[:500],
        "n_conflicts": len(conflicts),
        "missing": missing,
        "adm_target": adm_state,
        "bytes_copied": sum(c["bytes"] for c in copied + author_copied),
        "deleted_anything": False,
    }
    write_receipt(V.MOUNT, payload)
    vol_new.commit()

    print(json.dumps({k: v for k, v in payload.items() if k != "copied"}, ensure_ascii=False, indent=2))

    if conflicts:
        raise RuntimeError(
            f"{len(conflicts)} xung đột — không ghi đè. Xem receipt.json, kiểm tay rồi quyết định."
        )
    # Ba trạng thái của ADM, ba thông báo khác nhau. Trước đây cả ba đều ra
    # "1 xung đột", và người dùng đi tìm một cuộc đụng độ không tồn tại.
    if adm_state["classify"] == "adm":
        print(f"[adm] {V.CKPT} đã có ({adm_state['size'] / 1e9:.2f} GB) — không có gì để chuyển.", flush=True)
    elif adm_state["exists"]:
        raise RuntimeError(
            f"{V.CKPT} tồn tại nhưng chỉ {adm_state['size'] / 1e9:.2f} GB ⇒ **không phải** "
            "trọng số ADM (cần ~2,2 GB). Nhiều khả năng ai đó đã đẩy nhầm detector vào đó.\n"
            "Migration không xoá và không ghi đè: hãy tự đổi tên/kiểm tay rồi chạy lại, "
            "hoặc trỏ sang chỗ khác. Kiểm nhanh bằng `modal run -m dire_zero.modal_app::selfcheck`."
        )
    else:
        raise RuntimeError(
            "chưa có trọng số ADM ở đích và cũng không tìm thấy ở volume cũ.\n"
            "Migration chỉ chuyển được thứ đã tồn tại — hãy đẩy lên:\n"
            "  modal volume put dire-zero <256x256_diffusion_uncond.pt> ckpt/adm_lsun.pt"
        )
    return payload


@app.function(volumes=VOLUMES, cpu=2, timeout=1800)
def verify() -> dict:
    """Kiểm biên lai với đĩa thật — chạy sau `apply`, và chạy lại được bất cứ lúc nào."""
    size = Path(V.CKPT).stat().st_size if Path(V.CKPT).exists() else 0
    out = {
        "receipt": str(receipt_path(V.MOUNT)),
        **verify_receipt(V.MOUNT),
        # `layout_targets` chỉ nói "có tồn tại". ADM cần thêm **kích thước**:
        # một file 283 MB ở đúng đường dẫn trông y hệt một file 2,2 GB.
        "adm_target": {
            "path": V.CKPT,
            "exists": bool(size),
            "size": size,
            "classify": classify_ckpt_size(size) if size else "absent",
            "ok": size >= ADM_MIN_BYTES,
        },
        "author_detectors": sorted(p.name for p in Path(V.CKPT_AUTHOR).glob("*.pth"))
        if Path(V.CKPT_AUTHOR).is_dir() else [],
        "layout_targets": {k: os.path.exists(v) for k, v in V.LAYOUT.items()},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out
