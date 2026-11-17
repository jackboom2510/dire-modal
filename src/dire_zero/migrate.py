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
3. **Mặc định là ``plan``.** Muốn ghi thì phải gọi ``apply``; muốn ghi thật thì
   ``apply`` phải có ``confirm=True``. Cùng lý do như ``demo_train``: gõ thiếu
   một cờ không được phép tiêu tiền hay đổi dữ liệu.
4. **Có biên lai.** Mỗi lần ``apply`` ghi ``_migration/receipt.json``: nguồn,
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

#: Đuôi file không bao giờ chuyển: rác của quá trình tải/giải nén.
SKIP_SUFFIXES = (".downloading", ".part", ".tmp", ".partial")

#: Thư mục không chuyển: staging tạm và file bị giữ lại để kiểm tay.
SKIP_DIRS = (".staging", "_held", "__pycache__", ".ipynb_checkpoints")

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

    kind: str                # copy | skip | conflict
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
    def bytes_to_copy(self) -> int:
        return sum(a.size for a in self.to_copy)

    def summary(self) -> dict:
        return {
            "n_copy": len(self.to_copy),
            "n_skip": sum(1 for a in self.actions if a.kind == "skip"),
            "n_conflict": len(self.conflicts),
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


def discover_ckpts(roots: dict[str, str] | None = None) -> list[Candidate]:
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
                out.append(Candidate(path=str(p), size=p.stat().st_size))
    out.sort(key=lambda c: (-c.size, c.path))
    return out


def plan_copy(candidates: list[Candidate], target: str | Path, target_exists: bool) -> Plan:
    """Quyết định copy/skip/conflict cho **một** đích duy nhất."""

    target = str(target)
    if not candidates:
        return Plan([Action(kind="conflict", source="", target=target,
                            reason="không tìm thấy checkpoint nào trong volume cũ")])

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
    new = _tree_summary(Path(V.MOUNT))
    out = {
        "new_volume": {"name": V.VOLUME_NAME, "mount": V.MOUNT, "layout_version": V.LAYOUT_VERSION},
        "old": old,
        "ckpt_candidates": [asdict(c) for c in cands],
        "new": new,
        "layout": V.LAYOUT,
        "warnings": [],
    }
    if not cands:
        out["warnings"].append(
            "không thấy checkpoint nào trong volume cũ. Nếu bạn biết nó ở đường khác, "
            "thêm tên file vào CKPT_NAMES rồi chạy lại survey."
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
    cands = discover_ckpts()
    target = V.CKPT
    p = plan_copy(cands, target, Path(target).exists())

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
            "candidates": [asdict(c) for c in cands],
            "actions": [asdict(a) for a in p.actions],
            **p.summary(),
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
def apply(confirm: bool = False, data: bool = False) -> dict:
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

    cands = discover_ckpts()
    p = plan_copy(cands, V.CKPT, Path(V.CKPT).exists())
    for a in p.actions:
        if a.kind == "copy":
            copied.append(copy_verified(a.source, a.target, a.sha256_source))
        elif a.kind == "conflict":
            conflicts.append(asdict(a))
        else:
            skipped.append(asdict(a))

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
        "skipped": skipped[:500],
        "n_skipped": len(skipped),
        "conflicts": conflicts[:500],
        "n_conflicts": len(conflicts),
        "bytes_copied": sum(c["bytes"] for c in copied),
        "deleted_anything": False,
    }
    write_receipt(V.MOUNT, payload)
    vol_new.commit()

    print(json.dumps({k: v for k, v in payload.items() if k != "copied"}, ensure_ascii=False, indent=2))
    if conflicts:
        raise RuntimeError(
            f"{len(conflicts)} xung đột — không ghi đè. Xem receipt.json, kiểm tay rồi quyết định."
        )
    return payload


@app.function(volumes=VOLUMES, cpu=2, timeout=1800)
def verify() -> dict:
    """Kiểm biên lai với đĩa thật — chạy sau `apply`, và chạy lại được bất cứ lúc nào."""
    out = {
        "receipt": str(receipt_path(V.MOUNT)),
        **verify_receipt(V.MOUNT),
        "layout_targets": {k: os.path.exists(v) for k, v in V.LAYOUT.items()},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out
