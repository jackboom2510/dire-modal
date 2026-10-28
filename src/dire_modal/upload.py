"""Đưa dữ liệu **đã tải ở máy** lên Volume Modal, rồi dọn tiến độ cũ.

    modal run -m dire_modal.upload::main --tiers e2min                    # kế hoạch
    modal run -m dire_modal.upload::main --tiers e2min --mode push
    modal run -m dire_modal.upload::main --tiers e2min --mode push --clean-after
    modal run -m dire_modal.upload::main --tiers e2min --mode clean --yes

Vì sao có module này
-------------------
Trước đây hai chiều đều là Baidu → Volume (`download.py`). Khi dữ liệu đã nằm ở
máy thì hướng đúng là **máy → Volume**: byte đã có sẵn, không phải chờ Baidu
76.6 KiB/s lần nữa. `modal volume put` là CLI chạy ở máy, nên việc của module
này là quyết định **file nào cần đẩy** (hỏi Volume trước, không đẩy mù) và
**dọn `.staging` cũ** trên Volume.

Ba chi tiết dễ sai, đều đã thành guard/test:

1. **`modal volume put` dùng đường dẫn TÍNH TỪ GỐC VOLUME.** Volume `dire-raw`
   mount ở `/raw` trong container, nhưng CLI nhận `dire/test/...`, **không**
   `/raw/dire/test/...`. Đưa `/raw/...` vào là tạo ra `/raw/raw/...`.
2. **`modal volume put` đặt thư mục VÀO TRONG đích.** Đẩy `./data/raw/dire` tới
   `/raw/dire` cho ra `/raw/dire/dire/...`. Vì vậy ở đây đẩy **từng file** tới
   đúng đường dẫn đích — không phụ thuộc ngữ nghĩa đó, và resume được từng file.
3. **`.staging` trên Volume giờ là rác**, nhưng chỉ sau khi file tương ứng đã
   nằm trên Volume. Xoá trước khi đẩy xong là xoá bản duy nhất giữa hai nơi —
   nên `--mode clean` **kiểm trước**: thiếu file nào của tier thì từ chối, trừ
   khi nói rõ `--skip-check`.

Đường dẫn local giữ ĐÚNG cây của manifest (`dire/test/…`, `checkpoints/…`), nên
`_find_on_disk` (dùng chung với `local_fetch`) tìm được cả file do GUI tải về
một cây hơi khác, miễn tên khớp duy nhất.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .common import RAW_VOLUME, app, cpu_image, vol_raw
from .fetchcore import _demangle, _human
from .local_fetch import DEFAULT_DEST, _find_on_disk
from .manifest import File, expand, resolve_many

#: Vùng nội bộ của `/raw`. Tên khớp `download._is_internal`.
#: `.staging` = tiến độ tải của đường Modal, `.bench` = rác đo tốc độ,
#: `.state` = cache kiểm tra.
INTERNAL_AREAS = (".staging", ".bench", ".state")


# ===========================================================================
# Hàm THUẦN — test được với thư mục tạm, không cần Modal
# ===========================================================================
def internal_area_of(rel: str) -> str:
    """``".staging"`` / ``".bench"`` / ``".state"`` nếu ``rel`` thuộc vùng nội bộ."""
    first = rel.replace("\\", "/").lstrip("/").split("/", 1)[0]
    return first if first in INTERNAL_AREAS else ""


def index_tree(root: Path) -> dict:
    """Quét một gốc dữ liệu → ``{"files": {rel: size}, "internal": {vùng: bytes}}``.

    Tách khỏi function Modal để test chạy được trên thư mục tạm: ``/raw`` trong
    container là FUSE, nhưng hình dạng dữ liệu thì y hệt một cây thường.
    """
    files: dict[str, int] = {}
    internal: dict[str, int] = {}
    if not root.exists():
        return {"files": files, "internal": internal}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        try:
            size = p.stat().st_size
        except OSError:
            continue
        area = internal_area_of(rel)
        if not area:
            files[rel] = size
            continue
        internal[area] = internal.get(area, 0) + size
        if rel.startswith(f"{area}/_held"):
            held = f"{area}/_held"
            internal[held] = internal.get(held, 0) + size
    return {"files": files, "internal": internal}


def remote_path(local_rel: str) -> str:
    """Đường dẫn cho CLI ``modal volume put`` — tính từ **gốc Volume**.

    Volume mount ở ``/raw`` trong container, nhưng CLI không biết chuyện đó:
    ``dire/test/x.tar.gz`` ⇒ file nằm ở ``/raw/dire/test/x.tar.gz`` khi container
    nhìn. Đưa ``/raw/dire/...`` vào CLI là tạo ``/raw/raw/dire/...``.
    """
    return "/" + local_rel.strip("/")


def put_command(local: Path, local_rel: str, *, volume: str = RAW_VOLUME,
                force: bool = False, modal_bin: str = "modal") -> list[str]:
    """Lệnh đẩy MỘT file. ``local_rel`` là đường dẫn *trong* cây dữ liệu."""
    cmd = [modal_bin, "volume", "put", volume, str(local), remote_path(local_rel)]
    if force:
        cmd.append("--force")
    return cmd


def put_supports_force(modal_bin: str = "modal") -> bool:
    """Bản `modal` này có cờ ``--force`` cho `volume put` không?

    Hỏi một lần rồi nhớ. Đoán bừa mà cờ không tồn tại thì `modal` exit 2 và
    người dùng phải tự đoán xem lỗi do đâu — trong khi giá trị của cờ này chỉ
    là ghi đè một file đã có nhưng sai size.
    """
    try:
        cp = subprocess.run([modal_bin, "volume", "put", "--help"],
                            capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    return "--force" in (cp.stdout or "") + (cp.stderr or "")


def plan_upload(files: list[File], dest: Path,
                remote_files: dict[str, int]) -> tuple[list, list, list]:
    """So cây local với cây trên Volume → ``(cần đẩy, đã có, thiếu ở local)``.

    So **size tuyệt đối**, không so ``ls -l``: ở đây cả hai phía đều là file thật
    trên đĩa, không có phép làm tròn nào xen vào (khác hẳn đường Baidu).
    """
    todo: list[tuple[File, Path, int, str]] = []
    ok: list[tuple[File, Path, int]] = []
    absent: list[File] = []
    for f in files:
        local = _find_on_disk(dest, f.local)
        if local is None:
            absent.append(f)
            continue
        size = local.stat().st_size
        have = remote_files.get(f.local)
        if have == size:
            ok.append((f, local, size))
        elif have is None:
            todo.append((f, local, size, "chưa có trên Volume"))
        else:
            todo.append((f, local, size,
                         f"Volume có {_human(have)} ≠ local {_human(size)}"))
    return todo, ok, absent


def check_ready(files: list[File], remote_files: dict[str, int]) -> list[str]:
    """Vấn đề khiến việc xoá ``.staging`` là liều: file nào chưa nằm trên Volume."""
    problems = []
    for f in files:
        have = remote_files.get(f.local)
        if have is None:
            problems.append(f"thiếu: {f.local}")
    return problems


def clean_tree(root: Path, areas: tuple[str, ...], *, dry: bool = True) -> dict:
    """Xoá các vùng nội bộ (``.staging`` …) trong ``root``. Trả báo cáo.

    Cố ý **không** đụng tới dữ liệu thật: chỉ xoá đúng tên trong
    :data:`INTERNAL_AREAS`. Một lần gõ sai ``--areas`` không được phép xoá
    ``dire/``.
    """
    report: dict = {"dry": dry, "areas": [], "bytes": 0, "files": 0}
    for a in areas:
        a = a.strip().strip("/")
        if a not in INTERNAL_AREAS:
            raise ValueError(
                f"'{a}' không phải vùng nội bộ — chỉ nhận {list(INTERNAL_AREAS)}. "
                "Xoá dữ liệu thật phải làm bằng tay, có chủ đích."
            )
        p = root / a
        if not p.exists():
            continue
        stat = index_tree(p)
        nbytes = sum(stat["files"].values()) + sum(
            v for k, v in stat["internal"].items() if k != ".staging/_held")
        nfiles = len(stat["files"])
        held = stat["internal"].get(".staging/_held", 0)
        report["areas"].append({"area": a, "bytes": nbytes, "files": nfiles,
                                "held_bytes": held})
        report["bytes"] += nbytes
        report["files"] += nfiles
        if not dry:
            shutil.rmtree(p, ignore_errors=True)
            report["areas"][-1]["removed"] = not p.exists()
    return report


# ===========================================================================
# Mặt Modal: hỏi Volume và dọn nó — hai việc phải chạy TRONG container
# ===========================================================================
@app.function(image=cpu_image, volumes={"/raw": vol_raw}, timeout=1800)
def remote_index(root: str = "/raw") -> dict:
    """Cây dữ liệu hiện có trên Volume (chỉ tên + size; không đọc nội dung)."""
    return index_tree(Path(root))


@app.function(image=cpu_image, volumes={"/raw": vol_raw}, timeout=1800)
def clean(areas: str = ".staging", yes: bool = False, root: str = "/raw") -> dict:
    """Xoá vùng nội bộ trên Volume. ``yes=False`` ⇒ chỉ báo cáo (dry-run)."""
    rep = clean_tree(Path(root), tuple(a for a in areas.split(",") if a.strip()),
                     dry=not yes)
    if yes and rep["bytes"]:
        # Bắt buộc: không commit thì thay đổi không sống qua container này và
        # `download::verify` vẫn thấy file cũ ⇒ tưởng xoá hụt.
        vol_raw.commit()
        print(f"[clean] đã xoá {_human(rep['bytes'])} và commit Volume")
    return rep


# ===========================================================================
# Mặt máy local: quyết định, rồi gọi CLI `modal volume put`
# ===========================================================================
def _remote_index() -> dict:
    """Móc duy nhất để test thay thế (không patch được ``remote_index``)."""
    return remote_index.remote()


def _clean(areas: str, yes: bool) -> dict:
    """Móc duy nhất để test thay thế."""
    return clean.remote(areas=areas, yes=yes)


def _local_lister(dest: Path):
    """Cho ``DirSpec`` (tier `e4x`): liệt kê từ **đĩa local**, không hỏi Baidu."""
    def _list(remote_dir: str) -> list[str]:
        d = dest / remote_dir
        return sorted(p.name for p in d.glob("*") if p.is_file()) if d.is_dir() else []
    return _list


def _resolve_files(tiers: str, dest: Path) -> tuple[list[File], list[str]]:
    ents = resolve_many(tiers) if tiers else []
    if not ents:
        return [], [f"tier {tiers!r} không có file nào"]
    files, unresolved, blocked = expand(ents, lister=_local_lister(dest))
    notes = [f"⛔ NEVER chặn: {why}" for _f, why in blocked]
    notes += [f"⚠ không thấy dữ liệu local cho {d.remote_dir}" for d in unresolved]
    return files, notes


def _short(rel: str) -> str:
    """Đường dẫn dài in cả ra thì log xuống dòng; giữ phần đuôi phân biệt."""
    return rel if len(rel) <= 64 else "…" + rel[-63:]


def _print_plan(todo, ok, absent, internal) -> None:
    print(f"{'=' * 78}\nKẾ HOẠCH ĐẨY LÊN VOLUME\n{'=' * 78}")
    for f, local, size, why in todo:
        print(f"  ↑ {_human(size):>10}  {f.local}\n       ({why})")
    if ok:
        print(f"  = {len(ok)} file đã có trên Volume, đúng size → bỏ qua")
    for f in absent:
        print(f"  ✗ thiếu ở local: {f.local}")
    need = sum(s for _f, _l, s, _w in todo)
    print(f"{'-' * 78}\n  Đẩy : {len(todo)} file · {_human(need)}")
    print(f"  Skip: {len(ok)} file")
    if absent:
        print(f"  Thiếu local: {len(absent)} file  (tải trước: --mode fetch)")
    if internal:
        print("\n  Vùng nội bộ trên Volume (dọn được khi dữ liệu đã ở cả hai nơi):")
        for area, nbytes in sorted(internal.items()):
            if area.count("/"):
                continue
            print(f"    {area:<10} {_human(nbytes):>10}")


@app.local_entrypoint()
def main(
    mode: str = "plan",
    tiers: str = "e2min",
    dest: str = DEFAULT_DEST,
    volume: str = RAW_VOLUME,
    areas: str = ".staging",
    yes: bool = False,
    clean_after: bool = False,
    skip_check: bool = False,
    modal_bin: str = "modal",
) -> int:
    """``plan`` (mặc định) | ``push`` | ``clean``.

    ``plan`` không đổi gì cả ở hai phía. ``push`` chỉ đẩy file thiếu/sai size.
    ``clean`` xoá vùng nội bộ trên Volume, và **từ chối** nếu còn file của tier
    chưa nằm trên Volume (trừ ``--skip-check``).
    """
    dest_path = Path(_demangle(dest)).expanduser()
    areas = areas or ".staging"
    print(f"[upload] local  : {dest_path.resolve()}")
    print(f"[upload] volume : {volume}  → khai báo trong lệnh put là path TÍNH TỪ GỐC VOLUME")

    idx = _remote_index()
    remote_files: dict[str, int] = idx.get("files", {})
    internal: dict[str, int] = idx.get("internal", {})

    files, notes = _resolve_files(tiers, dest_path) if tiers else ([], [])
    for n in notes:
        print(f"  {n}")

    if mode == "clean":
        if not files and not skip_check:
            raise SystemExit(
                "[!] --mode clean cần biết cái gì PHẢI còn nguyên trên Volume trước "
                "khi xoá `.staging`.\n"
                "    Cách đúng:  --mode clean --tiers e2min --yes\n"
                "    Nếu chắc chắn (dữ liệu đã ở chỗ khác):  --skip-check --yes"
            )
        problems = check_ready(files, remote_files) if files else []
        if problems and not skip_check:
            print(f"\n[GUARD] {len(problems)} file của tier '{tiers}' CHƯA có trên Volume:")
            for p in problems[:10]:
                print(f"    {p}")
            raise SystemExit(
                "\n  Xoá `.staging` lúc này là xoá bản duy nhất còn lại trên cloud.\n"
                "  Đẩy trước:  --mode push --tiers …\n"
                "  Hoặc bỏ qua kiểm (đã có bản ở nơi khác):  --skip-check"
            )
        rep = _clean(areas, yes)
        _print_clean(rep, yes)
        return 0

    todo, ok, absent = plan_upload(files, dest_path, remote_files)
    _print_plan(todo, ok, absent, internal)

    if mode == "plan":
        print("\n[DRY-RUN] Chưa đẩy gì. Đẩy thật: thêm --mode push")
        if internal:
            print(f"          Dọn `.staging` cũ: --mode clean --tiers {tiers} --yes")
        return 0

    if mode != "push":
        raise SystemExit(f"[!] mode không hợp lệ: {mode!r} (plan | push | clean)")
    if absent:
        print(f"\n[GUARD] {len(absent)} file chưa có ở local — đẩy phần còn lại sẽ "
              "để Volume ở trạng thái nửa vời. Tải nốt rồi chạy lại.")
        return 1
    if not todo:
        print("\n[ok] Volume đã khớp local, không cần đẩy gì.")
    else:
        force_ok = put_supports_force(modal_bin)
        failed = []
        for i, (f, local, size, why) in enumerate(todo, 1):
            force = why.startswith("Volume có") and force_ok
            cmd = put_command(local, f.local, volume=volume, force=force,
                              modal_bin=modal_bin)
            print(f"\n[{i}/{len(todo)}] {_short(f.local)}  {_human(size)}")
            cp = subprocess.run(cmd, capture_output=True, text=True)
            if cp.returncode != 0:
                failed.append((f.local, (cp.stderr or cp.stdout or "").strip()[:400]))
                print(f"  ✗ exit {cp.returncode}: {failed[-1][1]}")
            else:
                print("  ✓")
        if failed:
            print(f"\n[!] {len(failed)}/{len(todo)} file thất bại:")
            for rel, err in failed[:10]:
                print(f"    {rel}\n      {err}")
            return 1
        print(f"\n[ok] đã đẩy {len(todo)} file · {_human(sum(t[2] for t in todo))}")

    # Kiểm lại sau khi đẩy: nếu Volume vẫn lệch thì đừng xoá gì.
    after = _remote_index().get("files", {})
    still = [(f, s, after.get(f.local)) for f, _l, s, _w in todo if after.get(f.local) != s]
    if still:
        print(f"\n[GUARD] sau khi đẩy vẫn lệch {len(still)} file → KHÔNG dọn staging:")
        for f, want, have in still[:10]:
            print(f"    {f.local}: local {want} B, Volume {have} B")
        return 1

    print("\n[ok] Volume khớp local từng byte.")
    if clean_after:
        print("\n[clean] dọn vùng nội bộ cũ trên Volume…")
        _print_clean(_clean(areas, yes), yes)
    else:
        print("     Dọn `.staging` cũ để bớt dung lượng: "
              f"--mode clean --tiers {tiers} --yes")
        print("     Hoặc gộp vào lần sau: --mode push --clean-after --yes")
    print("     CỔNG 2 (kiểm trên cloud): modal run -m dire_modal.download::verify")
    return 0


def _print_clean(rep: dict, yes: bool) -> None:
    if not rep["areas"]:
        print("[clean] không có vùng nội bộ nào để dọn.")
        return
    print(f"{'=' * 78}\nDỌN VÙNG NỘI BỘ TRÊN VOLUME{'  (thật)' if yes else '  (DRY-RUN)'}\n{'=' * 78}")
    for a in rep["areas"]:
        mark = "✓ đã xoá" if a.get("removed") else ("… sẽ xoá" if not yes else "✗ còn")
        extra = f"  ({_human(a['held_bytes'])} trong _held)" if a.get("held_bytes") else ""
        print(f"  {a['area']:<10} {_human(a['bytes']):>10} · {a['files']:>3} file  {mark}{extra}")
    print(f"{'-' * 78}\n  Tổng: {_human(rep['bytes'])} · {rep['files']} file")
    if not yes:
        print("  Chưa xoá gì. Thêm --yes để xoá thật.")
