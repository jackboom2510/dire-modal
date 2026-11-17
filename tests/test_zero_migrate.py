"""Test cho `dire_zero.migrate` — chuyển Volume cũ sang bố cục `dire-zero`.

Vì sao bộ test này tồn tại
--------------------------
Migration là đoạn code **duy nhất trong repo được phép ghi vào dữ liệu đã tải về**.
Một lỗi ở đây không làm chương trình đỏ: nó làm mất 2,5 GB checkpoint đã tải trong
9 giờ, hoặc ghi đè một bản tốt bằng một bản cụt, và không có thông báo nào.

Bốn khẳng định được canh:

1. **Không xoá.** Không hàm nào trong module gọi `unlink`/`rmtree` (kiểm bằng AST).
   `copy_verified` có `unlink` duy nhất cho file tạm `.migrating` của chính nó.
2. **Không ghi đè.** Đích đã có ⇒ `conflict`, không copy.
3. **Mặc định không ghi.** `apply` không có `confirm=True` ⇒ từ chối.
4. **Có kiểm sha hai đầu.** Copy xong, sha file đích phải bằng sha đã tính khi đọc.

Test chạy trên `tmp_path`, không cần Modal, không cần mạng.

Chạy::

    python3 -m pytest tests/test_zero_migrate.py -q
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from dire_zero import migrate as M
from dire_zero import volume as V


# --------------------------------------------------------------------------- #
# Bố cục: một nguồn sự thật
# --------------------------------------------------------------------------- #

def test_layout_is_one_volume_with_the_documented_subdirs():
    assert V.VOLUME_NAME == "dire-zero"
    assert V.MOUNT == "/zero"
    for key, path in V.LAYOUT.items():
        assert path.startswith(V.MOUNT + "/"), (key, path)
    assert V.CKPT.endswith("adm_lsun.pt")     # giữ tên file: ADM đã ở trên Volume
    # detector tác giả có chỗ riêng — không bao giờ trộn với ADM hay detector R2
    assert V.CKPT_AUTHOR == f"{V.MOUNT}/ckpt/author"
    assert V.RUNS_AUTHOR == f"{V.MOUNT}/runs/author"
    assert V.CKPT_AUTHOR != str(Path(V.CKPT).parent)


def test_layout_version_is_an_int_and_receipts_carry_it():
    assert isinstance(V.LAYOUT_VERSION, int) and V.LAYOUT_VERSION >= 1


def test_targets_helper_rebases_the_layout(tmp_path: Path):
    t = V.targets(tmp_path)
    assert t["ckpt"] == f"{tmp_path}/ckpt/adm_lsun.pt"
    assert t["manifest"] == f"{tmp_path}/data/manifest.tsv"
    assert V.targets("/")["data"] == "/data"


def test_old_volumes_are_only_named_never_written():
    """Volume cũ được khai để **đọc**; test này chặn việc thêm nó vào VOLUMES như đích."""
    assert V.OLD_VOLUMES == {"dire-raw": "/raw", "dire-ckpt": "/ckpt"}
    assert V.MOUNT not in V.OLD_VOLUMES.values()


# --------------------------------------------------------------------------- #
# Không xoá: kiểm bằng AST, không bằng lời
# --------------------------------------------------------------------------- #

def test_migrate_never_deletes_data():
    """Kiểm bằng AST: lệnh xoá duy nhất được phép là file tạm `.migrating`."""
    tree = ast.parse(Path(M.__file__).read_text(encoding="utf-8"))

    allowed = {("copy_verified", "unlink")}     # chỉ file tạm của chính nó
    offenders = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            else:
                continue
            if name in ("rmtree", "remove", "rmdir", "unlink", "os.remove", "os.unlink"):
                if (fn.name, name) not in allowed:
                    offenders.append((fn.name, name, node.lineno))
    assert not offenders, f"có lệnh xoá ngoài danh sách cho phép: {offenders}"


def test_copy_verified_does_not_leave_a_partial_target_behind(tmp_path: Path):
    src = tmp_path / "a.pt"
    src.write_bytes(b"x" * 1000)
    dst = tmp_path / "out" / "a.pt"
    M.copy_verified(src, dst)
    assert dst.read_bytes() == src.read_bytes()
    assert not (tmp_path / "out" / "a.pt.migrating").exists()


def test_copy_verified_refuses_a_source_whose_hash_is_wrong(tmp_path: Path):
    """Bắt lỗi trước khi thay file đích — không để một bản sai thành bản 'đã chuyển'."""
    src = tmp_path / "a.pt"
    src.write_bytes(b"abc")
    dst = tmp_path / "out" / "a.pt"
    with pytest.raises(ValueError, match="sha nguồn lệch"):
        M.copy_verified(src, dst, expect_sha="0" * 64)
    assert not dst.exists()
    assert not (tmp_path / "out" / "a.pt.migrating").exists()


def test_copy_verified_streams_and_reports_both_hashes(tmp_path: Path):
    src = tmp_path / "big.bin"
    src.write_bytes(os.urandom(3 * M.CHUNK + 17))     # vắt qua ranh giới chunk
    out = M.copy_verified(src, tmp_path / "d" / "big.bin")
    assert out["sha256_source"] == out["sha256_target"] == M.sha256_file(src)
    assert out["bytes"] == src.stat().st_size


# --------------------------------------------------------------------------- #
# Phát hiện checkpoint ở đường cũ
# --------------------------------------------------------------------------- #

def _sparse(path: Path, size: int) -> Path:
    """File **thưa** đúng kích thước: test được ngưỡng 2,2 GB mà không ghi 2,2 GB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)
    return path


def _touch(path: Path, size: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"z" * size)
    return path


def test_is_skipped_knows_staging_and_partial_downloads(tmp_path: Path):
    assert M.is_skipped("/raw/zero/.staging/x.tar.gz") == "thư mục tạm/giữ lại"
    assert M.is_skipped("/raw/x/a.downloading") == "đuôi tạm .downloading"
    assert M.is_skipped("/raw/x/_held/a.pt") == "thư mục tạm/giữ lại"
    assert M.is_skipped("/raw/x/a.pt") == ""


def test_discover_finds_all_six_historical_paths(tmp_path: Path):
    """Đây chính là lý do module này tồn tại: cùng một checkpoint ở nhiều đường."""
    raw, ckpt = tmp_path / "raw", tmp_path / "ckpt"
    for rel, size in [
        ("checkpoints/lsun_adm.pth", 300),
        ("ckpt/adm_lsun.pt", 301),
        ("zero/ckpt/adm_lsun.pt", 302),
        ("sources/ckpt/256x256_diffusion_uncond.pt", 303),
    ]:
        _touch(raw / rel, size)
    _touch(ckpt / "pretrained/lsun_adm.pth", 304)

    found = M.discover_ckpts({"dire-raw": str(raw), "dire-ckpt": str(ckpt)})
    names = sorted(Path(c.path).name for c in found)
    assert names == ["256x256_diffusion_uncond.pt", "adm_lsun.pt", "adm_lsun.pt",
                     "lsun_adm.pth", "lsun_adm.pth"]
    assert [c.size for c in found] == sorted([c.size for c in found], reverse=True)


def test_discover_ignores_staging_and_partial_files(tmp_path: Path):
    raw = tmp_path / "raw"
    _touch(raw / ".staging/adm_lsun.pt", 500)
    _touch(raw / "ckpt/adm_lsun.pt.downloading", 500)
    _touch(raw / "ckpt/adm_lsun.pt", 42)
    found = M.discover_ckpts({"dire-raw": str(raw)})
    assert [c.path for c in found] == [str(raw / "ckpt" / "adm_lsun.pt")]


def test_a_detector_named_like_a_checkpoint_is_not_accepted_as_adm(tmp_path: Path):
    """Ca thật trên volume: `checkpoints/lsun_adm.pth` chỉ 283 MB — đó là detector.

    Nếu không có sàn kích thước, `apply` sẽ bê nó vào `ckpt/adm_lsun.pt`, biên lai
    khai thành công, rồi `selfcheck` nổ `Missing key(s)` — hỏng **muộn** nhất có thể.
    """
    raw = tmp_path / "raw"
    _sparse(raw / "checkpoints" / "lsun_adm.pth", 282_534_013)
    _sparse(raw / "sources" / "256x256_diffusion_uncond.pt", 2_200_000_000)

    adm, rejected = M.discover_adm({"dire-raw": str(raw)})
    assert [Path(c.path).name for c in adm] == ["256x256_diffusion_uncond.pt"]
    assert [Path(r["path"]).name for r in rejected] == ["lsun_adm.pth"]
    assert "quá nhỏ" in rejected[0]["reason"]


def test_classify_ckpt_size_is_the_only_place_that_decides():
    assert M.classify_ckpt_size(282_534_013) == "author-detector"
    assert M.classify_ckpt_size(2_200_000_000) == "adm"
    assert M.classify_ckpt_size(42) == "too-small"
    assert M.classify_ckpt_size(M.ADM_MIN_BYTES) == "adm"


def test_discover_ignores_benchmark_copies(tmp_path: Path):
    """`.bench/k1` trên volume thật là bản sao checkpoint để đo băng thông."""
    raw = tmp_path / "raw"
    _sparse(raw / ".bench" / "k1" / "checkpoints" / "lsun_adm.pth", 282_534_013)
    _sparse(raw / ".state" / "x.pt", 2_200_000_000)
    assert M.discover_adm({"dire-raw": str(raw)}) == ([], [])


def test_author_detectors_go_to_their_own_directory(tmp_path: Path):
    raw = tmp_path / "raw"
    for name in ("lsun_adm", "lsun_pndm", "imagenet_adm", "celebahq_sdv2"):
        _sparse(raw / "checkpoints" / f"{name}.pth", 282_534_013)
    _touch(raw / "checkpoints" / "README.pth", 42)          # quá nhỏ ⇒ không phải

    found = M.discover_author_detectors({"dire-raw": str(raw)})
    assert [Path(c.path).stem for c in found] == ["celebahq_sdv2", "imagenet_adm", "lsun_adm", "lsun_pndm"]

    plan = M.plan_author(found, tmp_path / "zero" / "ckpt" / "author")
    assert plan.summary()["n_copy"] == 4
    # tên file = tên generator ⇒ giữ nguyên, không đổi thành adm_lsun.pt
    assert {Path(a.target).name for a in plan.to_copy} == {"lsun_adm.pth", "lsun_pndm.pth",
                                                          "imagenet_adm.pth", "celebahq_sdv2.pth"}


def test_plan_author_never_overwrites_with_a_different_size(tmp_path: Path):
    d = tmp_path / "author"
    _touch(d / "lsun_adm.pth", 300)
    p = M.plan_author([M.Candidate("/raw/checkpoints/lsun_adm.pth", 200)], d)
    assert p.conflicts and p.summary()["n_copy"] == 0

    same = M.plan_author([M.Candidate("/raw/checkpoints/lsun_adm.pth", 300)], d)
    assert same.summary()["n_skip"] == 1 and not same.conflicts


def test_discover_on_missing_root_is_empty_not_an_exception(tmp_path: Path):
    assert M.discover_ckpts({"dire-raw": str(tmp_path / "nope")}) == []


# --------------------------------------------------------------------------- #
# Kế hoạch: copy / skip / conflict
# --------------------------------------------------------------------------- #

def test_plan_copies_the_largest_candidate(tmp_path: Path):
    cands = [M.Candidate("/raw/a/adm_lsun.pt", 300), M.Candidate("/raw/b/adm_lsun.pt", 100)]
    p = M.plan_copy(cands, "/zero/ckpt/adm_lsun.pt", target_exists=False)
    assert p.summary()["n_copy"] == 1 and p.bytes_to_copy == 300
    copy = p.to_copy[0]
    assert copy.source == "/raw/a/adm_lsun.pt"
    # bản còn lại vẫn được ghi nhận, không bị bỏ im lặng
    assert any(a.kind == "skip" and a.source == "/raw/b/adm_lsun.pt" for a in p.actions)


def test_plan_refuses_to_overwrite_an_existing_target(tmp_path: Path):
    cands = [M.Candidate("/raw/ckpt/adm_lsun.pt", 300)]
    p = M.plan_copy(cands, "/zero/ckpt/adm_lsun.pt", target_exists=True)
    assert p.summary()["n_copy"] == 0
    assert p.conflicts and "KHÔNG ghi đè" in p.conflicts[0].reason


def test_missing_source_is_not_a_conflict_when_the_target_exists():
    """Ca thật, đã xảy ra: ADM đã ở đích, volume cũ không có bản nào.

    `apply` từng ném `1 xung đột — không ghi đè ... kiểm tay rồi quyết định`,
    trong khi sự thật là *không có gì để chép*. Một thông báo sai làm người dùng
    đi tìm một cuộc đụng độ không tồn tại.
    """
    p = M.plan_copy([], "/zero/ckpt/adm_lsun.pt", target_exists=True)
    assert not p.conflicts
    assert p.summary()["n_skip"] == 1 and p.summary()["n_missing"] == 0
    assert "verify" in p.actions[0].reason and "selfcheck" in p.actions[0].reason


def test_missing_source_without_target_asks_for_an_upload():
    p = M.plan_copy([], "/zero/ckpt/adm_lsun.pt", target_exists=False)
    assert not p.conflicts, "thiếu nguồn không phải xung đột"
    assert p.summary()["n_missing"] == 1
    assert "modal volume put" in p.missing[0].reason


def test_plan_data_maps_the_tree_and_reports_junk(tmp_path: Path):
    old = tmp_path / "old"
    _touch(old / "data/test/adm/a.png", 10)
    _touch(old / "data/manifest.tsv", 5)
    _touch(old / "maps/test/adm/a.npy", 7)
    _touch(old / "runs/dire.json", 3)
    _touch(old / "ckpt/adm_lsun.pt", 999)          # đi đường riêng
    _touch(old / ".staging/x.tar.gz", 100)         # phải bị bỏ, có lý do

    p = M.plan_data(old, tmp_path / "new")
    copies = {a.target for a in p.to_copy}
    assert str(tmp_path / "new/data/test/adm/a.png") in copies
    assert str(tmp_path / "new/maps/test/adm/a.npy") in copies
    assert str(tmp_path / "new/runs/dire.json") in copies
    assert not any("ckpt/adm_lsun.pt" in a.target for a in p.to_copy)
    assert any(a.kind == "skip" and ".staging" in a.source for a in p.actions)


def test_plan_data_skips_identical_and_flags_different_sizes(tmp_path: Path):
    old, new = tmp_path / "old", tmp_path / "new"
    _touch(old / "data/a.png", 10)
    _touch(new / "data/a.png", 10)                 # giống hệt ⇒ skip
    _touch(old / "data/b.png", 20)
    _touch(new / "data/b.png", 21)                 # khác ⇒ conflict

    p = M.plan_data(old, new)
    kinds = {(a.kind, Path(a.source).name) for a in p.actions}
    assert ("skip", "a.png") in kinds
    assert ("conflict", "b.png") in kinds
    assert p.summary()["n_copy"] == 0


def test_plan_data_on_a_missing_root_is_empty(tmp_path: Path):
    assert M.plan_data(tmp_path / "nope", tmp_path / "new").actions == []


# --------------------------------------------------------------------------- #
# Biên lai + verify
# --------------------------------------------------------------------------- #

def test_receipt_roundtrip_and_verify(tmp_path: Path):
    root = tmp_path / "zero"
    src = _touch(tmp_path / "old" / "a.pt", 64)
    dst = root / "ckpt" / "adm_lsun.pt"
    copied = M.copy_verified(src, dst)

    M.write_receipt(root, {"layout_version": V.LAYOUT_VERSION, "copied": [copied]})
    assert M.read_receipt(root)["layout_version"] == V.LAYOUT_VERSION

    v = M.verify_receipt(root)
    assert v["ok"] is True and v["n_checked"] == 1 and v["n_bad"] == 0
    # biến kết quả từng dòng không được đè lên biên lai (đã từng đè ⇒ None)
    assert v["layout_version"] == V.LAYOUT_VERSION


def test_verify_reports_a_missing_or_tampered_target(tmp_path: Path):
    root = tmp_path / "zero"
    src = _touch(tmp_path / "old" / "a.pt", 64)
    copied = M.copy_verified(src, root / "ckpt" / "adm_lsun.pt")
    M.write_receipt(root, {"layout_version": V.LAYOUT_VERSION, "copied": [copied]})

    (root / "ckpt" / "adm_lsun.pt").write_bytes(b"tampered")
    v = M.verify_receipt(root)
    assert v["ok"] is False and v["n_bad"] == 1 and v["bad"][0]["reason"] == "sha lệch"

    (root / "ckpt" / "adm_lsun.pt").unlink()
    assert M.verify_receipt(root)["bad"][0]["reason"] == "mất file"


def test_verify_without_receipt_is_not_ok(tmp_path: Path):
    """Không có biên lai ≠ mọi thứ đều ổn."""
    v = M.verify_receipt(tmp_path / "zero")
    assert v["ok"] is False and v["n_checked"] == 0


# --------------------------------------------------------------------------- #
# An toàn của entrypoint
# --------------------------------------------------------------------------- #

def test_apply_refuses_without_confirm():
    """Gõ thiếu một cờ không được phép đổi dữ liệu — cùng luật với `demo_train`.

    Đọc từ **source** chứ không `inspect.signature`: `modal.Function` bọc hàm gốc
    và không phải callable, nên test dựa vào signature sẽ vỡ khi Modal đổi API.
    """
    tree = ast.parse(Path(M.__file__).read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply")
    defaults = {
        a.arg: (ast.unparse(d) if d is not None else "<required>")
        for a, d in zip(fn.args.args[-len(fn.args.defaults):], fn.args.defaults)
    }
    assert defaults.get("confirm") == "False"
    # bê dữ liệu phải là lựa chọn rõ ràng, không phải mặc định
    assert defaults.get("data") == "False"
    assert defaults.get("author") == "True"      # detector tác giả: thuần bổ sung, cần cho R0
    # và thân hàm phải thực sự từ chối
    body = ast.unparse(fn)
    assert "confirm" in body and "RuntimeError" in body and "--confirm" in body


def test_apply_declares_cpu_only_and_a_bounded_timeout():
    """Migration là I/O, không được xin GPU; và phải có trần thời gian."""
    import ast

    tree = ast.parse(Path(M.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("survey", "plan", "apply", "verify"):
            kws = {
                kw.arg: ast.unparse(kw.value)
                for dec in node.decorator_list
                if isinstance(dec, ast.Call)
                for kw in dec.keywords
            }
            assert "gpu" not in kws, f"{node.name} xin GPU — không cần cho copy file"
            assert "timeout" in kws, f"{node.name} thiếu timeout"


def test_new_volume_is_the_only_writable_mount():
    assert set(M.VOLUMES) == {V.MOUNT, "/raw", "/ckpt"}
    assert M.VOLUMES[V.MOUNT] is M.vol_new


def test_json_payload_of_a_plan_is_serialisable(tmp_path: Path):
    """Plan được in ra JSON trong container; `asdict` lồng `Path` là lỗi im lặng."""
    cands = [M.Candidate("/raw/a/adm_lsun.pt", 10)]
    p = M.plan_copy(cands, "/zero/ckpt/adm_lsun.pt", False)
    payload = {"actions": [M.asdict(a) for a in p.actions], **p.summary()}
    assert json.loads(json.dumps(payload))["n_copy"] == 1
