"""Test cho `tools/put_resume.py` — công cụ đẩy dữ liệu lên Volume theo lô.

Vì sao bộ test này tồn tại
--------------------------
Công cụ này ra đời sau một lần mất trắng 22 GB: `modal volume put` gom mọi file
vào **một** request `VolumePutFiles` ở cuối context manager, nên 217 579/217 585
file rồi treo ⇒ không file nào vào FS. Ba khẳng định phải được canh:

1. **Chỉ đẩy phần thiếu.** Đẩy lại cả cây là 22 GB lần nữa; bỏ sót file đẩy dở
   (đúng tên, thiếu byte) là `checkdata` báo sha lệch ở bước sau.
2. **Mỗi lô là một lần ghi riêng.** Nếu ai đó "tối ưu" thành một lô duy nhất thì
   công cụ mất đúng lý do tồn tại — test đếm số lần `batch_upload`.
3. **`manifest.tsv` + `data_card.json` ở gốc không bị bỏ sót.** Đây chính là hai
   file đã thiếu trên Volume thật sau khi đẩy thủ công từng thư mục con.

Test chạy bằng Volume giả, không mạng, không cần đăng nhập Modal.

Chạy::

    python3 -m pytest tests/test_put_resume.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "put_resume.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("put_resume", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load_tool()


# --------------------------------------------------------------------------- #
# Volume giả
# --------------------------------------------------------------------------- #

class _Entry:
    def __init__(self, path: str, size: int = 0, kind: int = 1) -> None:
        self.path, self.size, self.type = path, size, kind


class FakeVolume:
    """Đủ giống `modal.Volume` để chạy `remote_files` / `push`, không có mạng."""

    def __init__(self, listing: dict[str, list] | None = None,
                 fail_names: set[str] | None = None, fail_until: int = 0) -> None:
        self.listing = listing or {}
        self.fail_names = fail_names or set()
        self.fail_until = fail_until          # số lần lỗi trước khi thành công
        self.batches: list[list[str]] = []
        self.attempts: list[int] = []

    def listdir(self, prefix: str):
        if prefix not in self.listing:
            raise FileNotFoundError(prefix)   # Modal ném lỗi khi thư mục chưa có
        return self.listing[prefix]

    class _Ctx:
        def __init__(self, vol: FakeVolume) -> None:
            self.vol = vol
            self.files: list[str] = []

        def __enter__(self):
            return self

        def put_file(self, local, remote) -> None:
            self.files.append(str(remote))

        def __exit__(self, exc_type, exc, tb) -> bool:
            names = {f.rsplit("/", 1)[-1] for f in self.files}
            n = len(self.vol.attempts)
            self.vol.attempts.append(n + 1)
            if names & self.vol.fail_names and n < self.vol.fail_until:
                raise ConnectionError("lô giả lập hỏng")   # như Modal: ném ở __exit__
            self.vol.batches.append(list(self.files))
            return False

    def batch_upload(self, force: bool = False):
        return FakeVolume._Ctx(self)


def _pngs(d: Path, n: int, size: int = 10) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = d / f"{i:016x}.png"
        p.write_bytes(b"x" * size)
        out.append(p)
    return out


# --------------------------------------------------------------------------- #
# 1. Chỉ đẩy phần thiếu
# --------------------------------------------------------------------------- #

def test_leaves_are_split_generator_dirs_and_skip_empty(tmp_path: Path):
    _pngs(tmp_path / "train" / "real", 2)
    _pngs(tmp_path / "train" / "pndm", 1)
    (tmp_path / "test" / "adm").mkdir(parents=True)          # rỗng ⇒ bỏ
    (tmp_path / "manifest.tsv").write_text("png\n", encoding="utf-8")

    assert [rel for rel, _ in M.leaves(tmp_path)] == ["train/pndm", "train/real"]
    assert [rel for rel, _ in M.leaves(tmp_path, "pndm")] == ["train/pndm"]


def test_missing_in_catches_both_absent_and_truncated_files(tmp_path: Path):
    files = _pngs(tmp_path / "train" / "real", 3, size=10)
    have = {files[0].name: 10,                                  # đã có, đúng size
            files[1].name: 4}                                   # có nhưng cụt
    missing = M.missing_in(files, have)
    assert [p.name for p in missing] == [files[1].name, files[2].name]


def test_missing_in_is_empty_when_everything_matches(tmp_path: Path):
    files = _pngs(tmp_path / "train" / "real", 2, size=7)
    assert M.missing_in(files, {p.name: 7 for p in files}) == []


# --------------------------------------------------------------------------- #
# 2. Mỗi lô là một lần ghi riêng
# --------------------------------------------------------------------------- #

def test_push_creates_one_batch_per_chunk(tmp_path: Path):
    files = _pngs(tmp_path, 7)
    vol = FakeVolume()
    uploaded, failed = M.push(vol, files, "data/train/real", chunk=3, retries=0)
    assert uploaded == 7 and failed == []
    assert len(vol.batches) == 3                    # 3 + 3 + 1
    assert [len(b) for b in vol.batches] == [3, 3, 1]
    assert vol.batches[0][0].startswith("data/train/real/")


def test_a_failing_batch_does_not_take_the_others_with_it(tmp_path: Path):
    files = _pngs(tmp_path, 6)
    vol = FakeVolume(fail_names={files[0].name}, fail_until=99)
    uploaded, failed = M.push(vol, files, "data/x", chunk=2, retries=0)
    assert uploaded == 4 and len(failed) == 1        # chỉ lô chứa file hỏng mất
    assert failed[0]["files"] == [files[0].name, files[1].name]
    assert len(vol.batches) == 2


def test_push_retries_before_giving_up(tmp_path: Path):
    files = _pngs(tmp_path, 2)
    vol = FakeVolume(fail_names={files[0].name}, fail_until=2)
    uploaded, failed = M.push(vol, files, "data/x", chunk=2, retries=2)
    assert uploaded == 2 and failed == []            # lần thứ ba mới thành công
    assert len(vol.attempts) == 3


def test_chunk_of_one_isolates_the_poison_file(tmp_path: Path):
    """`--chunk 1` là chế độ dùng để khoanh vùng file cứng đầu."""
    files = _pngs(tmp_path, 3)
    vol = FakeVolume(fail_names={files[1].name}, fail_until=99)
    uploaded, failed = M.push(vol, files, "data/x", chunk=1, retries=0)
    assert uploaded == 2 and len(failed) == 1
    assert failed[0]["files"] == [files[1].name]


# --------------------------------------------------------------------------- #
# 3. Metadata ở gốc cây
# --------------------------------------------------------------------------- #

def test_root_files_to_push_finds_the_two_files_at_the_root(tmp_path: Path):
    (tmp_path / "manifest.tsv").write_text("png\tlabel\n", encoding="utf-8")
    (tmp_path / "data_card.json").write_text("{}\n", encoding="utf-8")
    _pngs(tmp_path / "train" / "real", 1)

    got = M.root_files_to_push(tmp_path, {})
    assert [p.name for p in got] == ["manifest.tsv", "data_card.json"]

    sizes = {p.name: p.stat().st_size for p in got}
    assert M.root_files_to_push(tmp_path, sizes) == []          # đã có đủ
    sizes["manifest.tsv"] = 1                                    # cụt
    assert [p.name for p in M.root_files_to_push(tmp_path, sizes)] == ["manifest.tsv"]


def test_root_files_missing_on_disk_are_not_invented(tmp_path: Path):
    assert M.root_files_to_push(tmp_path, {}) == []


def test_remote_files_reads_names_and_sizes_and_ignores_directories():
    vol = FakeVolume(listing={"data/train/real": [
        _Entry("data/train/real/a.png", 10),
        _Entry("data/train/real/b.png", 20),
        _Entry("data/train/real/sub", 0, kind=2),
    ]})
    assert M.remote_files(vol, "data/train/real") == {"a.png": 10, "b.png": 20}


def test_remote_files_on_a_missing_directory_is_empty_not_an_exception():
    assert M.remote_files(FakeVolume(), "data/nope") == {}


# --------------------------------------------------------------------------- #
# Bất biến của công cụ
# --------------------------------------------------------------------------- #

def test_the_tool_never_deletes_anything():
    """Không có `unlink`/`rmtree`/`remove` nào — giống luật của `migrate`."""
    import ast

    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    banned = {"unlink", "rmtree", "remove", "rmdir"}
    offenders = [n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in banned]
    assert not offenders, offenders


def test_default_chunk_is_small_enough_to_be_useful():
    """Một lô phải nhỏ hơn nhiều so với 217 585 file của lần chạy thật."""
    src = TOOL.read_text(encoding="utf-8")
    assert '"--chunk", type=int, default=500' in src
