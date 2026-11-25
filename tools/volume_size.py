import os
import stat
from pathlib import Path

import modal

app = modal.App("dire-zero-runs-size")
volume = modal.Volume.from_name("dire-zero", create_if_missing=False)

@app.function(volumes={"/zero": volume}, cpu=1, timeout=1800)
def directory_size(path: str = "./gallery") -> dict:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {root}")

    total_bytes = 0
    file_count = 0

    for current, _dirs, names in os.walk(root):
        for name in names:
            item = Path(current) / name
            try:
                info = item.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                total_bytes += info.st_size
                file_count += 1

    return {
        "path": str(root),
        "files": file_count,
        "bytes": total_bytes,
        "GB": round(total_bytes / 1_000_000_000, 3),
        "GiB": round(total_bytes / (1024 ** 3), 3),
    }

@app.local_entrypoint()
def main(path: str = "/zero/runs"):
    print(directory_size.remote(path))