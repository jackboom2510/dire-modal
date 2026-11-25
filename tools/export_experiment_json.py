import os
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("dire-zero-export-json")
volume = modal.Volume.from_name("dire-zero", create_if_missing=False)


@app.function(volumes={"/zero": volume}, cpu=1, timeout=1800)
def find_json_files(root: str = "/zero/experiments") -> list[str]:
    base = "/zero/experiments"
    root = os.path.abspath(os.path.normpath(root))

    if root != base and not root.startswith(base + os.sep):
        raise ValueError("Chỉ cho phép duyệt bên trong /zero/experiments")
    if not os.path.isdir(root):
        raise FileNotFoundError(root)

    results = []
    for current, _dirs, names in os.walk(root):
        for name in names:
            if name.lower().endswith(".json"):
                full_path = os.path.join(current, name)
                results.append(os.path.relpath(full_path, "/zero").replace(os.sep, "/"))

    return sorted(results)


@app.local_entrypoint()
def main(root: str = "/zero/experiments", out: str = "./modal-runs/experiments/"):
    files = find_json_files.remote(root)
    destination = Path(out).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    modal_cli = shutil.which("modal")
    if not modal_cli:
        raise RuntimeError("Không tìm thấy Modal CLI trong PATH")

    print(f"Tìm thấy {len(files)} file JSON; lưu vào {destination}")
    for index, remote_path in enumerate(files, start=1):
        local_path = destination.joinpath(*remote_path.split("/"))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(files)}] {remote_path}")
        subprocess.run(
            [modal_cli, "volume", "get", "dire-zero", remote_path, str(local_path)],
            check=True,
        )
