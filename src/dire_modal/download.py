"""Tải .tar.gz từ Baidu/Rec vào Volume /raw.
Nguyên tắc: 1 file = 1 function call (tránh timeout 24h cho cả dataset).
"""

import hashlib
import os
import subprocess
from pathlib import Path

import modal

from .common import app, cpu_image, vol_raw, baidu_secret
from .manifest import File, resolve, BAIDU_ROOT


def _sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def _baidu_login():
    """Login mỗi container (BDUSS được inject qua Secret)."""
    bduss = os.environ["BDUSS"]
    stoken = os.environ.get("STOKEN", "")
    args = ["BaiduPCS-Go", "login", f"-bduss={bduss}"]
    if stoken:
        args.append(f"-stoken={stoken}")
    subprocess.run(args, check=True, capture_output=True)


# ---------- Dò cây remote (chạy 1 lần để điền BAIDU_ROOT) ----------
@app.function(
    image=cpu_image,
    secrets=[baidu_secret],
    volumes={"/raw": vol_raw},
    timeout=600,
)
def ls_remote(path: str = "/") -> list[str]:
    """modal run -m dire_modal.download::ls_remote --path /apps"""
    _baidu_login()
    out = subprocess.run(
        ["BaiduPCS-Go", "ls", path],
        check=True,
        capture_output=True,
        text=True,
    )
    print(out.stdout)
    return out.stdout.splitlines()


# ---------- Tải 1 file ----------
@app.function(
    image=cpu_image,
    secrets=[baidu_secret],
    volumes={"/raw": vol_raw},
    timeout=6 * 3600,  # 6h/file đã quá dư
    cpu=0.25,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=30.0),
)
def fetch_one(f: dict) -> dict:
    """Tải 1 file từ Baidu. Resumable: skip nếu đã tồn tại & hợp lệ."""
    remote = f"{BAIDU_ROOT.rstrip('/')}/{f['remote']}"
    local = Path("/raw") / f["local"]
    local.parent.mkdir(parents=True, exist_ok=True)

    # Resume: nếu file tồn tại & gzip -t pass thì bỏ qua
    if local.exists() and local.stat().st_size > 0:
        chk = subprocess.run(["gzip", "-t", str(local)], capture_output=True)
        if chk.returncode == 0:
            print(f"[SKIP] {local.name} đã tồn tại và hợp lệ")
            return {
                "local": str(local),
                "status": "skipped",
                "size": local.stat().st_size,
                "sha256": _sha256(local),
            }
        local.unlink()  # file hỏng → xoá tải lại

    _baidu_login()
    # Baidu chậm → tải xuống /tmp (SSD container) rồi mv sang Volume
    tmp = Path("/tmp") / local.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["BaiduPCS-Go", "d", "--saveto", str(tmp.parent), remote],
        check=True,
    )
    # BaiduPCS-Go lưu theo basename
    downloaded = tmp.parent / Path(remote).name
    downloaded.rename(local)

    # Verify
    chk = subprocess.run(["gzip", "-t", str(local)], capture_output=True)
    if chk.returncode != 0:
        local.unlink(missing_ok=True)
        raise RuntimeError(f"gzip -t FAIL: {local}")

    size = local.stat().st_size
    digest = _sha256(local)
    vol_raw.commit()  # bắt buộc để file sống qua container khác
    print(f"[OK] {local.name}  {size/1e9:.2f} GB  sha256={digest[:12]}…")
    return {"local": str(local), "status": "ok", "size": size, "sha256": digest}


# ---------- Entry: tải nhiều tier ----------
@app.function(
    image=cpu_image,
    volumes={"/raw": vol_raw},
    timeout=24 * 3600,
)
def main(tiers: str = "ckpt,A", concurrency: int = 4) -> None:
    """
    modal run -m dire_modal.download::main --tiers ckpt,A --concurrency 4

    Chú ý: concurrency cao không giúp vì Baidu bottleneck ở phía server.
    4-8 là ngọt.
    """
    files: list[File] = []
    for t in tiers.split(","):
        files.extend(resolve(t.strip()))
    print(f"[plan] {len(files)} files across tiers={tiers}")

    payloads = [{"remote": f.remote, "local": f.local} for f in files]
    results = list(fetch_one.map(payloads, order_outputs=False))

    # Ghi manifest ra Volume /raw
    import csv, datetime

    out = Path("/raw/_manifest.tsv")
    write_header = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        if write_header:
            w.writerow(["timestamp", "local", "size_bytes", "sha256", "status"])
        now = datetime.datetime.utcnow().isoformat()
        for r in results:
            w.writerow([now, r["local"], r["size"], r["sha256"], r["status"]])
    vol_raw.commit()
    print(f"[done] manifest → {out}")


# ---------- Verify toàn bộ ----------
@app.function(
    image=cpu_image,
    volumes={"/raw": vol_raw},
    timeout=2 * 3600,
    cpu=2.0,
)
def verify() -> None:
    """modal run -m dire_modal.download::verify"""
    bad = []
    for p in Path("/raw").rglob("*.tar.gz"):
        chk = subprocess.run(["gzip", "-t", str(p)], capture_output=True)
        tag = "OK" if chk.returncode == 0 else "FAIL"
        print(f"[{tag}] {p}")
        if chk.returncode != 0:
            bad.append(str(p))
    if bad:
        raise SystemExit(f"{len(bad)} file hỏng: {bad}")
    print("All OK.")
