#!/usr/bin/env python3
"""Kiểm máy trước khi tạo HF Space — tập trung vào lỗi **không phải lỗi của code**.

Vì sao có tệp này
-----------------
`hf repo create` từng chết ở máy Windows bằng một traceback 30 dòng kết thúc ở::

    File "C:\\...\\ssl.py", line 707, in create_default_context
    FileNotFoundError: [Errno 2] No such file or directory

Không có tên tệp, không có biến nào được nhắc. Nguyên nhân: `httpx` đọc
`os.environ["SSL_CERT_FILE"]` khi dựng SSL context, và biến đó **có** nhưng trỏ
tới một tệp không tồn tại (thường là đường dẫn kiểu MSYS `/mingw64/ssl/...` do
Git Bash/conda đặt, hoặc một `cacert.pem` đã bị xoá khi nâng cấp conda).

Lỗi này không liên quan tới token, tới quyền, hay tới `huggingface_hub`; sửa
`SSL_CERT_FILE` là xong. Nhưng traceback không nói thế, nên mỗi người mất vài
chục phút để đoán. Tool này đoán thay, trong 1 giây.

Chạy::

    python tools/hf_space_preflight.py --space <user>/dire-demo
    python tools/hf_space_preflight.py --json

Mã thoát: ``0`` nếu không có vấn đề chặn, ``1`` nếu có (biến CA trỏ tệp không
tồn tại, hoặc không có token khi cần tạo repo).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

#: Ba biến mà thư viện HTTP đọc để tìm CA bundle. `httpx` dùng `SSL_CERT_FILE`,
#: `requests` dùng `REQUESTS_CA_BUNDLE`, `curl` dùng `CURL_CA_BUNDLE` — một máy
#: Windows có Git Bash/conda thường có cả ba, và cả ba có thể cùng sai.
CA_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_DIR")

#: Đường dẫn kiểu MSYS/Cygwin: `/c/Users/...`, `/mingw64/ssl/...`, `/usr/ssl/...`.
#: Python bản Windows **không** hiểu chúng (nó hiểu là "ổ hiện tại + thư mục `c`").
_MSYS = re.compile(r"^(/(?:c|d|e|f|g)/|/mingw64/|/usr/|/mingw32/|/clang64/)", re.IGNORECASE)

FIXES = {
    "git-bash": "unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE SSL_CERT_DIR",
    "cmd": "set SSL_CERT_FILE=\nset REQUESTS_CA_BUNDLE=\nset CURL_CA_BUNDLE=",
    "powershell": ("Remove-Item Env:SSL_CERT_FILE,Env:REQUESTS_CA_BUNDLE,"
                   "Env:CURL_CA_BUNDLE -ErrorAction SilentlyContinue"),
}


def looks_like_msys(value: str) -> bool:
    """Đường dẫn kiểu Git Bash mà chương trình Windows không mở được."""
    return bool(_MSYS.match(value.replace("\\", "/")))


def ca_status(env: dict[str, str], exists=None) -> list[dict]:
    """Trạng thái từng biến CA. Không bao giờ ném — đầu vào là môi trường lạ."""
    exists = exists or os.path.exists
    out = []
    for var in CA_VARS:
        raw = (env.get(var) or "").strip()
        if not raw:
            continue
        msys = looks_like_msys(raw)
        try:
            # `exists` nhận **chuỗi** (mặc định `os.path.exists`); test nào truyền
            # `Path.exists` cũng chạy được vì ta thử lại bằng `Path` khi cần.
            ok = bool(exists(raw))
        except (OSError, ValueError, AttributeError):
            try:
                ok = bool(exists(Path(raw)))
            except Exception:
                ok = False
        reason = ""
        if not ok:
            reason = ("đường dẫn kiểu MSYS — Python bản Windows không mở được"
                      if msys else "tệp/thư mục không tồn tại")
        out.append({"var": var, "value": raw, "ok": ok, "msys": msys, "reason": reason})
    return out


def certifi_bundle(import_module=None) -> str:
    """Đường dẫn CA bundle của `certifi`, nếu có — phương án thay thế sạch nhất."""
    import_module = import_module or __import__
    try:
        mod = import_module("certifi")
        path = str(mod.where())
    except Exception:
        return ""
    return path if os.path.exists(path) else ""


def find_token(env: dict[str, str], home: Path | None = None) -> str:
    """Tên biến/tệp đang giữ token — **không** trả token.

    Token không bao giờ được in ra: tool này chạy trong terminal, và terminal thì
    bị chụp màn hình, bị dán vào chat, bị đưa vào log CI.
    """
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if (env.get(var) or "").strip():
            return var
    home = home or Path.home()
    for p in (home / ".cache" / "huggingface" / "token",
              home / ".huggingface" / "token"):
        if p.is_file() and p.read_text(encoding="utf-8", errors="ignore").strip():
            return str(p)
    return ""


def build_report(space: str, env: dict[str, str] | None = None,
                 exists=None, which=None, home: Path | None = None) -> dict:
    env = dict(os.environ if env is None else env)
    which = which or shutil.which
    ca = ca_status(env, exists)
    broken = [c for c in ca if not c["ok"]]
    token = find_token(env, home)
    report = {
        "space": space,
        "ca": ca,
        "ca_broken": broken,
        "certifi": certifi_bundle(),
        "token": token,
        "git": which("git") or "",
        "hf": which("hf") or which("huggingface-cli") or "",
        "blocked": bool(broken),
        "warnings": [],
    }
    if not token:
        report["warnings"].append(
            "không thấy token HF (`HF_TOKEN`, hoặc tệp `~/.cache/huggingface/token`) — "
            "cần cho `hf repos create` và cho `git push` lên Space private"
        )
    if not report["hf"]:
        report["warnings"].append("không thấy `hf` trong PATH — tạo Space bằng web UI cũng được")
    if not report["git"]:
        report["warnings"].append("không thấy `git` trong PATH — không đẩy được nội dung lên Space")
    return report


def render(report: dict) -> str:
    lines: list[str] = []
    say = lines.append
    say(f"KIỂM MÁY CHO HF SPACE — {report['space'] or '(chưa đặt tên)'}")

    if report["ca"]:
        say("\nBiến CA trong môi trường:")
        for c in report["ca"]:
            mark = "OK  " if c["ok"] else "ĐỎ  "
            tail = f"  ← {c['reason']}" if c["reason"] else ""
            say(f"  [{mark}] {c['var']} = {c['value']}{tail}")
    else:
        say("\nBiến CA trong môi trường: không đặt biến nào (bình thường).")

    if report["ca_broken"]:
        say("\nVÌ SAO `hf repo create` CHẾT:")
        say("  `httpx` đọc biến CA ở trên khi dựng SSL context. Biến có nhưng tệp")
        say("  không mở được ⇒ `FileNotFoundError: [Errno 2] ...` **không kèm tên tệp**,")
        say("  ở `ssl.py`/`_config.py` — traceback không nhắc tới biến nào.")
        say("\nSửa (chọn một):")
        say("  Git Bash   : " + FIXES["git-bash"])
        say("  cmd.exe    : " + FIXES["cmd"].replace("\n", " && "))
        say("  PowerShell : " + FIXES["powershell"])
        if report["certifi"]:
            say(f"  hoặc trỏ cả ba biến vào CA bundle có thật: {report['certifi']}")
        say("  (đặt biến trong phiên hiện tại là đủ; nếu nó nằm trong `.bashrc`/biến")
        say("   hệ thống thì sửa ở đó, nếu không lần sau mở terminal lại lỗi)")

    say("\nToken HF: " + (f"có ({report['token']})" if report["token"] else "KHÔNG THẤY"))
    say("git: " + (report["git"] or "KHÔNG THẤY") + "   ·   hf: " + (report["hf"] or "KHÔNG THẤY"))

    if report["warnings"]:
        say("\nCảnh báo:")
        for w in report["warnings"]:
            say(f"  - {w}")

    say("\nLệnh tạo Space (sau khi hết ĐỎ):")
    say(f"  hf repos create {report['space'] or '<user>/dire-demo'} --repo-type space --space-sdk gradio")
    say("  # `hf repo` (số ít) đã deprecated; `hf repos` là lệnh hiện hành.")
    say("\nKhông muốn dùng CLI: tạo bằng web UI (không đọc biến CA nào của máy):")
    say("  https://huggingface.co/new-space   → SDK: Gradio · Hardware: CPU basic")
    say("  rồi: git clone https://huggingface.co/spaces/<user>/dire-demo /tmp/space")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="kiểm máy trước khi tạo HF Space")
    ap.add_argument("--space", default="", help="<user>/dire-demo; trống thì chỉ kiểm máy")
    ap.add_argument("--json", action="store_true", help="in JSON thay cho văn bản")
    args = ap.parse_args(argv)

    report = build_report(args.space)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report))
    return 1 if report["blocked"] else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
