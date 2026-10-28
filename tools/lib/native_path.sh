#!/usr/bin/env bash
# tools/lib/native_path.sh — đổi đường dẫn POSIX ↔ dạng mà chương trình WINDOWS đọc được.
#
# Vì sao cần file này
# -------------------
# Trên Git Bash (MSYS2), `pwd` trả về dạng POSIX:
#
#     /d/Documents/Code/dire-modal/tools/manim
#
# MSYS2 thường tự đổi chuỗi đó thành `D:\Documents\...` khi gọi một chương trình
# Windows thật (python.exe, ffmpeg.exe, manim.exe). NHƯNG repo này **tắt** việc
# đổi đó (`MSYS_NO_PATHCONV=1`, `MSYS2_ARG_CONV_EXCL='*'`) vì cần gửi nguyên văn
# đường dẫn container `/raw` cho `modal`.
#
# Hệ quả: mọi lệnh gọi chương trình Windows phải tự đổi đường dẫn. Không đổi thì
# Windows nhận nguyên chuỗi `/d/Documents/...` và hiểu là "ổ hiện tại + thư mục
# tên `d`" — đúng lỗi đã xảy ra khi render video:
#
#     FileNotFoundError: D:\d\Documents\Code\dire-modal\tools\manim\scenes.py not found
#                              ^^^ thêm một tầng thư mục không tồn tại
#
# Cách dùng
# ---------
#     . tools/lib/native_path.sh
#     manim --media_dir "$(native_path "$MEDIA_ABS")" ...
#
# `cygpath -m` trả về dạng "mixed": `D:/Documents/...` — dùng được cho CẢ chương
# trình Windows (manim, ffmpeg) VÀ cho bash (`mv`, `test -f`). Không có `cygpath`
# (Linux, macOS, container) thì hàm trả nguyên đầu vào: không đổi gì, không lỗi.
#
# ⚠ Giới hạn có chủ ý: KHÔNG dùng hàm này cho đối số dành cho **container Linux**
# (`modal run --path /raw`). Những đường dẫn đó phải giữ dạng `/raw`; đổi sang
# `D:/raw` là lỗi ngược lại, và `download._demangle()` tồn tại để chữa đúng lỗi đó.

# Đang chạy trong MSYS/Cygwin? (Linux/macOS ⇒ sai)
is_windows_shell() {
  case "$(uname -s 2>/dev/null || echo unknown)" in
    MINGW*|MSYS*|CYGWIN*) return 0 ;;
    *) return 1 ;;
  esac
}

native_path() {
  # Không tham số ⇒ trả rỗng thay vì treo chờ stdin.
  [ "$#" -ge 1 ] || { printf '%s' ""; return 1; }

  # Rỗng thì trả rỗng: "$(native_path "")" phải là "", không phải "/current/dir".
  [ -n "$1" ] || { printf '%s' ""; return 0; }

  if command -v cygpath >/dev/null 2>&1; then
    # `|| printf` : cygpath lỗi (đường dẫn lạ, quyền) thì vẫn trả đầu vào, để
    # lỗi thật hiện ra ở chỗ dùng chứ không bị nuốt ở đây.
    cygpath -m -- "$1" 2>/dev/null || printf '%s' "$1"
  else
    printf '%s' "$1"
  fi
}
