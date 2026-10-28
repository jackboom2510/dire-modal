#!/usr/bin/env bash
# TIỆN ÍCH — render video giải thích, không phải quy trình thực nghiệm.
#
#   bash tools/manim/render.sh                       # nháp 480p15, tất cả scene
#   bash tools/manim/render.sh --quality h           # 1080p60
#   bash tools/manim/render.sh --scene TrainMechanism
#   bash tools/manim/render.sh --check               # liệt kê scene, KHÔNG render
#   bash tools/manim/render.sh --gif                 # kèm GIF nhỏ để nhúng README
#
# Vì sao script này tồn tại (thay vì gõ `manim` tay):
#   * `manim` cần đúng 3 thứ dễ quên: PYTHONPATH tới `src/` (data.py đọc manifest),
#     thư mục output, và tên scene hợp lệ — gõ sai tên thì manim in một stack
#     trace dài thay vì "không có scene đó".
#   * Đường dẫn phải đổi cho Windows (xem khối "MSYS" ngay dưới).
#
# MSYS / Git Bash — hai chiều ngược nhau
# -------------------------------------
# Repo tắt path conversion (`MSYS_NO_PATHCONV=1`) để gửi nguyên văn `/raw` cho
# `modal`. Nhưng `manim` và `ffmpeg` là chương trình WINDOWS THẬT: đưa chúng
# `/d/Documents/Code/dire-modal/...` thì chúng đọc thành
# `D:\d\Documents\Code\...` — thêm một tầng thư mục không tồn tại:
#
#     FileNotFoundError: D:\d\Documents\Code\dire-modal\tools\manim\scenes.py not found
#
# Nên ở đây:
#   * file scene được truyền **tương đối** (`tools/manim/scenes.py`) — không có
#     tiền tố `/` nào để bị hiểu nhầm, và đúng cho mọi hệ điều hành;
#   * các đường dẫn tuyệt đối còn lại đi qua `native_path()` (`tools/lib/native_path.sh`)
#     ⇒ `D:/...` khi có `cygpath`, nguyên văn khi ở Linux/macOS.
set -euo pipefail

export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

# shellcheck source=tools/lib/native_path.sh
. "$HERE/../lib/native_path.sh"

# manim đặt tên thư mục đầu ra theo ĐỘ PHÂN GIẢI, không theo chữ cái `-q`:
# `-ql` → `480p15`, `-qh` → `1080p60`. Bản trước glob theo `"$QUALITY"` nên
# KHÔNG BAO GIỜ khớp ⇒ mp4 render xong vẫn nằm trong `_manim/`, script in "Xong"
# mà `docs/media/` trống (bẫy im lặng, phát hiện khi chạy thử với manim giả).
quality_dir() {
  case "$1" in
    l) printf '%s' "480p15" ;;
    m) printf '%s' "720p30" ;;
    h) printf '%s' "1080p60" ;;
    p) printf '%s' "1440p60" ;;
    k) printf '%s' "2160p60" ;;
    *) printf '%s' "" ;;
  esac
}

QUALITY="${QUALITY:-l}"
SCENE="${SCENE:-all}"
MEDIA_DIR="${MEDIA_DIR:-$ROOT/docs/media}"
GIF=0
CHECK=0

while [ $# -gt 0 ]; do
  case "$1" in
    --quality) QUALITY="${2:?--quality cần giá trị l|m|h|p|k}"; shift 2 ;;
    --scene)   SCENE="${2:?--scene cần tên scene}"; shift 2 ;;
    --gif)     GIF=1; shift ;;
    --check)   CHECK=1; shift ;;
    -h|--help)
      sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "[!] tham số lạ: $1" >&2; exit 2 ;;
  esac
done

# --- đường dẫn cho chương trình native -------------------------------------
# `HERE` nằm trong `ROOT` (thường lệ) ⇒ truyền tương đối. Symlink hoặc cài lạ thì
# quay về tuyệt đối đã đổi cho Windows.
if [ "${HERE#"$ROOT"/}" != "$HERE" ]; then
  SCENES_ARG="${HERE#"$ROOT"/}/scenes.py"
else
  SCENES_ARG="$(native_path "$HERE/scenes.py")"
fi
if [ ! -f "$SCENES_ARG" ] && [ ! -f "$HERE/scenes.py" ]; then
  echo "[!] không thấy scenes.py cạnh render.sh ($HERE)" >&2
  exit 1
fi

# --- danh sách scene: đọc từ source, không chép tay ------------------------
# Chép tay thì thêm scene mới là script lặng lẽ bỏ qua nó.
mapfile -t SCENES < <(grep -oE '^class ([A-Za-z_][A-Za-z0-9_]*)\(Scene\)' \
  "$HERE/scenes.py" | sed -E 's/^class ([A-Za-z0-9_]+)\(Scene\)/\1/')

if [ "${#SCENES[@]}" -eq 0 ]; then
  echo "[!] không tìm thấy scene nào trong $HERE/scenes.py" >&2
  exit 1
fi

if [ "$CHECK" = "1" ]; then
  printf '%s\n' "${SCENES[@]}"
  exit 0
fi

if [ "$SCENE" != "all" ]; then
  found=0
  for s in "${SCENES[@]}"; do [ "$s" = "$SCENE" ] && found=1; done
  if [ "$found" -eq 0 ]; then
    echo "[!] không có scene '$SCENE'. Có: ${SCENES[*]}" >&2
    exit 2
  fi
  SCENES=("$SCENE")
fi

# --- kiểm môi trường TRƯỚC khi render ---------------------------------------
if ! command -v manim >/dev/null 2>&1; then
  cat >&2 <<'EOF'
[!] Không thấy lệnh `manim`.

  Linux (Debian/Ubuntu):
    sudo apt install -y ffmpeg libcairo2-dev libpango1.0-dev pkg-config build-essential
    python3 -m pip install manim

  Windows: chỉ cần ffmpeg, phần còn lại có wheel sẵn
    winget install Gyan.FFmpeg
    py -m pip install manim
    # nếu vẫn đòi build cairo: conda install -c conda-forge manim

  Modal (tái lập được, không cần cài gì ở máy): xem tools/manim/README.md §3

Lưu ý: manim cần numpy>=2.1, còn image `cpu_image`/`gpu_image` của repo pin
numpy==1.26.4 cho torch. ĐỪNG cài manim vào hai image đó — phải là image riêng.
EOF
  exit 1
fi

mkdir -p "$MEDIA_DIR"
# Tuyệt đối hoá để `native_path` và các glob phía dưới cùng nhìn một chỗ.
MEDIA_DIR="$(cd "$MEDIA_DIR" && pwd)"
MEDIA_NATIVE="$(native_path "$MEDIA_DIR")"

# PYTHONPATH là BIẾN MÔI TRƯỜNG, MSYS không đổi hộ ⇒ phải tự đổi. Không đổi thì
# python.exe thấy `/d/.../src` (không tồn tại) và `import dire_modal` fail với
# thông điệp "No module named dire_modal" — trông như thiếu cài đặt.
SRC_NATIVE="$(native_path "$ROOT/src")"
export PYTHONPATH="$SRC_NATIVE${PYTHONPATH:+:$PYTHONPATH}"

# Trace log của TrainMechanism (nếu có) — scene tự tìm docs/media/demo_train_log.json
if [ ! -f "${DIRE_LOSS_LOG:-$MEDIA_DIR/demo_train_log.json}" ]; then
  echo "[i] chưa có log training ⇒ TrainMechanism sẽ hiện khung 'chưa có log'"
  echo "    (đúng ý: scene KHÔNG vẽ đường loss giả). Có log thì đặt ở:"
  echo "    $MEDIA_NATIVE/demo_train_log.json  hoặc đặt DIRE_LOSS_LOG=<path>"
fi

for s in "${SCENES[@]}"; do
  echo "[>] $s  (-q$QUALITY)  ← $SCENES_ARG"
  if ! manim -q"$QUALITY" --media_dir "$MEDIA_NATIVE/_manim" \
        --output_file "$s" "$SCENES_ARG" "$s"; then
    cat >&2 <<EOF
[!] manim không render được scene '$s'.

    Nếu log trên có 'FileNotFoundError: ...scenes.py not found' với đường dẫn
    lạ dạng 'D:\\d\\Documents\\...' (một tầng 'd' thừa) thì đó là MSYS đang
    không đổi đường dẫn hộ: kiểm \`command -v cygpath\` và
    \`bash -c '. tools/lib/native_path.sh; native_path "\$PWD"'\`.

    Đường dẫn đang dùng: $SCENES_ARG
EOF
    exit 1
  fi
done

# --- gom file ra docs/media/ ------------------------------------------------
shopt -s nullglob
QD="$(quality_dir "$QUALITY")"
MOVED=0
for s in "${SCENES[@]}"; do
  cand=()
  if [ -n "$QD" ]; then
    for f in "$MEDIA_DIR"/_manim/videos/*/"$QD"/*"$s"*.mp4; do
      cand+=("$f")
    done
  fi
  if [ "${#cand[@]}" -eq 0 ]; then
    # Dự phòng: manim đổi tên thư mục độ phân giải giữa các bản ⇒ tìm theo tên
    # scene, bỏ qua `partial_movie_files` (đó là mảnh dở, không phải video).
    while IFS= read -r f; do cand+=("$f"); done < <(
      find "$MEDIA_DIR/_manim" -type f -name "*$s*.mp4" \
        -not -path "*partial_movie_files*" 2>/dev/null
    )
  fi
  if [ "${#cand[@]}" -eq 0 ]; then
    echo "[!] không thấy mp4 của $s trong $MEDIA_NATIVE/_manim — bỏ qua" >&2
    continue
  fi
  newest="$(ls -t "${cand[@]}" | head -1)"
  mv -f "$newest" "$MEDIA_DIR/$s.mp4"
  MOVED=$((MOVED + 1))
  echo "[ok] $MEDIA_NATIVE/$s.mp4"
done
if [ "$MOVED" -eq 0 ]; then
  echo "[!] KHÔNG có mp4 nào được gom ra docs/media — manim có chạy xong không?" >&2
  exit 1
fi

if [ "$GIF" = "1" ]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[!] --gif cần ffmpeg" >&2; exit 1
  fi
  # GIF chỉ để nhúng README: 12 fps, rộng 640. Giữ dưới ~2 MB/scene.
  for s in "${SCENES[@]}"; do
    [ -f "$MEDIA_DIR/$s.mp4" ] || continue
    # `ffmpeg` cũng là binary Windows ⇒ đường dẫn phải ở dạng native.
    ffmpeg -loglevel error -y -i "$(native_path "$MEDIA_DIR/$s.mp4")" \
      -vf "fps=12,scale=640:-1:flags=lanczos" \
      "$(native_path "$MEDIA_DIR/$s.gif")"
    size=$(wc -c <"$MEDIA_DIR/$s.gif")
    echo "[gif] $s.gif  ($((size / 1024)) KiB)"
    if [ "$size" -gt 2000000 ]; then
      echo "      ⚠ >2 MB — cân nhắc cắt scene hoặc giảm fps trước khi commit" >&2
    fi
  done
fi

cat <<EOF

Xong. Còn lại:
  * xem nhanh: mở $MEDIA_NATIVE/<Scene>.mp4
  * MP4 1080p KHÔNG commit (vài chục MB); GIF nhỏ thì commit để nhúng README
  * dữ liệu cho TrainMechanism: docs/media/demo_train_log.json
EOF
