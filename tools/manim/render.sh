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
#   * Git Bash (MSYS2) đổi `/raw` thành `C:/Program Files/Git/raw`; ở đây ta
#     truyền đường dẫn tuyệt đối nội bộ nên phải tắt conversion.
set -euo pipefail

export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

QUALITY="${QUALITY:-l}"
SCENE="${SCENE:-all}"
MEDIA_DIR="${MEDIA_DIR:-$ROOT/docs/media}"
GIF=0
CHECK=0

while [ $# -gt 0 ]; do
  case "$1" in
    --quality) QUALITY="${2:?--quality cần giá trị l|m|h|k}"; shift 2 ;;
    --scene)   SCENE="${2:?--scene cần tên scene}"; shift 2 ;;
    --gif)     GIF=1; shift ;;
    --check)   CHECK=1; shift ;;
    -h|--help)
      sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "[!] tham số lạ: $1" >&2; exit 2 ;;
  esac
done

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
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Trace log của TrainMechanism (nếu có) — scene tự tìm docs/media/demo_train_log.json
if [ ! -f "${DIRE_LOSS_LOG:-$MEDIA_DIR/demo_train_log.json}" ]; then
  echo "[i] chưa có log training ⇒ TrainMechanism sẽ hiện khung 'chưa có log'"
  echo "    (đúng ý: scene KHÔNG vẽ đường loss giả). Có log thì đặt ở:"
  echo "    $MEDIA_DIR/demo_train_log.json  hoặc đặt DIRE_LOSS_LOG=<path>"
fi

for s in "${SCENES[@]}"; do
  echo "[>] $s  (-q$QUALITY)"
  manim -q"$QUALITY" --media_dir "$MEDIA_DIR/_manim" \
        --output_file "$s" "$HERE/scenes.py" "$s"
done

# --- gom file ra docs/media/ ------------------------------------------------
shopt -s nullglob
for s in "${SCENES[@]}"; do
  for f in "$MEDIA_DIR"/_manim/videos/scenes/"$QUALITY"/*"$s"*.mp4; do
    mv -f "$f" "$MEDIA_DIR/$s.mp4"
    echo "[ok] $MEDIA_DIR/$s.mp4"
  done
done

if [ "$GIF" = "1" ]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[!] --gif cần ffmpeg" >&2; exit 1
  fi
  # GIF chỉ để nhúng README: 12 fps, rộng 640. Giữ dưới ~2 MB/scene.
  for s in "${SCENES[@]}"; do
    [ -f "$MEDIA_DIR/$s.mp4" ] || continue
    ffmpeg -loglevel error -y -i "$MEDIA_DIR/$s.mp4" \
      -vf "fps=12,scale=640:-1:flags=lanczos" "$MEDIA_DIR/$s.gif"
    size=$(wc -c <"$MEDIA_DIR/$s.gif")
    echo "[gif] $s.gif  ($((size / 1024)) KiB)"
    if [ "$size" -gt 2000000 ]; then
      echo "      ⚠ >2 MB — cân nhắc cắt scene hoặc giảm fps trước khi commit" >&2
    fi
  done
fi

cat <<EOF

Xong. Còn lại:
  * xem nhanh: mở $MEDIA_DIR/<Scene>.mp4
  * MP4 1080p KHÔNG commit (vài chục MB); GIF nhỏ thì commit để nhúng README
  * dữ liệu cho TrainMechanism: docs/media/demo_train_log.json
EOF
