#!/usr/bin/env bash
# ============================================================================
# TÁI LẬP TỪ ĐẦU (R1+R2+R3) — KHÔNG đọc một byte nào từ dire/** của tác giả
# ============================================================================
# Khác `scripts/01_repro.sh`: bản đó nạp DIRE map ĐÃ PHÁT HÀNH rồi chấm bằng
# detector đã phát hành ⇒ đó là "chấm lại" (R0). Ở đây:
#
#   R3  tự sinh ảnh      ADM LSUN-Bedroom, seed mình chọn (gen.json ghi lại)
#   R1  tự tính DIRE map ADM + DDIM S=20, mốc {0,50,…,950}, NFE = 2S
#   R2  tự train detector ResNet-50, 1 logit + BCE, chọn epoch trên val
#
# Đặt ở `tools/` chứ không phải `scripts/`: quy ước repo là `scripts/` đúng
# HAI quy trình (có test canh), và đây là tiện ích vận hành.
#
#   bash tools/scratch_repro.sh                      # mặc định: in kế hoạch, không tốn tiền
#   bash tools/scratch_repro.sh --stage gates        # kiểm 9 cổng vàng
#   bash tools/scratch_repro.sh --stage fetch        # tải LSUN + checkpoint ADM (local)
#   bash tools/scratch_repro.sh --stage bench        # ĐO sec/step (bắt buộc trước GPU)
#   bash tools/scratch_repro.sh --stage gen          # sinh ảnh (GPU)
#   bash tools/scratch_repro.sh --stage dire         # tính DIRE map (GPU)
#   bash tools/scratch_repro.sh --stage train        # train detector (GPU)
#   bash tools/scratch_repro.sh --stage all          # fetch → bench → gen → dire → train
#
# Tải tay (không BaiduPCS-Go) rồi chấm lại / đối chiếu recons:
#   bash tools/scratch_repro.sh --stage list          # in danh sách + vị trí trong data/raw
#   bash tools/scratch_repro.sh --stage verify-data   # kiểm cấu trúc tar/pth đã tải
#   bash tools/scratch_repro.sh --stage push          # modal volume put từng file
#   bash tools/scratch_repro.sh --stage verify-recons # B: so recon của mình vs tác giả
#   bash tools/scratch_repro.sh --stage r0            # C: eval R0 trên dữ liệu đã đẩy
#
# Không chạy GPU khi cổng G8 (sec/step chưa đo) còn đỏ — đó là chỗ mọi ước
# lượng tiền trong repo đang dựa vào một giả định 0,35 s/step chưa kiểm.
# ============================================================================
set -euo pipefail

# Git Bash (MSYS2) đổi `/raw` → `C:/Program Files/Git/raw` trước khi Modal thấy.
# Đã nổ thật một lần với tools/tree.sh; bật vô điều kiện, Linux/macOS bỏ qua.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*' 

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="plan"
N_GEN=500          # số ảnh sinh
N_REAL=500         # số ảnh thật
SEED_BASE=1000     # seed gốc; train/test suy ra và ghi vào gen.json
DDIM_STEPS=20
EPOCHS=40
FORCE=0
NEED="a,gallery,e2min"   # nhóm dữ liệu cho tải tay (xem tools/manual_data.py list)
DEST="${DIRE_LOCAL_ROOT:-data/raw}"
PY_HELP=0
usage() { sed -n '3,34p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --n-gen) N_GEN="$2"; shift 2 ;;
    --n-real) N_REAL="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --steps) DDIM_STEPS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --need) NEED="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage ;;
    *) echo "tham số lạ: $1" >&2; exit 2 ;;
  esac
done

# ---- hằng số dùng chung -----------------------------------------------------
ADM_URL="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
LSUN_URL="http://dl.yf.io/lsun/objects/bedroom_val_lmdb.zip"
: "${DIRE_LOCAL_ROOT:?đặt DIRE_LOCAL_ROOT (vd. D:/dire-data), xem README}"
STATE="$DIRE_LOCAL_ROOT/scratch"
MANIFEST="$ROOT/manifests/scratch_sha256.tsv"
SEC_FILE="$STATE/sec_per_step.json"

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
die()  { printf '\033[31m[CHẶN]\033[0m %s\n' "$*" >&2; exit 3; }

# G0 ở dạng kiểm được ngay trong bash: không tham chiếu dire/ ở đâu cả.
g0_clean() {
  # Ghép mẫu từ hai mảnh để chính file này không tự khớp với nó.
  local pat; pat="$(printf '/raw/%s' dire)"
  local hit=0
  for f in "$ROOT/src/dire_modal/scratch_gen.py" "$ROOT/src/dire_modal/scratch_train.py"; do
    [[ -f "$f" ]] || continue
    if grep -nF -- "$pat" "$f" | grep -qv 'assert_no_published'; then hit=1; fi
  done
  return $hit
}

# ---- kế hoạch ---------------------------------------------------------------
plan() {
  say "KẾ HOẠCH TÁI LẬP TỪ ĐẦU (R1+R2+R3) — chưa tốn đồng nào"
  cat <<EOF

  dữ liệu thật   LSUN-Bedroom val        $N_REAL ảnh
  ảnh sinh       ADM (unconditional)     $N_GEN ảnh, seed base $SEED_BASE
  DIRE           DDIM S=$DDIM_STEPS, mốc cuối $((1000 - 1000 / DDIM_STEPS))
  NFE/ảnh        $((2 * DDIM_STEPS))   (đảo ngược $DDIM_STEPS + tái tạo $DDIM_STEPS)
  NFE tổng map   $((2 * DDIM_STEPS * (N_GEN + N_REAL)))
  detector       40 epoch tối đa, batch 64, Adam 1e-4, 1 logit + BCE

  Không nằm trong "scratch" và phải ghi rõ khi báo cáo:
    - trọng số diffusion (ADM) là của OpenAI, không phải của mình;
    - LSUN chỉ được dùng cho nghiên cứu;
    - kiến trúc ResNet-50 lấy từ torchvision.

  Bước tiếp theo:
    bash tools/scratch_repro.sh --stage gates     # kiểm cổng (rẻ, không GPU)
EOF
}

# ---- G0..G8 ----------------------------------------------------------------
gates() {
  say "KIỂM CỔNG VÀNG"
  local fail=0
  chk() { if [[ "$1" == "ok" ]]; then echo "  [xanh] $2"; else echo "  [ ĐỎ ] $2 — $3"; fail=1; fi; }

  if g0_clean; then chk ok "G0 không tham chiếu dire/**"; else chk no "G0 không tham chiếu dire/**" "có chuỗi dire/ trong code"; fi

  if [[ -f "$STATE/ckpt/adm_lsun.pt" ]]; then chk ok "G1 có checkpoint ADM"; else chk no "G1 có checkpoint ADM" "chạy --stage fetch"; fi

  local n_pin=0
  [[ -f "$MANIFEST" ]] && n_pin="$(grep -vc '^#' "$MANIFEST" || true)"
  if [[ "${n_pin:-0}" -gt 0 ]]; then
    chk ok "G1 sha256 đã ghim ($n_pin dòng)"
  else
    chk no "G1 sha256 đã ghim" "chạy --stage fetch rồi commit $MANIFEST"
  fi

  if [[ -f "$SEC_FILE" ]]; then
    chk ok "G8 sec/step đã ĐO ($(sed -n 's/.*"sec_per_step": *\([0-9.]*\).*/\1/p' "$SEC_FILE"))"
  else
    chk no "G8 sec/step đã ĐO" "chạy --stage bench (mọi ước lượng tiền đang dựa vào 0,35 giả định)"
  fi

  echo
  if [[ $fail -ne 0 && $FORCE -eq 0 ]]; then
    die "còn cổng đỏ. Sửa điều kiện trước khi bật GPU (hoặc --force nếu bạn biết mình đang bỏ qua gì)."
  fi
  [[ $fail -eq 0 ]] && say "TẤT CẢ CỔNG XANH"
}

# ---- tải dữ liệu ------------------------------------------------------------
fetch() {
  say "TẢI DỮ LIỆU (local) → $DIRE_LOCAL_ROOT"
  mkdir -p "$STATE/ckpt" "$STATE/lsun" "$(dirname "$MANIFEST")"

  pull() {  # url dest label
    local url="$1" dest="$2" label="$3"
    if [[ -s "$dest" ]]; then
      echo "  [có]    $label  $(du -h "$dest" | cut -f1)"
    else
      echo "  [tải]   $label  ← $url"
      curl -fL --retry 3 --retry-delay 5 -C - -o "$dest.part" "$url"
      mv "$dest.part" "$dest"
    fi
    local sha size
    sha="$(sha256sum "$dest" | cut -d' ' -f1)"
    size="$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest")"
    if [[ -f "$MANIFEST" ]] && grep -q "^$label" "$MANIFEST"; then
      local want; want="$(awk -F'\t' -v l="$label" '$1==l{print $3}' "$MANIFEST")"
      [[ "$want" == "$sha" ]] || die "sha256 lệch cho $label: manifest $want, thực tế $sha"
      echo "  [khớp]  $label  sha ${sha:0:16}…"
    else
      printf '%s\t%s\t%s\t%s\n' "$label" "$size" "$sha" "$dest" >> "$MANIFEST"
      echo "  [ghim]  $label  sha ${sha:0:16}…  (đã ghi $MANIFEST — commit dòng này)"
    fi
  }

  pull "$ADM_URL"  "$STATE/ckpt/adm_lsun.pt" "adm_lsun.pt"
  pull "$LSUN_URL" "$STATE/lsun/bedroom_val_lmdb.zip" "bedroom_val_lmdb.zip"

  if [[ ! -f "$STATE/lsun/bedroom_val_lmdb/data.mdb" ]]; then
    echo "  [giải nén] lmdb…"
    unzip -q -o "$STATE/lsun/bedroom_val_lmdb.zip" -d "$STATE/lsun"
  fi
  if [[ ! -d "$STATE/lsun/png" || -z "$(ls -A "$STATE/lsun/png" 2>/dev/null)" ]]; then
    python3 "$ROOT/tools/lsun_to_png.py" \
      --lmdb "$STATE/lsun/bedroom_val_lmdb" --out "$STATE/lsun/png" \
      --n "$N_REAL" --manifest "$ROOT/manifests/lsun_val.tsv" \
      || die "giải nén LMDB lỗi — cần: pip install lmdb pillow"
  fi
  say "XONG: checkpoint + $N_REAL ảnh thật"
  echo "Bước tiếp: đẩy lên Volume (chỉ chạy khi bạn đồng ý tốn thời gian tải):"
  echo "  modal volume put dire-raw \"$STATE/ckpt/adm_lsun.pt\" ckpt/adm_lsun.pt"
  echo "  modal volume put dire-raw \"$STATE/lsun/png\" scratch/real"
}

# ---- đo sec/step ------------------------------------------------------------
bench() {
  say "ĐO sec/step THẬT (G8) — 3 ảnh, NFE = $((2 * DDIM_STEPS * 3))"
  modal run -m dire_modal.scratch_gen::dire --limit 3 --steps "$DDIM_STEPS" \
    | tee "$STATE/bench.log"
  python3 - "$STATE/bench.log" "$SEC_FILE" <<'PY'
import json, re, sys
log, out = sys.argv[1], sys.argv[2]
text = open(log, encoding="utf-8", errors="replace").read()
m = re.search(r's_per_image"?\s*[:=]\s*([0-9.]+)', text)
if not m:
    raise SystemExit("[!] log không có s_per_image — sửa scratch_gen.dire để in số đo")
sec = float(m.group(1)) / (40)   # /NFE
json.dump({"sec_per_step": round(sec, 6), "source": "bench 3 ảnh",
           "nfe": 120, "raw": m.group(0)},
          open(out, "w"), ensure_ascii=False, indent=2)
print(f"[G8] sec/step = {sec:.4f}")
PY
}

# ---- các bước GPU -----------------------------------------------------------
gen()   { gates; say "SINH ẢNH (R3)"; modal run -m dire_modal.scratch_gen::generate \
            --n "$N_GEN" --seed-base "$SEED_BASE" --steps "$DDIM_STEPS"; }
dire()  { gates; say "TÍNH DIRE MAP (R1)"; modal run -m dire_modal.scratch_gen::dire \
            --limit "$N_GEN" --steps "$DDIM_STEPS"; }
train() { gates; say "TRAIN DETECTOR (R2)"; modal run -m dire_modal.scratch_train::train \
            --epochs "$EPOCHS"; }

# ---- tải tay: in danh sách / kiểm / đẩy lên Volume -------------------------
list_data() {
  say "DANH SÁCH TẢI TAY → $DEST"
  python3 "$ROOT/tools/manual_data.py" list --need "$NEED" --dest "$DEST"
}

verify_data() {
  say "KIỂM CHỨNG OFFLINE (không gọi mạng)"
  python3 "$ROOT/tools/manual_data.py" verify --need "$NEED" --dest "$DEST" --strict
}

push_data() {
  say "ĐẨY LÊN VOLUME dire-raw"
  python3 "$ROOT/tools/manual_data.py" push --need "$NEED" --dest "$DEST" --execute
}

# ---- B: đối chiếu reconstruction của mình với recons/ của tác giả ----------
verify_recons() {
  say "ĐỐI CHIẾU RECONSTRUCTION (cần images/ + recons/ + ADM trên /raw)"
  modal run -m dire_modal.scratch_verify::main --n "${RECON_N:-10}" --steps "$DDIM_STEPS"
}

# ---- C: chấm lại R0 trên dữ liệu vừa đẩy lên -------------------------------
r0() {
  say "CHẤM LẠI R0 (map + checkpoint của tác giả) — nhãn R0, KHÔNG phải tái lập"
  modal run -m dire_modal.eval::main --tiers "${R0_TIERS:-e2min}"
}

case "$STAGE" in
  plan)  plan ;;
  gates) gates ;;
  fetch) fetch ;;
  bench) bench ;;
  gen)   gen ;;
  dire)  dire ;;
  train) train ;;
  list)          list_data ;;
  verify-data)   verify_data ;;
  push)          push_data ;;
  verify-recons) verify_recons ;;
  r0)            verify_data; r0 ;;
  all)   fetch; bench; gen; dire; train
         say "XONG — đọc /ckpt/scratch/scratch_adm/train.json; nhãn được phép nói: R1+R2+R3" ;;
  *)     echo "stage lạ: $STAGE" >&2; usage ;;
esac
