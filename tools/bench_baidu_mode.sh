#!/usr/bin/env bash
# TIỆN ÍCH — đo `--mode` của BaiduPCS-Go, không phải quy trình thực nghiệm.
# Đo `--mode` của BaiduPCS-Go trên MỘT file nhỏ: locate | pcs | stream.
#
#   bash tools/bench_baidu_mode.sh
#   FILE=dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz bash tools/bench_baidu_mode.sh
#
# Vì sao: ba mode là BA API khác nhau của Baidu với lịch sử throttle khác nhau.
# Nếu `pcs` nhanh gấp đôi `locate` thì cả bài toán đổi. Mỗi lần đo ~6 phút với
# file 26.57 MB ở 76.6 KiB/s, nên chạy trên file nhỏ, không trên `e2`.
#
# ⚠ CHỈ chạy khi tài khoản đang bình thường: đo trên tài khoản vừa bị throttle
# sẽ cho mọi mode cùng một số ≈ 0 và dẫn tới kết luận sai.
#
# Cách làm: sửa `baidu_mode` trong src/dire_modal/tiers.yaml rồi chạy
# `download::bench_concurrency` với đúng 1 file — script này in sẵn lệnh.
set -euo pipefail

FILE="${FILE:-dire/test/lsun_bedroom/lsun_bedroom/real.tar.gz}"
YAML="src/dire_modal/tiers.yaml"

for MODE in locate pcs stream; do
  printf '\n\033[1m== mode=%s ==\033[0m\n' "$MODE"
  python3 - "$MODE" "$YAML" <<'PY'
import re, sys, pathlib
mode, yaml_path = sys.argv[1], sys.argv[2]
p = pathlib.Path(yaml_path)
s = p.read_text(encoding="utf-8")
s = re.sub(r'^baidu_mode:.*$', f'baidu_mode: "{mode}"', s, flags=re.M)
p.write_text(s, encoding="utf-8")
print(f"[i] {yaml_path}: baidu_mode = {mode}")
PY
  modal run -m dire_modal.download::bench_concurrency \
    --paths "$FILE" --limit 1 --ks 1
done

printf '\n\033[1m== trả lại mặc định ==\033[0m\n'
python3 - "$YAML" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text(encoding="utf-8")
s = re.sub(r'^baidu_mode:.*$', 'baidu_mode: ""', s, flags=re.M)
p.write_text(s, encoding="utf-8")
print("[i] baidu_mode = \"\" (mặc định của client)")
PY
cat <<'EOF'

Cách đọc: so KB/s của 3 mode.
  * Một mode nhanh hơn rõ rệt (>1.3×)  → đặt nó làm mặc định trong tiers.yaml.
  * Cả ba như nhau                      → trần là của TÀI KHOẢN, không phải API.
                                          Đừng mất thêm thời gian; đi đường HF Hub
                                          (archive::push_hf + download::from_hf).
EOF
