# Chạy Table 4, Table 5 và Figure 4 theo subset tăng dần

Subset là **manifest/index riêng**, không phải bản sao ảnh. Ảnh nguồn trong
`/zero/data` không bị sửa hay copy. `::subset` tạo snapshot bất biến dưới
`/zero/subsets/<subset_id>/per-cell-<n>.json`.

`per_cell` là số ảnh tối đa cho mỗi cell `(split, generator)`. Với ma trận hiện
tại có 10 cell cần cho Table 4/5 — train real/ADM, val real/ADM, test real và
ADM/DDPM/iDDPM/PNDM/SDv2 — `--per-cell 200` tạo 2.000 ảnh. Chọn theo SHA-256
xếp hạng ổn định trong từng cell; cùng `subset_id` và seed thì mốc 400 chứa đủ
mẫu của mốc 200. Cell có ít hơn yêu cầu giữ toàn bộ mẫu.

## Tạo snapshot trước khi GPU

```sh
modal run -m dire_zero.modal_app::subset \
  --subset-id lsun-pilot --per-cell 200 --seed 3407
```

Mở rộng cùng subset, không ghi đè snapshot trước:

```sh
modal run -m dire_zero.modal_app::subset \
  --subset-id lsun-pilot --per-cell 400 --seed 3407
```

Dùng lại cùng `subset_id`, seed và `run_id` qua các mốc để cache DIRE được tái
sử dụng. Mỗi mốc train detector độc lập và lưu ledger riêng; không tiếp tục
weights của mốc nhỏ hơn như thể đó là run cùng protocol.

## Table 4

```sh
T4_RUN=table4-lsun-pilot
for n in 200 400; do
  for s in 5 10 20 50; do
    modal run -m dire_zero.modal_app::table4 \
      --run-id "$T4_RUN" --steps "$s" \
      --subset-id lsun-pilot --per-cell "$n"
  done
done
```

Mỗi S có cache DIRE riêng; tăng `per_cell` trong cùng run sẽ chỉ bổ sung ảnh
mới vào cache S tương ứng. Ledger của mốc nằm tại
`experiments/table4/<run_id>/subsets/per-cell-<n>/table4.json`.

## Table 5 và Figure 4

```sh
T5_RUN=table5-lsun-pilot
for n in 200 400; do
  modal run -m dire_zero.modal_app::table5_prepare \
    --run-id "$T5_RUN" --subset-id lsun-pilot --per-cell "$n"
  for rep in REC RGB RGB_DIRE DIRE; do
    modal run -m dire_zero.modal_app::table5_train \
      --run-id "$T5_RUN" --representation "$rep" \
      --subset-id lsun-pilot --per-cell "$n"
  done
  for condition in clean blur_sigma1 blur_sigma2 blur_sigma3 jpeg_q65 jpeg_q30; do
    for generator in adm ddpm iddpm pndm sdv2; do
      modal run -m dire_zero.modal_app::figure4_robustness \
        --run-id "$T5_RUN" --condition-name "$condition" \
        --generator-name "$generator" \
        --subset-id lsun-pilot --per-cell "$n"
    done
  done
done
```

Table 5 chuẩn bị inputs và cache tăng dần trong cùng run; model/ledger tách theo
mốc. Figure 4 dùng detector đã train đúng mốc subset. Đầu ra của mỗi mốc nằm
trong thư mục `subsets/per-cell-<n>/` và ghi rõ `subset_status`; chỉ mốc có toàn
bộ số hàng nguồn mới được đánh dấu `full-data`. Các mốc nhỏ hơn là pilot, không
được trích dẫn như kết quả đầy đủ của paper.

Để bao phủ toàn bộ ma trận hiện tại, dùng `--per-cell 40000` (các cell nhỏ hơn
được lấy trọn). Các run đầy đủ vẫn cần hoàn thành đủ S/representation/condition,
đúng checkpoint và protocol; subset không làm giảm yêu cầu đó.
