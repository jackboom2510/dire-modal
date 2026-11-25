# Runbook: Table 4, Table 5 và Figure 4

**Trạng thái:** runner và cấu trúc artefact đã được thêm; chưa chạy GPU trong repo.  
**Nguyên tắc:** mọi đầu ra ở namespace mới `/zero/experiments/<experiment>/<run_id>`; không ghi đè `/zero/maps`, `/zero/rebuild/maps` hay các run cũ.

## Protocol được mã hóa

- **Table 4:** `S = 5, 10, 20, 50`; mỗi S được tính map riêng và train detector mới trên ADM. Chấm các cột `adm, ddpm, iddpm, pndm, sdv2`, với ngưỡng cố định `0.5` khi lập bảng ACC/AP. Generator tại LSUN-Bedroom test là `sdv2`, không phải `sdv1`.
- **Table 5:** train riêng trên ADM cho `REC`, `RGB`, `RGB_DIRE`, `DIRE`, tất cả dùng cùng nguồn ảnh/split và `S=20`. RGB/REC ở `[0,1]`; DIRE giữ thang `[0,2]`; `RGB_DIRE` nối kênh theo thứ tự RGB rồi DIRE (6 kênh).
- **Figure 4:** clean, Gaussian blur σ=`1,2,3`, JPEG quality=`65,30`. Mỗi condition bắt đầu từ ảnh nguồn sạch; DIRE được tính lại từ ảnh đã biến đổi. Report AP theo generator, không gộp các cột thành một điểm.
- Detector khởi tạo ImageNet ResNet-50, mặc định 400 epochs, batch 32, learning rate `1e-4`, crop 224. Checkpoint ImageNet phải do người vận hành cung cấp và được hash vào protocol.

**Chốt generator theo dữ liệu hiện có:** cột là `sdv2` từ `images/test/lsun_bedroom/sdv2.tar.gz`; không phải `sdv1` từ ImageNet hoặc archive DIRE đã xuất bản. Ledger và hình sẽ ghi `sdv2` rõ ràng. Nếu bảng tham chiếu bên ngoài yêu cầu SDv1, không diễn giải kết quả SDv2 thành kết quả SDv1.

## Preconditions — kiểm trước khi tốn GPU

1. `/zero/data/<split>/<generator>` phải có ảnh 256×256 và đã qua `checkdata`/manifest audit.
2. Table 4/5 yêu cầu `real, adm` ở train/val, cùng `real` và đủ năm generator ở test. Runner dừng trước bước tính nếu thiếu cell.
3. ImageNet ResNet-50 state dict (torchvision naming, có `conv1.weight`) phải nằm tại `/zero/ckpt/resnet50_imagenet.pth`. Có thể upload bằng:

   ```bash
   modal volume put dire-zero /path/to/resnet50-IMAGENET1K-V2.pth ckpt/resnet50_imagenet.pth
   ```

4. Chọn `run_id` mới riêng cho thí nghiệm; thư mục model đã có sẽ không bị ghi đè. Việc có map S=20 trong namespace khác không làm runner tự nhận chúng là map cho S khác.

> **Chưa phải tuyên bố tái hiện bit-for-bit.** Paper không mô tả đầy đủ cách chuẩn hóa/fusion cho `RGB & DIRE`, và phần văn bản không làm rõ thứ tự resize–perturbation hoặc có tính lại DIRE sau perturbation hay không. Mã hiện dùng nối kênh 6 chiều, perturb ảnh nguồn 256² trước bước crop detector, rồi tính lại DIRE. Mỗi lựa chọn được ghi trong metadata để có thể đối chiếu với repo/checkpoint tác giả trước khi gọi là exact reproduction.

## Table 4 — mỗi lệnh chạy một giá trị S

Dùng chung một `run_id`, nhưng gọi một lần cho từng S:

```bash
modal run -m dire_zero.modal_app::table4 --run-id table4-<id> --steps 5
modal run -m dire_zero.modal_app::table4 --run-id table4-<id> --steps 10
modal run -m dire_zero.modal_app::table4 --run-id table4-<id> --steps 20
modal run -m dire_zero.modal_app::table4 --run-id table4-<id> --steps 50
```

Mỗi S có thư mục riêng `experiments/table4/<id>/S<S>/`; ledger `table4.json` liệt kê `missing_steps` tới khi đủ bốn S. S=5/10/50 cần map riêng; tuyệt đối không dùng map S=20 cho các S đó. Với Table 4/5, map train/val chỉ cần `real+ADM`; các generator khác chỉ cần map ở test. Không cần tạo train map cho `iddpm`, `pndm`, `stylegan` nếu chúng không nằm trong train protocol.

### Tái sử dụng map S=20 đã có

Có thể truyền `--source-map-run-id` để Table 4 S=20 đọc DIRE map trực tiếp trong namespace rebuild, không copy và không ghi đè map cũ:

```bash
modal run -m dire_zero.modal_app::table4 --run-id table4-<id> --steps 20 --source-map-run-id rebuild-2026-10-14
```

Trước khi huấn luyện, runner khóa checkpoint ADM, manifest, `steps_mixed`, `_run_config.json` và đường dẫn nguồn trong `dire.json`. Nếu map nguồn chưa đủ mọi ảnh trong protocol hoặc provenance không khớp, runner dừng — không tự tính lại DIRE và không train trên tập con. **Inventory trước đó mới xác nhận mốc 1.000 ảnh mỗi ô** (ví dụ train/real và train/adm có 40.000 ảnh nhưng 1.000 map), nên cache hiện vẫn chưa đủ full-dataset. Trong protocol của run này, cột test là `sdv2` theo archive LSUN-Bedroom hiện có; không yêu cầu `sdv1` và không gán kết quả `sdv2` thành SDv1.

Khi đủ, tải ledger và dựng CSV/Markdown:

```bash
modal volume get dire-zero experiments/table4/table4-<id>/table4.json ./table4.json
python tools/experiment_tables.py --table4 ./table4.json --out ./paper-tables
```

## Table 5 — tạo inputs một lần, train bốn detector riêng

```bash
modal run -m dire_zero.modal_app::table5_prepare --run-id table5-<id>
modal run -m dire_zero.modal_app::table5_train --run-id table5-<id> --representation REC
modal run -m dire_zero.modal_app::table5_train --run-id table5-<id> --representation RGB
modal run -m dire_zero.modal_app::table5_train --run-id table5-<id> --representation RGB_DIRE
modal run -m dire_zero.modal_app::table5_train --run-id table5-<id> --representation DIRE
```

`table5_prepare` tính triplet một lần ở S=20 rồi lưu tensor riêng theo representation. Mỗi detector train chỉ trên ảnh thật + ADM; evaluator chấm từng generator test riêng. `table5.json` báo representation/cột còn thiếu. RGB+DIRE dùng stem ResNet 6 kênh; ImageNet conv1 được khởi tạo bằng cách lặp hai nhánh RGB và chia đôi trọng số, chi tiết này được ghi lại trong source và cần xác nhận theo implementation tác giả nếu yêu cầu đối chiếu chính xác.

Lưu ý: map `.npy` chỉ giữ `|RGB − reconstruction|`, không có reconstruction để tạo representation REC. Do đó chỉ inventory map cũ không đủ để dựng REC; nếu không có reconstruction đã lưu đúng protocol thì `table5_prepare` vẫn phải chạy phép tính diffusion để lấy reconstruction (không ghi vào namespace map cũ). Không gọi đây là “tái sử dụng hoàn toàn không tính DIRE”.

Tải ledger rồi kết xuất bảng:

```bash
modal volume get dire-zero experiments/table5/table5-<id>/table5.json ./table5.json
python tools/experiment_tables.py --table5 ./table5.json --out ./paper-tables
```

## Subset pilot và tăng dần theo mốc lồng nhau

Nếu muốn chạy thử trước khi xử lý toàn bộ dữ liệu, hãy tạo snapshot riêng **trước**
Table 4/5/Figure 4. Subset là index trỏ tới ảnh gốc trên Volume, không copy ảnh;
chi tiết lệnh và cách chạy tăng dần 2.000 → 4.000 ảnh nằm trong
[`incremental-paper-subsets.md`](plans/research/incremental-paper-subsets.md). Cùng `subset_id`,
seed và run ID được khóa xuyên suốt ba runner; mỗi cỡ subset có ledger/model riêng.
Cache tensor/DIRE Table 5 dùng chung trong run và chỉ bổ sung biểu diễn cho ảnh
mới. Kết quả pilot được gắn nhãn, và bảng chuẩn không chấp nhận pilot như kết quả
full-data; dùng `--allow-partial` chỉ để kiểm tra, không để trích dẫn kết quả.

## Figure 4 — dùng model RGB và DIRE đã train sạch

```bash
Chạy mỗi condition × generator như shard độc lập để timeout không làm mất các ô đã xong:

```bash
for c in clean blur_sigma1 blur_sigma2 blur_sigma3 jpeg_q65 jpeg_q30; do
  for g in adm ddpm iddpm pndm sdv2; do
    modal run -m dire_zero.modal_app::figure4_robustness --run-id table5-<id> --condition-name "$c" --generator-name "$g"
  done
done
modal volume get dire-zero experiments/figure4/table5-<id>/figure4.json ./figure4.json
python tools/figures.py f4-robustness --results ./figure4.json --out figs --label R1+R2
```

Runner ghi từng condition/generator vào shard bất biến, resume shard đã xong, và ledger `missing_cells` cho tới khi đủ 30 cell. JSON giữ AP, scores, labels và sample IDs theo condition × model × generator. Hình PDF/PNG có sidecar nguồn và cảnh báo protocol.

### Giới hạn hiện tại của Figure 4

Paper nêu CNNDetection* và PatchForensics* làm baseline RGB. Runner này hiện chỉ so hai detector ResNet-50 của dự án (RGB và DIRE), không nạp/tái tạo hai baseline nói trên; JSON và caption đánh dấu rõ, nên hình này **không** phải tái hiện đầy đủ Figure 4 của paper. Cũng không được diễn giải kết quả như “perfect performance” trước khi chạy và kiểm dữ liệu.

## Kiểm tra sau khi chạy

- `table4.json`: đủ `5,10,20,50`; mỗi S có năm cột `status=ran`.
- `table5.json`: đủ 4 representation; mỗi representation có năm generator test.
- `figure4.json`: đủ 6 condition × 2 detector × 5 generator; counts real/fake và sample IDs phải khớp.
- So sánh config, hash checkpoint, sample IDs và split trước khi dựng bảng/hình. Giữ score-level artefacts; không nhập số paper bằng tay.
