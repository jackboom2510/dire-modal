# ADR-0006 — Mức tái lập: "chấm lại" khác "tái lập"

- **Trạng thái**: Accepted (2026-11-16)
- **Liên quan**: `spec.md` §2.1 · `src/dire_core/provenance.py` ·
  `tests/test_provenance.py` · ADR-0004 (retrain để đối chiếu)

## Bối cảnh

Repo tự giới thiệu là "tái lập DIRE". Nhưng đường eval mặc định
(`scripts/01_repro.sh`) chạy **ResNet-50 của tác giả** trên **DIRE map của tác
giả** rồi so với **bảng của tác giả**. Cả ba thứ quyết định con số đều không do
mình tạo:

| Thành phần | Nguồn | Muốn tự làm |
|---|---|---|
| DIRE map `\|x − x̂\|` (`dire/**`) | tác giả tính, ADM-LSUN + DDIM S=20 | ADM 2,5 GB + GPU ⇒ $22–150 |
| Detector `*.pth` | tác giả train | FR-8 ⇒ $4–56 |
| Ảnh real + ảnh generator (`images/**`) | tác giả tải/sinh | generator + seed; 3 API đóng ⇒ không bao giờ |

Khớp `32/32` ô trong tình huống đó nghĩa là: **đường đo của mình** (Resize 256 →
CenterCrop 224 → Normalize, đọc `fc.weight`, logit → `P(fake)`, ACC/AP) khớp
đường đo của họ. Đó là điều cần biết — nếu harness sai thì mọi số về sau đều
sai — nhưng nó **không** là bằng chứng về phương pháp DIRE.

Điểm nguy hiểm: nếu tác giả tính DIRE map sai, kết quả của mình vẫn đẹp, **vì
mình dùng đúng map sai đó**. Loại sai này không có exception nào báo, và một
câu chữ quá tay trong README sẽ biến nó thành tuyên bố khoa học.

## Quyết định

1. **Ba trục độc lập, không phải một cái thang.** `dire_maps`, `detector`,
   `images`. Nhãn của một lần chạy là **phép hợp** các thành phần tự làm:
   `R0` · `R1` · `R2` · `R1+R2` · `R1+R2+R3`. Lý do không dùng "cấp 1/2/3":
   train lại detector được mà **không** cần tính lại map (FR-8 dùng map phát
   hành), nên xếp R2 dưới R1 là sai về bản chất.

2. **Nhãn suy ra từ cờ, không khai bằng tay.**
   `eval::main` nhận `--dire-maps-self` / `--detector-self` / `--images-self`,
   mặc định **False cả ba**. `Claim` là `frozen dataclass`; muốn có nhãn khác
   phải đi qua `claim()`.

3. **Nhãn đi vào artefact, không chỉ vào lời nói.** `run.json → reproduction`
   (nhãn, thành phần của ai, chứng minh gì, **không** chứng minh gì, còn thiếu
   gì để lên mức) + cột `data_source` trong `dire.csv`. Báo cáo in khối này
   **trước** bảng số.

4. **Từ ngữ bị ràng buộc.** R0 chỉ được nói "khớp số paper"; cấm "tái lập
   DIRE", "tái lập Table 2". `tests/test_provenance.py` kiểm cả README/spec/
   plan/runbook: không được có "tái lập 100%", và tiêu đề README phải mang
   nhãn R0/"chấm lại".

## Hệ quả

- `scripts/01_repro.sh` được gọi là **quy trình 1 — chấm lại (R0)**. Tên file
  giữ nguyên (`01_repro.sh`) để không phá lệnh đã có trong docs/CI.
- Mọi bảng eval in kèm mức; `report` cảnh báo nếu gộp nhiều mức trong một bảng.
- `run.json` cũ (trước ADR này) được đọc là `"?"` — "không ghi", **không** mặc
  định R0. Hai câu khác nhau: "đã ghi là R0" và "quên ghi".
- Cờ "tự làm" là **tự khai**: code không kiểm được "map này do mình tính". Vì
  thế banner in thêm dòng ⚠ khi có cờ bật, và `git_sha` + `tiers` vẫn nằm trong
  `run.json` để người khác đối chiếu.

## Phương án bị loại

| Phương án | Vì sao không |
|---|---|
| Giữ nguyên cách gọi "tái lập", thêm ghi chú ở cuối docs | Ghi chú cuối không đọc; tiêu đề mới là thứ được trích |
| Một con số "mức độ tái lập" 0–100 % | Gộp ba trục độc lập thành một số ⇒ không diễn đạt được "train lại được nhưng map vẫn của tác giả" |
| Ghi nhãn bằng tay trong `run.json` | Cùng loại lỗi với điền 0 khi thiếu số: tự khai thì không kiểm được |
| Bỏ hẳn đường R0, chỉ làm R1+R2 | $26–206 và hàng chục giờ GPU; R0 vẫn có giá trị riêng (kiểm harness) |

## Chi phí

$0 — ADR này chỉ thêm metadata + kiểm tra, không chạy model, không tải byte nào.
