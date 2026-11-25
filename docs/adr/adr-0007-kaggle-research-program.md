# ADR-0007 — Chương trình nghiên cứu hai track và chuyển giao chạy sang Kaggle

- **Trạng thái:** Accepted — khung phạm vi đã thống nhất; protocol chi tiết còn mở.
- **Ngày:** 2026-10-14
- **Liên quan:** [`GLOSSARY.md`](../../GLOSSARY.md), [`docs/plans/README.md`](../plans/README.md), [`docs/workflows/README.md`](../workflows/README.md), [`docs/plans/research/incremental-paper-subsets.md`](../plans/research/incremental-paper-subsets.md), [`platforms/kaggle/notebook.ipynb`](../../platforms/kaggle/notebook.ipynb), [`tools/kaggle_maps_inventory.py`](../../tools/kaggle_maps_inventory.py)

## Bối cảnh

Các run hiện có tập trung vào các subset và chưa hoàn tất mọi đối chứng của paper. Table 5 đã có runner cho REC, RGB, RGB_DIRE và DIRE, nhưng ledger hiện có trong repo mới ghi protocol/inputs, chưa có kết quả train hoàn chỉnh. Ảnh hưởng của residual tuyệt đối (`abs`) so với residual có dấu cũng cần được kiểm tra, không được suy ra trước từ metric của paper.

Song song, người dùng muốn mở rộng câu hỏi nghiên cứu: dùng miền ImageNet để train detector, khảo sát generator diffusion mới, thử ViT và kết hợp ResNet–ViT, đồng thời benchmark các generator được paper nhắc tới trên miền huấn luyện thống nhất. Các kết quả mới này không cùng loại với tái lập protocol paper.

Kaggle đã có notebook trong repo, nhưng notebook tự mô tả là smoke test/dev nhỏ và ghi các giới hạn phiên, GPU và persistence. Vì quota/quy định thực tế có thể thay đổi, cần kiểm tra lại trước khi xếp lịch run; chưa có bằng chứng rằng toàn bộ kế hoạch có thể chạy liên tục trong một phiên Kaggle.

Inventory theo manifest/check-in (chưa xác minh archive trên Kaggle hoặc số ảnh thực tế) cho thấy `dire/train/imagenet/` hiện có `real+ADM`; `sdv1` chỉ được khai báo ở val/test ImageNet, nên tuyệt đối không chuyển các sample đó sang train. Manifest cũng khai báo DIRE train maps cho CelebA-HQ `real+sdv2` và checkpoint `celebahq_sdv2.pth`, cùng test maps CelebA-HQ riêng. Người dùng chọn chuyển thí nghiệm train ngoài ADM sang CelebA-HQ-SD-v2 để tránh sinh ảnh ImageNet độc lập; đây là track/domain riêng, không thay thế ImageNet baseline `real+ADM`. Việc dùng map có sẵn có thể tránh tính DIRE mới nếu archive đầy đủ và provenance hợp lệ, nhưng cardinality, split IDs, khả năng validation và khả dụng trên Kaggle chưa được xác minh.

Manifest inventory do người dùng cung cấp (không được checkout truy cập trực tiếp) ghi 28.000 train real và 40.000 train SDv2; test có 1.000 real, 500 DALL-E2, 1.000 IF/Imagen, 100 Midjourney, 1.000 StarGAN và 1.000 SDv2. Tổng tám archive là 5.769.522.447 byte (~5,77 GB thập phân). Đây là counts trong archive, chưa phải số ảnh unique đã xác minh.

Cùng báo cáo cho thấy basename candidate `0`–`999` xuất hiện ở cả train/test SDv2 và có duplicate candidate basenames bên trong train archive. Người dùng xác nhận đây là chỉ số cục bộ, không phải ID ảnh nguồn; vì vậy overlap basename không phải bằng chứng leakage. Bản đầu của inventory tool chỉ so sánh member đầu tiên khi basename lặp, nên các cờ `map_bytes_identical=false` đó không bao quát mọi member. Tool đã được sửa trên nhánh hiện tại để tổng hợp mọi member cho từng basename và trả thêm paths/counts. Khi tạo validation subset, dùng full archive-relative member path làm locator ổn định thay vì basename; vẫn cần kiểm exact content/provenance trước khi tuyên bố train/test độc lập.

Counts test cũng cho thấy tỷ lệ fake khác nhau theo generator (DALL-E2 500/1.500, Imagen 1.000/2.000, Midjourney 100/1.100, StarGAN 1.000/2.000). Vì AP phụ thuộc prevalence, metric macro-AP đã thống nhất cần được chuẩn hóa theo prior chung hoặc diễn giải cùng một kiểm tra độ nhạy; chưa chọn lại metric hay protocol trong ADR.

Khảo sát source-level ban đầu về [official DiT](https://github.com/facebookresearch/DiT) cho thấy repo đã archived, code/weights ghi license CC-BY-NC, mô hình được mô tả là latent diffusion transformer với checkpoint class-conditional ImageNet ở 256/512. [`sample.py`](https://github.com/facebookresearch/DiT/blob/main/sample.py) tạo latent 4-channel ở `image_size/8`, nhận nhãn lớp và decode qua VAE. Trong khi đó `dire_zero` hiện dùng ADM U-Net trên ảnh 3-channel ở pixel space. Đây **không phải thay thế drop-in**; VAE, latent inversion/reconstruction và class-conditioning có thể đổi residual. Chưa có benchmark công bằng nào trong repo cho thấy DiT nhanh hơn ADM/U-Net.

## Quyết định

1. **Tổ chức thành hai track riêng.**
   - `paper-reproduction track`: hoàn tất đối chứng biểu diễn (REC, RGB, RGB_DIRE, DIRE), đánh giá residual tuyệt đối/có dấu và benchmark theo protocol paper.
   - `research-extension track`: các câu hỏi mới về ImageNet, generator holdout, ViT và ResNet–ViT.
   Baseline/protocol của track đối chiếu được khóa trước khi mở rộng kết luận từ track mới. Không gộp các metric của hai track thành một tuyên bố tái lập.

2. **Giữ miền dữ liệu tách biệt.** LSUN-Bedroom là track đối chiếu với các run/paper hiện có; benchmark các generator paper-listed trên miền detector đang train dùng cùng training domain nhưng sample IDs test tách khỏi train/validation. ImageNet là track riêng cho detector/đánh giá theo miền ImageNet. Không trộn sample giữa hai miền để tạo một split không phân biệt.

3. **Giữ generator mới làm holdout ở cấp họ; tách train ngoài ADM thành research-extension.** Toàn bộ một họ và biến thể liên quan được để ngoài train/validation, dành cho test generalization. Các sample IDs test luôn tách khỏi train/validation. Giữ ImageNet baseline `real+ADM` theo train maps hiện có. Người dùng chọn thí nghiệm ngoài ADM trên CelebA-HQ-SD-v2, tái sử dụng DIRE maps `real+sdv2` có sẵn và huấn luyện detector mới; không sinh ảnh hay tính DIRE lại cho nhánh này. Không dùng checkpoint `celebahq_sdv2.pth` thay cho detector mới. Validation được tách cố định ở mức 10% riêng trong mỗi train class (`real` và `sdv2`), dùng deterministic hash của full archive-relative member path; không dùng basename counter. Lock file hashes, hash salt/rounding rule và actual counts trước training. Detector mới dùng ResNet-50 khởi tạo ImageNet với checkpoint/hash được khóa trước run; ViT/fusion là so sánh kiến trúc riêng. Người dùng chọn giữ trọn các họ `dalle2`, `if` (Imagen), `midjourney`, `stargan` làm holdout và báo cáo riêng; sample IDs test mới của `sdv2` là seen-generator control riêng. Không trộn CelebA-HQ với ImageNet/LSUN; cần xác minh archive, provenance, split/sample IDs và family aliases trước compute.

4. **Dùng Kaggle làm môi trường thực thi mục tiêu, Modal làm fallback.** Người dùng chấp thuận chạy qua nhiều phiên Kaggle và đồng ý đưa maps lên một Kaggle Dataset riêng, versioned/immutable, mount read-only; kèm manifest hashes, counts/IDs và split/generator metadata. Chia job theo giai đoạn có thể resume, lưu checkpoint/ledger/output ra artifact bền vững riêng ngoài phiên Kaggle, và dùng Modal nếu giới hạn quota/phiên hoặc hardware chặn một stage. Trước khi chạy, xác minh quota/tài nguyên tài khoản; không xem con số ghi trong notebook cũ là bảo đảm hiện hành.

5. **Khảo sát DiT như một backbone diffusion thay thế U-Net trong research-extension.** Đây là hướng nghiên cứu được yêu cầu khảo sát, chưa phải quyết định đưa vào pipeline chính. Giữ ADM/U-Net làm mốc paper; mọi residual được tạo từ DiT là một biến thể mới, báo cáo riêng. Chỉ đề xuất pilot sau khi kiểm tra latent/VAE, conditioning, checkpoint/license và khả năng inversion/reconstruction phù hợp.

## Hệ quả

- Mọi bảng kết quả phải ghi track, training domain, representation, split/sample IDs, generator, kiến trúc, checkpoint, seed và trạng thái run.
- Train maps có 28k real/40k SDv2 theo inventory được cung cấp. Người dùng chọn dùng class-weighted loss để giữ toàn bộ mẫu nhưng cân bằng tổng đóng góp real/fake; class weights sẽ được tính từ counts/member IDs đã xác minh, không hardcode theo inventory chưa kiểm toán.
- Với bốn họ CelebA-HQ holdout, metric chính là macro-average AP theo họ, trong đó mỗi generator AP được tính với sample weights chuẩn hóa tổng khối lượng real/fake về prior 50:50. Báo cáo thêm AP thô theo prevalence archive, AUROC từng generator, counts real/fake và paired stratified bootstrap CI 95%: resample chung real test IDs qua bốn họ, resample fake IDs riêng theo họ, rồi tính lại weighted AP và macro-average. Seed variability (khi số seed được chốt) báo cáo riêng hoặc qua tầng bootstrap bổ sung. ACC tại threshold 0.5 là metric phụ; không chọn/tune threshold trên test.
- Kết quả trên ImageNet không thay thế baseline LSUN; benchmark generator paper trên cùng training domain không được reuse ảnh train làm test.
- Kết quả generator holdout chỉ có ý nghĩa nếu danh sách holdout và ranh giới train/validation được khóa trước khi chọn checkpoint/hyperparameter.
- Kaggle notebook hiện tại là điểm khởi đầu để smoke-test môi trường, chưa phải pipeline chuyển giao hoàn chỉnh. Phải có bước xác nhận dữ liệu/quota, kiểm tra resume, artifact persistence và smoke run trước các lượt dài.
- Người dùng có các map archives dạng tar riêng theo split/generator trên máy local. `tools/kaggle_maps_inventory.py` tạo manifest read-only gồm archive hashes, member counts/names và cảnh báo candidate-ID overlap; `--hash-members` tùy chọn kiểm byte trùng chính xác. Công cụ không trích xuất, sửa, upload dữ liệu hay gán validation split; chưa chạy trên archive thật.
- ADR này không khẳng định metric sẽ bằng paper hoặc đã có cải thiện; không có run/benchmark mới nào được thực hiện khi ghi ADR.

## Các quyết định còn mở

ADR chỉ khóa ranh giới chương trình, không khóa protocol chi tiết. Các quyết định sau phải được làm rõ trước khi viết lịch chạy hoặc khởi động compute:

- Split/sample IDs, số lượng theo lớp/domain và nguồn/giấy phép của các generator mới.
- Danh sách generator holdout và quy tắc phân nhóm họ/biến thể; giữ toàn bộ họ ngoài train/validation.
- Với thí nghiệm CelebA-HQ-SD-v2 đã chọn: người dùng cho biết các map archives dạng tar ở máy local gồm train `real+sdv2` và test maps (real, bốn họ holdout, SDv2 control). Chạy lại `tools/kaggle_maps_inventory.py` phiên bản đã sửa hoặc trích member paths từ manifest hiện có; giải nghĩa basename overlap/duplicate candidate IDs (không mặc định là leakage); xác minh archive/provenance, sample IDs/cardinality, quyền sử dụng; dùng deterministic hash trên full archive-relative member path (không dùng basename counter) để tạo validation IDs ở tỷ lệ 10%/class đã chốt; khóa hash salt/rounding và counts thực sau audit; tính class weights từ counts/member IDs đã kiểm tra theo weighted-loss; xác minh taxonomy/aliases, content overlap và ranh giới train/validation/test, giữ đủ biến thể của bốn họ holdout. Xác minh checkpoint/hash ImageNet init dùng cho ResNet-50, cùng seed/training budget. Đóng gói snapshot bất biến thành Kaggle Dataset riêng để mount qua nhiều phiên, rồi kiểm tra resume/artifact persistence. Checkpoint `celebahq_sdv2.pth` chỉ là tham chiếu/đối chiếu, không thay detector mới.
- Quy trình đối chiếu chính xác Table 5 và residual signed-vs-absolute, gồm preprocessing và cách kết hợp RGB_DIRE.
- ViT cụ thể, pretrained initialization, budget tuning và tiêu chí so sánh công bằng với ResNet; kiểu kết hợp ResNet–ViT (late score fusion hay feature-level fusion).
- DiT: xác định mục tiêu ưu tiên (latency/VRAM hay detector quality); lớp/nguồn checkpoint phù hợp; class labels nhất quán cho real và fake; VAE/latent protocol; cách đảo ngược và tái tạo; phép so latency/VRAM/quality với ADM trên cùng ảnh, cùng hardware và cùng protocol. Nếu không có nhãn lớp hoặc checkpoint phù hợp, hoãn pilot DiT thay vì đổi giao thức âm thầm.
- Số seed và training budget sẽ chốt sau preflight dữ liệu/tài nguyên trên Kaggle; người dùng yêu cầu chưa khóa các giá trị này trước preflight. Metric/CI đã khóa là macro-AP chuẩn hóa prior (chính), AP thô và AUROC per-generator, counts, paired stratified bootstrap CI 95%, ACC@0.5 (phụ). Sau preflight vẫn cần khóa checkpoint-selection/early-stopping, initialization hash và quy tắc dừng trước khi run.
- Dataset Kaggle bất biến, kiểm tra quyền sử dụng, dung lượng, quota thực tế, resume và export artifact.

## Phương án bị loại

| Phương án | Vì sao không chọn |
|---|---|
| Gộp ImageNet và LSUN vào một train/test pool | Làm mờ domain shift và không còn phép so sánh tách bạch với các run hiện có. |
| Dùng generator mới trong train rồi tuyên bố đây là unseen-generator test | Làm mất tính chất holdout và gây leakage trong câu hỏi generalization. |
| Chỉ tiếp tục subset reproduction, không có track mở rộng | Không trả lời các câu hỏi mới về miền ImageNet, generator mới và kiến trúc transformer mà người dùng muốn khảo sát. |
| Giả định Kaggle notebook hiện tại đã đủ cho training toàn diện | Notebook chỉ kiểm smoke path; giới hạn phiên, quota và persistence cần được giải quyết/test riêng. |

## Chi phí

Không phát sinh chi phí compute khi ghi quyết định này. Chi phí dữ liệu/train chỉ được lập sau khi hoàn tất các quyết định mở và preflight tài nguyên.
