# Giải thích Stage 10 và đối chiếu với kết quả DIRE (ICCV 2023)

## Tóm tắt

Report Stage 10 **do người dùng cung cấp** ghi nhận kết quả rất cao cho detector CelebA-HQ-SDv2 trên bốn generator holdout: macro balanced-prior AP trung bình theo ba seed là **99,934%** (SD quần thể giữa seed **0,0455 điểm phần trăm**). Nhưng ACC tại ngưỡng cố định 0,5 chỉ từ **72,5% đến 96,9%** theo họ; StarGAN và IF yếu hơn đáng kể. Điều này cho thấy khả năng xếp hạng rất tốt theo AP/AUROC nhưng quyết định nhị phân tại threshold 0,5 chưa ổn định/tốt tương ứng.

Đây **không phải** tái hiện kết quả tác giả: model được huấn luyện trên CelebA-HQ với SD-v2, test trên CelebA-HQ với bốn họ khác; paper huấn luyện theo protocol DiffusionForensics/LSUN-ADM và báo cáo tám generator khác. Chỉ có thể nói các con số Stage 10 cao về mặt số học; không thể kết luận chúng vượt hoặc tái hiện paper trong so sánh trực tiếp.

## 1. Stage 10 đang đo điều gì?

Theo report được lưu nguyên văn ở [`celebahq-stage10-evaluation.json`](celebahq-stage10-evaluation.json):

- Scope là **CelebA-HQ-SD-v2 research extension**, không phải paper-reproduction track.
- Bốn họ holdout chính: `dalle2`, `if` (Imagen), `midjourney`, `stargan`. `sdv2` là **seen-generator control**, không tính vào metric chính.
- Mỗi họ được so với cùng 1.000 ảnh thật; số fake thay đổi theo họ.
- Metric chính: macro-average AP giữa bốn họ, với sample weights đặt tổng trọng số real/fake thành 50:50 trong từng họ. Sau đó mỗi họ có trọng số bằng nhau trong macro.
- AP thô theo prevalence cũng được báo cáo. AUROC và ACC@0.5 là metric phụ.
- Report khai bootstrap percentile 95%, 2.000 replicate; các real IDs được resample chung giữa họ, fake IDs resample riêng từng họ. Các CI đó là theo từng seed; độ biến thiên giữa ba seed được báo riêng.
- Report khai `training_started=false`, `gpu_workload_started=true`, `image_contents_read=true`, `mounted_input_write_attempted=false`, và test không được dùng chọn checkpoint. Đây là **các khai báo trong report người dùng**, chưa được kiểm tra độc lập từ Kaggle/checkpoint/prediction files.

## 2. Kết quả Stage 10 được report

Bảng dưới là trung bình theo ba seed; tất cả đều là report do người dùng cung cấp, không phải kết quả agent xác minh trên Kaggle.

| Test family | Real / fake | Fake prevalence | Balanced-prior AP | Raw-prevalence AP | AUROC | ACC@0.5 trung bình | ACC@0.5 theo seed |
|---|---:|---:|---:|---:|---:|---:|---:|
| DALL·E 2 | 1.000 / 500 | 33,3% | 99,898% | 99,809% | 99,893% | 92,11% | 87,8–95,5% |
| IF / Imagen | 1.000 / 1.000 | 50,0% | 99,924% | 99,924% | 99,920% | 82,28% | 79,8–86,3% |
| Midjourney | 1.000 / 100 | 9,1% | 99,993% | 99,933% | 99,993% | 96,94% | 96,6–97,4% |
| StarGAN | 1.000 / 1.000 | 50,0% | 99,921% | 99,921% | 99,922% | 72,53% | 57,3–85,3% |
| SD-v2 control (seen) | 1.000 / 1.000 | 50,0% | 99,994% | 99,994% | 99,993% | 96,57% | 96,1–97,3% |

**Macro primary trên bốn holdout (không gồm SD-v2):**

| Seed | Macro balanced-prior AP | Reported 95% bootstrap CI |
|---|---:|---:|
| 3407 | 99,9110% | 99,8565–99,9553% |
| 3408 | 99,9975% | 99,9944–99,9996% |
| 3409 | 99,8935% | 99,8350–99,9411% |
| Mean | **99,9340%** | — |
| Population SD across seeds | **0,0455 percentage points** | — |

CI theo seed mô tả bootstrap lấy mẫu ảnh trong seed đó; SD đo biến thiên của ba điểm seed. Hai đại lượng không thay thế nhau. Report gốc có sáu upper endpoints ở một số CI per-family hơi lớn hơn 100% do sai số dấu phẩy động. Report nguồn được giữ nguyên; các endpoint đó không nên được diễn giải như xác suất hợp lệ.

### Vì sao AP rất cao nhưng ACC thấp hơn?

- **AP/AUROC đo thứ hạng trên nhiều ngưỡng.** Gần 100% nghĩa là fake thường được xếp cao hơn real trong các cặp so sánh.
- **ACC@0.5 dùng đúng một ngưỡng cố định.** Một ảnh có score sai phía của 0,5 bị tính sai, ngay cả khi thứ hạng tổng thể vẫn gần hoàn hảo. Vì threshold không được tune trên test, ACC thấp không bị che bằng điều chỉnh hậu nghiệm.
- Do đó, AP/AUROC cao cùng ACC thấp gợi ý ngưỡng 0,5 không phù hợp với score distribution của một số family/seed hoặc còn lỗi phân loại ở vùng ngưỡng. Chỉ từ aggregate metrics không thể phân biệt calibration shift với loại lỗi cụ thể; cần score-level prediction artifact để kiểm tra. Không nên hiệu chỉnh ngưỡng bằng test.
- Các dấu hiệu rõ nhất là IF: AP/AUROC ≈99,92% nhưng ACC trung bình 82,28%; StarGAN: AP/AUROC ≈99,92% nhưng ACC 72,53%, với StarGAN seed range 57,3–85,3%. Đây là lý do không nên tóm gọn Stage 10 thành “độ chính xác gần 100%”.
- Raw AP và balanced-prior AP khác nhiều nhất ở Midjourney vì fake chỉ chiếm 100/1.100 mẫu (9,1%); DALL·E 2 có fake prevalence 33,3%. Balanced prior làm AP các họ dễ so sánh hơn trong mục tiêu đã chọn. Với IF và StarGAN, prevalence là 50:50 nên hai AP gần như trùng nhau.

## 3. Kết quả paper của tác giả

Tham chiếu trong repo là [`src/dire_modal/paper_table2.tsv`](../../src/dire_modal/paper_table2.tsv), ghi rõ các số là của tác giả, Table 2, arXiv v1; không phải kết quả của dự án này. Dòng DIRE huấn luyện trên ADM, ACC/AP theo tám cột test của Table 2 là:

| Generator test trong paper | ADM | DDPM | iDDPM | PNDM | SD-v1 | SD-v2 | LDM | VQ-Diffusion | Trung bình |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **DIRE (ADM), ACC / AP** | 100/100 | 100/100 | 100/100 | 99,7/100 | 99,7/100 | 100/100 | 100/100 | 100/100 | **99,9/100** |

Các giá trị paper được in theo phần trăm và làm tròn một chữ số ở ACC trung bình; paper không đưa CI/đa seed cho dòng này trong bảng. Theo phần thiết lập paper được tổng hợp trong repo, đây là detector DIRE huấn luyện trên subset ADM của DiffusionForensics và đánh giá trên các test generator tương ứng trong benchmark đó. Có thêm khác biệt quy ước ở đúng điểm score `0,5`: code tác giả phân lớp bằng `score > 0.5`, còn Stage 10 helper dùng fake nếu `score >= 0.5` (và điều kiện bổ sung cho real). Chỉ mẫu score đúng bằng 0,5 bị ảnh hưởng, nhưng cần ghi nhận nếu so protocol chính xác.

Paper cũng có kết quả khác miền ở Table 3: detector train trên LSUN-ADM đạt **90,2% ACC / 97,9% AP** trên ADM ImageNet; và **97,2% ACC / 99,8% AP** trên SD-v1 ImageNet. Chúng cho thấy kết quả thay đổi theo miền/dataset và không thể suy rộng chỉ từ một điểm trung bình Table 2.

## 4. Đối chiếu đúng phạm vi — và giới hạn của phép so

| Thuộc tính | Stage 10 (report người dùng) | Paper DIRE Table 2 (số tác giả) |
|---|---|---|
| Track / mục tiêu | Research extension CelebA-HQ-SDv2 | Benchmark paper DIRE |
| Domain / split | Train CelebA-HQ real + SD-v2; test CelebA-HQ real và holdout generators | DiffusionForensics theo protocol paper, chủ yếu LSUN-Bedroom ở Table 2 |
| Generator train của detector | SD-v2 trên CelebA-HQ | ADM trên LSUN-Bedroom cho dòng headline DIRE (ADM) |
| Test families chính | DALL·E 2, IF, Midjourney, StarGAN; SD-v2 control riêng | ADM, DDPM, iDDPM, PNDM, SD-v1, SD-v2, LDM, VQ-Diffusion |
| Metric chính so trong báo cáo | Macro AP 50:50 prior giữa bốn family holdout | Bảng báo ACC/AP theo từng cột; headline average ACC 99,9%, AP 100% |
| Số lần chạy / bất định | 3 seed theo report; bootstrap CI ảnh theo seed và SD qua seed | Table 2 báo một giá trị tổng hợp/đã làm tròn; không có CI/đa seed ở bảng |

Vì khác domain, generator train, test families, prevalence và metric aggregation, **99,934% Stage 10 không phải phép đo cùng đại lượng với 99,9% ACC / 100% AP của paper**. Cũng không nên lấy 99,934 trừ 99,9 để kết luận hơn/kém. Một bên là macro AP cân prior trên bốn family CelebA-HQ holdout; bên kia là ACC/AP từng generator trên benchmark paper, với một detector train ADM-LSUN.

Điểm tương đồng duy nhất ở mức khái quát là cả hai báo cáo AP/AUROC rất cao trong những điều kiện riêng của chúng. Nhưng Stage 10 ACC thấp hơn đáng kể ở IF/StarGAN cho thấy thresholded classification chưa tương tự các ô headline của paper. Điều này không bác bỏ paper và cũng không xác nhận paper; nó là một kết quả extension cần audit score distribution, provenance, split separation và artifacts trước khi diễn giải rộng.

## 5. Provenance và cách trích dẫn an toàn

- Stage 10 values: “Theo report evaluation do người dùng cung cấp; chưa independently verified against Kaggle artifacts.” Archive: `docs/reports/celebahq-stage10-evaluation.json`, SHA-256 `191b40ad405fbe064f9a0863d48f3f644db597d0475d003b869166290502b833`.
- Paper values: “Số của tác giả DIRE (ICCV 2023), Table 2; không phải số của Stage 10.” Repo giữ tham chiếu tại `src/dire_modal/paper_table2.tsv`.
- Audit P0 hiện vẫn yêu cầu xác minh PDF/revision và supplementary material với nguồn authoritative; vì vậy các con số paper ở đây là đối chiếu theo reference table đã lưu trong repo, không phải xác nhận mới từ artifact gốc của publisher.
- Không gọi Stage 10 là tái hiện paper, không gọi ACC@0.5 là gần 100%, không dùng test để tune threshold/checkpoint, và không gộp SD-v2 control vào macro holdout.
