# Kế hoạch tái hiện đầy đủ thực nghiệm DIRE (ICCV 2023)

> **Trạng thái:** kế hoạch nghiên cứu, chưa chạy thực nghiệm quy mô đầy đủ. Ở lượt
> này không khởi động GPU, không tải dữ liệu và không tạo số liệu. Mục tiêu là thay
> cách nhìn "demo với trần thời gian/tiền" bằng một chương trình tái hiện có thể
> chạy đến hết, giữ nguyên chuẩn mực về nguồn gốc và phân tích được sai khác.
>
> **Ranh giới khoa học:** "bỏ giới hạn thời gian và chi phí" có nghĩa là không
> cắt số ảnh/epoch, không dừng vì trần USD của demo và không đổi giao thức để rẻ
> hơn. Nó **không** có nghĩa là bỏ cổng kiểm tra, dùng test để chọn ngưỡng, bỏ qua
> giấy phép, hoặc gọi một thí nghiệm gần giống là tái hiện chính xác. Tài nguyên
> chỉ được tiêu khi có xác nhận riêng cho giai đoạn chạy và tài khoản/hardware đã
> được cấu hình; tài liệu này tự nó không khởi chạy gì.

## 1. Mục tiêu và định nghĩa hoàn thành

Mục tiêu là tái hiện **các thực nghiệm được báo cáo trong bài DIRE**, không chỉ
một đường `ADM → DIRE → detector`: dựng dữ liệu/split, tạo hoặc xác nhận các
triplet (ảnh nguồn, reconstruction, DIRE), huấn luyện detector cho từng hàng
thử nghiệm, chạy các bảng tổng quát hoá và ablation, đo độ bền với biến đổi ảnh,
và sinh báo cáo/artefact có thể truy nguyên.

Hoàn thành khi:

1. Mọi hàng/cột của bảng thực nghiệm và hình có số liệu trong paper được ánh xạ
   tới một cấu hình máy đọc được, dữ liệu đã kiểm kê, và một lệnh chạy lại.
2. Có dữ liệu đủ đúng split và số lượng theo giao thức; dữ liệu phát hành được
   đối chiếu với hash/count, dữ liệu tự sinh có checkpoint, cấu hình, seed và
   metadata đầy đủ.
3. Các model/DIRE map được tạo trong clean-room (R1/R2) tách biệt khỏi artefact
   phát hành của tác giả (R0). Không trộn các nguồn trong cùng một con số.
4. Mọi run lưu cấu hình hiệu dụng, commit, environment, nguồn/sha256, seed, split,
   checkpoint cha, log, metric từng ảnh và artefact trung gian; chạy lại độc lập
   cho kết quả trong sai số số học đã định trước.
5. Báo cáo cuối có đầy đủ bảng, hình, khoảng tin cậy, sai khác so paper, và mục
   **Hạn chế**. Không dùng câu "chính xác 99 %"; số paper luôn chú thích
   *"số của paper (ICCV 2023), không phải của demo này"*.

Đây là tái hiện thực nghiệm, không phải cam kết bitwise-identical: CUDA/cuDNN,
GPU, thư viện, thứ tự phép toán và nguồn ảnh có thể làm kết quả khác. Sai khác
phải được đo và giải thích, không được xoá bằng cách chọn seed thuận lợi.

## 2. Hai đích song song — không đánh đồng

### A. Đối chiếu với thực nghiệm gốc

Dùng đúng các split, loại dữ liệu, sampler, số bước, preprocessing, kiến trúc,
khởi tạo, optimizer, epoch/early stopping và metric ghi trong paper + repo nguồn
`ZhendongWang6/DIRE`. Reuse dữ liệu/weight phát hành chỉ ở các hàng được paper
phép dùng; ghi nguồn rõ. Đây là đường kiểm xem giao thức mình triển khai có tái
hiện được kết quả báo cáo hay không.

### B. Chương trình clean-room của repo này

Không copy/import `third_party/DIRE`, `guided_diffusion` hay
`src/dire_modal/ddim.py`. Dùng implementation `dire_zero` đã viết độc lập; trọng
số ADM công khai được phép dùng. Với mỗi đường chạy, gắn provenance theo đúng
thành phần thực sự tự tạo:

| Nhãn | Thành phần tự làm | Có thể tuyên bố |
|---|---|---|
| R0 | không có; map, detector và ảnh đều phát hành | chấm lại / kiểm harness, không phải tái hiện |
| R1 | tự tính DIRE map | map do implementation này tính |
| R2 | tự train detector | detector do implementation này train |
| R1+R2 | tự tính map và tự train detector; ảnh nguồn vẫn phát hành | tái hiện một phần có nêu rõ nguồn ảnh |
| R1+R2+R3 | tự tính map, tự train detector, tự sinh ảnh | toàn bộ các thành phần của pipeline được dựng lại, nhưng vẫn không tự train ADM trừ khi chạy gói tùy chọn dưới đây |

Không ghi nhãn R3 cho ảnh có `fake_source: author`. `ours_adm` nếu được tạo bằng
ADM công khai là **in-distribution**; không được diễn giải là kiểm tra tổng quát
hoá sang generator độc lập.

### C. Huấn luyện generator từ đầu (tuỳ chọn mở rộng, không thay đường A)

Có thể train ADM hoặc generator khác từ đầu để nghiên cứu độ phụ thuộc vào nguồn
checkpoint và tạo thêm mẫu. Đây là thí nghiệm mới, không phải điều kiện để tái
hiện paper nếu paper dùng checkpoint generator phát hành. Ghi riêng: dữ liệu,
license, cấu hình, số step, FLOP/GPU, checkpoint, seed, FID/ảnh kiểm tra và
provenance. Không gộp kết quả này vào bảng tái hiện paper.

## 3. Ma trận thực nghiệm cần đóng

Bảng dưới là phạm vi cần truy nguyên từ paper/repo trước khi chạy; mọi ô phải có
trạng thái `planned / data-ready / ran / audited / blocked`, không điền số giả.
Các tên thử nghiệm/chi tiết cuối cùng được xác nhận từ đúng phiên bản paper và
repo đã ghim, không suy diễn từ bảng tóm tắt trong tài liệu nội bộ.

| Nhóm | Thực nghiệm | Đầu ra cần có |
|---|---|---|
| M0 | Kiểm tra pipeline / đối chiếu R0 | So sánh preprocessing, điểm ảnh, metric và artefact phát hành; phát hiện sai trước khi tốn GPU |
| M1 | Kết quả chính: train detector trên từng training generator, test trên từng generator trong cùng tập/domain | Toàn bộ hàng × cột của bảng tổng quát hoá chính (paper Table 2); mỗi model riêng, không gộp generator thành lớp giả chung |
| M2 | Cross-dataset / cross-domain | Các cấu hình train LSUN ↔ ImageNet và các cặp paper nêu (Table 3); split/nhãn/nguồn ảnh tách biệt |
| M3 | Số bước DDIM `S` | **Có runner, chưa chạy:** `::table4 --steps 5,10,20,50` (mỗi lần một S); map riêng và detector riêng; ledger báo ô còn thiếu (Table 4) |
| M4 | Biểu diễn đầu vào | **Có runner, chưa chạy:** `::table5_prepare` + `::table5_train` cho REC/RGB/RGB_DIRE/DIRE trên cùng split (Table 5); exact fusion cần xác nhận với mã nguồn tác giả |
| M5 | Trị tuyệt đối | So `x − x̂` với `|x − x̂|` cùng dữ liệu/model/protocol (Table 6) |
| M6 | Robustness | **Có runner, chưa chạy:** Figure 4 conditions clean/blur σ=1,2,3/JPEG q=65,30, AP và sample scores; CNNDetection*/PatchForensics* baselines vẫn thiếu |
| M7 | Kiểm tra/ablation bổ sung của paper | Đọc danh mục bảng/hình trong paper và repo đã ghim; đưa từng protocol vào ma trận trước khi bắt đầu lượt đầy đủ |
| X1 | Generator mới / tự train | Chỉ là mở rộng; bảng riêng, không thay thế generator gốc hoặc ô thiếu trong M1–M7 |

**Tính toàn vẹn của bảng:** detector được train riêng cho từng generator/hàng
mà paper yêu cầu. Các generator chỉ có test split không được đưa vào train.
Ngưỡng chỉ chọn trên validation; test không được dùng để chọn threshold,
epoch, seed, hyperparameter hay checkpoint. `ours_adm` in-distribution phải
được ghi ngay trong tên/config/báo cáo.

## 4. Dữ liệu: kiểm kê trước khi chạy

Nguồn làm việc đã nêu là ổ `F:` với cây `DiffusionForensics`; root cần truyền qua
`DIRE_LOCAL_ROOT` (hỗ trợ `F:/…`, `F:\…`, `/f/…`). Không tải lại dữ liệu nếu đã
có; không xoá byte đã tải; chỉ bỏ `.staging` sau khi bản đã được xác minh trên
Volume. Bước này **chưa xác nhận** rằng mọi tệp paper đều hiện có trên máy.

### Cổng dữ liệu D0 — inventory có thể tái lập

- Chạy inventory trên ổ cục bộ; liệt kê archive, bytes, file/member, split,
generator, kích thước/shape và hash. Không kết luận thiếu dữ liệu từ danh sách
bị cắt hoặc chỉ từ kích thước archive.
- Đối chiếu với Table 1 và các supplementary material của paper, cùng layout
  phát hành trong repo tác giả. Tạo `manifests/paper_data_matrix.tsv` với một
  dòng cho mỗi `split × domain × generator × kind(images/recons/DIRE)`, trường
  `expected_count`, `observed_count`, `source`, `sha256`, `license`, `status`.
  Không biết số lượng/nguồn thì `null` + lý do, không `0`.
- Ghi sai khác chính tả/alias (ví dụ `stylegan_official`, typo archive) trong
  bảng alias có kiểm thử; không tự sửa tên nguồn mà không để lại ánh xạ.
- Xác nhận split bằng ID/hash ảnh, tránh trùng ảnh real giữa train/val/test.
  Cố định chính xác sampling/prompt nếu paper chỉ công bố một phần dữ liệu.
- Tạo mirror Volume sau khi kiểm hash; `F:` vẫn là nguồn local, không đổi hay
  xoá dữ liệu tải tay. Tệp >4 GiB không đi qua FAT32.
- Rà license của LSUN, ImageNet, generator weights, ảnh và dịch vụ trước khi
  mirror/chia sẻ. Không công khai dữ liệu có giới hạn nghiên cứu.

**Exit D0:** bảng đủ căn cứ cho toàn bộ ma trận; mọi ô thiếu có lý do, người chịu
trách nhiệm và phương án (tự tái sinh đúng protocol / xin nguồn / đánh dấu không
thể tái hiện). Không chạy model trước exit.

## 5. Quy trình vận hành theo giai đoạn

### P0 — Ghim nguồn và hợp đồng khoa học

1. Ghim commit/tag của paper PDF, mã nguồn tác giả, checkpoint public, config,
   dataset release và commit repo này. Lưu sha256, URL/citation, ngày lấy.
2. Đọc từng bảng/hình, trích protocol vào config YAML và `paper_data_matrix`;
   reviewer thứ hai đối chiếu độc lập.
3. Chuyển mọi giá trị chưa công bố sang `null` kèm lý do; không điền theo suy
   đoán. Chỗ paper và code bất đồng được ghi thành câu hỏi nghiên cứu, không
   âm thầm chọn một bên.

### P1 — Hoàn tất kiểm kê dữ liệu (D0)

Thực hiện mục 4, sau đó đưa dữ liệu đã xác minh lên storage phù hợp. Không fetch
lại nếu bytes đã có. Ghi inventory trước/sau upload và xác nhận sha256 từ Volume.

### P2 — Chốt clean-room implementations và scientific self-check

1. Xác nhận `dire_zero` độc lập: schedule, U-Net, checkpoint mapping strict,
   inversion/reconstruction, NFE, normalization, map scale, save/load.
2. Chạy `selfcheck` trên GPU thật với checkpoint ADM; kiểm key/shape, NaN/Inf,
   NFE `2S`, deterministic seed, một ảnh end-to-end, compare `x0`, `x̂0`, DIRE
   map với reference implementation (reference chỉ dùng như oracle kiểm thử,
   không import/copy vào sản phẩm).
3. So sánh map pixel-wise trên curated set từ nhiều domain/generator; báo MAE,
   max error, correlation, sai số theo channel, dynamic range và sai khác do
   precision. Sai khác không được che bằng ảnh PNG.
4. Chốt transformation đúng cho RGB, resize/crop, range `[0,2]`; `.npy` là nguồn
   chuẩn, PNG chỉ là visualization.

**Exit P2:** test GPU thật qua; không còn sai lệch chưa giải thích; checkpoint,
software, precision, hardware và seed được ghi.

### P3 — Khôi phục triplet và tạo DIRE maps

- Với tất cả ảnh phát hành có source/reconstruction/DIRE: xác thực basename,
  count, split, shapes, channel order, dtype/range và SHA; đối chiếu lại công
  thức từ subset. Không chấm map phát hành như R1.
- Tính lại map từ ảnh nguồn bằng ADM-LSUN public checkpoint và DDIM `S=20` theo
  protocol paper; cả inversion và reconstruction dùng cùng lịch; lưu đúng
  `[0,2]`, 3 kênh. Dùng pipeline restart/resume idempotent, trạng thái theo từng
  sample, checksum; lỗi một shard không buộc chạy lại shard đã xác minh.
- Cache map riêng theo `image_hash + checkpoint_hash + sampler + S + config`
  để không nhầm các ablation. Không dùng `--skip-existing` nếu metadata key
  không trùng tuyệt đối.
- Chạy số bước khác như các cache khác nhau; dữ liệu crop/resize chỉ áp dụng ở
  đúng công đoạn theo paper.

**Exit P3:** số lượng map khớp inventory từng split/generator; không có file mồ
côi, duplicate ID hoặc map thiếu metadata.

### P4 — Train detector đầy đủ

- Dùng ResNet-50 / ImageNet initialization và preprocessing/augmentation/optimizer,
  batch, learning rate, epoch/early-stopping đúng repo paper đã ghim.
- Train mỗi detector trên đúng generator và split của paper; validation mỗi
  epoch; chọn best checkpoint theo validation như protocol. Test chạy sau khi
  config/checkpoint đã đóng băng.
- Ghi loss, val metrics từng epoch, checkpoint mỗi mốc, optimizer state, seed,
  số sample/step, learning rate, thời gian thực, GPU, phiên bản thư viện và
  hash map. Hỗ trợ resume chính xác.
- Tạo hai lượt độc lập với seed đã định trước khi paper không quy định seed;
  không chọn seed đẹp. Nếu paper chỉ có một seed, báo rõ lượt bổ sung là kiểm
  độ nhạy, không thay số chính.

### P5 — Evaluate nguyên ma trận, một lần test

- Đóng băng model/hyperparameter trước khi chấm test. Chạy toàn bộ test row ×
  generator column, domain, table/ablation.
- Lưu `scores.jsonl`/CSV theo sample ID, label, score/logit, model/config hash,
  split và provenance; metric aggregate tính lại được từ điểm từng ảnh.
- Tính ACC tại threshold đúng quy ước paper, AP, AUC, balanced accuracy nếu cần,
  confusion matrix, CI 95% cho đúng estimator/threshold; ghi `null` nếu metric
  không định nghĩa được. Threshold chỉ từ val; không hiện `p_fake` ở UI khi
  `gate != passed`.
- Bootstrap lấy mẫu ở đúng đơn vị độc lập (ảnh/prompt/seed tuỳ dữ liệu); nếu
  nhiều ảnh cùng prompt hoặc checkpoint, lưu group ID để tránh CI quá hẹp.
- Báo cả paper value và reproduction value cạnh nhau; mỗi paper value ghi
  *"số của paper (ICCV 2023), không phải của demo này"*. Không diễn đạt thành
  "chính xác 99 %".

### P6 — Robustness, ablation và generator mở rộng

- Dùng một tập test đã đóng băng; sinh bản blur/JPEG theo đúng cường độ paper,
  seed cố định và metadata biến đổi. Chạy toàn bộ baseline/representation cần
  so. Báo cả degradation tuyệt đối và tương đối, theo generator/domain.
- Sau khi M0–M7 có thể chạy, mở rộng generator theo danh sách ưu tiên: các
  checkpoint có sẵn trong release (ADM, iDDPM, PNDM, StyleGAN theo split), rồi
  generator chỉ test (DDPM, LDM, SDv1/SDv2, VQ-Diffusion và các generator khác
  đã kiểm kê). Generator nào không có train split giữ nguyên vai trò test.
- Nếu train model generator mới: ưu tiên kiến trúc/dataset thuộc license nghiên
  cứu; giữ `fake_source` và generator checkpoint lineage. Không gọi sample là
  `author`; không gộp với hàng paper; ghi riêng kết quả zero-shot/cross-generator.
- Có thể train ADM từ đầu như một thử nghiệm bổ sung, nhưng đó là dự án con có
  config/checkpoint/dataset riêng. Không cần làm nếu mục tiêu là tái hiện protocol
  paper với checkpoint công khai; cũng không được gọi model đã train khác seed
  là checkpoint gốc của tác giả.

### P7 — Audit độc lập, báo cáo và phát hành

1. Auditor chạy lại từ clean environment bằng manifest, không dùng cache không
   được khai báo; đối chiếu output hashes, counts, test suite và sample scores.
2. Rà rò rỉ split, threshold/test selection, nguồn ảnh, nguồn map/checkpoint,
   gate và các claim trên README/demo.
3. Sinh bảng/figure từ artifact, không nhập số bằng tay. Ghi số liệu paper và
   kết quả của dự án thành hai cột rõ nguồn. Xuất PDF có LaTeX, trích dẫn và mục
   **Hạn chế**.
4. Công bố code/config/manifest/metadata và các artifact được phép chia sẻ;
   không tải lên hay phát lại dataset hạn chế quyền.

**Exit P7:** reviewer có thể tái dựng mọi bảng/hình từ artefact machine-readable,
hoặc thấy rõ ô nào không tái hiện được và vì sao.

## 6. Tài nguyên và tiến độ — không còn giới hạn demo

Không ấn định số ngày hay USD trước inventory và benchmark; không đưa ra một con
số chi phí suy đoán. Không cắt mẫu/epoch để lọt trần. Sau P1/P2, lập dự toán dựa
trên số ảnh đếm thật và phép đo tại chỗ:

```
DIRE GPU time = Σ_images (2 × S × measured_seconds_per_forward)
training time = measured_step_time × steps_per_epoch × epochs × model_count
storage = bytes of verified source + recon + map + checkpoints + logs + replicas
```

Báo cả accelerator-hours, wall-clock (có thể chạy shard song song), peak VRAM,
storage, egress và tiền theo báo giá hiện hành của nơi chạy. Tính dự phòng cho
retry/shard lỗi nhưng không nhân một hệ số tùy tiện. Tiến hành theo queue:

1. self-check và pilot nhỏ để xác nhận correctness/throughput (pilot không vào
   paper table);
2. run đầy đủ theo shard, restartable, chạy song song theo resource capacity;
3. kiểm hash/count rồi mới giải phóng compute cho giai đoạn kế.

Bỏ cost cap trong ứng dụng không đồng nghĩa cho phép container chạy vô hạn:
phải có checkpoint/resume, timeout hợp lý mỗi job, heartbeat, idempotency, quota
account và xác nhận chi tiêu trước lệnh tạo hạ tầng. Đây là cơ chế an toàn vận
hành, không phải giới hạn khoa học. Tài liệu này chưa phê duyệt hay tạo cloud
resource.

## 7. Muốn mở rộng `dire_zero` trước lượt đầy đủ

Các entrypoint hiện có trong `src/dire_zero/modal_app.py` là bước khởi đầu
(`selfcheck`, `bench`, `gen`, `dire`, `train`, `evaluate`, `report`), không phải
bằng chứng rằng toàn bộ protocol đã hoàn thiện. Trước P3–P7 cần:

- cấu hình paper versioned, ma trận jobs sinh tự động và validate đủ ô;
- image/reconstruction/DIRE triplet validator và inventory manifest;
- loader/checkpoint interface cho các generator ngoài ADM (mỗi model có adapter
  riêng, test shape/range/licensing; không đoán tương thích);
- representation/ablation runner cho RGB, REC, RGB&DIRE, signed residual và
  robustness blur/JPEG;
- trainer khớp exact schedule/early-stop/config, resume checkpoint đầy đủ;
- evaluator chạy matrix, group-aware bootstrap, score-level artifacts, cùng
  provenance/gates;
- orchestration có shard ledger, retry, checksum, progress, cancellation và
  resume mà không đếm mẫu hai lần;
- report builder tạo tất cả paper tables/figures từ configs + artifacts;
- test GPU thật và integration test Modal trước full-scale launch.

**Không** giải quyết bằng cách nới test, bỏ cost guard mà không thay orchestration,
đổi nhãn R-level, hoặc thêm các generator vào một bài toán nhị phân gộp. Thay
đổi code làm ở PR riêng sau khi kế hoạch được duyệt.

## 8. Lệnh khởi động thực tế sau khi kế hoạch được duyệt

Chưa chạy các lệnh này. Trước tiên inventory local; sau đó ghim config và chỉ
mới thực hiện từng phase khi exit criteria của phase trước đã qua.

```bash
# 0. Không tải lại; kiểm kê root mà người dùng đã chuẩn bị tay
export DIRE_LOCAL_ROOT='F:/DiffusionForensics'
python -m dire_zero dataset --root "$DIRE_LOCAL_ROOT" --dataset lsun_bedroom \
  --license research-only --splits train val test

# 1. Xem kế hoạch hiện tại của pipeline, xác nhận checkpoint/split đã đặt đúng
modal run -m dire_zero.modal_app::selfcheck
modal run -m dire_zero.modal_app::bench

# 2. Pilot correctness (chỉ validation/debug, không tính vào paper result)
# Các tham số exact phải lấy từ paper_config đã được audit; chưa có config đó thì dừng.

# 3. Full matrix: lệnh production sẽ sinh từ manifest/config đã duyệt,
#    không hard-code --n demo và không dùng `generator=auto` để gộp nhiều hàng.
```

## 9. Tài liệu nguồn và traceability

- Wang et al., *DIRE for Diffusion-Generated Image Detection*, ICCV 2023, paper: <https://arxiv.org/abs/2303.09295>.
- Mã nguồn tác giả: <https://github.com/ZhendongWang6/DIRE>. Trước khi dùng làm
  reference, ghim commit SHA và ghi rõ file/hàm tham chiếu trong `manifests/`.
- Tài liệu nội bộ để chuyển plan thành thao tác: `docs/dire-from-scratch.md`
  (protocol paper), `docs/paper-code-walkthrough.md` (paper ↔ code),
  `docs/from-zero-clean-room.md` (`dire_zero`), `docs/dataset-own.md` (inventory
  ổ F:) và `docs/adr-0006-reproduction-levels.md` (provenance).

Mọi chi tiết chưa đối chiếu với phiên bản paper/repo đã ghim là **chưa xác nhận**,
không phải mặc định đúng. Các bảng số trong paper nếu được trích trong báo cáo
luôn mang nhãn *"số của paper (ICCV 2023), không phải của demo này"*.
