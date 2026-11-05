# Ý tưởng video manim — DIRE, giải thích cho người chưa biết gì

- **Trạng thái:** đề xuất (chưa chốt) · **Ngày:** 2026-10-28
- **Liên quan:** `tools/manim/` (5 scene đã có code), `docs/adr-0003-demo-training.md`,
  `docs/runbook-free.md` §8b
- **Mục đích của tài liệu này:** chốt *nội dung và thứ tự* trước khi viết thêm
  code. Code hiện có là 5 scene rời; tài liệu này biến chúng thành **một bộ phim
  có mạch**, và chỉ ra chỗ còn thiếu dữ liệu.

---

## 1. Video này để làm gì (và KHÔNG để làm gì)

**Để:** một người chưa biết DIRE xem 6 phút hiểu được *vì sao* một ảnh do
diffusion sinh ra lại "để lại dấu vết", và *dự án này* đã kiểm chứng điều đó
bằng cách nào.

**KHÔNG để:** trình bày kết quả của dự án. Hiện repo **chưa có một dòng số nào
chạy thật** trên dữ liệu đầy đủ. Mọi con số trong video vì thế thuộc đúng một
trong ba loại, và **luôn có nhãn trên khung hình**:

| Loại | Nhãn bắt buộc | Ví dụ |
|---|---|---|
| Số của paper | "số của paper (ICCV 2023) — chưa phải số đo của mình" | ACC 99.9 % trên ADM |
| Số đo của repo | "đo 2026-10-25" + nguồn | 76.6 KiB/s của Baidu |
| Số demo | "demo — KHÔNG phải số của paper" | val_acc của 2 000 ảnh × 5 epoch |

Không có loại thứ tư. Một khung hình đẹp mà không rõ số từ đâu là lỗi, không
phải phong cách.

**Ba bản xuất từ cùng một bộ scene:**

| Bản | Dài | Dùng cho |
|---|---|---|
| **Teaser** | 75–90 s | GIF nhúng `README.md`, mở đầu bài nói |
| **Explainer** | 6–7 phút | Video chính, YouTube/HF, thuyết trình |
| **Clip lẻ** | 30–60 s mỗi chương | Chèn vào tài liệu, trả lời câu hỏi "DIRE là gì?" |

---

## 2. Mạch kể: 8 chương

Ý tưởng xương sống: **một câu hỏi, một câu trả lời phản trực giác, một bằng
chứng.** Không đi theo cấu trúc "giới thiệu → phương pháp → kết quả" vì nó
khiến video thành bản đọc báo cáo.

```
Ch0  Hook           10s   "Ảnh này do AI sinh ra. Làm sao biết?"
Ch1  Vì sao khó     30s   Ảnh fake ngày càng không phân biệt được bằng mắt
Ch2  Ý tưởng DIRE   75s   Model sinh ra ảnh ⇒ nó "thuộc" ảnh của nó
Ch3  Detector       50s   từ ảnh → DIRE map → 1 logit → P(fake)
Ch4  Bằng chứng     45s   Table 2 của paper (có nhãn "của paper")
Ch5  Dự án làm gì   60s   đường đi dữ liệu thật + những cái bẫy đã gặp
Ch6  Train thật     45s   đường loss thật của demo_train (có nhãn "demo")
Ch7  Sự cố fc.weight 40s  một lỗi thật: hai quy ước lớp cuối
Ch8  Giới hạn       25s   video này KHÔNG chứng minh gì thay cho số đo
```

Tổng ≈ **6:20**.

### Ch0 — Hook (10 s) · *chưa có, cần viết*

Khung mở: hai ảnh cạnh nhau, một thật một do diffusion sinh, **không gắn nhãn**.
Chữ: *"Một trong hai ảnh này không có thật."* Giữ 3 giây. Rồi hiện nhãn đúng
lên từng ảnh ⇒ *"Mắt người không phân biệt được. Nhưng model thì có."*

Vì sao mở như vậy: đặt câu hỏi *trước* khi giải thích, và chứng minh luôn vấn đề
là có thật chứ không phải lo xa.

Việc cần: hai ảnh mẫu (LSUN thật + một ảnh ADM). **Chưa có** — xem §6.

### Ch1 — Vì sao khó (30 s) · *chưa có, cần viết*

Ba dòng chữ hiện dần, không hình phức tạp:

1. *"Ảnh do AI sinh không còn lỗi nhìn thấy được."*
2. *"Mọi dấu vết đều nằm ở tầng thống kê, không ở pixel."*
3. *"⇒ Cần một detector, không cần một người tinh mắt."*

Chương này là **nhịp nghỉ** giữa hook và phần kỹ thuật. Giữ đơn giản: nền tối,
ba khối chữ, không chuyển động thừa.

### Ch2 — Ý tưởng DIRE (75 s) · *đã có `DiffusionTheory`, cần siết lại*

Đây là chương quyết định người xem có hiểu hay không. Bốn nhịp hiện tại giữ
nguyên về nội dung, nhưng cần thêm **một câu chốt bằng lời** sau nhịp 4:

> *"Ảnh do model sinh ra nằm trên chính manifold của model đó — nên khi bắt model
> tái tạo nó, model tái tạo rất tốt. Ảnh thật thì không."*

Đó là toàn bộ DIRE trong hai câu. Nếu người xem chỉ nhớ một câu, phải là câu này.

| Nhịp | Hình | Chữ trên khung |
|---|---|---|
| 1 | `x₀` bị phun thành đám mây điểm Gauss | *"Forward: thêm nhiễu dần"* + `q(xₜ│xₜ₋₁) = N(√(1−βₜ)·xₜ₋₁, βₜI)` |
| 2 | mũi tên ngược, màu "ok" | *"Reverse bằng DDIM: một đường đi DUY NHẤT"* → *"tất định ⇒ đảo ngược được"* |
| 3 | `x → z_T → x̂ → │x−x̂│` (3 kênh) | *"Inversion rồi tái tạo. Hiệu = DIRE map"* |
| 4 | đường cong manifold + 2 điểm (fake ở trên, real lệch) | *"fake: residual NHỎ · real: residual LỚN"* ⇒ *"DIRE(real) > DIRE(fake)"* |

Nhịp 4 **bắt buộc** có dòng: *"trực giác hình học — paper đo bằng thực nghiệm,
không chứng minh"*. Đã có trong code.

### Ch3 — Detector (50 s) · *đã có `DirePipeline`, ổn*

`ảnh RGB → DIRE map → ResNet-50 → 1 logit → sigmoid → P(fake)`, kèm ba "cạm bẫy"
thật của repo: 1 logit (không phải 2 lớp), `Resize 256 → CenterCrop 224 →
Normalize(ImageNet)`, và demo HF Spaces chạy ResNet **trực tiếp trên ảnh** (thiếu
bước DIRE ⇒ khác hẳn detector trong paper).

Gợi ý thêm 1 beat (10 s): một khung nhỏ *"nếu bỏ bước DIRE map"* với dấu ✗ đỏ —
đây là điều người xem dễ hiểu sai nhất khi dùng demo.

### Ch4 — Bằng chứng (45 s) · *đã có `PaperTable2`, cần thêm 1 nhịp*

Hiện tại: 8 thanh ACC của hàng `lsun_adm`, trung bình tính tại chỗ (**99.92**).

Thêm nhịp 2 (~15 s): **4 detector**, mỗi cái một con số trung bình —
`lsun_adm 99.92 · lsun_pndm 98.67 · lsun_iddpm 98.60 · lsun_stylegan 97.91`
(tất cả tính từ TSV, không chép). Câu dẫn: *"Bốn detector, bốn cách train khác
nhau, cùng một kết luận."*

Nhãn góc dưới là **bắt buộc**: *"số của paper — CHƯA phải số đo của mình"*.

### Ch5 — Dự án làm gì (60 s) · *đã có `DataPipeline`*

Hai phần, đã có số thật:

1. **Đường đi:** Baidu (76.6 KiB/s — trần của *tài khoản*, không phải của
   container) → staging trên máy → `volume put` → `verify` (`gzip -t` + sha256,
   `.pth` thì size + sha256) → eval.
2. **Bẫy `ls -l`:** `119.22MB` → quy đổi ngược `125 011 230` B, file thật
   `125 015 131` B, lệch **3 901 B (0.003 %)** ⇒ code cũ đòi tải lại file đã đủ.
   Đây là bài học dạy được: *làm tròn là một dạng sai số, và sai số phải được
   mang theo trong kiểu dữ liệu*.

Nhịp kết: **15 cặp × 4 detector = 60 container** (tính từ manifest, không gõ).

### Ch6 — Train thật (45 s) · *đã có `TrainMechanism`, đang chờ log*

Đường loss **thật** của `demo_train` (2 000 ảnh × 5 epoch, L4, seed 3407, 1
logit). Nếu chưa có log: khung *"chưa có log thật — scene này KHÔNG vẽ đường cong
giả"*.

Giá trị của chương này không phải "xem nó train", mà là **cho thấy repo không vẽ
số**: khán giả thấy một chỗ trống được báo đúng cách thì tin phần còn lại.

### Ch7 — Sự cố `fc.weight` (40 s) · *chưa có, đề xuất thêm*

Chương "debug story" — thật 100 %, và là chương dễ hiểu nhất với người làm kỹ
thuật:

```
size mismatch for fc.weight: [1, 2048] vs [2, 2048]
```

Hai khối cạnh nhau: checkpoint tác giả (1 logit + BCE, `fc.weight [1,2048]`) và
code cũ (2 lớp + softmax, `[2,2048]`). Rồi: *"không phải dữ liệu hỏng — là hai
quy ước"* → *"và một cặp lỗi làm mất cả 60 lần chấm"*.

Vì sao nên có: nó minh hoạ **tính trung thực kỹ thuật** của repo mạnh hơn mọi
lời giới thiệu, và người xem học được một bài học dùng lại được.

### Ch8 — Giới hạn (25 s) · *chưa có, cần viết*

Ba dòng, không hình:

1. *"Video này giải thích cơ chế. Nó không phải bằng chứng."*
2. *"Số của dự án: chưa có. Khi có, nó sẽ nằm ở `eval::report` với Δ so với paper."*
3. *"Cái đang có: đường đo, guard chi phí, và một chỗ trống được ghi đúng."*

Kết ở đây chứ không kết ở "DIRE đạt 99.9 %" — vì câu đó sẽ là nói dối.

---

## 3. Ngôn ngữ thị giác

Đã có trong `tools/manim/style.py`, giữ nguyên:

| Thành phần | Quy ước |
|---|---|
| Nền / chữ | `#0E1116` / `#E8EAED` |
| **real** | xanh dương `#4C9AFF` |
| **fake** | đỏ `#FF6B6B` |
| **DIRE map / residual** | vàng `#FFC857` |
| **ResNet** | tím `#8B7CF6` |
| đo được / đã kiểm | xanh lá `#4ADE80` |
| cảnh báo, nhãn "chưa đo" | hổ phách `#FBBF24` |
| Font | `DejaVu Sans` (phủ dấu tiếng Việt), đổi được qua `DIRE_FONT` |

Nguyên tắc chuyển động:

- **Mỗi khung chỉ một ý.** Chữ hiện bằng `Write`, hình bằng `FadeIn`; không
  dùng hiệu ứng xoay/nhấp nháy.
- **Không chuyển động khi đang có số.** Số phải đứng yên ≥ 1.5 s để đọc được.
- **Không `MathTex`** (cần TeX Live 1–2 GB) — công thức viết bằng `Text` + Unicode.
- Mỗi scene kết thúc bằng `wait(≥ 2)` để GIF không bị cắt cụt (đã có test).

---

## 4. Đã có gì / cần gì

| Chương | Scene class | Code | Dữ liệu | Render được? |
|---|---|---|---|---|
| Ch2 | `DiffusionTheory` | ✅ | không cần | ✅ |
| Ch3 | `DirePipeline` | ✅ | không cần | ✅ |
| Ch4 | `PaperTable2` | ✅ | `paper_table2.tsv` ✅ | ✅ |
| Ch5 | `DataPipeline` | ✅ | manifest ✅ | ✅ |
| Ch6 | `TrainMechanism` | ✅ | **thiếu `demo_train_log.json`** | ✅ (hiện khung "chưa có log") |
| Ch0 | `Hook` | ❌ | **cần 2 ảnh mẫu** | — |
| Ch1 | `WhyHard` | ❌ | không cần | — |
| Ch4b | (thêm nhịp 4 detector vào `PaperTable2`) | ⚠ sửa | TSV ✅ | — |
| Ch7 | `FcWeightIncident` | ❌ | hằng số đã biết (1 vs 2 logit) | — |
| Ch8 | `Limits` | ❌ | không cần | — |

**Chỗ thiếu dữ liệu, theo mức độ chặn:**

1. **2 ảnh mẫu cho Ch0** — không có thì hook yếu. Nguồn rẻ nhất: 1 ảnh LSUN thật +
   1 ảnh ADM, lấy từ `dire/test/lsun_bedroom/lsun_bedroom/{real,adm}.tar.gz`
   **đã có trên Volume** ⇒ 0 đồng, 0 giờ tải thêm. Cần một script nhỏ trích ra
   `docs/media/still/`.
2. **`demo_train_log.json` cho Ch6** — cần chạy `demo_train --mode run`, mà nó
   cần tier `train` (5.74 GB, ~20–22 h ở 76.6 KiB/s). Đây là hạng mục **duy nhất
   trong kế hoạch này cần tải dữ liệu**.
3. Ảnh gallery cho HF Spaces (Ch4 nếu muốn chèn ảnh thật) — chưa có, và không
   bắt buộc cho video.

---

## 5. Kế hoạch sản xuất: 4 đợt

| Đợt | Việc | Điều kiện vào | Kết quả |
|---|---|---|---|
| **P0** | Render 5 scene hiện có ở 480p15, xem trên máy, sửa bố cục | không cần gì | 5 mp4 nháp + danh sách chỗ chữ tràn/lệch |
| **P1** | Viết `Hook`, `WhyHard`, `Limits`, `FcWeightIncident`; thêm nhịp 4 detector vào `PaperTable2` | P0 xong (đã biết bố cục thật) | đủ 8 chương, Ch6 vẫn ở trạng thái "chưa có log" |
| **P2** | Trích 2 ảnh mẫu từ tar trên Volume ⇒ Ch0 hoàn chỉnh | `e2` có trên Volume | hook thật |
| **P3** | Có `demo_train_log.json` ⇒ Ch6 thật; render 1080p60; cắt teaser + GIF | tier `train` đã tải | bộ phim hoàn chỉnh |

Thứ tự này cố ý đặt **P0 trước P1**: sửa bố cục khi đã nhìn thấy khung hình thật
rẻ hơn nhiều so với sửa 8 scene viết mù.

---

## 6. Cách kiểm (không chỉ "xem bằng mắt")

Đã có, giữ:

- `tests/test_manim_scenes.py` — scene dựng được **không cần manim** (stub có
  whitelist API), mọi scene `play` + `wait`, `TrainMechanism` có/không có log.
- Không có số nào viết thẳng trong `scenes.py`; đối chiếu từng ô với TSV.
- `render.sh` chạy thật với manim giả: mp4 phải ra, `-qh` phải khớp `1080p60`,
  không được đưa đường dẫn POSIX cho binary Windows.

Đề xuất thêm (P1):

- **Thời lượng tối thiểu** mỗi scene (đếm `wait` + số `play`) để không có scene
  3 giây vô nghĩa.
- **Không có khung trống**: mỗi scene phải `play` ≥ 3 lần.
- **Nhãn nguồn bắt buộc** với scene có số (AST: scene nào gọi `bar/curve_from_points`
  thì phải gọi `badge/source_note`).

---

## 7. Rủi ro và phương án

| Rủi ro | Phương án |
|---|---|
| Chữ tiếng Việt bị tràn khung (dài hơn tiếng Anh) | P0 render nháp trước; dùng `Text` có `set_width` khi cần |
| Font thiếu glyph ⇒ ô vuông, chỉ thấy sau render | `DIRE_FONT` đổi được; P0 kiểm bằng mắt một lần |
| Máy Windows không có cairo/pango | đã có đường Modal (image riêng) trong `tools/manim/README.md` §3; hoặc chỉ dùng GIF do người khác render |
| `demo_train` chưa chạy ⇒ Ch6 trống | đã thiết kế trạng thái "chưa có log" trông như thiếu, không vẽ bù |
| Video bị đọc thành "kết quả của dự án" | 3 loại nhãn + Ch8 nói thẳng; nhãn nằm trong chính file log |
| Render 1080p60 tốn thời gian trên máy yếu | teaser/GIF ở 480p15 là đủ cho README; 1080p chỉ render lần cuối |

---

## 8. Tiêu chí "xong"

1. 8 chương render được, tổng 5–7 phút, không khung nào có số vô nguồn.
2. Mỗi con số trên khung hình truy được về một file trong repo (TSV / manifest /
   `loss_log.json`) hoặc một dòng trong `docs/` có ngày đo.
3. Teaser 75–90 s + GIF < 2 MB/scene, nhúng được `README.md`.
4. Người chưa biết DIRE, xem một lần, kể lại được: *"model sinh ảnh thì tái tạo
   ảnh của nó tốt hơn ảnh thật, nên chênh lệch là dấu vết"*.
5. Không có câu nào trong video khẳng định dự án đã đạt được một con số.

---

## 9. Việc cần bạn quyết

| # | Câu hỏi | Mặc định mình đề xuất |
|---|---|---|
| 1 | Giọng kể: phụ đề tiếng Việt, hay có voice-over? | **Phụ đề** (voice-over cần thu âm, và sửa lời là phải thu lại) |
| 2 | Có Ch7 (sự cố `fc.weight`) không? | **Có** — chương thật nhất và dễ hiểu nhất |
| 3 | Chờ có log rồi mới render 1080p, hay render ngay bản Ch6 trống? | **Render ngay**; Ch6 render lại sau khi có log (chỉ 1 scene) |
| 4 | Thời lượng mục tiêu | **6–7 phút**; teaser 75–90 s |
