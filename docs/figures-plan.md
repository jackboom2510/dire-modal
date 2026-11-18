# Kế hoạch figure — dựng lại hình của paper, và những hình paper không có

- **Trạng thái:** kế hoạch (chưa có hình nào được sinh) · **Ngày:** 2026-11-16
- **Liên quan:** `docs/dataset-own.md` (dữ liệu + nhãn R0/R1+R2/R1+R2+R3),
  `docs/from-zero-clean-room.md` (đường chạy `dire_zero`),
  `tools/triplet_view.py` (đã có: panel gốc | tái tạo | DIRE + kiểm chứng),
  `docs/manim-video-plan.md` (video, khác mục đích)
- **Mục đích:** chốt *hình nào*, *số nào*, *nhãn nào* **trước** khi viết code vẽ.
  Một hình vẽ sai nhãn tệ hơn không có hình: nó đi vào báo cáo và không ai kiểm lại.

---

## 1. Năm nguyên tắc, không thương lượng

1. **Mỗi hình phải sinh lại được bằng đúng một lệnh**, từ dữ liệu đã có trong
   repo/Volume, với `--seed` ghi trong tên file. Hình dán tay không tính.
2. **Mọi hình mang nhãn mức tái lập** ngay trên khung: `R0` (map + detector của
   tác giả), `R1+R2` (map + detector của mình, ảnh của tác giả), `R1+R2+R3`
   (ảnh cũng của mình). Nhãn nằm trong **góc dưới trái**, cỡ 7 pt, không thể
   crop mất mà không thấy.
3. **Số của paper không được vẽ lại như số của mình.** Nếu hình có đối chiếu với
   Table 2, vẽ số paper bằng **nét đứt + xám**, chú thích "số của paper
   (ICCV 2023)"; số đo của mình bằng nét liền + màu. Không có loại thứ ba.
4. **Mỗi hình có file số liệu đi kèm** (`figs/data/F0x.json`): mọi điểm vẽ ra đều
   truy được về `scores.npy`/`eval.json` nào, `n`, seed, thời điểm đo.
5. **Không có hình nào vẽ từ số chưa đo.** Chỗ nào chưa có số thì ghi `null` +
   lý do, không điền 0, không nội suy, không "mô phỏng cho đẹp".

Hệ quả kỹ thuật: `matplotlib` **không** được vào `src/dire_zero` — package đó
đang tự khai chỉ phụ thuộc numpy/torch/PIL, và `tests/test_zero_isolation.py`
canh đúng điều đó. Toàn bộ code vẽ nằm ở `tools/figures.py` (một file, một entry
point), đọc `.npy`/`.json`/`.tsv` do `dire_zero` sinh ra.

**Đã có (từ 2026-11-16):** `tools/figures.py` với 8 lệnh, `tests/test_figures_tool.py`
(27 test), `tests/test_notebooks.py` (16 test) và 4 notebook trong `notebooks/`.
Chạy `python tools/figures.py list` để xem lệnh và đầu vào từng hình.

## 2. Figure của paper — kiểm kê đủ 10 hình

| # | Nội dung | Paper dùng gì | Mình có gì | Khả thi | Nhãn |
|---|---|---|---|---|---|
| F1 | Teaser: 1 ảnh thật + 4 ảnh sinh (DDPM, iDDPM, ADM, PNDM) kèm DIRE | map + ảnh gốc | `images/test/lsun_bedroom/*` + `dire/test/...` | **đợt 1** (0 GPU) | R0 |
| F2 | Sơ đồ khái niệm: `p_r`, `p_g`, `x_r → x_r'`, `x_g → x_g'` | vẽ tay | không cần dữ liệu | **đợt 0** | — |
| F3 | Pipeline: `x0 → (DDIM inversion) → x_T → (DDIM gen) → x0' → DIRE → classifier` | vẽ | `src/dire_zero/schedule.py` (mốc thật) | **đợt 0** | — |
| F4 | Độ bền: Gaussian blur σ=1,2,3 · JPEG q=65,30, AP theo từng mức | map tính lại trên ảnh đã làm hỏng | **phải tự tính** — `dire/` là map cố định, không làm hỏng lại được | đợt 2 (GPU) | R1+R2 |
| F5 | Noise pattern + phổ tần số của DIRE (thật vs sinh) | map + FFT | `dire/` có sẵn ⇒ FFT được ngay | **đợt 1** (0 GPU) | R0 |
| F6 | App C: DDPM + iDDPM | triplet | `recons/` không có ddpm/iddpm | đợt 2 | R1+R2 |
| F7 | App C: ADM + PNDM | triplet | `recons/` **không có adm** (chỉ real, ldm, pndm) | một nửa được | R1+R2 |
| F8 | App C: SD-v1 + SD-v2 | triplet | `images/test/.../{sdv2}` có, `sdv1` không (có `sdv1` ở imagenet) | đợt 2 | R1+R2 |
| F9 | App C: LDM + VQ-Diffusion | triplet | `recons/` có `ldm`; `vqdiffusion` chỉ có ảnh | một nửa được | R1+R2 |
| F10 | App C: ImageNet — ADM + SD-v1 | triplet | `images/test/imagenet/*` + `dire/test/imagenet/...` | đợt 2 (khác phân bố) | R1+R2 |

**Điểm phải nói rõ trong mọi caption của F1/F5/F6–F10:** paper vẽ **triplet thật**
(`x`, `x̂`, `DIRE`). Mình chỉ có **map** (`dire/`) và **ảnh gốc** (`images/`), còn
`recons/` chỉ phủ `real`, `ldm`, `pndm`. Vì vậy hai cột "ảnh tái tạo" chỉ vẽ được
ở 3 generator đó; chỗ khác phải **bỏ cột**, không được tự tính `x̂ = x − DIRE`
(phép đó sai dấu và sai thang đo, sẽ tạo ra một hình trông rất thuyết phục mà vô nghĩa).

### 2.1 Việc phải kiểm trước khi vẽ F1: map có khớp ảnh không?

Giả định ngầm của F1 là "file thứ *i* trong `dire/test/lsun_bedroom/adm.tar.gz` là
map của file thứ *i* trong `images/test/lsun_bedroom/adm.tar.gz`". **Chưa ai kiểm
điều này.** Nếu sai, hình teaser sẽ ghép map của ảnh A lên ảnh B — nhìn vẫn "ra
hình" và hoàn toàn sai.

Cách kiểm, 0 GPU, trước khi vẽ:

```bash
python -m dire_zero peek --tar ".../dire/test/lsun_bedroom/lsun_bedroom/adm.tar.gz"
python -m dire_zero peek --tar ".../images/test/lsun_bedroom/adm.tar.gz"
```

- cùng số member và cùng thứ tự tên ⇒ ghép theo thứ tự được, ghi vào
  `figs/data/F01_pairing.json` (`method: "order"`, `n_members`, sha của danh sách tên);
- khác số/thứ tự ⇒ **dừng**, chuyển sang ghép theo `recons/` (đường `triplet_view.py`
  đã làm được), và F1 chỉ còn 3 generator — hoặc bỏ F1.

Với `real`/`ldm`/`pndm`, `tools/triplet_view.py --check` đã kiểm được `|x − x̂|`
so với map tác giả ⇒ đó là **bằng chứng ghép đúng**, dùng làm chuẩn đối chiếu.

## 3. Đợt làm, theo thứ tự phụ thuộc dữ liệu

### Đợt 0 — không cần dữ liệu, không cần GPU (làm được ngay hôm nay)

| Hình | Vì sao làm trước |
|---|---|
| F3 pipeline | Là hình duy nhất mọi bài nói đều cần; vẽ bằng `schedule.py` nên trục thời gian đúng số thật (`timesteps(1000,20)`) |
| F2 khái niệm | Giải thích *vì sao* DIRE ≠ 0; không phụ thuộc kết quả |
| A1 sơ đồ dữ liệu (mục 5) | Chống nhầm nhãn R0/R1+R2/R3 — hình này bảo vệ mọi hình khác |

Đầu ra: `figs/F02_concept.pdf`, `figs/F03_pipeline.pdf`, `figs/A01_provenance.pdf`.
Cả ba là hình **vẽ**, không phải hình **đo**: không được mang nhãn R, nhưng phải
ghi "schematic".

### Đợt 1 — R0, chỉ đọc `dire/` + `recons/`, **0 đồng GPU**

F5 (noise + FFT) và F1 (teaser) chỉ cần giải nén map và ảnh. Đây là toàn bộ giá
trị nhìn thấy được của dự án mà **không tiêu một xu GPU**. Nếu chỉ làm được một
đợt, làm đợt này.

### Đợt 2 — R1+R2, cần GPU (A10G) và `sec/step` đã đo

F4 (robustness), F6–F10 (gallery triplet), A1/A2/A4/A5/A7 (mục 5). Mọi hình ở
đợt này tốn NFE, nên **phải có `bench.json` trước** — không có số đo `sec/step`
thì không có hình nào kèm chi phí, và không được đoán.

### Đợt 3 — R1+R2+R3, cần tự sinh ảnh

Chỉ khi cần generator **không có** trong cây dữ liệu. Không cần cho bất kỳ hình
nào ở trên, vì `images/test/lsun_bedroom/` đã có 15 generator.

## 4. Đặc tả từng hình gốc

### F3 — Pipeline (vẽ, không đo)

- Trục: `t = 0 … 1000` với 20 mốc thật của `timesteps(1000, 20)`; nhãn mốc lấy
  từ `dire_zero.schedule`, **không gõ tay**.
- Ba băng: `x_0` (ảnh), `x_T` (nhiễu), `x_0'` (tái tạo); mũi tên hai chiều,
  nhãn `I(·)` (Eq. 8) và `R(·)` (Eq. 5).
- Góc dưới phải: `S = 20`, `NFE/ảnh = 2S = 40`.
- Bẫy: vẽ `x_T` là "nhiễu trắng" nhưng đúng ra phải là ảnh nhiễu **có cấu trúc
  còn lại của `x_0`** — vẽ sai chỗ này làm người đọc hiểu nhầm inversion là
  "thêm nhiễu ngẫu nhiên".

### F5 — Noise pattern + phổ tần số

- Trái: 2 cột `thật` / `sinh`, 4 hàng ảnh DIRE đã chuẩn hoá percentile 1–99.
- Phải: `|FFT2(DIRE)|` đã `fftshift`, log-scale, cùng `vmin/vmax` cho **mọi**
  panel (khác thang ⇒ "sáng hơn" chỉ là do chuẩn hoá, không phải do dữ liệu).
- Thêm (không có trong paper): đường **radial power spectrum** trung bình ± CI
  theo 32 vòng bán kính, vẽ ở panel dưới — định lượng hoá câu "frequency bands
  are more abundant", thứ paper chỉ nói bằng mắt.
- Bẫy: DIRE map float16 `[0, 2]`; nếu chuẩn hoá mỗi ảnh riêng thì mọi ảnh đều
  "giống nhau". Chuẩn hoá **theo từng lớp**, ghi rõ trong caption.

### F1 — Teaser

- 6 cột × 2 hàng: hàng trên `ảnh gốc | DIRE (đã tô màu)`, hàng dưới `ảnh gốc |
  DIRE` cho lớp còn lại; hoặc 1 hàng 5 panel như paper (thật, DDPM, iDDPM, ADM, PNDM).
- Chọn ảnh bằng **seed cố định**, không "chọn ảnh đẹp nhất" — chọn theo tiêu chí
  thì phải khai tiêu chí; chọn theo mắt là cherry-pick.
- Colorbar chung, đơn vị **giá trị DIRE trước chuẩn hoá**, không phải `[0,1]`.

### F4 — Robustness

- Trục x: mức biến dạng (blur σ ∈ {1, 2, 3}, JPEG q ∈ {65, 30}, cộng `clean`).
- Trục y: AP (%) — paper báo AP, không báo ACC cho hình này.
- Mỗi mức: tính lại DIRE trên **ảnh đã làm hỏng** rồi chấm bằng detector đã train.
  Đây là lý do F4 **không làm được ở đợt 1**: map tác giả là map của ảnh sạch.
- Bẫy: làm hỏng ảnh **trước** khi resize/crop (giống CNNDetection) hay **sau**?
  Hai thứ cho hai đường cong khác nhau. Chọn "sau resize, trước crop", ghi rõ.

### F6–F10 — Gallery triplet

Dùng lại `tools/triplet_view.py` (đã có) thay vì viết mới: nó đã kiểm `|x − x̂|`
so với map tác giả và ghi rõ chỗ map lưu uint8 nên phải so bằng tương quan.
Việc cần thêm chỉ là: ghép 4–6 panel thành figure nhiều cột, chữ A/B/C, và cắt
bớt cột "recon" ở generator không có `recons/`.

## 5. Figure nên có **ngoài** paper

Paper có 10 hình, nhưng cả 10 đều trả lời "DIRE trông thế nào" và "nó tốt hơn
baseline không". Nó **không** trả lời: phân bố điểm ra sao, ngưỡng lấy ở đâu,
tốn bao nhiêu, sai ở đâu, và dữ liệu có sạch không. Đó chính là những câu mà một
báo cáo tái lập phải trả lời.

| # | Hình | Câu hỏi nó trả lời | Vì sao paper thiếu | Dữ liệu | Nhãn |
|---|---|---|---|---|---|
| **A1** | **Sơ đồ nguồn gốc & nhãn** (provenance): ô nào là của tác giả, ô nào của mình, ứng với R0/R1+R2/R3 | "Con số này là của ai?" | Paper không cần — họ là tác giả | không cần | — |
| **A2** | **Phân bố điểm DIRE theo 15 generator** (ridge/violin + ngưỡng) | Tách lớp **mạnh đến đâu**, generator nào khó | Paper chỉ có teaser định tính | `scores.npy` theo generator | R1+R2 |
| **A3** | **ROC/PR kèm CI bootstrap** theo generator | ACC 99,9 % có ý nghĩa gì với `n` này | Paper báo ACC/AP điểm, không có CI | `metrics.evaluate` (đã có `bootstrap_ci`) | R1+R2 |
| **A4** | **Đường cong chất lượng–chi phí theo S** (S=5,10,20,50): AP và **NFE/ảnh** trên hai trục | 20 bước có đáng không | Table 4 chỉ có AP, không có chi phí | `bench.json` + eval theo S | R1+R2 |
| **A5** | **"Euler gap"**: biên độ DIRE theo `t` trong quá trình invert, và theo S | Kiểm App. B bằng số, không bằng lời | App. B là lý thuyết thuần | map trung gian theo `t` | R1 |
| **A6** | **Audit dữ liệu**: ma trận số ảnh (split × generator), tỉ lệ lớp, số trùng, số rò rỉ | Dữ liệu có đủ và sạch không | Paper không phải lo | `manifest.tsv` + `data_card.json` | — |
| **A7** | **Calibration** (reliability diagram + ECE) | Xác suất 0,9 có đúng 90 % không | Paper không dùng cho deployment | `scores.npy` + `labels` | R1+R2 |
| **A8** | **Ca sai điển hình** (FN/FP) kèm DIRE và lý do | Mô hình sai ở đâu, khi nào | Paper chỉ khoe đúng | top-k sai theo score | R1+R2 |

Ưu tiên nếu thời gian có hạn: **A1 → A6 → A2 → A3** (hai cái đầu 0 GPU, hai cái
sau chỉ cần `dire/` + detector R0).

Chi tiết ba hình đáng đầu tư nhất:

### A1 — Provenance (vẽ, 0 dữ liệu)

Bốn cột `dữ liệu thật · ảnh giả · map DIRE · detector`; mỗi ô tô màu **xanh =
của mình**, **xám = của tác giả**. Ba hàng R0 / R1+R2 / R1+R2+R3, mỗi hàng tô
đúng những ô được phép nhận. Hình này là **hàng rào** cho mọi hình khác: nó nói
ngay "R1+R2 nghĩa là gì" mà không cần đọc chú thích.

### A6 — Audit dữ liệu

- Panel trái: heatmap `split × generator` (số ảnh), ghi số tuyệt đối trong ô.
- Panel phải: thanh tỉ lệ `thật : giả` mỗi split, đường đỏ 5:1 (ngưỡng `checkdata`).
- Chú thích bắt buộc: `manifest_sha256` + thời điểm `ingest` ⇒ hình này chứng
  minh **đúng bộ dữ liệu nào** đã dùng, thứ mà một bảng số không làm được.
- Bẫy: đừng vẽ số **ước lượng** từ dung lượng archive. Chỉ số `ingest` đếm thật.

### A2 — Phân bố điểm

- Mỗi generator một hàng (ridge plot), trục x là score của detector; thật màu
  xám ở **mọi** hàng để so được.
- In kèm **AUC từng generator** ở mép phải, và `n`.
- Bẫy: nếu vẽ chung một histogram gộp 15 generator, hình sẽ đẹp và vô dụng —
  nó che đúng chỗ cần thấy (generator yếu).

## 6. Quy ước kỹ thuật

| Thứ | Quy ước | Vì sao |
|---|---|---|
| Code | `python tools/figures.py <tên> --out figs/` — đã có | một entry point; `src/dire_zero` giữ nguyên phụ thuộc tối thiểu |
| Định dạng | `.pdf` (vector) cho báo cáo, `.png` 300 dpi cho README/web | PDF nhúng chữ LaTeX; PNG để xem nhanh |
| Kích thước | cột đơn 3,5 in; cột đôi 7,2 in; font ≥ 7 pt | Chữ nhỏ hơn 7 pt là không đọc được khi in |
| Font | không đổi font mặc định; `mathtext` cho công thức | Tránh hình "đẹp" nhưng lệch font với bài |
| Màu | cùng một generator = cùng một màu **trong mọi hình** | Người đọc học bảng màu một lần |
| Panel | chữ `(a) (b) (c)` góc trên trái, in đậm | Chuẩn của paper |
| Nhãn R | góc dưới trái, 7 pt, `R0` / `R1+R2` / `R1+R2+R3` | Không crop mất được |
| Nguồn | `figs/data/<F>.json` cạnh mỗi hình | Truy ngược mọi điểm vẽ |
| Git | commit `.pdf`, `.json`; `.png` > 500 KB và mọi `.npy` thì không | Repo không phình; số liệu tái sinh được |
| Ngôn ngữ | nhãn trục **tiếng Anh**, caption **tiếng Việt** | Dùng lại được nếu nộp báo; caption đọc trong repo |

## 7. Định nghĩa "xong" cho một hình

Một hình chỉ được coi là xong khi có **đủ sáu** thứ:

- [ ] lệnh sinh lại trong docstring, chạy từ repo sạch;
- [ ] `figs/data/<F>.json` có `n`, `seed`, `label`, `source` (đường dẫn tuyệt đối
      tới `scores.npy`/`eval.json`), `created`;
- [ ] nhãn mức R hiển thị trên hình;
- [ ] caption nói rõ **cái gì là của tác giả** (nếu dùng `dire/`, `recons/`, checkpoint);
- [ ] nếu có so với paper: số paper vẽ khác kiểu (nét đứt, xám) + ghi nguồn;
- [ ] `figs/README.md` có một dòng: hình → lệnh → nhãn.

## 7b. Đã có gì, chạy thế nào (cập nhật 2026-11-16)

| Thứ | Trạng thái |
|---|---|
| `tools/figures.py` | **có** — 8 lệnh: `f2-concept`, `f3-pipeline`, `a1-provenance`, `f5-noise`, `f1-teaser`, `a6-audit`, `a2-scores`, `a3-roc` |
| F2, F3, A1 | **chạy được ngay**, không cần dữ liệu (`01_hinh_schematic.ipynb`) |
| F5, F1 | **có code**, cần map đã giải nén (`02_hinh_R0_tu_dire.ipynb`) |
| A6 | **có code**, cần `manifest.tsv` từ `ingest` |
| A2, A3 | **có code**, cần `scores.npy`/`labels.npy` từ một lần eval thật |
| F4, F6–F10, A4, A5, A7, A8 | **chưa có code** — vẫn theo đợt 2 như bảng trên |

```bash
python tools/figures.py list                      # xem lệnh + đầu vào từng hình
python tools/figures.py f3-pipeline --out figs    # hình vẽ, chạy ngay
python tools/figures.py f5-noise --out figs --maps "$VOL/maps/test" --label R0   # cần map
jupyter lab notebooks/                            # hoặc đi theo notebook
```

Bốn luật đã được cài bằng test, không chỉ bằng lời:

* notebook **không** được tự vẽ (`plt.`/`imshow(`/`savefig(`) — phải gọi `tools/figures.py`,
  nếu không thì có hai bản hình lệch nhau (`tests/test_notebooks.py`);
* mọi hình **phải** có sidecar với `kind`, `claim_label`, `sources`, `cmd`;
  thiếu một khoá là `_save` ném lỗi (`tests/test_figures_tool.py`);
* mã màu/đường dẫn cá nhân (`F:/…`) không được nằm trong notebook;
* `figs/README.md` sinh từ sidecar, nên bảng không thể lệch hình.

Stub matplotlib ở `tests/fixtures/mpl_stub/` để CI kiểm được đường vẽ mà không
phải cài matplotlib — và stub **mô phỏng đúng** `subplots(1,1) → Axes` (không
phải mảng), vì kiểu sai đó làm test xanh trên stub và đỏ trên máy thật.

## 8. Việc chưa làm được, và sẽ không giả vờ làm được

- **F4 và F6–F10 cần GPU** và cần detector đã train; hiện chưa có lần chạy thật nào.
- **Loader cho `checkpoints/*.pth` chưa có** ⇒ chưa có hình nào thuộc R0 dùng
  checkpoint tác giả. F1/F5 đợt 1 chỉ dùng **map** tác giả, không dùng detector.
- **`recons/` thiếu `adm`** ⇒ dòng headline của paper không có cột ảnh tái tạo
  nếu không tự sinh — và nếu tự sinh thì nhãn đổi thành R1+R2+R3.
- **Chưa đo `sec/step`** ⇒ không có hình nào trục USD. A4 dùng NFE, đó là số
  đếm được, không phải số đoán.
- **Cây dữ liệu thật chưa được đọc lần nào** (mọi test đều dựng cây giả) ⇒ F1/F5
  là hai hình đầu tiên sẽ chạm dữ liệu thật; phải chạy `peek` trước như mục 2.1.
