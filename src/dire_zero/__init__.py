"""DIRE from zero — một bản viết lại độc lập, tự chứa, của toàn bộ phép đo DIRE.

Vì sao có package này
---------------------
`dire_core` + `dire_modal` là bản chạy được của dự án, nhưng nó *thừa hưởng*:

* `guided_diffusion` (clone repo tác giả) để dựng U-Net,
* `vendor/networks/resnet.py` (chép từ repo tác giả) cho detector,
* và vì thế, một câu hỏi không bao giờ trả lời được: nếu code tác giả có lỗi
  ở chỗ mình không đọc, kết quả của mình cũng sai theo.

`dire_zero` trả lời câu đó bằng cách **không dùng một dòng nào** của họ: U-Net,
lịch khuếch tán, DDIM, DIRE map, ResNet-50, cách tính ACC/AP, cách chia
train/test — tất cả viết lại từ công thức trong bài báo. Thứ duy nhất dùng lại
là **trọng số** ADM (`256x256_diffusion_uncond.pt`), vì đó là *công cụ*: train
lại nó tốn ~250k step × batch 128 trên 8×A100.

Điều gì chứng minh bản viết lại là đúng
---------------------------------------
Không phải lời hứa, mà bốn lớp kiểm tra chạy được:

1. **Hình dạng tham số** — `tests/fixtures/adm_lsun_state_keys.json` là 566
   cặp *tên → shape* của checkpoint ADM (dữ liệu, không phải code). Test đòi
   ánh xạ tên của mình ↔ tên của checkpoint là **song ánh** và khớp shape
   từng cái; thiếu một key là fail, không phải cảnh báo.
2. **Số học** — `ddim_timesteps(1000, 20)`, `ᾱ_t` tại các mốc, NFE/ảnh = 2S,
   ngưỡng `[0, 2]` của map: so với giá trị đóng.
3. **Parity số** — khi có bản tham chiếu trong máy, test nạp *cùng bộ trọng số*
   vào hai mô hình rồi so đầu ra từng phần tử (xem `tests/test_zero_parity.py`,
   tự skip nếu không có bản tham chiếu).
4. **Cổng chạy thật** — `modal run -m dire_zero.modal_app::selfcheck` nạp
   checkpoint 2,5 GB với `strict=True` và chạy một ảnh thật; sai một tensor là
   `RuntimeError`, không có "chắc là ổn".

Package tự chứa: chỉ phụ thuộc `numpy`, `torch`, `Pillow`. Không import
`dire_core`, `dire_modal`, `dire_local`, `guided_diffusion`. Có test canh việc đó.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
