# CLAUDE.md

> File này cố ý **ngắn**. Nó không lặp lại nội dung của các file khác — chỉ nêu
> điểm dễ sai nhất và trỏ tới chỗ đúng.

Đọc theo thứ tự: [AGENT.md](AGENT.md) → [instructions.md](instructions.md) →
[plan.md](plan.md) → [skills.md](skills.md).

## Bối cảnh trong một câu

Pipeline phát hiện ảnh do diffusion sinh ra bằng **DIRE**, chạy trên Modal, dữ
liệu từ Baidu, ngân sách **0đ** — mọi quyết định phải trả lời được "tốn bao
nhiêu credit".

## Năm điều hay bị làm sai nhất ở repo này

1. **Không thêm `from __future__ import annotations` vào file có `modal.parameter`.**
   PEP 563 biến annotation thành chuỗi; Modal đọc `__annotations__` thô và chết
   lúc **import** với `AttributeError: 'str' object has no attribute '__name__'`.
   (`compute_dire.py` đã từng chết đúng như vậy.)

2. **Tar trong dữ liệu này không nén**, dù tên là `.tar.gz`. Luôn `"r:*"`.

3. **`sha256` của container không nói được hai tập ảnh có giống nhau hay không.**
   Tar chứa `mtime`/owner/thứ tự. Phải so từng member.

4. **Kiểm harness trước khi tin một test FAIL.** Trong repo này harness tự viết
   đã cho kết quả sai mười lần. Đọc bảng bẫy ở `instructions.md` §2.

5. **Mọi thứ chạm GPU phải qua `preflight_cost()` trước `.map()`/`.remote()`**,
   và test phải chứng minh **thứ tự** đó (đếm lời gọi, không chỉ kiểm số).

## Quy ước riêng của repo

- Tài liệu và comment: **tiếng Việt**, giải thích **vì sao** (nhất là vì sao
  cách làm hiển nhiên lại sai).
- Policy nằm ở `src/dire_modal/tiers.yaml`. `configs/tiers.yaml` chỉ là bảng
  tra cứu, **code không đọc**.
- Số liệu phải là số đo, hoặc ghi rõ "ước lượng". Không bịa.
- Stub Modal phải mô phỏng **hành vi gây lỗi**, không được dễ dãi hơn thật.
- Không raise ra khỏi `.map()`; trả `{"status": ...}` và retry bên trong.
- Không xoá `.staging` (đó là tiến độ tải) hay bất cứ gì trong `/raw`.

## Lệnh kiểm trước khi báo "xong"

```bash
./scripts/07_check_imports.sh          # cổng 0, 0 credit
python3 -m pytest tests/ -q            # 75 test (2 skip nếu thiếu modal)
python3 -m pyflakes src/dire_modal/*.py tests/
```

Nếu chưa có venv: `python3 -m venv /tmp/venv && /tmp/venv/bin/pip install -r requirements-dev.txt`.

## Khi viết PR

Nêu: **triệu chứng → nguyên nhân → cách sửa → test** (kèm "X test mới, Y fail
trên `main` cũ") → **phần chưa làm**. Nếu harness của mình sai, nói rõ ra.

Nếu cần thêm tài liệu mới: kiểm xem có file nào đã nói điều đó chưa. Repo này
đã có `AGENT.md`, `instructions.md`, `plan.md`, `skills.md`, `README.md`,
`docs/runbook-free.md`, `docs/adr-000*.md` — thêm file thứ tám cho cùng một nội
dung là tạo chỗ để lệch nhau.
