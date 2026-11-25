# GLOSSARY — DIRE research and experiments

> Thuật ngữ miền nghiên cứu và dữ liệu của dự án. File này chỉ định nghĩa khái niệm và ranh giới sử dụng; không thay cho protocol, đặc tả, runbook hay kết quả đo.

## A

### ABS residual / absolute residual

Residual lấy độ lớn theo từng pixel giữa ảnh nguồn và ảnh tái tạo. Trong DIRE, đây là biểu diễn không giữ dấu của chênh lệch. Ảnh hưởng thực nghiệm của lựa chọn này so với residual có dấu chưa được kết luận trong dự án.

## C

### Cross-dataset evaluation

Đánh giá detector trên một dataset/domain khác với miền dùng để huấn luyện. Kết quả này trả lời câu hỏi khác với đánh giá in-domain và được báo cáo riêng.

## D

### DIRE

**Diffusion Reconstruction Error**: biểu diễn sai số tái tạo tạo từ ảnh nguồn và ảnh ADM tái tạo tương ứng. Trong phương pháp DIRE, biểu diễn này là độ lớn residual theo pixel. DIRE là đầu vào/biểu diễn cho detector, không phải bản thân detector và không phải một ảnh giả được sinh độc lập.

### DIRE representation

Cách biểu diễn mẫu dùng làm đầu vào cho detector. Các thí nghiệm phải ghi rõ đang dùng `DIRE`, `RGB`, `REC` hay biểu diễn kết hợp; không mặc định chúng tương đương.

### DiT (Diffusion Transformer)

Kiến trúc transformer dùng làm denoiser trong một mô hình diffusion; trong nghiên cứu này được khảo sát như một backbone tái tạo thay thế U-Net. Dùng DiT sẽ tạo một biến thể reconstruction-error cần đánh giá riêng, không mặc định là cùng phương pháp hoặc tương đương với DIRE của paper.

## G

### Generator holdout / held-out generator

Trong chương trình nghiên cứu này, holdout ở cấp **họ generator**: cả họ và các biến thể liên quan đều bị để ngoài train/validation, dành cho đánh giá tổng quát hóa. Sample IDs test luôn phải tách khỏi train/validation, kể cả với generator đã thấy.

### Seen-generator test

Bài test dùng generator đã xuất hiện trong train hoặc validation nhưng sample IDs được giữ riêng. Đây là đánh giá trên ảnh mới từ generator đã thấy, không phải đánh giá generator holdout.

## I

### In-domain evaluation

Đánh giá trên cùng miền/dataset với train nhưng dùng sample IDs riêng, không xuất hiện trong train/validation. Không đồng nghĩa với việc kiểm tra trên đúng các ảnh đã huấn luyện.

## M

### Matched-domain benchmark

So sánh detector trên các generator khi dữ liệu huấn luyện và test thuộc cùng miền dữ liệu, nhưng các sample ID test tách khỏi train/validation. “Cùng miền” không có nghĩa là tái sử dụng cùng ảnh ở hai split.

## R

### REC representation

Biểu diễn dùng ảnh ADM tái tạo tương ứng với ảnh nguồn. Đây là một điều kiện đầu vào riêng trong nhóm thí nghiệm ablation; không đồng nhất với DIRE residual.

### RGB representation

Biểu diễn dùng ảnh màu RGB nguồn làm đầu vào detector. Một detector huấn luyện trên RGB là điều kiện thí nghiệm riêng, không thể mặc định thay cho detector được huấn luyện trên DIRE.

### RGB_DIRE representation

Biểu diễn kết hợp thông tin ảnh RGB nguồn và DIRE. Cách kết hợp, thứ tự kênh, range và preprocessing phải được khóa trong protocol thí nghiệm; glossary không quy định cách cài đặt.

### Research-extension track

Nhánh thí nghiệm mới ngoài giao thức tác giả được yêu cầu đối chiếu, như ImageNet-domain detector, generator holdout mới hoặc kiến trúc ViT. Kết quả của nhánh này được báo cáo riêng, không trình bày như kết quả tái lập của paper.

## T

### Training domain

Miền/dataset được dùng để xây các split train/validation của detector. Trong chương trình nghiên cứu đã thống nhất, LSUN-Bedroom giữ vai trò track đối chiếu paper; ImageNet là track riêng. Không gộp chúng thành một split không phân biệt domain.

### Two-track study

Tổ chức nghiên cứu thành (1) **paper-reproduction track** để hoàn tất các đối chứng theo giao thức paper và (2) **research-extension track** để kiểm tra câu hỏi/phương pháp mới. Hai track có câu hỏi, protocol, provenance và báo cáo riêng.

## V

### ViT detector

Vision Transformer dùng làm **detector ảnh** trong thí nghiệm kiến trúc. Khác với DiT: ViT detector phân loại representation đầu vào, còn DiT là denoiser bên trong mô hình diffusion.

### Paper-reproduction track

Nhánh thực nghiệm nhằm đối chiếu các protocol/kết quả paper, bao gồm những điều kiện còn thiếu như REC/RGB/RGB_DIRE/DIRE và residual có dấu/không dấu. “Đối chiếu” không đảm bảo metric trùng tuyệt đối với paper; mọi sai khác phải được báo cáo cùng protocol và dữ liệu.
