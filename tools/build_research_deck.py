#!/usr/bin/env python3
"""Build the editable DIRE research deck and a matching PDF from checked-in evidence.

The deck uses native PowerPoint text/shapes for editability. Mathematical
expressions are typeset from LaTeX into crisp transparent formula images; the
exact sources are kept in platforms/site/dire-research-equations.tex. No Modal
calls, training, data downloads, or cost-incurring actions are performed.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "platforms/site"
ILLUSTRATIONS = SITE / "illustrations"
OUT_PPTX = SITE / "dire-research-complete.pptx"
OUT_PDF = SITE / "dire-research-complete.pdf"
EQ_SOURCE = SITE / "dire-research-equations.tex"

sys.path.insert(0, str(ROOT / "platforms/prototype"))
from data import (  # noqa: E402
    EVAL_PATH, GALLERY, PAPER_TABLE2_PATH, REPORT_PATH, TABLE4_PATH,
    TABLE5_ROOT, FIGURE4_ROOT, read_json, table4, table4_rows,
)

# Presentation palette: warm paper + deep navy + teal; amber is reserved for
# caveats and unverified evidence, never used as a positive performance signal.
C = {
    "bg": "F5F7F5", "paper": "FFFFFF", "ink": "122C3A", "navy": "102A3A",
    "navy2": "193F50", "muted": "526A76", "line": "D8E2E3", "teal": "147C80",
    "mint": "55B7A6", "gold": "D59A35", "goldp": "FCF2DB", "red": "A8484E",
    "redp": "FBECEE", "bluep": "EAF2F5", "greenp": "EAF5F0", "purple": "7057A3",
    "grayp": "EDF1F1", "white": "FFFFFF", "deepmuted": "B8C8CF",
}
PW, PH = 960.0, 540.0  # points; 13.333 x 7.5 inches, 16:9
FONT_PPT = "Aptos"
FONT_PDF = "DIREDejaVu"
FONT_PDF_BOLD = "DIREDejaVu-Bold"
FONT_PDF_ITALIC = FONT_PDF
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont(FONT_PDF, str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont(FONT_PDF_BOLD, str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily(FONT_PDF, normal=FONT_PDF, bold=FONT_PDF_BOLD,
                              italic=FONT_PDF, boldItalic=FONT_PDF_BOLD)



# Visible copy is localized to Vietnamese at render time; proper names, identifiers,
# equations, percentages and citations are intentionally left intact.
VI_COPY = {
    "TECHNICAL DEEP DIVE  ·  RESEARCH UPDATE": "PHÂN TÍCH KỸ THUẬT CHUYÊN SÂU  ·  CẬP NHẬT NGHIÊN CỨU",
    "PAPER-REPORTED": "SỐ LIỆU DO BÀI BÁO CÔNG BỐ",
    "REPO-RECORDED": "SỐ LIỆU GHI TRONG REPO",
    "Reconstruction": "tái tạo",
    "Diffusion Reconstruction Error": "Sai số tái tạo khuếch tán (DIRE)",
    "DIFFUSION / U-NET": "KHUẾCH TÁN / U-NET",
    "RESNET-50 BLOCK": "KHỐI RESNET-50",
    "OBJECTIVE": "HÀM MỤC TIÊU",
    "APPENDIX A": "PHỤ LỤC A",
    "APPENDIX B": "PHỤ LỤC B",
    "APPENDIX C": "PHỤ LỤC C",
    "APPENDIX D": "PHỤ LỤC D",
    "APPENDIX E": "PHỤ LỤC E",
    "APPENDIX F": "PHỤ LỤC F",
    "APPENDIX G": "PHỤ LỤC G",
    "MISSING ≠ ZERO": "THIẾU DỮ LIỆU ≠ 0",
    "Editable PPTX + PDF  ·  equations typeset from LaTeX source": "PPTX + PDF có thể chỉnh sửa  ·  công thức được dàn từ mã LaTeX",
    "repo-recorded": "ghi trong repo",
    "provenance": "nguồn gốc dữ liệu",
    "live state": "trạng thái hiện thời",
    "Snapshot của checkout": "Ảnh chụp checkout",
    "TABLE 4 LEDGER": "SỔ KẾT QUẢ TABLE 4",
    "PER TEST CELL": "MỖI Ô KIỂM THỬ",
    "GALLERY RECORDS": "MẪU TRONG THƯ VIỆN",
    "HARD CAP": "GIỚI HẠN CỨNG",
    "$30 total": "Trần $30",
    "Có trong repo": "Có trong repo",
    "Chưa có ledger": "Chưa có sổ kết quả",
    "Chưa xác minh": "Chưa xác minh",
    "Live Volume": "Volume hiện thời",
    "Claim": "Kết luận",
    "Paper": "Bài báo",
    "Repo artifact": "Artifact repo",
    "Nguồn công bố": "Nguồn công bố",
    "CẢNH BÁO PROVENANCE": "CẢNH BÁO NGUỒN GỐC",
    "TRAIN": "HUẤN LUYỆN",
    "UNSEEN TEST": "KIỂM THỬ CHƯA GẶP",
    "DETECTOR": "BỘ PHÂN LOẠI",
    "Shortcut risk": "Nguy cơ học tắt",
    "Research target": "Mục tiêu nghiên cứu",
    "likelihood": "khả năng hợp lý",
    "Inverse / trajectory": "Phép nghịch đảo / quỹ đạo",
    "Hệ quả lý thuyết": "Hệ quả lý thuyết",
    "Vị trí với DIRE": "Vai trò trong DIRE",
    "Không dùng làm I/R của DIRE": "Không dùng làm phép đảo/tái tạo của DIRE",
    "không có inverse tích hợp": "không có phép nghịch đảo tích hợp",
    "GAN inversion là tối ưu bổ sung": "nghịch đảo GAN cần tối ưu bổ sung",
    "Không có I/R gốc": "Không có phép đảo/tái tạo gốc",
    "inverse chính xác (lý tưởng)": "nghịch đảo chính xác (lý tưởng)",
    "R∘I ≈ identity": "R∘I ≈ đồng nhất",
    "token/codebook + decoder": "token/codebook + bộ giải mã",
    "nhiều bước latent": "nhiều bước trong không gian ẩn",
    "Có trong benchmark paper": "Có trong benchmark của bài báo",
    "denoise U-Net trên pixel": "U-Net khử nhiễu trên pixel",
    "Backbone I/R dùng trong DIRE": "Mạng nền cho phép đảo/tái tạo của DIRE",
    "Autoregressive, EBM và flow-matching được ghi là phạm vi còn thiếu trong tài liệu tổng quan HTML; không suy rộng từ bảng này.": "Tự hồi quy, EBM và flow-matching còn ngoài phạm vi tài liệu tổng quan HTML; không suy rộng từ bảng này.",
    "image": "ảnh",
    "encoder → latent distribution": "encoder → phân phối ẩn",
    "decoder": "bộ giải mã",
    "generator": "bộ sinh",
    "nhiễu tiềm ẩn": "nhiễu trong không gian ẩn",
    "ảnh được sinh": "ảnh sinh",
    "latent posterior": "hậu nghiệm trong không gian ẩn",
    "posterior": "hậu nghiệm",
    "prior": "phân phối tiên nghiệm",
    "codec": "bộ mã hóa/giải mã",
    "latent": "không gian ẩn",
    "VQ-Diffusion and latent diffusion": "VQ-Diffusion và latent diffusion",
    "Normalizing flow": "Dòng chuẩn hóa (normalizing flow)",
    "residual": "sai số tái tạo",
    "Change of variables": "Công thức đổi biến",
    "Change of variables: bijective mapping makes exact likelihood tractable.": "Đổi biến: ánh xạ song ánh giúp tính mật độ xác suất chính xác.",
    "Global context, used at selected resolutions.": "Ngữ cảnh toàn cục, dùng tại một số độ phân giải.",
    "Norm + SiLU  ·  stabilize / nonlinearity": "Chuẩn hóa + SiLU  ·  ổn định / phi tuyến",
    "Conv 3×3  ·  local spatial mixing": "Conv 3×3  ·  trộn lân cận không gian",
    "+ time embedding  ·  condition on t": "+ nhúng thời gian  ·  điều kiện hóa theo t",
    "Conv 3×3  ·  refine features": "Conv 3×3  ·  tinh chỉnh đặc trưng",
    "Residual: y = x + F(x) · đường gradient ngắn": "Residual: y = x + F(x) · rút ngắn đường truyền gradient",
    "Q, K, V projections / across spatial positions": "Phép chiếu Q, K, V / trên các vị trí không gian",
    "Global context, used at selected resolutions.": "Ngữ cảnh toàn cục, dùng tại một số độ phân giải.",
    "DIFFUSION MATHEMATICS": "TOÁN HỌC KHUẾCH TÁN",
    "ADM CONFIGURATION": "CẤU HÌNH ADM",
    "DDIM EQUATIONS": "CÁC PHƯƠNG TRÌNH DDIM",
    "DDIM INVERSION": "PHÉP ĐẢO DDIM",
    "Ablation paper": "Phân tích loại bỏ của bài báo",
    "ABLATION PAPER": "PHÂN TÍCH LOẠI BỎ · BÀI BÁO",
    "PROVENANCE": "NGUỒN GỐC",
    "PAPER EVIDENCE": "BẰNG CHỨNG TỪ BÀI BÁO",
    "PROJECT MEASUREMENTS": "SỐ ĐO CỦA DỰ ÁN",
    "GALLERY & PROVENANCE": "THƯ VIỆN & NGUỒN GỐC",
    "TABLE 4 PILOT": "PILOT TABLE 4",
    "PAPER VS PILOT": "BÀI BÁO SO VỚI PILOT",
    "PAPER VS PROJECT TABLE 5": "TABLE 5 · BÀI BÁO SO VỚI DỰ ÁN",
    "FIGURE 4": "HÌNH 4",
    "SUBSET / RESUMPTION": "TẬP CON / TIẾP TỤC CHẠY",
    "BA NGHIÊN CỨU": "BA NHÓM THÍ NGHIỆM",
    "CLEAN-ROOM IMPLEMENTATION": "TRIỂN KHAI ĐỘC LẬP",
    "VERIFICATION LESSONS": "BÀI HỌC KIỂM CHỨNG",
    "DATASET & GENERATORS": "TẬP DỮ LIỆU & BỘ SINH",
    "DEMO & PROTOTYPE": "DEMO & BẢN THỬ NGHIỆM",
    "BUDGET & SAFEGUARDS": "NGÂN SÁCH & BẢO VỆ",
    "PAPER NUMBERS": "SỐ LIỆU BÀI BÁO",
    "PILOT DATA": "DỮ LIỆU PILOT",
    "PILOT PROTOCOL": "GIAO THỨC PILOT",
    "PROTOCOL COMPARISON": "SO SÁNH GIAO THỨC",
    "FINGERPRINTS": "DẤU VÂN TAY SỐ",
    "IMPLEMENTATION MAP": "SƠ ĐỒ TRIỂN KHAI",
    "EVIDENCE GATES": "CỔNG BẰNG CHỨNG",
    "REFERENCES": "TÀI LIỆU THAM KHẢO",
    "The strongest deliverable today is a transparent evidence system — not a full reproduction": "Sản phẩm vững nhất hiện nay là hệ thống bằng chứng minh bạch — chưa phải tái lập toàn bộ",
    "R1+R2 report records a broader run — but bookkeeping counters need reconciliation": "Báo cáo R1+R2 ghi nhận lượt chạy rộng hơn — nhưng cần đối soát bộ đếm",
    "The R1+R2 artifact records heterogeneous scores across generator-trained runs": "Hiện vật R1+R2 ghi nhận điểm số không đồng nhất giữa các lượt huấn luyện theo bộ sinh",
    "ADM held-out eval reports AP=0.911 on the self-computed-map path": "Đánh giá ADM trên tập giữ lại ghi AP=0,911 ở luồng tự tính map",
    "Three experiment families share the subset identity but vary different factors": "Ba nhóm thí nghiệm dùng chung định danh tập con nhưng thay đổi các yếu tố khác nhau",
    "Ba experiment families share the subset identity but vary different factors": "Ba nhóm thí nghiệm dùng chung định danh tập con nhưng thay đổi các yếu tố khác nhau",
    "Input representation": "Biểu diễn đầu vào",
    "Corruption robustness": "Độ bền vững trước biến đổi",
    "inputs only; no results": "chỉ có đầu vào; chưa có kết quả",
    "no result ledger": "chưa có sổ kết quả",
    "Dimension": "Khía cạnh",
    "each condition × generator → measured AP shard → result ledger": "mỗi điều kiện × bộ sinh → AP đo được → sổ kết quả",
    "Planned matrix: 30 condition×generator shards (6×5), each comparing 2 project-trained detectors = 60 model-condition metrics. Current output: none found. Runner is not the paper's CNNDetection/PatchForensics baseline reproduction.": "Ma trận dự kiến: 30 ô điều kiện×bộ sinh (6×5), mỗi ô so sánh 2 detector huấn luyện trong dự án = 60 metric. Chưa tìm thấy đầu ra. Runner không tái lập baseline CNNDetection/PatchForensics của bài báo.",
    "Table 2 source values: complete 4 × 8 DIRE block (ACC / AP, %)": "Số liệu nguồn Table 2: đầy đủ khối DIRE 4 × 8 (ACC / AP, %)",
    "Hashes identify artifacts but do not prove binaries are available or live state is current": "Hash định danh hiện vật nhưng không chứng minh còn tệp nhị phân hay trạng thái hiện thời",
    "Summary of paper, repository artifacts, known caveats and pilot constraints": "Tổng hợp bài báo, hiện vật repo, giới hạn đã biết và ràng buộc pilot",
    "DIRE provides a reproducible representation hypothesis;": "DIRE đưa ra giả thuyết về một biểu diễn có thể tái lập;",
    "the repository now contains a pilot ledger and gallery, but evidence boundaries still matter.": "repo hiện có sổ pilot và thư viện ảnh, nhưng vẫn phải phân định rõ bằng chứng.",
    "Theory": "Lý thuyết",
    "Evidence": "Bằng chứng",
    "Next": "Tiếp theo",
    "A deterministic diffusion round trip yields a structured residual.": "Vòng khuếch tán tất định tạo ra sai số tái tạo có cấu trúc.",
    "Paper results and repo results must stay in separate lanes.": "Kết quả bài báo và kết quả repo phải được trình bày tách biệt.",
    "Resolve dates, live state and metrics before more compute.": "Đối chiếu ngày, trạng thái hiện thời và số đo trước khi chạy thêm.",
    "Setting": "Thiết lập",
    "Value": "Giá trị",
    "Interpretation": "Diễn giải",
    "Model family": "Họ mô hình",
    "Input size": "Kích thước đầu vào",
    "Diffusion schedule": "Lịch khuếch tán",
    "Width / depth": "Độ rộng / độ sâu",
    "Attention": "Attention",
    "Output mode": "Chế độ đầu ra",
    "Weight provenance": "Nguồn gốc trọng số",
    "Pixel-space, unconditional": "Không gian pixel, vô điều kiện",
    "3-channel image tensor": "Tensor ảnh 3 kênh",
    "DIRE paper uses respaced DDIM S=20": "Bài DIRE dùng DDIM rút gọn S=20",
    "ADM config recorded in source": "Cấu hình ADM được ghi trong mã nguồn",
    "64 channels per head": "64 kênh mỗi head",
    "unconditional bedroom model": "mô hình phòng ngủ vô điều kiện",
    "not trained by this project": "không do dự án này huấn luyện",
    "REPOSITORY-CHECKED HASH IN LEDGER": "HASH ĐƯỢC ĐỐI CHIẾU TRONG SỔ REPO",
    "DDPM · stochastic reverse": "DDPM · chiều ngược ngẫu nhiên",
    "DDIM · deterministic path": "DDIM · quỹ đạo tất định",
    "η = 0 loại thành phần noise bổ sung.": "η = 0 loại bỏ thành phần nhiễu bổ sung.",
    "Cùng xₜ và protocol → cùng x₀.": "Cùng xₜ và giao thức → cùng x₀.",
    "Dạng rút gọn DDIM khi η=0; không có thành phần nhiễu ngẫu nhiên.": "Dạng DDIM rút gọn khi η=0; không thêm thành phần nhiễu ngẫu nhiên.",
    "sampling · t giảm": "lấy mẫu · t giảm",
    "inversion · t tăng": "đảo · t tăng",
    "NFE = number of function evaluations of the diffusion network.": "NFE = số lượt đánh giá mạng khuếch tán.",
    "low cost": "chi phí thấp",
    "paper reference": "mốc tham chiếu bài báo",
    "high cost": "chi phí cao",
    "Input": "Đầu vào",
    "Invert": "Đảo",
    "Reconstruct": "Tái tạo",
    "Residual": "Sai số",
    "RGB image": "ảnh RGB",
    "DIRE map": "bản đồ DIRE",
    "Absolute error, pixel-wise and channel-wise; output is a 3-channel image.": "Sai số tuyệt đối theo từng pixel và kênh; đầu ra là ảnh 3 kênh.",
    "Train detector": "Bộ phân loại huấn luyện trên",
    "Every cell is shown as ACC / AP (%), reported by the authors. These are not measurements from the project.": "Mỗi ô ghi ACC / AP (%), do tác giả báo cáo; đây không phải số đo của dự án.",
    "PAPER-REPORTED · ICCV 2023 TABLE 2": "SỐ LIỆU BÀI BÁO · TABLE 2 · ICCV 2023",
    "Table 2 headline is specific to its dataset, split, reference weights and evaluation protocol.": "Kết quả Table 2 phụ thuộc tập dữ liệu, cách chia, trọng số tham chiếu và giao thức đánh giá.",
    "Train source": "Nguồn huấn luyện",
    "Lowest shown cell: ACC 76.9% · cross-domain generalization is a real limitation.": "Ô thấp nhất: ACC 76,9% · tổng quát hóa khác miền là giới hạn thực sự.",
    "Steps": "Số bước",
    "Representation": "Biểu diễn",
    "Generator": "Bộ sinh",
    "Test generator": "Bộ sinh kiểm thử",
    "Protocol field": "Trường giao thức",
    "Repo-recorded value": "Giá trị ghi trong repo",
    "Subset / seed": "Tập con / seed",
    "Sampling": "Cách lấy mẫu",
    "Images / cell": "Ảnh / ô",
    "Train generator": "Bộ sinh huấn luyện",
    "Test generators": "Các bộ sinh kiểm thử",
    "Detector initialization": "Khởi tạo bộ phân loại",
    "Training actually recorded": "Huấn luyện được ghi nhận",
    "Diffusion reference hash": "Hash checkpoint khuếch tán",
    "Manifest hash": "Hash manifest",
    "Run tag": "Mã lượt chạy",
    "Best epoch": "Epoch tốt nhất",
    "TEST N": "CỠ MẪU KIỂM THỬ",
    "ACC · VAL THRESHOLD": "ACC · NGƯỠNG CHỌN TRÊN VAL",
    "AUDIT FLAG:": "CẢNH BÁO KIỂM TOÁN:",
    "Do not recompute, combine or call these counters a reconciled total.": "Không tính lại, cộng gộp hoặc gọi các bộ đếm này là tổng đã đối soát.",
    "Repository inventory adds DALL·E 2, IF, projected GAN, hybrids and other tags.": "Danh mục repo có thêm DALL·E 2, IF, projected GAN, mô hình lai và các nhãn khác.",
    "Surface": "Giao diện",
    "Current purpose": "Mục đích hiện tại",
    "Boundary": "Giới hạn",
    "Current checkout": "Checkout hiện tại",
    "HF Spaces / Gradio": "HF Spaces / Gradio",
    "Public explainer, gallery, optional DIRE endpoint": "Trang giải thích công khai, thư viện ảnh, endpoint DIRE tùy chọn",
    "Endpoint computes maps; no detector prediction on raw RGB": "Endpoint tính map; không phân loại ảnh RGB thô",
    "Streamlit fallback": "Bản dự phòng Streamlit",
    "Demo backup UI": "Giao diện demo dự phòng",
    "Shares public demo constraints": "Giữ cùng các giới hạn của demo công khai",
    "Private research prototype": "Bản thử nghiệm nghiên cứu riêng",
    "Paper refs + repo ledgers + gallery + image preview": "Tham chiếu bài báo + sổ repo + gallery + xem trước ảnh",
    "No Modal calls, no checkpoints, no prediction": "Không gọi Modal, không checkpoint, không dự đoán",
    "On-demand / generator": "Theo yêu cầu / bộ sinh",
    "Future roadmap": "Kế hoạch tương lai",
    "UPLOAD ≠ SCORE · RGB MUST NOT GO TO DIRE DETECTOR": "TẢI ẢNH ≠ ĐIỂM SỐ · RGB THÔ KHÔNG VÀO DIRE",
    "PILOT HARD CAP": "GIỚI HẠN CỨNG CỦA PILOT",
    "all GPU cost / accrued + active + new": "tổng chi phí GPU / đã phát sinh + đang chạy + phát sinh mới",
    "Before any new GPU job": "Trước mọi job GPU mới",
    "Check Modal app status, Volume ledger/artifacts, account spend and remaining headroom.": "Kiểm tra trạng thái app Modal, sổ/hiện vật Volume, chi tiêu tài khoản và hạn mức còn lại.",
    "Stop only on evidence": "Chỉ dừng khi có căn cứ",
    "Stop only to protect the cap or with evidence of a stuck/non-progressing job; inspect artifacts first.": "Chỉ dừng để bảo vệ hạn mức hoặc khi có bằng chứng job bị kẹt; kiểm tra hiện vật trước.",
    "The deck build is offline/read-only. It did not launch, stop, upload, modify or delete Modal data.": "Tạo slide ngoại tuyến/chỉ đọc; không khởi chạy, dừng, tải lên, sửa hay xóa dữ liệu Modal.",
    "Mechanism": "Cơ chế",
    "Domain shift": "Dịch chuyển miền",
    "Robustness": "Độ bền vững",
    "Threat model": "Mô hình đe dọa",
    "Data breadth": "Độ bao phủ dữ liệu",
    "Uncertainty": "Độ bất định",
    "Reconcile chronology": "Đối soát trình tự thời gian",
    "Verify commit / artifact timestamps and provenance.": "Xác minh mốc thời gian commit/hiện vật và nguồn gốc.",
    "Read-only Modal audit": "Kiểm toán Modal chỉ đọc",
    "Check live apps, Volume ledgers, spend and headroom.": "Kiểm tra app đang hoạt động, sổ Volume, chi tiêu và hạn mức.",
    "Audit metrics": "Kiểm toán số đo",
    "Repair report counter mismatch and CI inconsistency.": "Sửa lệch bộ đếm trong báo cáo và bất nhất của khoảng tin cậy.",
    "Verify checkpoints": "Xác minh checkpoint",
    "Locate binaries; validate hashes and preprocessing.": "Tìm tệp nhị phân; kiểm tra hash và tiền xử lý.",
    "Complete ledgers": "Hoàn thiện sổ kết quả",
    "Table 5 + Figure 4 only if cap permits.": "Chỉ chạy Table 5 + Hình 4 nếu hạn mức cho phép.",
    "Inference safely": "Suy luận an toàn",
    "Add cost-guarded on-demand path; keep image handling explicit.": "Thêm luồng theo yêu cầu có chặn chi phí; quy định rõ cách xử lý ảnh.",
    "Do not start new GPU jobs until steps 01–03 establish valid headroom under $30.": "Không chạy job GPU mới cho đến khi bước 01–03 xác nhận còn hạn mức dưới trần $30.",
    "Component": "Thành phần",
    "Reference dependency": "Phụ thuộc tham chiếu",
    "Clean-room counterpart": "Bản triển khai độc lập",
    "Verification idea": "Cách kiểm chứng",
    "Failure surface": "Điểm có thể lỗi",
    "How it misleads": "Cách gây hiểu nhầm",
    "Guard / correction": "Chốt chặn / cách sửa",
    "Paper benchmark": "Benchmark bài báo",
    "Repository extensions / uncertainty": "Mở rộng repo / bất định",
    "Paper Table 4": "Table 4 của bài báo",
    "Repo pilot Table 4": "Pilot Table 4 của repo",
    "Dataset / scale": "Tập dữ liệu / quy mô",
    "Detector training": "Huấn luyện bộ phân loại",
    "Use": "Mục đích sử dụng",
    "Artifact": "Hiện vật",
    "Recorded SHA-256 (prefix)": "SHA-256 ghi nhận (tiền tố)",
    "What is / is not verified": "Điều đã / chưa xác minh",
    "Stage": "Thiết lập",
    "Epochs run": "Epoch đã chạy",
    "Val AP": "AP trên val",
    "Val threshold": "Ngưỡng trên val",
    "Crop": "Crop",
    "Before": "Trước khi chạy",
    "Per cell": "Theo từng ô",
    "After": "Sau khi chạy",
    "For claims": "Khi đưa ra kết luận",
    "Data / Volume": "Dữ liệu / Volume",
    "manifest · maps / ckpt · run ledger": "manifest · maps / checkpoint · sổ chạy",
    "preprocess · detector / score · batch · provenance": "tiền xử lý · bộ phân loại / điểm · batch · nguồn gốc",
    "CLI + local API": "CLI + API cục bộ",
    "Modal wrappers / remote GPU paths": "lớp bọc Modal / luồng GPU từ xa",
    "Other surfaces:": "Các giao diện khác:",
    "PAPER BENCHMARK": "BENCHMARK BÀI BÁO",
    "REPO EXTENSIONS / UNCERTAINTY": "MỞ RỘNG REPO / BẤT ĐỊNH",
    "MANIFEST-ONLY": "CHỈ CÓ MANIFEST",
    "MAP INVENTORY": "TỒN KHO MAP",
    "DDIM SETTING": "THIẾT LẬP DDIM",
    "NFE IN REPORT": "NFE TRONG BÁO CÁO",
    "MANIFEST": "MANIFEST",
    "ORIGINAL · ADM / test · R1+R2": "ẢNH GỐC · ADM / test · R1+R2",
    "DIRE MAP · S=20 · display PNG": "DIRE MAP · S=20 · PNG hiển thị",
    "clean": "sạch",
    "baseline": "đối chứng",
    "Gaussian": "Gaussian",
    "compression": "nén",
    "NOT RUN / NO LEDGER": "CHƯA CHẠY / CHƯA CÓ SỔ",
    "PILOT · n=400 PER GENERATOR CELL · REPO LEDGER": "PILOT · n=400 MỖI Ô BỘ SINH · SỔ REPO",
    "AP · test": "AP · tập kiểm thử",
    "n≈1,000 per test cell; paper provides no CIs. Differences of a few tenths should not be overstated.": "n≈1.000 mỗi ô kiểm thử; bài báo không cung cấp CI. Không nên phóng đại chênh lệch vài phần mười.",
    "This is a project evaluation on self-computed DIRE maps with a from-scratch detector; it is not an absolute Table 2 comparison.": "Đây là đánh giá dự án trên DIRE map tự tính và detector huấn luyện từ đầu; không phải so sánh trực tiếp với Table 2.",
    "The source protocol uses SD-v2; that differs from this paper SD-v1 table. These columns are deliberately not combined.": "Giao thức nguồn dùng SD-v2, khác SD-v1 trong bảng bài báo; cố ý không gộp hai cột này.",
    "Related work, not a result or feature implemented by this project's prototype.": "Công trình liên quan; không phải kết quả hay tính năng đã triển khai trong prototype.",
    "Paper protocol and full test data; S=20→50 multiplies NFE by 2.5. These rows are paper-reported, not the project's Table 4 ledger.": "Giao thức và dữ liệu kiểm thử đầy đủ của bài báo; các dòng này là số liệu công bố, không phải sổ pilot Table 4 của dự án.",
    "Paper protocol": "Giao thức bài báo",
    "Paper reference": "Tham chiếu bài báo",
    "Pilot measurement candidate; time provenance unresolved": "Số đo pilot; nguồn gốc thời gian chưa đối soát",
    "Paper and pilot use different generators and execution scales": "Bài báo và pilot dùng bộ sinh, quy mô thực thi khác nhau",
    "Paper full-data S sweep uses SD-v1; repo pilot uses SD-v2 and 200/cell with S=50 missing.": "Quét S toàn bộ của bài báo dùng SD-v1; pilot repo dùng SD-v2 và 200 ảnh/ô.",
    "Paper Table 4 shows an S-dependent trade-off; do not substitute it for pilot data": "Table 4 bài báo cho thấy đánh đổi theo S; không thay bằng dữ liệu pilot",
    "Paper Table 5 supports DIRE over RGB or reconstruction alone": "Table 5 bài báo ủng hộ DIRE hơn RGB hoặc ảnh tái tạo riêng lẻ",
    "The paper's representation ablation is available; this project's Table 5 remains unmeasured": "Bài báo có ablation biểu diễn; Table 5 của dự án chưa được đo",
    "The project has a robustness protocol, but no Figure 4 results ledger": "Dự án có giao thức độ bền vững nhưng chưa có sổ kết quả Hình 4",
    "Changing dataset domain can be harder than changing generator family": "Đổi miền dữ liệu có thể khó hơn đổi họ bộ sinh",
    "Author-reported mean ACC ranges from 97.9% to 99.9% across training generators": "ACC trung bình do tác giả báo cáo nằm trong khoảng 97,9%–99,9% theo bộ sinh huấn luyện",
    "Paper Table 2 reports strong cross-generator results for four DIRE detectors": "Table 2 báo cáo khả năng tổng quát hóa chéo bộ sinh cao ở bốn detector DIRE",
    "Paper Figure 4 reports qualitative robustness to blur and JPEG compression": "Hình 4 báo cáo định tính độ bền vững trước làm mờ và nén JPEG",
    "AEROBLADE shares the reconstruction-error intuition but uses an autoencoder": "AEROBLADE cùng trực giác sai số tái tạo nhưng dùng autoencoder",
    "R1+R2 means self-computed maps and self-trained detector — not self-generated images": "R1+R2 nghĩa là tự tính map và tự huấn luyện detector — không phải tự sinh ảnh",
    "An archived ADM test pair links an original image to a display-only DIRE map": "Một cặp ADM đã lưu nối ảnh gốc với DIRE map chỉ dùng để hiển thị",
    "Pilot protocol shares a fixed 200-per-cell subset across S settings": "Giao thức pilot dùng chung tập con 200 ảnh/ô cho các mức S",
    "The reverse step is not an algebraically exact inverse after finite-step discretization, clipping, or finite precision. That non-zero reconstruction error is the representation being measured.": "Bước ngược không phải phép đảo đại số chính xác khi số bước hữu hạn, có clipping hoặc sai số số học. Sai số tái tạo khác không chính là biểu diễn được đo.",
    "Paper's qualitative claim: DIRE changes little; RGB baselines degrade more. The plot's pixel values are not tabulated in the cited paper copy, so this deck does not invent exact AP points.": "Nhận định định tính của bài báo: DIRE thay đổi ít, baseline RGB suy giảm nhiều hơn. Bản bài báo trích dẫn không công bố số pixel của đồ thị nên slide không tự đặt điểm AP.",
    "R1+R2 means self-computed maps and self-trained detector — not self-generated images": "R1+R2 nghĩa là tự tính map và tự huấn luyện bộ phân loại — không phải tự sinh ảnh",
    "Paper Table 4 and project Table 4 are not directly comparable experiments": "Table 4 bài báo và pilot dự án không phải thí nghiệm so sánh trực tiếp",
    "The strongest deliverable today is a transparent evidence system — not a full reproduction": "Sản phẩm vững nhất hiện nay là hệ thống bằng chứng minh bạch — chưa phải tái lập toàn bộ",
    "High paper scores do not close the mechanism, domain and deployment questions": "Điểm cao trong bài báo chưa giải đáp câu hỏi về cơ chế, miền dữ liệu và triển khai",
    "A completed scientific cell requires metrics, labels, counts and compatible provenance": "Một ô thí nghiệm hoàn chỉnh cần chỉ số, nhãn, số lượng và nguồn gốc tương thích",
    "DIRE ResNet emits one logit; sigmoid converts it to P(fake)": "ResNet DIRE tạo một logit; sigmoid chuyển thành P(fake)",
    "Recorded test AP remains high at S=5/10/20; S=50 is absent, not extrapolated": "AP kiểm thử đã ghi nhận tại S=5/10/20; không vẽ ngoại suy",
    "Pooled cells reveal weaker PNDM AP as S rises in this pilot snapshot": "Trong pilot này, AP của PNDM giảm khi S tăng",
    "Paper Table 4 and project Table 4 are not directly comparable experiments": "Table 4 bài báo và Table 4 dự án không phải hai thí nghiệm so sánh trực tiếp",
    "Table 5 inputs are staged; this checkout contains no project result ledger": "Đã chuẩn bị đầu vào Table 5; checkout này chưa có sổ kết quả dự án",
    "The project has a robustness protocol, but no Figure 4 results ledger": "Dự án có giao thức độ bền vững nhưng chưa có sổ kết quả Hình 4",
    "One manifest locks sample identity across Table 4, Table 5 and Figure 4": "Một manifest khóa định danh mẫu dùng chung cho Table 4, Table 5 và Hình 4",
    "Three experiment families share the subset identity but vary different factors": "Ba nhóm thí nghiệm dùng chung định danh tập con nhưng thay đổi các yếu tố khác nhau",
    "The clean-room path independently rewrites U-Net, DDIM, DIRE, ResNet and metrics": "Luồng triển khai độc lập viết lại U-Net, DDIM, DIRE, ResNet và metric",
    "Several silent-but-plausible bugs were found only by deriving the pipeline again": "Chỉ khi suy ra lại pipeline mới phát hiện được một số lỗi âm thầm nhưng có vẻ hợp lý",
    "Paper benchmark and repository extensions must remain separate": "Benchmark bài báo và phần mở rộng repo phải được tách biệt",
    "The research dashboard is read-only; upload preview is not model inference": "Dashboard nghiên cứu chỉ đọc; xem trước ảnh tải lên không phải suy luận mô hình",
    "The all-in GPU cap remains $30; live spend/headroom is unknown": "Trần tổng chi phí GPU vẫn là $30; chi tiêu và hạn mức còn lại chưa rõ",
    "High paper scores do not close the mechanism, domain and deployment questions": "Điểm cao trong bài báo chưa giải đáp câu hỏi về cơ chế, miền dữ liệu và triển khai",
    "Resolve evidence and cost gates before turning the dashboard into an inference service": "Giải quyết cổng bằng chứng và chi phí trước khi biến dashboard thành dịch vụ suy luận",
    "Primary sources and project evidence used in this deck": "Nguồn chính và bằng chứng dự án được dùng trong bộ slide",
    "Complete Table 4 pilot metrics from the checked-in ledger": "Toàn bộ metric pilot Table 4 từ sổ đã lưu trong repo",
    "Independent detector training metadata by S": "Metadata huấn luyện detector độc lập theo S",
    "Complete Table 4 pilot metrics from the checked-in ledger": "Toàn bộ metric pilot Table 4 từ sổ đã lưu trong repo",
    "Paper and pilot use different generators and execution scales": "Bài báo và pilot dùng bộ sinh, quy mô thực thi khác nhau",
    "A completed scientific cell requires metrics, labels, counts and compatible provenance": "Một ô thí nghiệm hoàn chỉnh cần metric, nhãn, số lượng và nguồn gốc tương thích",
    "Study path: source data → map cache → detector → ledger → report / display": "Luồng nghiên cứu: dữ liệu nguồn → cache map → detector → sổ kết quả → báo cáo / hiển thị",
    "Manifest + subset": "Manifest + tập con",
    "sample identity / seed / SHA": "định danh mẫu / seed / SHA",
    "DIRE generation": "Tính DIRE",
    "ADM + DDIM / S steps × 2": "ADM + DDIM / 2S bước",
    "Representation": "Biểu diễn",
    "DIRE / RGB / REC / per-image cache": "DIRE / RGB / REC / cache theo ảnh",
    "ResNet + metrics": "ResNet + metric",
    "split / threshold / ACC · AP · AUC": "tập chia / ngưỡng / ACC · AP · AUC",
    "Artifact handoff: immutable input protocol → result ledger → sourced tables / plots → slides": "Bàn giao hiện vật: giao thức đầu vào bất biến → sổ kết quả → bảng / đồ thị có nguồn → slide",
    "Before": "Trước",
    "Per cell": "Theo ô",
    "After": "Sau",
    "For claims": "Khi kết luận",
    "The project's representation ablation is available; this project's Table 5 remains unmeasured": "Bài báo có ablation biểu diễn; Table 5 của dự án chưa được đo",
    "The paper's representation ablation is available; this project's Table 5 remains unmeasured": "Bài báo có ablation biểu diễn; Table 5 của dự án chưa được đo",
    "Table 2 source values: complete 4 × 8 DIRE block (ACC / AP, %)": "Số liệu nguồn Table 2: đầy đủ khối DIRE 4 × 8 (ACC / AP, %)",
    "Author-reported mean ACC ranges from 97.9% to 99.9% across training generators": "ACC trung bình do tác giả báo cáo nằm trong khoảng 97,9%–99,9% theo bộ sinh huấn luyện",
    "Changing dataset domain can be harder than changing generator family": "Đổi miền dữ liệu có thể khó hơn đổi họ bộ sinh",
    "The paper's qualitative claim: DIRE changes little; RGB baselines degrade more.": "Nhận định định tính của bài báo: DIRE thay đổi ít; baseline RGB suy giảm nhiều hơn.",
    "The paper's qualitative claim: DIRE changes little; RGB baselines degrade more. The plot's pixel values are not tabulated in the cited paper copy, so this deck does not invent exact AP points.": "Nhận định định tính của bài báo: DIRE thay đổi ít; baseline RGB suy giảm nhiều hơn. Bài báo không công bố số pixel trong đồ thị nên slide không tự đặt điểm AP.",
    "The paper's representation ablation is available; this project's Table 5 remains unmeasured": "Bài báo có ablation biểu diễn; Table 5 của dự án chưa được đo",
    "Paper Table 4 and project Table 4 are not directly comparable experiments": "Table 4 bài báo và pilot dự án không phải thí nghiệm so sánh trực tiếp",
    "Table 5 inputs are staged; this checkout contains no project result ledger": "Đầu vào Table 5 đã chuẩn bị; checkout này chưa có sổ kết quả dự án",
    "The project has a robustness protocol, but no Figure 4 results ledger": "Dự án có giao thức độ bền vững nhưng chưa có sổ kết quả Hình 4",
    "AEROBLADE shares the reconstruction-error intuition but uses an autoencoder": "AEROBLADE cùng trực giác sai số tái tạo nhưng dùng autoencoder",
    "DIRE ResNet emits one logit; sigmoid converts it to P(fake)": "ResNet DIRE tạo một logit; sigmoid chuyển thành P(fake)",
    "Paper Table 2 reports strong cross-generator results for four DIRE detectors": "Table 2 báo cáo khả năng tổng quát hóa chéo bộ sinh cao ở bốn detector DIRE",
    "Paper Table 5 supports DIRE over RGB or reconstruction alone": "Table 5 bài báo ủng hộ DIRE hơn RGB hoặc ảnh tái tạo riêng lẻ",
    "Paper Figure 4 reports qualitative robustness to blur and JPEG compression": "Hình 4 báo cáo định tính độ bền vững trước làm mờ và nén JPEG",
    "A deterministic diffusion round trip yields a structured residual.": "Vòng khuếch tán tất định tạo ra sai số tái tạo có cấu trúc.",
    "Paper results and repo results must stay in separate lanes.": "Kết quả bài báo và số đo repo phải được trình bày riêng.",
    "Resolve dates, live state and metrics before more compute.": "Đối chiếu nguồn, trạng thái và metric trước khi diễn giải thêm.",
    "Property": "Thuộc tính",
    "Model family": "Họ mô hình",
    "Dataset / scale": "Tập dữ liệu / quy mô",
    "Detector training": "Huấn luyện bộ phân loại",
    "Train source": "Nguồn huấn luyện",
    "Test generator": "Bộ sinh kiểm thử",
    "Test generators": "Các bộ sinh kiểm thử",
    "Train generator": "Bộ sinh huấn luyện",
    "Steps": "Số bước",
    "Dimension": "Khía cạnh",
    "Value": "Giá trị",
    "Interpretation": "Diễn giải",
    "Setting": "Thiết lập",
    "Transform": "Phép biến đổi",
    "Domain / score": "Miền / điểm số",
    "Compute": "Tính toán",
    "Decision": "Quyết định",
    "Stage": "Giai đoạn",
    "Epochs run": "Epoch đã chạy",
    "Best epoch": "Epoch tốt nhất",
    "Val AP": "AP trên tập xác thực",
    "Val threshold": "Ngưỡng trên tập xác thực",
    "Threshold": "Ngưỡng",
    "Current purpose": "Mục đích hiện tại",
    "Boundary": "Giới hạn",
    "Source": "Nguồn",
    "Artifact": "Hiện vật",
    "Reported": "Được báo cáo",
    "results": "kết quả",
    "metrics": "metric",
    "training": "huấn luyện",
    "evaluation": "đánh giá",
    "threshold": "ngưỡng",
    "generator": "bộ sinh",
    "model": "mô hình",
    "layers": "các lớp",
    "layer": "lớp",
    "blocks": "các khối",
    "block": "khối",
    "attention": "attention",
    "features": "đặc trưng",
    "feature": "đặc trưng",
    "image": "ảnh",
    "images": "ảnh",
    "input": "đầu vào",
    "output": "đầu ra",
    "reconstruction": "tái tạo",
    "residual": "sai số tái tạo",
    "source": "nguồn",
    "result": "kết quả",
    "report": "báo cáo",
    "project": "dự án",
    "paper": "bài báo",
    "current": "hiện tại",
    "missing": "chưa có",
    "complete": "đầy đủ",
    "full": "toàn bộ",
    "only": "chỉ",
    "with": "với",
    "without": "không có",
    "across": "xuyên",
    "between": "giữa",
    "per": "mỗi",
    "clean": "sạch",
    "blur": "làm mờ",
    "compression": "nén",
    "baseline": "đối chứng",
    "representation": "biểu diễn",
    "sample": "mẫu",
    "samples": "mẫu",
    "records": "bản ghi",
    "average": "trung bình",
    "accuracy": "độ chính xác",
    "score": "điểm số",
    "robustness": "độ bền vững",
    "references": "tài liệu tham khảo",
    "implementation": "triển khai",
    "independent": "độc lập",
    "identity": "đồng nhất",
    "shortcut": "nhánh tắt",
    "global": "toàn cục",
    "local": "cục bộ",
    "stochastic": "ngẫu nhiên",
    "deterministic": "tất định",
    "trained": "được huấn luyện",
    "train": "huấn luyện",
    "Train": "Huấn luyện",
    "test": "kiểm thử",
    "Test": "Kiểm thử",
    "tested": "được kiểm thử",
    "uses": "dùng",
    "use": "dùng",
    "used": "được dùng",
    "learns": "học",
    "learned": "đã học",
    "emits": "tạo ra",
    "converts": "chuyển thành",
    "reports": "báo cáo",
    "reported": "được báo cáo",
    "remains": "vẫn",
    "remain": "vẫn",
    "requires": "cần",
    "supports": "hỗ trợ",
    "compares": "so sánh",
    "compare": "so sánh",
    "degrade": "suy giảm",
    "degrades": "suy giảm",
    "include": "bao gồm",
    "includes": "bao gồm",
    "available": "có sẵn",
    "currently": "hiện nay",
    "actual": "thực tế",
    "author": "tác giả",
    "authors": "các tác giả",
    "Report": "Báo cáo",
    "results": "kết quả",
    "scores": "điểm số",
    "score": "điểm số",
    "metric": "chỉ số",
    "metrics": "các chỉ số",
    "maps": "bản đồ",
    "reconstruct": "tái tạo",
    "inversion": "phép đảo",
    "architecture": "kiến trúc",
    "protocol": "giao thức",
    "evaluation": "đánh giá",
    "detector": "bộ phân loại",
    "detectors": "các bộ phân loại",
    "models": "các mô hình",
    "dataset": "tập dữ liệu",
    "datasets": "các tập dữ liệu",
    "domain": "miền",
    "reference": "tham chiếu",
    "references": "tài liệu tham khảo",
    "property": "thuộc tính",
    "properties": "các thuộc tính",
    "dimension": "khía cạnh",
    "dimensions": "các khía cạnh",
    "setting": "thiết lập",
    "settings": "các thiết lập",
    "value": "giá trị",
    "values": "các giá trị",
    "inputs": "đầu vào",
    "outputs": "đầu ra",
    "layers": "các lớp",
    "blocks": "các khối",
    "step": "bước",
    "steps": "các bước",
    "samples": "mẫu",
    "records": "bản ghi",
    "record": "bản ghi",
    "cell": "ô",
    "cells": "các ô",
    "scale": "quy mô",
    "execution": "thực thi",
    "maximum": "tối đa",
    "about": "khoảng",
    "real": "thật",
    "fake": "giả",
    "unmeasured": "chưa đo",
    "measured": "được đo",
    "baseline": "đối chứng",
    "mild": "nhẹ",
    "medium": "vừa",
    "strong": "mạnh",
    "heavy": "mạnh",
    "compression": "nén",
    "Global": "Toàn cục",
    "and": "và",
    "or": "hoặc",
    "of": "của",
    "the": "",
    "a": "",
    "an": "",
    "in": "trong",
    "on": "trên",
    "at": "tại",
    "from": "từ",
    "with": "với",
    "without": "không có",
    "for": "cho",
    "by": "theo",
    "across": "qua",
    "between": "giữa",
    "under": "dưới",
    "over": "trên",
    "to": "tới",
    "than": "hơn",
    "as": "như",
    "each": "mỗi",
    "all": "toàn bộ",
    "only": "chỉ",
    "not": "không",
    "is": "là",
    "are": "là",
    "was": "đã",
    "were": "đã",
    "has": "có",
    "have": "có",
    "can": "có thể",
    "does": "làm",
    "this": "này",
    "these": "này",
    "those": "đó",
    "their": "của chúng",
    "its": "của nó",
    "into": "thành",
    "where": "nơi",
    "when": "khi",
    "because": "vì",
    "while": "trong khi",
    "but": "nhưng",
    "yet": "tuy nhiên",
    "instead": "thay vào đó",
    "higher": "cao hơn",
    "lower": "thấp hơn",
    "complete": "hoàn chỉnh",
    "full": "đầy đủ",
    "qualitative": "định tính",
    "quantitative": "định lượng",
    "one": "một",
    "two": "hai",
    "first": "đầu tiên",
    "second": "thứ hai",
    "means": "có nghĩa là",
    "range": "khoảng",
    "Diffusion Reconstruction Error · nền tảng toán học, kiến trúc mô hình,\\nđối chiếu paper và trạng thái pilot theo artifacts trong repo": "Sai số tái tạo khuếch tán (DIRE) · nền tảng toán học, kiến trúc mô hình,\\nđối chiếu bài báo và trạng thái thử nghiệm theo hiện vật trong kho mã",
    "Editable PPTX + PDF  ·  equations typeset from LaTeX source": "PPTX + PDF có thể chỉnh sửa · công thức được dàn trang từ mã LaTeX",
    "Train and test can differ in p_g: recognizing one generator's fingerprint is not the same as general synthetic-image detection.": "Phân phối bộ sinh p_g có thể khác giữa huấn luyện và kiểm thử: nhận diện dấu vết của một bộ sinh không đồng nghĩa phát hiện ảnh tổng hợp nói chung.",
    "A model may learn generator fingerprints rather than a robust cue.": "Mô hình có thể học dấu vân tay của bộ sinh thay vì dấu hiệu bền vững.",
    "Find a representation that remains useful when test generator ≠ train generator.": "Tìm biểu diễn vẫn hữu ích khi bộ sinh kiểm thử khác bộ sinh huấn luyện.",
    "VAE latent diffusion": "Khuếch tán trong không gian ẩn VAE",
    "Pixel diffusion / ADM": "Khuếch tán trên pixel / ADM",
    "Autoregressive, EBM and flow-matching models are outside the HTML survey's scope; do not generalize from this table.": "Mô hình tự hồi quy, EBM và flow-matching nằm ngoài phạm vi tài liệu HTML; không khái quát hóa từ bảng này.",
    "Better reconstruction ↔ posterior in latent space is close to the prior; the encoder only approximates the posterior.": "Tái tạo tốt hơn ↔ hậu nghiệm trong không gian ẩn gần với phân phối tiên nghiệm; bộ mã hóa chỉ xấp xỉ hậu nghiệm.",
    "VAE is the latent-space encoder/decoder inside latent diffusion models.": "VAE là bộ mã hóa/giải mã không gian ẩn bên trong mô hình khuếch tán tiềm ẩn.",
    "GAN learns an adversarial generator; it has no explicit image-to-latent inverse.": "GAN học bộ sinh đối kháng, không có phép nghịch đảo ảnh–không gian ẩn tường minh.",
    "A basic GAN has no integrated likelihood or inverse; latent recovery requires separate optimization.": "GAN cơ bản không có mật độ xác suất hay phép nghịch đảo tích hợp; cần tối ưu riêng để tìm không gian ẩn.",
    "An ideal normalizing flow is exactly invertible — eliminating DIRE reconstruction-error signal.": "Dòng chuẩn hóa lý tưởng có phép nghịch đảo chính xác — làm mất tín hiệu sai số tái tạo DIRE.",
    "IDEAL CONSEQUENCE / Choose I=f and R=f⁻¹ (or vice versa): R(I(x))=x for all x, so DIRE(x)=0. This is a mathematical implication of exact invertibility — not an experimental result in the DIRE paper.": "HỆ QUẢ LÝ TƯỞNG / Chọn I=f và R=f⁻¹ (hoặc ngược lại): R(I(x))=x với mọi x, nên DIRE(x)=0. Đây là hệ quả toán học của khả năng nghịch đảo chính xác, không phải kết quả thực nghiệm trong bài DIRE.",
    "VQ-Diffusion and latent diffusion move the diffusion trajectory into a different space.": "VQ-Diffusion và khuếch tán tiềm ẩn đưa quỹ đạo khuếch tán sang không gian khác.",
    "LDM diffuses continuously in VAE latent space.": "LDM khuếch tán liên tục trong không gian ẩn VAE.",
    "Discrete diffusion over codebook tokens; quantization error belongs to the representation. This model is in the paper's generator benchmark.": "Khuếch tán rời rạc trên token codebook; sai số lượng tử hóa thuộc về biểu diễn. Mô hình này có trong benchmark bộ sinh của bài báo.",
    "VAE encoder → latent-space DDIM trajectory → VAE decoder. The DIRE paper still uses pixel-space ADM as its reconstruction reference.": "Bộ mã hóa VAE → quỹ đạo DDIM trong không gian ẩn → bộ giải mã VAE. Bài DIRE vẫn dùng ADM trên pixel làm tham chiếu tái tạo.",
    "The model-space difference matters: the detected generator is not the reconstruction backbone.": "Khác biệt về không gian mô hình rất quan trọng: bộ sinh được phát hiện không phải backbone tái tạo.",
    "Same backbone, schedule and timesteps; inversion is approximate, not stochastic.": "Giữ nguyên backbone, lịch khuếch tán và timestep; phép đảo là xấp xỉ, không phải ngẫu nhiên.",
    "The residual distributions of real and generated images must differ; this must be measured.": "Phân phối sai số tái tạo của ảnh thật và ảnh tổng hợp phải khác nhau; cần đo kiểm.",
    "ADM uses a linear β_t schedule; cumulative ᾱ_t gradually reduces the signal.": "ADM dùng lịch β_t tuyến tính; ᾱ_t tích lũy làm tín hiệu suy giảm dần.",
    "ε in the forward process explains the noise target during diffusion training; DDIM inference chooses a deterministic path.": "ε trong quá trình thuận là mục tiêu nhiễu khi huấn luyện diffusion; suy luận DDIM chọn quỹ đạo tất định.",
    "ADM uses epsilon prediction. Timestep embedding supplies t to every block; the simplified loss is a practical training form related to ELBO.": "ADM dự đoán epsilon. Nhúng timestep cung cấp t cho mọi khối; hàm mất mát rút gọn là dạng huấn luyện thực dụng liên hệ với ELBO.",
    "ADM uses an encoder–bottleneck–decoder with multi-scale skip connections": "ADM dùng bộ mã hóa–nút thắt–giải mã với các kết nối tắt đa tỉ lệ",
    "Timestep embedding enters each stage; attention is placed at 32×32, 16×16 and 8×8 feature maps.": "Nhúng timestep đi vào từng tầng; attention được đặt tại bản đồ đặc trưng 32×32, 16×16 và 8×8.",
    "SKIP CONNECTIONS · preserve shallow detail across the bottleneck": "KẾT NỐI TẮT · giữ chi tiết tầng nông qua nút thắt",
    "Down: reduce resolution, increase representation capacity. Bottleneck: global context. Up: restore detail through skips. This is an architecture diagram, not a tensor visualization.": "Đi xuống: giảm độ phân giải, tăng năng lực biểu diễn. Nút thắt: ngữ cảnh toàn cục. Đi lên: khôi phục chi tiết qua kết nối tắt. Đây là sơ đồ kiến trúc, không phải trực quan hóa tensor.",
    "Block errors learn local transforms; attention exchanges global information.": "Các khối sai số học phép biến đổi cục bộ; attention trao đổi thông tin toàn cục.",
    "Q, K, V projections / across spatial positions": "Phép chiếu Q, K, V / trên các vị trí không gian",
    "Local filters highlight vertical edges, horizontal edges and detail; each kernel produces a different feature map.": "Bộ lọc cục bộ làm nổi cạnh dọc, cạnh ngang và chi tiết; mỗi kernel tạo một bản đồ đặc trưng khác nhau.",
    "The main path transforms features; the shortcut adds them back before ReLU to preserve information and gradient flow.": "Nhánh chính biến đổi đặc trưng; nhánh tắt cộng lại trước ReLU để duy trì luồng thông tin và gradient.",
    "Toy-image illustration; not a feature map or attention score extracted from ADM/ResNet.": "Minh họa trên ảnh tổng hợp; không phải bản đồ đặc trưng hay điểm attention trích từ ADM/ResNet.",
    "The reverse step is not an algebraically exact inverse after finite-step discretization, clipping, or finite precision. That non-zero reconstruction error is the representation being measured.": "Bước ngược không phải phép đảo đại số chính xác khi rời rạc hóa số bước hữu hạn, clipping hoặc sai số số học. Sai số tái tạo khác không chính là biểu diễn được đo.",
    "The listed largest change: SD-v1 ACC 87.0 → 99.7 (percentage points).": "Mức thay đổi lớn nhất được liệt kê: ACC của SD-v1 tăng từ 87,0 lên 99,7 (điểm phần trăm).",
    "The result supports the chosen representation; it does not establish a universal mechanism.": "Kết quả ủng hộ biểu diễn được chọn, nhưng không chứng minh một cơ chế phổ quát.",
    "Gallery PNG is display-only: [0,2] → uint8 with gamma 0.5, RGB channels preserved. Do not feed that PNG back as authoritative data or infer source-map encoding from an extension.": "PNG trong gallery chỉ dùng để hiển thị: [0,2] → uint8 với gamma 0,5, giữ nguyên kênh RGB. Không dùng PNG này làm dữ liệu chuẩn hoặc suy đoán cách mã hóa map nguồn từ phần mở rộng tệp.",
    "Bottleneck counts [3, 4, 6, 3]; final pooled feature vector has 2,048 components. The DIRE checkpoint convention is fc: 2,048 → 1 logit.": "Các khối bottleneck có số lượng [3, 4, 6, 3]; vector đặc trưng sau pooling có 2.048 thành phần. Quy ước checkpoint DIRE: fc 2.048 → 1 logit.",
    "Projection W_s is used when stride or channel count changes.": "Dùng phép chiếu W_s khi stride hoặc số kênh thay đổi.",
    "The residual path supports gradient flow; it does not itself guarantee generalization.": "Đường residual hỗ trợ luồng gradient nhưng không tự bảo đảm khả năng tổng quát hóa.",
    "Label convention: y=1 denotes generated/fake in the binary detector.": "Quy ước nhãn: y=1 chỉ ảnh tổng hợp/giả trong bộ phân loại nhị phân.",
    "Training uses BCEWithLogitsLoss for numerical stability.": "Huấn luyện dùng BCEWithLogitsLoss để ổn định số học.",
    "Why one logit matters": "Vì sao một logit quan trọng",
    "The published head shape is [1, 2048]; a 2-logit softmax head is not interchangeable.": "Đầu ra được công bố có kích thước [1, 2048]; không thể thay bằng head softmax 2 logit.",
    "Why ImageNet initialization": "Vì sao khởi tạo ImageNet",
    "Transfer learning initializes visual features; the pilot protocol records ImageNet initialization and an independent detector for each S.": "Học chuyển giao khởi tạo đặc trưng thị giác; giao thức pilot ghi nhận khởi tạo ImageNet và một bộ phân loại riêng cho mỗi S.",
    "Area / discrete sum along precision–recall; threshold-free ranking summary.": "Diện tích / tổng rời rạc theo đường precision–recall; tóm tắt thứ hạng không phụ thuộc ngưỡng.",
    "Area under ROC; compare cautiously under domain shift.": "Diện tích dưới ROC; cần thận trọng khi so sánh nếu miền dữ liệu thay đổi.",
    "Protocol guard: choose epoch and threshold on validation; test is only for final evaluation.": "Ràng buộc giao thức: chọn epoch và ngưỡng trên tập xác thực; tập kiểm thử chỉ dùng đánh giá cuối.",
    "self-computed DIRE": "DIRE tự tính",
    "self-trained classifier": "bộ phân loại tự huấn luyện",
    "own generator / data": "bộ sinh / dữ liệu tự sở hữu",
    "Repo label reported: R1+R2. The diffusion checkpoint and dataset source are not project-trained; fake images are marked author-sourced, so do not claim R3.": "Nhãn ghi trong repo: R1+R2. Checkpoint diffusion và nguồn tập dữ liệu không do dự án huấn luyện; ảnh giả được đánh dấu là lấy từ tác giả, vì vậy không tuyên bố R3.",
    "RGB + DIRE can underperform DIRE alone: extra features may invite shortcut learning.": "RGB + DIRE có thể kém hơn DIRE riêng: đặc trưng bổ sung có thể khuyến khích học lối tắt.",
    "CLEAN · ADM inversion + reconstruction": "ADM · phép đảo và tái tạo",
    "LDM · VAE encode + decode": "LDM · mã hóa và giải mã VAE",
    "2S diffusion network evaluations": "2S lượt đánh giá mạng khuếch tán",
    "Broad generated-image benchmark": "Benchmark ảnh tổng hợp đa dạng",
    "Related work; not a result or feature implemented in the prototype.": "Công trình liên quan; không phải kết quả hay tính năng đã triển khai trong bản thử nghiệm.",
    "Recorded value": "Giá trị ghi nhận",
    "ImageNet ResNet-50 · seed 3407": "ResNet-50 ImageNet · seed 3407",
    "batch 32 · lr 1e−4 · crop 224 · early stopping": "batch 32 · lr 1e−4 · crop 224 · dừng sớm",
    "Different: SD-v1 vs SD-v2; paper full data vs pilot; paper vs 200-per-cell; documented training protocol differs. No delta = reproduction claim.": "Khác biệt: SD-v1 so với SD-v2; dữ liệu đầy đủ của bài báo so với pilot; bài báo so với 200 ảnh mỗi ô; giao thức huấn luyện được ghi nhận cũng khác. Không diễn giải chênh lệch thành tuyên bố tái lập.",
    "six-channel paired input": "đầu vào ghép cặp sáu kênh",
    "three-channel reconstruction-error map": "bản đồ sai số tái tạo ba kênh",
    "Repo protocol: S=20 · 40 NFE/image · 256×256 · float16 input. Inputs materialized; no Table 5 training/evaluation ledger found. REC needs reconstruction data, not only a reconstruction-error .npy.": "Giao thức repo: S=20 · 40 NFE/ảnh · 256×256 · đầu vào float16. Đầu vào đã được chuẩn bị; chưa tìm thấy sổ huấn luyện/đánh giá Table 5. REC cần dữ liệu tái tạo, không chỉ tệp sai số tái tạo .npy.",
    "The source protocol uses SD-v2, unlike SD-v1 in the paper table; these columns are intentionally not merged.": "Giao thức nguồn dùng SD-v2, khác SD-v1 trong bảng bài báo; hai cột này được giữ riêng.",
    "A manifest locks shared sample identities for Table 4, Table 5 and Figure 4": "Một manifest khóa định danh mẫu dùng chung cho Table 4, Table 5 và Hình 4",
    "NESTED SNAPSHOT 400 / CELL": "ẢNH CHỤP LỒNG 400 / Ô",
    "increase cap, keep IDs": "Tăng giới hạn, giữ ID",
    "Manifest-only: source images stay unchanged; seed 3407 · sha256-ranked-cell-v1 · cell counts in ledger.": "Chỉ cập nhật manifest: ảnh nguồn không đổi; seed 3407 · sha256-ranked-cell-v1 · số mẫu mỗi ô được ghi trong sổ.",
    "REC · RGB · RGB_DIRE · DIRE": "REC · RGB · RGB_DIRE · DIRE",
    "Paper and pilot use different generators and execution scales": "Bài báo và pilot dùng bộ sinh cùng quy mô thực thi khác nhau",
    "ImageNet init, batch 32, LR 1e−4, early stopped": "Khởi tạo ImageNet, batch 32, LR 1e−4, dừng sớm",
}

# Additional standalone scientific/report vocabulary; phrase translations above take priority.
VI_COPY.update({
    "generated": "tạo sinh", "generators": "các bộ sinh", "fingerprint": "dấu vân tay", "fingerprints": "dấu vân tay",
    "robust": "bền vững", "cue": "dấu hiệu", "same": "cùng", "different": "khác", "useful": "hữu ích",
    "Gaussian": "Gauss", "sampling": "lấy mẫu", "noise": "nhiễu", "noisy": "nhiễu", "path": "quỹ đạo",
    "schedule": "lịch", "timestep": "bước thời gian", "timesteps": "các bước thời gian", "inference": "suy luận",
    "forward": "thuận", "reverse": "ngược", "deterministic": "tất định", "determinist": "tất định",
    "supervision": "giám sát", "target": "mục tiêu", "embedding": "nhúng", "loss": "hàm mất mát",
    "resolution": "độ phân giải", "stage": "tầng", "stages": "các tầng", "bottleneck": "nút thắt",
    "context": "ngữ cảnh", "projection": "phép chiếu", "projections": "các phép chiếu", "positions": "vị trí",
    "kernel": "kernel", "main": "chính", "information": "thông tin", "preserve": "duy trì", "preserves": "duy trì",
    "extracted": "trích xuất", "selected": "được chọn", "absolute": "tuyệt đối", "signed": "có dấu",
    "largest": "lớn nhất", "change": "mức thay đổi", "percentage": "phần trăm", "points": "điểm",
    "depends": "phụ thuộc", "fixed": "cố định", "ranking": "xếp hạng", "summary": "tóm tắt",
    "cautiously": "thận trọng", "caution": "thận trọng", "shift": "dịch chuyển", "guard": "ràng buộc",
    "choose": "chọn", "chosen": "được chọn", "epoch": "epoch", "validation": "xác thực", "final": "cuối",
    "published": "được công bố", "head": "đầu ra", "shape": "kích thước", "interchangeable": "có thể thay thế",
    "initialization": "khởi tạo", "transfer": "chuyển giao", "visual": "thị giác", "independent": "độc lập",
    "binary": "nhị phân", "denotes": "chỉ", "numerical": "số học", "stability": "ổn định",
    "area": "diện tích", "discrete": "rời rạc", "along": "theo", "recall": "recall", "carefully": "thận trọng",
    "related": "liên quan", "supports": "ủng hộ", "checkpoint": "checkpoint", "dataset": "tập dữ liệu",
    "source": "nguồn", "fake": "giả", "marked": "được đánh dấu", "claim": "tuyên bố", "includes": "bao gồm",
    "tests": "kiểm thử", "blur": "làm mờ", "conditions": "điều kiện", "transformations": "phép biến đổi",
    "untested": "chưa được kiểm thử", "adaptive": "thích ứng", "attacker": "đối thủ", "optimization": "tối ưu hóa",
    "current": "hiện nay", "closed": "đóng", "models": "mô hình", "multiple": "nhiều", "seeds": "seed",
    "comparisons": "so sánh", "overread": "diễn giải quá mức", "tenths": "phần mười", "point": "điểm",
    "patience": "patience", "stopped": "dừng", "early": "sớm", "executed": "đã thực thi", "planned": "dự kiến",
    "matrix": "ma trận", "condition": "điều kiện", "shards": "phân mảnh", "comparing": "so sánh",
    "output": "đầu ra", "found": "được tìm thấy", "runner": "trình chạy", "reproduction": "tái lập",
    "observed": "đã quan sát", "split": "cách chia", "weights": "trọng số", "reference": "tham chiếu",
    "specific": "đặc thù", "headline": "kết quả chính", "roles": "vai trò", "prompt": "prompt", "specified": "được quy định",
    "inventory": "danh mục", "adds": "bổ sung", "hybrids": "mô hình lai", "tags": "nhãn", "assignments": "phân loại",
    "uncertain": "chưa chắc chắn", "names": "tên gọi", "evidence": "bằng chứng", "exact": "chính xác",
    "folder": "thư mục", "name": "tên", "merge": "gộp", "mean": "trung bình", "many": "nhiều", "don't": "không",
    "classes": "lớp", "repeatability": "khả năng lặp lại", "checks": "kiểm tra", "labels": "nhãn",
    "caveat": "lưu ý", "missingness": "dữ liệu thiếu", "status": "trạng thái", "config": "cấu hình",
    "materialized": "đã chuẩn bị", "epochs": "epoch", "full-data": "toàn bộ dữ liệu", "early-stopped": "dừng sớm",
    "clean": "sạch", "raw": "thô", "preserved": "được giữ nguyên", "authoritative": "chuẩn", "infer": "suy ra",
    "encoding": "mã hóa", "extension": "phần mở rộng", "counts": "số lượng", "pooled": "sau pooling",
    "vector": "vector", "components": "thành phần", "convention": "quy ước", "stride": "stride", "changes": "thay đổi",
    "flow": "luồng", "itself": "tự thân", "guarantee": "bảo đảm", "generalization": "khả năng tổng quát hóa",
    "records": "ghi nhận", "every": "mỗi", "ROC": "ROC", "domain": "miền", "underperform": "kém hơn",
    "extra": "bổ sung", "may": "có thể", "invite": "dẫn đến", "learning": "việc học", "shortcut": "lối tắt",
    "listed": "được liệt kê", "establish": "xác lập", "universal": "phổ quát", "mechanism": "cơ chế",
    "gamma": "gamma", "feed": "đưa", "back": "ngược", "denominator": "mẫu số", "reconciled": "đối soát",
    "definitions": "định nghĩa", "separate": "riêng", "inconsistent": "không nhất quán", "centered": "tập trung",
    "near": "gần", "corrected": "được sửa", "separately": "riêng", "per-generator": "theo bộ sinh",
    "recorded": "được ghi nhận", "threshold": "ngưỡng", "size": "kích thước", "protocol": "giao thức",
    "documented": "được ghi nhận", "delta": "độ chênh", "paired": "ghép cặp", "needs": "cần",
    "none": "không có", "such": "như", "projected": "chiếu", "model-closed": "đóng mô hình",
    "family": "họ mô hình", "record": "bản ghi", "actual": "thực tế", "reconstruct": "tái tạo",
    "token": "token", "density": "mật độ", "likelihood": "hợp lý", "integrated": "tích hợp", "trajectory": "quỹ đạo",
    "continuous": "liên tục", "quantization": "lượng tử hóa", "diffuses": "khuếch tán", "encoder": "bộ mã hóa",
    "decoder": "bộ giải mã", "encode": "mã hóa", "decode": "giải mã", "difference": "khác biệt",
    "matters": "quan trọng", "detected": "được phát hiện", "measurement": "phép đo", "measurements": "các phép đo",
    "artifacts": "hiện vật", "artifact": "hiện vật", "cost": "chi phí", "budget": "ngân sách",
})

# Exact sentence translations keep technical claims readable and unambiguous.
VI_COPY.update({
    "Autoregressive, EBM and flow-matching are recorded as out of scope in the HTML survey; do not generalize from this table.": "Các mô hình tự hồi quy, EBM và flow-matching nằm ngoài phạm vi tài liệu tổng quan HTML; không khái quát hóa từ bảng này.",
    "Autoregressive, EBM and flow-matching models are outside the HTML survey's scope; do not generalize from this table.": "Các mô hình tự hồi quy, EBM và flow-matching nằm ngoài phạm vi tài liệu tổng quan HTML; không khái quát hóa từ bảng này.",
    "DIRE does not ask which model is 'stronger'; it needs a reproducible round trip, not a perfect inverse.": "DIRE không hỏi mô hình nào “mạnh hơn”; phương pháp cần phép đi-về có thể tái lập, không cần nghịch đảo hoàn hảo.",
    "A single encoder/decoder pass; no learned DDIM trajectory.": "Một lượt mã hóa/giải mã; không có quỹ đạo DDIM được học.",
    "GAN basic has no probability density or integrated inverse; latent recovery requires separate optimization.": "GAN cơ bản không có mật độ xác suất hay phép nghịch đảo tích hợp; cần tối ưu riêng để khôi phục không gian ẩn.",
    "The basic GAN has no probability density or integrated inverse; latent recovery requires separate optimization.": "GAN cơ bản không có mật độ xác suất hay phép nghịch đảo tích hợp; cần tối ưu riêng để khôi phục không gian ẩn.",
    "ADM uses a linear β_t schedule; cumulative ᾱ_t reduces the signal over time.": "ADM dùng lịch β_t tuyến tính; ᾱ_t tích lũy làm tín hiệu suy giảm theo thời gian.",
    "ε in the forward process explains the noise target when training diffusion; DDIM inference selects a deterministic path.": "ε trong quá trình thuận là mục tiêu nhiễu khi huấn luyện diffusion; suy luận DDIM chọn quỹ đạo tất định.",
    "ADM uses epsilon-prediction. Timestep embedding provides t to every block; the simplified loss is a practical training form related to ELBO.": "ADM dự đoán epsilon. Nhúng timestep cung cấp t cho mọi khối; hàm mất mát rút gọn là dạng huấn luyện thực dụng có liên hệ với ELBO.",
    "Same model, schedule and timesteps; inversion is approximate, not stochastic.": "Giữ nguyên mô hình, lịch và timestep; phép đảo là xấp xỉ, không ngẫu nhiên.",
    "Same backbone, schedule and timesteps; inversion is approximate, not stochastic.": "Giữ nguyên backbone, lịch và timestep; phép đảo là xấp xỉ, không ngẫu nhiên.",
    "The residual distributions of real and generated images must differ; this must be measured.": "Phân phối sai số tái tạo của ảnh thật và ảnh tạo sinh phải khác nhau; cần đo kiểm.",
    "The result supports the chosen representation; it does not establish a universal mechanism.": "Kết quả ủng hộ biểu diễn được chọn, nhưng không xác lập cơ chế phổ quát.",
    "Projection W_s is used when stride or channel count changes.": "Dùng phép chiếu W_s khi stride hoặc số kênh thay đổi.",
    "Residual path supports gradient flow; it does not itself guarantee generalization.": "Đường residual hỗ trợ luồng gradient nhưng không tự bảo đảm khả năng tổng quát hóa.",
    "Label convention: y=1 denotes generated/fake in the binary detector.": "Quy ước nhãn: y=1 chỉ ảnh tạo sinh/giả trong bộ phân loại nhị phân.",
    "Training uses BCEWithLogitsLoss for numerical stability.": "Huấn luyện dùng BCEWithLogitsLoss để ổn định số học.",
    "Why one logit matters": "Vì sao một logit quan trọng",
    "The published head shape is [1, 2048]; a 2-logit softmax head is not interchangeable.": "Kích thước head được công bố là [1, 2048]; không thể thay bằng head softmax 2 logit.",
    "Why ImageNet initialization": "Vì sao khởi tạo ImageNet",
    "Transfer learning initializes visual features; the pilot protocol records ImageNet initialization and an independent detector for each S.": "Học chuyển giao khởi tạo đặc trưng thị giác; giao thức pilot ghi nhận khởi tạo ImageNet và bộ phân loại riêng cho từng S.",
    "Protocol guard: choose epoch and threshold on validation; test is only for final evaluation.": "Ràng buộc giao thức: chọn epoch và ngưỡng trên tập xác thực; tập kiểm thử chỉ dùng để đánh giá cuối.",
    "Every cell is shown as ACC / AP (%), reported by the authors. These are not measurements from the project.": "Mỗi ô ghi ACC / AP (%), do tác giả báo cáo; đây không phải số đo của dự án.",
    "Table 2 headline is specific to its dataset, split, reference weights and evaluation protocol.": "Kết quả chính của Table 2 gắn với tập dữ liệu, cách chia, trọng số tham chiếu và giao thức đánh giá cụ thể.",
    "n≈1,000 per test cell; paper provides no CIs. Differences of a few tenths should not be overstated.": "n≈1.000 cho mỗi ô kiểm thử; bài báo không cung cấp khoảng tin cậy. Không nên phóng đại khác biệt chỉ vài phần mười.",
    "Lowest shown cell: ACC 76.9% · cross-domain generalization is a real limitation.": "Ô thấp nhất được trình bày: ACC 76,9% · khả năng tổng quát hóa chéo miền là một giới hạn thực sự.",
    "Paper protocol and full test data; S=20→50 multiplies NFE by 2.5. These rows are paper-reported, not the project's Table 4 ledger.": "Giao thức bài báo và dữ liệu kiểm thử đầy đủ; S=20→50 làm NFE tăng 2,5 lần. Đây là số liệu bài báo, không phải sổ Table 4 của dự án.",
    "Related work, not a result or feature implemented by this project's prototype.": "Công trình liên quan, không phải kết quả hay tính năng đã triển khai trong bản thử nghiệm của dự án.",
    "Do not recompute, combine or call these counters a reconciled total.": "Không tính lại, gộp hoặc gọi các bộ đếm này là tổng đã đối soát.",
    "This is a project evaluation on self-computed DIRE maps with a from-scratch detector; it is not an absolute Table 2 comparison.": "Đây là đánh giá của dự án trên bản đồ DIRE tự tính với bộ phân loại huấn luyện từ đầu; không phải phép so sánh trực tiếp với Table 2.",
    "The source protocol uses SD-v2; that differs from this paper SD-v1 table. These columns are deliberately not combined.": "Giao thức nguồn dùng SD-v2, khác SD-v1 trong bảng bài báo; hai cột này được giữ riêng có chủ ý.",
    "each condition × generator → measured AP shard → result ledger": "mỗi điều kiện × bộ sinh → phần AP đã đo → sổ kết quả",
    "Planned matrix: 30 condition×generator shards (6×5), each comparing 2 project-trained detectors = 60 model-condition metrics. Current output: none found. Runner is not the paper's CNNDetection/PatchForensics baseline reproduction.": "Ma trận dự kiến: 30 ô điều kiện×bộ sinh (6×5), mỗi ô so sánh 2 bộ phân loại do dự án huấn luyện = 60 chỉ số mô hình–điều kiện. Hiện chưa tìm thấy đầu ra. Trình chạy không tái lập baseline CNNDetection/PatchForensics của bài báo.",
    "Manifest-only: source images stay unchanged; seed 3407 · sha256-ranked-cell-v1 · cell counts in ledger.": "Chỉ cập nhật manifest: ảnh nguồn không đổi; seed 3407 · sha256-ranked-cell-v1 · số mẫu mỗi ô được ghi trong sổ.",
    "LSUN-Bedroom: ADM · DDPM · iDDPM · PNDM · LDM · SD-v1 · SD-v2 · VQ-Diffusion\n\nImageNet: ADM · SD-v1\n\nTrain / test roles and prompt conditions are specified in the paper.": "LSUN-Bedroom: ADM · DDPM · iDDPM · PNDM · LDM · SD-v1 · SD-v2 · VQ-Diffusion\n\nImageNet: ADM · SD-v1\n\nVai trò huấn luyện/kiểm thử và điều kiện prompt được quy định trong bài báo.",
    "Report inventory adds DALL·E 2, IF, projected GAN, hybrids and other tags.\n\nSome family assignments are uncertain or model-closed. Names such as diff-projectedgan are not evidence of architecture.": "Danh mục báo cáo bổ sung DALL·E 2, IF, GAN chiếu, mô hình lai và các nhãn khác.\n\nMột số phân loại họ mô hình chưa chắc chắn hoặc thuộc mô hình đóng. Tên như diff-projectedgan không phải bằng chứng về kiến trúc.",
    "Do not merge out-of-paper generators into a paper mean or claim exact source architecture from a folder name.": "Không gộp bộ sinh ngoài phạm vi bài báo vào giá trị trung bình, hoặc suy ra kiến trúc chính xác chỉ từ tên thư mục.",
    "Euler error, clipping and data-manifold explanations are not isolated experimentally.": "Sai số Euler, clipping và giả thuyết về đa tạp dữ liệu chưa được kiểm chứng riêng bằng thực nghiệm.",
    "Table 3 includes 76.9% ACC in a cross-domain cell.": "Table 3 có một ô chéo miền với ACC 76,9%.",
    "Paper tests selected blur/JPEG conditions; many transformations remain untested.": "Bài báo thử nghiệm một số điều kiện làm mờ/JPEG; nhiều phép biến đổi vẫn chưa được kiểm tra.",
    "No adaptive attacker or post-processing adversarial optimization.": "Chưa đánh giá đối thủ thích ứng hoặc tối ưu hóa đối kháng sau xử lý.",
    "2023 generators/prompts do not cover current closed / transformer / flow models.": "Các bộ sinh/prompt năm 2023 chưa bao quát mô hình đóng, transformer hoặc flow hiện nay.",
    "No multiple seeds / paper CIs for many comparisons; don't overread tenths of a point.": "Nhiều so sánh không có nhiều seed hoặc khoảng tin cậy từ bài báo; không diễn giải quá mức chênh lệch vài phần mười điểm.",
    "Ledger protocol target says 400 epochs with patience 5; actual run stopped early at 12–15 epochs. Do not present target epochs as executed epochs.": "Giao thức trong sổ đặt mục tiêu 400 epoch với patience 5; lượt chạy thực tế dừng sớm ở epoch 12–15. Không trình bày số epoch mục tiêu như số epoch đã chạy.",
})

VI_COPY.update({
    "AUDIT FLAG: report.dire.n = 5,900; n_new = 5,869; n_reused = 2,431. The latter two sum to 8,300, not 5,900. NFE=234,760 equals 5,869×40, not 5,900×40. Keep these fields separate until their definitions / denominator are reconciled.": "CẢNH BÁO KIỂM TOÁN: report.dire.n = 5.900; n_new = 5.869; n_reused = 2.431. Hai số sau cộng thành 8.300, không phải 5.900. NFE=234.760 bằng 5.869×40, không phải 5.900×40. Giữ các trường riêng biệt cho đến khi đối soát được định nghĩa và mẫu số.",
    "AUDIT FLAG: eval.json lists ACC=78.1% at threshold 0.50029 (chosen on val), but `acc_ci95=[0.47,0.529]` is centered near 50%, inconsistent with that ACC. Do not publish this CI until corrected. At threshold 0.5, ACC is separately 50.0%.": "CẢNH BÁO KIỂM TOÁN: eval.json ghi ACC=78,1% tại ngưỡng 0,50029 (chọn trên tập xác thực), nhưng `acc_ci95=[0.47,0.529]` tập trung gần 50%, không nhất quán với ACC đó. Không công bố khoảng tin cậy này trước khi sửa. Tại ngưỡng 0,5, ACC riêng được ghi là 50,0%.",
    "Repo metrics, not paper Table 2; fake cell n=700 and report warnings limit direct interpretation.": "Chỉ số repo, không phải Table 2; ô ảnh giả n=700 và cảnh báo giới hạn diễn giải.",
    "Repo protocol: S=20 · 40 NFE/image · 256×256 · float16 inputs. Inputs materialized; no Table 5 training/evaluation ledger found. REC needs reconstruction data, not residual .npy alone.": "Giao thức repo: S=20 · 40 NFE/ảnh · 256×256 · đầu vào float16. Đầu vào đã được chuẩn bị; chưa tìm thấy sổ huấn luyện/đánh giá Table 5. REC cần dữ liệu ảnh tái tạo, không chỉ tệp sai số .npy.",
    "Different: SD-v1 vs SD-v2; paper full data vs pilot; paper vs 200-per-cell; documented training protocol differs. No delta = reproduction claim.": "Khác biệt: SD-v1 so với SD-v2; dữ liệu đầy đủ của bài báo so với pilot; bài báo so với 200 ảnh mỗi ô; giao thức huấn luyện được ghi nhận cũng khác. Không xem độ chênh là tuyên bố tái lập.",
    "Gallery PNG is display-only: [0,2] → uint8 with gamma 0.5, RGB channels preserved. Do not feed that PNG back as authoritative data or infer source-map encoding from an extension.": "PNG trong gallery chỉ dùng hiển thị: [0,2] → uint8 với gamma 0,5, giữ nguyên kênh RGB. Không dùng PNG làm dữ liệu chuẩn hoặc suy ra cách mã hóa map nguồn từ phần mở rộng tệp.",
})

VI_COPY.update({
    "Train và test có thể khác p_g: nhận diện dấu vết của một generator không đồng nghĩa phát hiện ảnh sinh nói chung.": "Phân phối p_g có thể khác giữa huấn luyện và kiểm thử: nhận diện dấu vết của một bộ sinh không đồng nghĩa phát hiện ảnh tạo sinh nói chung.",
    "AUDIT FLAG: eval.json lists ACC=78.1% at threshold 0.50029 (chosen on val), but `acc_ci95=[0.47,0.529]` is centered near 50%, inconsistent with that ACC. Do not publish this CI until corrected. At threshold 0.5, ACC is separately 50.0%.": "CẢNH BÁO KIỂM TOÁN: eval.json ghi ACC=78,1% tại ngưỡng 0,50029 (chọn trên tập xác thực), nhưng `acc_ci95=[0.47,0.529]` tập trung gần 50%, không nhất quán với ACC đó. Không công bố khoảng tin cậy này trước khi sửa. Tại ngưỡng 0,5, ACC riêng được ghi là 50,0%.",
    "Residual bottleneck counts [3, 4, 6, 3]; final pooled feature vector has 2,048 components. The DIRE checkpoint convention is fc: 2,048 → 1 logit.": "Các khối bottleneck residual có số lượng [3, 4, 6, 3]; vector đặc trưng sau pooling có 2.048 thành phần. Quy ước checkpoint DIRE: fc 2.048 → 1 logit.",
    "Transfer learning initializes visual features; pilot protocol records ImageNet init and an independent detector per S.": "Học chuyển giao khởi tạo đặc trưng thị giác; giao thức pilot ghi nhận khởi tạo ImageNet và một bộ phân loại riêng cho từng S.",
    "Protocol guard: choose epoch and threshold on validation; test is for final evaluation only.": "Ràng buộc giao thức: chọn epoch và ngưỡng trên tập xác thực; tập kiểm thử chỉ dùng cho đánh giá cuối.",
    "The basic GAN has no probability density or integrated inverse; latent recovery requires separate optimization.": "GAN cơ bản không có mật độ xác suất hay phép nghịch đảo tích hợp; cần tối ưu riêng để khôi phục không gian ẩn.",
    "VQ-Diffusion and latent diffusion move the diffusion trajectory into another space.": "VQ-Diffusion và khuếch tán tiềm ẩn đưa quỹ đạo khuếch tán sang không gian khác.",
    "Discrete diffusion over codebook tokens; quantization error belongs to the representation. This model is in the paper's generator benchmark.": "Khuếch tán rời rạc trên token codebook; sai số lượng tử hóa thuộc về biểu diễn. Mô hình này có trong benchmark bộ sinh của bài báo.",
    "VAE encoder → latent DDIM trajectory → VAE decoder. The DIRE paper still uses pixel-space ADM as its reconstruction reference.": "Bộ mã hóa VAE → quỹ đạo DDIM trong không gian ẩn → bộ giải mã VAE. Bài DIRE vẫn dùng ADM trên pixel làm tham chiếu tái tạo.",
    "Example on the ADM detector: Table 6 compares raw signed residual with |residual|; this is a paper result, not a pilot.": "Ví dụ trên bộ phân loại ADM: Table 6 so sánh sai số có dấu thô với |sai số|; đây là kết quả bài báo, không phải pilot.",
    "The toy-image illustration; not a feature map or attention score extracted from ADM/ResNet.": "Minh họa trên ảnh tổng hợp; không phải bản đồ đặc trưng hay điểm attention trích từ ADM/ResNet.",
    "This reconstruction is intentionally blurred to make the residual visible; it is not ADM or a measurement on a real image.": "Ảnh tái tạo được làm mờ có chủ ý để residual dễ quan sát; đây không phải ADM hay phép đo trên ảnh thật.",
    "In checkout there is an R1+R2 report, Table 4 pilot and 64 gallery samples; Table 5/Figure 4 have no result ledger.": "Trong bản mã nguồn có báo cáo R1+R2, pilot Table 4 và 64 mẫu thư viện; Table 5/Hình 4 chưa có sổ kết quả.",
    "Checkpoint not present in checkout; gallery image hashes verify only display PNGs, not the source map.": "Checkpoint không có trong bản mã nguồn; hash ảnh gallery chỉ xác nhận PNG hiển thị, không thay thế map nguồn.",
    "Git metadata · report.json:data.created · developer session date": "Siêu dữ liệu Git · report.json:data.created · ngày phiên làm việc",
    "The displayed data retain source labels. Until timeline / Modal Volume is confirmed, do not call them independently verified results.": "Số liệu hiển thị giữ nguyên nhãn nguồn. Cho đến khi xác nhận dòng thời gian và Modal Volume, không gọi đây là kết quả được kiểm chứng độc lập.",
    "A checkpoint is not in the checkout; gallery image hashes only verify display PNGs, not the source map.": "Checkpoint không có trong bản mã nguồn; hash ảnh gallery chỉ xác minh PNG hiển thị, không phải map nguồn.",
    "R-level · preprocessing · checkpoint hash · split / threshold plan": "Mức R · tiền xử lý · hash checkpoint · cách chia / kế hoạch ngưỡng",
    "Complete ledger · no empty class · repeatability / hash checks": "Sổ đầy đủ · không có lớp rỗng · kiểm tra khả năng lặp lại / hash",
})

VI_COPY.update({
    "Diffusion Reconstruction Error · nền tảng toán học, kiến trúc mô hình,\nđối chiếu paper và trạng thái pilot theo artifacts trong repo": "Sai số tái tạo khuếch tán (DIRE) · nền tảng toán học, kiến trúc mô hình,\nđối chiếu bài báo và trạng thái thí điểm theo hiện vật trong kho mã",
    "Editable PPTX + PDF  ·  equations typeset from LaTeX source": "PPTX + PDF có thể chỉnh sửa · công thức được dàn trang từ mã LaTeX",
    "Trong checkout có report R1+R2, pilot Table 4 và 64 mẫu gallery; Table 5/Figure 4 chưa có sổ kết quả.": "Trong bản mã nguồn có báo cáo R1+R2, pilot Table 4 và 64 mẫu thư viện; Table 5/Hình 4 chưa có sổ kết quả.",
    "Checkpoint chưa có trong checkout; hash ảnh gallery chỉ xác nhận PNG hiển thị, không thay thế map nguồn.": "Checkpoint chưa có trong bản mã nguồn; hash chỉ xác nhận PNG hiển thị, không thay thế bản đồ nguồn.",
    "Báo cáo R1+R2; đánh giá ADM trên tập giữ lại; pilot Table 4; 64 mẫu gallery có metadata.": "Báo cáo R1+R2; đánh giá ADM trên tập giữ lại; pilot Table 4; 64 mẫu thư viện có siêu dữ liệu.",
    "Table 5 mới có giao thức/đầu vào; Figure 4 chưa có kết quả. Không suy diễn S=50 cho pilot.": "Table 5 mới có giao thức/đầu vào; Hình 4 chưa có kết quả. Không ngoại suy S=50 cho pilot.",
    "Train và test có thể khác p_g: nhận diện dấu vết của một generator không đồng nghĩa phát hiện ảnh sinh nói chung.": "Phân phối p_g có thể khác giữa huấn luyện và kiểm thử: nhận diện dấu vết của một bộ sinh không đồng nghĩa phát hiện ảnh tạo sinh nói chung.",
    "Shortcut risk": "Nguy cơ học lối tắt",
    "Research target": "Mục tiêu nghiên cứu",
    "A round trip should be reproducible but is not a perfect inverse.": "Phép đi-về cần tái lập được nhưng không phải nghịch đảo hoàn hảo.",
    "Tái tạo tốt hơn ↔ latent posterior gần prior; encoder chỉ xấp xỉ posterior.": "Tái tạo tốt hơn ↔ hậu nghiệm trong không gian ẩn gần tiên nghiệm; bộ mã hóa chỉ xấp xỉ hậu nghiệm.",
    "Một lượt encoder/decoder; không có quỹ đạo DDIM được học.": "Một lượt mã hóa/giải mã; không có quỹ đạo DDIM được học.",
    "VAE là codec không gian ẩn bên trong mô hình khuếch tán tiềm ẩn.": "VAE là bộ mã hóa/giải mã không gian ẩn bên trong mô hình khuếch tán tiềm ẩn.",
    "GAN cơ bản không có mật độ xác suất hay phép nghịch đảo tích hợp; cần tối ưu riêng để tìm latent.": "GAN cơ bản không có mật độ xác suất hay phép nghịch đảo tích hợp; cần tối ưu riêng để tìm không gian ẩn.",
    "Khác biệt không gian mô hình rất quan trọng: generator bị phát hiện không phải backbone tái tạo.": "Khác biệt về không gian mô hình rất quan trọng: bộ sinh được phát hiện không phải backbone tái tạo.",
    "Khuếch tán rời rạc trên token codebook; sai số lượng tử hóa thuộc về biểu diễn. Mô hình này có trong benchmark generator của bài báo.": "Khuếch tán rời rạc trên token codebook; sai số lượng tử hóa thuộc về biểu diễn. Mô hình này nằm trong bộ kiểm chuẩn bộ sinh của bài báo.",
    "Encoder VAE → quỹ đạo DDIM trong latent → decoder VAE. Bài DIRE vẫn dùng ADM trên pixel làm phép tái tạo tham chiếu.": "Bộ mã hóa VAE → quỹ đạo DDIM trong không gian ẩn → bộ giải mã VAE. Bài DIRE vẫn dùng ADM trên pixel làm tham chiếu tái tạo.",
    "VQ-Diffusion và latent diffusion đưa quỹ đạo khuếch tán sang không gian khác": "VQ-Diffusion và khuếch tán tiềm ẩn đưa quỹ đạo sang không gian khác",
    "ε ở forward process giải thích noise target khi train diffusion; DDIM inference chọn đường determinist.": "ε trong quá trình thuận là mục tiêu nhiễu khi huấn luyện diffusion; suy luận DDIM chọn quỹ đạo tất định.",
    "Supervision tự sinh: biết ε vì chính quy trình forward vừa thêm nó.": "Tín hiệu giám sát tự sinh: biết ε vì chính quá trình thuận đã thêm nhiễu này.",
    "ADM dùng epsilon-prediction. Timestep embedding cung cấp t cho mọi khối; loss rút gọn là dạng huấn luyện thực dụng liên quan ELBO.": "ADM dự đoán epsilon. Nhúng timestep cung cấp t cho mọi khối; hàm mất mát rút gọn là dạng huấn luyện thực dụng liên hệ với ELBO.",
    "Timestep embedding đi vào từng stage; attention được đặt tại các bản đồ đặc trưng 32×32, 16×16, 8×8.": "Nhúng timestep đi vào từng tầng; attention được đặt tại bản đồ đặc trưng 32×32, 16×16 và 8×8.",
    "SKIP CONNECTIONS · giữ chi tiết nông qua bottleneck": "KẾT NỐI TẮT · giữ chi tiết tầng nông qua nút thắt",
    "Global context, used at selected resolutions.": "Ngữ cảnh toàn cục, dùng tại một số độ phân giải.",
    "Minh họa trên ảnh toy; không phải feature map hay attention score được trích từ ADM/ResNet.": "Minh họa trên ảnh tổng hợp; không phải bản đồ đặc trưng hay điểm attention trích từ ADM/ResNet.",
    "The result supports the chosen representation; it does not establish a universal mechanism.": "Kết quả ủng hộ biểu diễn được chọn, nhưng không xác lập cơ chế phổ quát.",
    "Paper's qualitative claim: DIRE changes little; RGB baselines degrade more. The plot's pixel values are not tabulated in the cited paper copy, so this deck does not invent exact AP points.": "Nhận định định tính của bài báo: DIRE thay đổi ít; các đối chứng RGB suy giảm nhiều hơn. Bản bài báo trích dẫn không có bảng số liệu của đồ thị, nên không tự tạo điểm AP.",
    "Repo report label: R1+R2. The diffusion checkpoint and source dataset are not project-trained; fake images are marked author-sourced, so do not claim R3.": "Nhãn báo cáo trong repo: R1+R2. Checkpoint diffusion và tập dữ liệu nguồn không do dự án huấn luyện; ảnh giả được ghi nhận là do tác giả cung cấp, vì vậy không tuyên bố R3.",
    "Lớn hơn không tự động có nghĩa 'fake'; classifier học phân phối map từ dữ liệu có nhãn.": "Giá trị lớn hơn không tự động đồng nghĩa “ảnh giả”; bộ phân loại học phân phối bản đồ từ dữ liệu có nhãn.",
    "Ví dụ minh họa trên detector ADM: Table 6 so sánh raw signed residual với |residual|; đây là paper result, không phải pilot.": "Ví dụ trên bộ phân loại ADM: Table 6 so sánh sai số có dấu thô với |sai số|; đây là kết quả bài báo, không phải pilot.",
    "The source protocol uses SD-v2; that differs from this paper SD-v1 table. These columns are deliberately not combined.": "Giao thức nguồn dùng SD-v2, khác SD-v1 trong bảng bài báo; hai cột được giữ riêng có chủ ý.",
    "Ảnh gốc ADM kiểm thử và bản đồ DIRE được lưu thành một cặp có nguồn gốc": "Ảnh ADM trong tập kiểm thử và bản đồ DIRE được lưu thành một cặp có nguồn gốc",
    "Đây là cặp ảnh/map lưu trong gallery, không phải lượt inference mới. File tái tạo và map .npy gốc chưa có trong checkout; PNG map chỉ dùng để hiển thị.": "Đây là cặp ảnh/bản đồ trong thư viện, không phải lượt suy luận mới. Ảnh tái tạo và bản đồ .npy gốc chưa có trong bản mã nguồn; PNG chỉ dùng để hiển thị.",
    "Chỉ cập nhật manifest: ảnh nguồn không đổi; seed 3407 · sha256-ranked-cell-v1 · số mẫu mỗi ô được ghi trong sổ.": "Chỉ cập nhật manifest: ảnh nguồn không đổi; seed 3407 · sha256-ranked-cell-v1 · số mẫu mỗi ô ghi trong sổ.",
    "Related work, not a result or feature implemented in the prototype.": "Công trình liên quan, không phải kết quả hay tính năng đã triển khai trong bản thử nghiệm.",
    "R-level · preprocessing · checkpoint hash · split / threshold plan": "Mức R · tiền xử lý · hash checkpoint · cách chia / kế hoạch ngưỡng",
})

VI_COPY.update({
    "repo": "kho mã", "checkout": "bản mã nguồn", "gallery": "thư viện ảnh", "metadata": "siêu dữ liệu",
    "hash": "mã băm", "map": "bản đồ", "maps": "bản đồ", "benchmark": "bộ kiểm chuẩn",
    "pixel": "điểm ảnh", "pixels": "điểm ảnh", "pipeline": "quy trình", "backbone": "mạng nền",
    "diffusion": "khuếch tán", "pilot": "thí điểm", "attention": "attention", "self-attention": "tự chú ý",
    "codebook": "bảng mã", "latent": "không gian ẩn", "threshold": "ngưỡng", "detector": "bộ phân loại",
    "input": "đầu vào", "output": "đầu ra", "real": "thật", "fake": "giả", "flow-matching": "flow-matching",
    "timestep": "timestep", "epsilon-prediction": "dự đoán epsilon", "sampling": "lấy mẫu", "inference": "suy luận",
    "feature map": "bản đồ đặc trưng", "attention score": "điểm attention", "display-only": "chỉ để hiển thị",
    "gallery samples": "mẫu thư viện ảnh", "metadata gallery": "siêu dữ liệu thư viện ảnh",
    "Repo metrics": "Chỉ số kho mã", "repo-recorded": "ghi trong kho mã",
    "Paper evidence": "Bằng chứng bài báo", "paper-reported": "do bài báo công bố",
})

VI_COPY.update({
    "Area under ROC; compare with caution under domain shift.": "Diện tích dưới ROC; cần thận trọng khi so sánh giữa các miền dữ liệu.",
    "Đi xuống: giảm resolution, tăng khả năng biểu diễn.  Bottleneck: context toàn cục.  Đi lên: khôi phục chi tiết qua skip.  Đây là sơ đồ kiến trúc, không phải tensor visualization.": "Đi xuống: giảm độ phân giải, tăng năng lực biểu diễn. Nút thắt: ngữ cảnh toàn cục. Đi lên: khôi phục chi tiết qua kết nối tắt. Đây là sơ đồ kiến trúc, không phải ảnh trực quan hóa tensor.",
    "xₜ₋₁ được lấy mẫu từ phân phối Gaussian mỗi bước.\nƯu điểm: sampling gốc.\nHệ quả cho DIRE: stochastic noise làm residual thay đổi nếu không cố định path.": "Ở mỗi bước, xₜ₋₁ được lấy mẫu từ phân phối Gauss.\nƯu điểm: cơ chế lấy mẫu nguyên bản.\nHệ quả với DIRE: nhiễu ngẫu nhiên làm sai số thay đổi nếu không cố định quỹ đạo.",
    "η = 0 loại thành phần noise bổ sung.\nCùng xₜ và protocol → cùng x₀.\nPhép reverse ODE có thể đi theo chiều tăng timestep.": "η = 0 loại bỏ thành phần nhiễu bổ sung.\nCùng xₜ và giao thức → cùng x₀.\nODE ngược có thể đi theo chiều timestep tăng.",
    "Ví dụ minh họa trên detector ADM: Table 6 so sánh raw signed residual với |residual|; đây là paper result, không phải pilot.": "Ví dụ trên bộ phân loại ADM: Table 6 so sánh sai số có dấu thô với |sai số|; đây là kết quả bài báo, không phải pilot.",
    "Table 4 bài báo · repo Table 4 ledger · different split size, bộ sinh and huấn luyện giao thức": "Table 4 của bài báo · sổ Table 4 của kho mã · khác quy mô chia tập, bộ sinh và giao thức huấn luyện",
    "Table 4 paper · project Table 4 ledger · different split size, generators and training protocol": "Table 4 của bài báo · sổ Table 4 của dự án · khác quy mô chia tập, bộ sinh và giao thức huấn luyện",
    "SỔ KẾT QUẢ TABLE 4 · _subset_protocol.json · MANIFEST SHA-256": "SỔ TABLE 4 · _subset_protocol.json · MANIFEST SHA-256",
    "Toàn bộ metric pilot Table 4 từ sổ đã lưu trong repo": "Toàn bộ chỉ số pilot Table 4 từ sổ trong kho mã",
    "Table 4 bài báo cho thấy đánh đổi theo S; không thay bằng dữ liệu pilot": "Table 4 của bài báo cho thấy sự đánh đổi theo S; không thay bằng số liệu pilot.",
    "Table 2 báo cáo khả năng tổng quát hóa chéo bộ sinh cao ở bốn detector DIRE": "Table 2 báo cáo khả năng tổng quát hóa chéo bộ sinh cao ở bốn bộ phân loại DIRE.",
    "ACC trung bình do tác giả báo cáo nằm trong khoảng 97,9%–99,9% theo bộ sinh huấn luyện": "ACC trung bình do tác giả báo cáo nằm trong khoảng 97,9%–99,9% tùy bộ sinh huấn luyện.",
    "Table 5 bài báo ủng hộ DIRE hơn RGB hoặc ảnh tái tạo riêng lẻ": "Table 5 của bài báo cho thấy DIRE tốt hơn RGB hoặc ảnh tái tạo riêng lẻ.",
    "Bài báo và pilot dùng bộ sinh cùng quy mô thực thi khác nhau": "Bài báo và pilot dùng các bộ sinh, quy mô thực thi khác nhau.",
    "The source protocol uses SD-v2; that differs from this paper SD-v1 table. These columns are deliberately not combined.": "Giao thức nguồn dùng SD-v2, khác với SD-v1 trong bảng bài báo; hai cột được giữ riêng có chủ ý.",
    "S5/S10/S20 train.json · MANIFEST giao thức hashes": "train.json của S5/S10/S20 · hash manifest giao thức",
    "flow-matching": "flow-matching", "normalizing flow": "dòng chuẩn hóa", "tensor visualization": "trực quan hóa tensor",
    "metric": "chỉ số", "metrics": "các chỉ số", "tag": "nhãn", "Run tag": "Nhãn lượt chạy",
    "sha256-ranked-cell-v1": "sha256-ranked-cell-v1", "seed": "seed", "timeline": "dòng thời gian", "generated": "tạo sinh",
    "Paper Table 4 · repo Table 4 ledger · different split size, generator and training protocol": "Table 4 của bài báo · sổ Table 4 của kho mã · khác quy mô chia tập, bộ sinh và giao thức huấn luyện",
    "Paper Table 4 · repo Table 4 protocol + ledger": "Table 4 của bài báo · giao thức và sổ kết quả Table 4 của kho mã",
    "Paper Table 4 and project Table 4 are not directly comparable experiments": "Table 4 của bài báo và Table 4 của dự án không phải các thí nghiệm so sánh trực tiếp",
    "25 · Hạn chế & research questions": "25 · Hạn chế và câu hỏi nghiên cứu",
    "RESEARCH QUESTIONS": "CÂU HỎI NGHIÊN CỨU", "Why one logit matters": "Vì sao một logit quan trọng",
    "Re-score": "Chấm điểm lại", "Recompute map": "Tính lại bản đồ", "Retrain detector": "Huấn luyện lại bộ phân loại",
    "Regenerate source images": "Tạo lại ảnh nguồn", "author map + author detector": "bản đồ và bộ phân loại của tác giả",
    "self-computed DIRE": "DIRE tự tính", "self-trained classifier": "bộ phân loại tự huấn luyện", "own generator / data": "bộ sinh / dữ liệu tự sở hữu",
})

VI_COPY.update({
    "Q, K, V projections\nacross spatial positions": "Phép chiếu Q, K, V\ntrên các vị trí không gian",
    "raw pixels": "điểm ảnh gốc",
    "float [−1,1]\ncenter crop/resize 256\ndiffusion domain": "float [−1,1]\ncắt giữa/đổi kích thước 256\nmiền khuếch tán",
    "|x − x̂|\nfloat [0,2], 3 channels\nauthoritative map array": "|x − x̂|\nfloat [0,2], 3 kênh\nmảng bản đồ chuẩn",
    "resize 256 → crop 224\nImageNet normalize\ndetector domain": "đổi kích thước 256 → cắt 224\nchuẩn hóa ImageNet\nmiền bộ phân loại",
    "Đi xuống: giảm resolution, tăng khả năng biểu diễn.  Bottleneck: context toàn cục.  Đi lên: khôi phục chi tiết qua skip.  Đây là sơ đồ kiến trúc, không phải tensor visualization.": "Đi xuống: giảm độ phân giải, tăng năng lực biểu diễn. Nút thắt: ngữ cảnh toàn cục. Đi lên: khôi phục chi tiết qua kết nối tắt. Đây là sơ đồ kiến trúc, không phải ảnh trực quan hóa tensor.",
    "A completed scientific cell requires metrics, labels, counts and compatible provenance": "Một ô thí nghiệm hoàn chỉnh cần chỉ số, nhãn, số lượng và nguồn gốc tương thích",
    "complete ledger · no empty class · repeatability / hash checks": "sổ đầy đủ · không có lớp rỗng · kiểm tra khả năng lặp lại / mã băm",
    "No empty class · all labels represented · repeatability checks": "Không có lớp rỗng · đủ mọi nhãn · kiểm tra khả năng lặp lại",
    "A completed cell requires a metric, labels, counts and compatible provenance.": "Một ô thí nghiệm hoàn chỉnh cần chỉ số, nhãn, số lượng và nguồn gốc tương thích.",
    "Mốc ghi nhận trong kho mã nằm sau ngày phiên làm việc": "Các mốc ghi nhận trong kho mã muộn hơn ngày phiên làm việc",
    "Có file + ledger\nnhưng dòng thời gian chưa khớp": "Có tệp và sổ kết quả\nnhưng dòng thời gian chưa khớp",
    "Mạng có thể học dấu vân tay Bộ sinh thay vì dấu hiệu bền vững.": "Mạng có thể học dấu vân tay của bộ sinh thay vì dấu hiệu bền vững.",
    "Tìm biểu diễn có ích khi Bộ sinh Kiểm thử ≠ Bộ sinh Huấn luyện.": "Tìm biểu diễn hữu ích khi bộ sinh kiểm thử khác bộ sinh huấn luyện.",
    "Dòng chuẩn hóa (normalizing flow) lý tưởng nghịch đảo chính xác — làm mất tín hiệu Sai số DIRE": "Dòng chuẩn hóa có phép nghịch đảo chính xác — trong trường hợp lý tưởng, DIRE bằng không.",
    "Các lớp biến đổi ảnh thành đặc trưng theo không gian": "Các lớp biến đổi ảnh thành đặc trưng không gian",
    "Bài DIRE vẫn dùng ADM trên pixel làm tham chiếu tái tạo.": "Bài DIRE vẫn dùng ADM trên điểm ảnh làm mô hình tham chiếu để tái tạo.",
    "AEROBLADE cùng trực giác sai số tái tạo nhưng dùng autoencoder": "AEROBLADE có trực giác tương tự về sai số tái tạo nhưng dùng bộ mã hóa tự động",
    "A completed scientific cell requires metrics, labels, counts and compatible provenance": "Một ô thí nghiệm hoàn chỉnh cần chỉ số, nhãn, số lượng và nguồn gốc tương thích",
    "A completed scientific cell requires a metric, labels, counts and compatible provenance": "Một ô thí nghiệm hoàn chỉnh cần chỉ số, nhãn, số lượng và nguồn gốc tương thích",
})

def _vi_copy(text: str) -> str:
    """Translate prose without cascading replacements into Vietnamese or identifiers."""
    import re
    # Exact full-copy entries take precedence, including prose that contains a
    # filename or slash-form label which must remain untouched as an identifier.
    exact = VI_COPY.get(text)
    if exact is None:
        exact = next((v for k, v in VI_COPY.items() if k.lower() == text.lower() and " " in k), None)
    if exact is not None:
        return exact
    # Keep repository paths, filenames, URLs, and author citation fragments intact,
    # while still translating any surrounding explanatory prose.
    protected = []
    def hold(match):
        protected.append(match.group(0))
        return f"\u0000{len(protected)-1}\u0000"
    text = re.sub(r"\b(?:docs|modal-runs|src|tools|platforms)/[\w./-]+|https?://\S+|\b\w+\.(?:json|pdf|tsv|npy|png|md|py)\b|\bet al\.", hold, text)

    # Multi-word phrases win first and are held so later glossary entries cannot
    # rewrite words inside the Vietnamese translation (or technical terms).
    held_phrases = []
    for en, vi in sorted(((k, v) for k, v in VI_COPY.items() if " " in k),
                         key=lambda item: len(item[0]), reverse=True):
        pattern = rf"(?<![\w-]){re.escape(en)}(?![\w-])"
        def hold_translation(_match, replacement=vi):
            held_phrases.append(replacement)
            return f"\u0001{len(held_phrases)-1}\u0001"
        text = re.sub(pattern, hold_translation, text, flags=re.IGNORECASE)

    # Translate single terms in one pass. This avoids accidental cascades such as
    # replacing English fragments that happen to occur in a translated Vietnamese word.
    stopwords = set()
    singles = {k.lower(): v for k, v in VI_COPY.items() if " " not in k and k.lower() not in stopwords}
    if singles:
        choices = sorted(singles, key=len, reverse=True)
        pattern = r"(?<![\w-])(?:" + "|".join(re.escape(k) for k in choices) + r")(?![\w-])"
        text = re.sub(pattern, lambda m: singles[m.group(0).lower()], text, flags=re.IGNORECASE)
    for i, value in enumerate(held_phrases):
        text = text.replace(f"\u0001{i}\u0001", value)
    for i, value in enumerate(protected):
        text = text.replace(f"\u0000{i}\u0000", value)
    return text

def hx(code: str) -> str:
    return C.get(code, code).lstrip("#")


def ppt_rgb(code: str) -> RGBColor:
    v = hx(code)
    return RGBColor.from_string(v)


def pdf_rgb(code: str):
    v = hx(code)
    return colors.HexColor("#" + v)


def inches(pt: float):
    return Inches(pt / 72.0)


def load_rows(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    lines = (line for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#"))
    return list(csv.DictReader(lines, delimiter=delimiter))


PAPER_ROWS = load_rows(PAPER_TABLE2_PATH)
PAPER_DETECTORS = ["lsun_adm", "lsun_pndm", "lsun_iddpm", "lsun_stylegan"]
PAPER_GENS = ["adm", "ddpm", "iddpm", "pndm", "sdv1", "sdv2", "ldm", "vqdiffusion"]
PAPER_LABELS = {"adm": "ADM", "ddpm": "DDPM", "iddpm": "iDDPM", "pndm": "PNDM",
                "sdv1": "SD-v1", "sdv2": "SD-v2", "ldm": "LDM", "vqdiffusion": "VQ-Diff."}
PAPER_DET_LABELS = {"lsun_adm": "ADM", "lsun_pndm": "PNDM", "lsun_iddpm": "iDDPM", "lsun_stylegan": "StyleGAN"}
PAPER_MAP = {(r["ckpt"], r["generator"]): (float(r["acc"]), float(r["ap"]))
             for r in PAPER_ROWS}
LEDGER = table4()
T4_ROWS = table4_rows(LEDGER)
REPORT = read_json(REPORT_PATH) or {}
EVAL = read_json(EVAL_PATH) or {}
GALLERY_SAMPLE = json.loads((GALLERY / "adm_004006349103115c.json").read_text(encoding="utf-8"))
T4_RUNS = {s: block for s, block in (LEDGER or {}).get("by_steps", {}).items()}

# Verify only the display PNGs committed beside each gallery metadata record.
# The source .npy hash is an identifier in metadata, not fetched/re-verified here.
GALLERY_HASH_OK: dict[str, bool] = {}
for p in sorted(GALLERY.glob("*.json")):
    meta = read_json(p)
    if not meta or not isinstance(meta.get("name"), str):
        continue
    ok = True
    for field, suffix in (("png_orig", "_orig.png"), ("png_diff", "_diff.png")):
        image = GALLERY / f"{meta['name']}{suffix}"
        expected = meta.get("sha256", {}).get(field)
        if not image.is_file() or hashlib.sha256(image.read_bytes()).hexdigest() != expected:
            ok = False
    GALLERY_HASH_OK[meta["name"]] = ok


def s_metrics(s: int, gen: str) -> dict[str, Any] | None:
    block = T4_RUNS.get(str(s))
    if not block:
        return None
    return block.get("by_test_generator", {}).get(gen, {}).get("metrics")


def paper_value(det: str, gen: str, metric: str) -> float:
    return PAPER_MAP[(det, gen)][0 if metric == "acc" else 1]


def gallery_pair_for(generator: str) -> tuple[Path, Path] | None:
    """Return a display pair for a generator; never invent reconstructions."""
    for meta_path in sorted(GALLERY.glob(f"{generator}_*.json")):
        meta = read_json(meta_path)
        name = (meta or {}).get("name")
        if not isinstance(name, str) or Path(name).name != name:
            continue
        orig, dire = GALLERY / f"{name}_orig.png", GALLERY / f"{name}_diff.png"
        if orig.is_file() and dire.is_file():
            return orig, dire
    return None


class Page:
    """Single page rendered in parallel as native PPTX elements and PDF vectors."""
    def __init__(self, owner: "Deck", section: str, title: str, source: str,
                 notes: str, dark: bool = False, cover: bool = False):
        self.owner, self.section, self.title = owner, section, title
        self.dark, self.cover = dark, cover
        self.slide = owner.prs.slides.add_slide(owner.prs.slide_layouts[6])
        self.pdf = owner.pdf
        self.pageno = len(owner.prs.slides)
        self.slide.background.fill.solid()
        self.slide.background.fill.fore_color.rgb = ppt_rgb("navy" if dark else "bg")
        self.pdf.setFillColor(pdf_rgb("navy" if dark else "bg"))
        self.pdf.rect(0, 0, PW, PH, stroke=0, fill=1)
        self.source = source
        self.notes = notes
        self._header()
        owner.pages.append(self)

    def _header(self):
        if self.cover:
            return
        ink = "white" if self.dark else "ink"
        muted = "deepmuted" if self.dark else "muted"
        self.text(42, 23, 850, 16, self.section.upper(), 9.2, "mint", bold=True, char_space=True)
        self.text(42, 43, 875, 56, self.title, 19.5, ink, bold=True, valign="mid")
        self.line(42, 108, 918, 108, "navy2" if self.dark else "line", 0.8)
        self.line(42, 505, 918, 505, "navy2" if self.dark else "line", 0.8)
        self.text(43, 512, 812, 16, self.source, 7.4, muted)
        self.text(858, 512, 58, 15, f"{self.pageno:02d}", 8, muted, align="right")

    def rect(self, x: float, y: float, w: float, h: float, fill: str = "paper",
             stroke: str | None = None, radius: float = 10, linew: float = 0.8):
        shape = self.slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
            inches(x), inches(y), inches(w), inches(h))
        shape.fill.solid(); shape.fill.fore_color.rgb = ppt_rgb(fill)
        shape.line.color.rgb = ppt_rgb(stroke or fill); shape.line.width = Pt(linew)
        # Softer native rounded-corner adjustment where supported.
        try:
            shape.adjustments[0] = max(0.05, min(0.35, radius / max(1.0, min(w, h))))
        except Exception:
            pass
        self.pdf.setFillColor(pdf_rgb(fill)); self.pdf.setStrokeColor(pdf_rgb(stroke or fill))
        self.pdf.setLineWidth(linew)
        self.pdf.roundRect(x, PH - y - h, w, h, min(radius, h / 3, w / 3), stroke=1 if stroke else 0, fill=1)
        return shape

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = "line", width: float = 1):
        sh = self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
            inches(x1), inches(y1), inches(x2), inches(y2))
        sh.line.color.rgb = ppt_rgb(color); sh.line.width = Pt(width)
        self.pdf.setStrokeColor(pdf_rgb(color)); self.pdf.setLineWidth(width)
        self.pdf.line(x1, PH - y1, x2, PH - y2)

    def arrow(self, x1: float, y1: float, x2: float, y2: float, color: str = "teal", width: float = 1.8):
        self.line(x1, y1, x2, y2, color, width)
        # PowerPoint arrowhead stays a native, editable line property.
        from pptx.oxml.xmlchemy import OxmlElement
        connector = self.slide.shapes[-1]
        ln = connector.get_or_add_ln()
        tail = OxmlElement("a:tailEnd")
        tail.set("type", "triangle"); tail.set("w", "med"); tail.set("len", "med")
        ln.append(tail)
        # ReportLab counterpart: draw a filled arrowhead in slide coordinates.
        angle = math.atan2(y2-y1, x2-x1); length = 7
        a1, a2 = angle + 2.48, angle - 2.48
        pts = [(x2, y2), (x2 + length*math.cos(a1), y2 + length*math.sin(a1)),
               (x2 + length*math.cos(a2), y2 + length*math.sin(a2))]
        path = self.pdf.beginPath(); path.moveTo(pts[0][0], PH-pts[0][1])
        for x, y in pts[1:]: path.lineTo(x, PH-y)
        path.close(); self.pdf.setFillColor(pdf_rgb(color)); self.pdf.setStrokeColor(pdf_rgb(color))
        self.pdf.drawPath(path, stroke=0, fill=1)

    def polygon(self, pts: list[tuple[float, float]], fill: str, stroke: str | None = None):
        # Pptx freeform is editable; construct in slide-coordinate units.
        builder = self.slide.shapes.build_freeform(inches(pts[0][0]), inches(pts[0][1]), scale=1)
        # build_freeform coordinates are local EMU increments. Safer native triangles use a
        # triangle primitive only for arrowheads; arbitrary polygons are primarily PDF diagrams.
        try:
            for x, y in pts[1:]: builder.add_line_segments([(inches(x-pts[0][0]), inches(y-pts[0][1]))], close=False)
            shp = builder.convert_to_shape(inches(0), inches(0), inches(1), inches(1))
            shp.fill.solid(); shp.fill.fore_color.rgb = ppt_rgb(fill)
            shp.line.color.rgb = ppt_rgb(stroke or fill)
        except Exception:
            # A small circle is a usable arrow marker if a client rejects freeform geometry.
            self.circle(pts[0][0], pts[0][1], 2.2, fill)
        p = self.pdf.beginPath(); p.moveTo(pts[0][0], PH-pts[0][1])
        for x, y in pts[1:]: p.lineTo(x, PH-y)
        p.close()
        self.pdf.setFillColor(pdf_rgb(fill)); self.pdf.setStrokeColor(pdf_rgb(stroke or fill))
        self.pdf.drawPath(p, stroke=1 if stroke else 0, fill=1)

    def circle(self, cx: float, cy: float, r: float, fill: str, stroke: str | None = None, linew: float = 0.8):
        shp = self.slide.shapes.add_shape(MSO_SHAPE.OVAL, inches(cx-r), inches(cy-r), inches(2*r), inches(2*r))
        shp.fill.solid(); shp.fill.fore_color.rgb = ppt_rgb(fill)
        shp.line.color.rgb = ppt_rgb(stroke or fill); shp.line.width = Pt(linew)
        self.pdf.setFillColor(pdf_rgb(fill)); self.pdf.setStrokeColor(pdf_rgb(stroke or fill)); self.pdf.setLineWidth(linew)
        self.pdf.circle(cx, PH-cy, r, stroke=1 if stroke else 0, fill=1)

    def text(self, x: float, y: float, w: float, h: float, text: str, size: float = 16,
             color: str = "ink", bold: bool = False, italic: bool = False,
             align: str = "left", valign: str = "top", char_space: bool = False):
        text = _vi_copy(str(text))
        shape = self.slide.shapes.add_textbox(inches(x), inches(y), inches(w), inches(h))
        tf = shape.text_frame; tf.clear(); tf.word_wrap = True
        tf.margin_left = Inches(0.015); tf.margin_right = Inches(0.015)
        tf.margin_top = Inches(0.0); tf.margin_bottom = Inches(0.0)
        tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "mid": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[valign]
        lines = text.split("\n")
        for i, line_text in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line_text
            p.space_before = Pt(0); p.space_after = Pt(0)
            p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
            p.font.name = FONT_PPT; p.font.size = Pt(size); p.font.bold = bold; p.font.italic = italic
            p.font.color.rgb = ppt_rgb(color)
            for run in p.runs:
                run.font.name = FONT_PPT; run.font.size = Pt(size); run.font.bold = bold
                run.font.italic = italic; run.font.color.rgb = ppt_rgb(color)
        # PDF counterpart: use Paragraph for robust Vietnamese wrapping.
        pdf_font = FONT_PDF_BOLD if bold else FONT_PDF_ITALIC if italic else FONT_PDF
        raw = html.escape(text).replace("\n", "<br/>")
        style = ParagraphStyle(
            name=f"p{self.pageno}-{int(x)}-{int(y)}", fontName=pdf_font,
            fontSize=size, leading=size*1.23, textColor=pdf_rgb(color),
            alignment={"left": TA_LEFT, "center": TA_CENTER, "right": TA_RIGHT}[align],
            spaceAfter=0, spaceBefore=0, splitLongWords=1,
        )
        para = Paragraph(raw, style)
        _, used_h = para.wrap(w, h)
        if used_h > h + 1.5:
            self.owner.overflow.append((self.pageno, "text", text[:80], used_h, h))
        para.drawOn(self.pdf, x, PH-y-min(used_h, h))
        return shape

    def badge(self, x: float, y: float, w: float, label: str, fill: str = "bluep",
              color: str = "teal", size: float = 9.2):
        self.rect(x, y, w, 22, fill, radius=11)
        self.text(x+4, y+3, w-8, 16, label, size, color, bold=True, align="center", valign="mid")

    def card(self, x: float, y: float, w: float, h: float, heading: str, body: str,
             accent: str = "teal", fill: str = "paper", head_size: float = 16,
             body_size: float = 12.2):
        self.rect(x, y, w, h, fill, stroke="line", radius=10)
        self.rect(x, y, 5, h, accent, radius=2)
        self.text(x+17, y+13, w-31, 27, heading, head_size, "ink", bold=True)
        self.text(x+17, y+44, w-32, h-52, body, body_size, "muted")

    def image(self, path: Path, x: float, y: float, w: float, h: float,
              alt: str = "", contain: bool = True):
        with Image.open(path) as im:
            iw, ih = im.size
        ratio = min(w/iw, h/ih) if contain else max(w/iw, h/ih)
        dw, dh = iw*ratio, ih*ratio
        dx, dy = x+(w-dw)/2, y+(h-dh)/2
        pic = self.slide.shapes.add_picture(str(path), inches(dx), inches(dy), inches(dw), inches(dh))
        try: pic._element._nvXxPr.cNvPr.set("descr", alt)
        except Exception: pass
        self.pdf.drawImage(str(path), dx, PH-dy-dh, dw, dh, preserveAspectRatio=True, mask="auto")
        return pic

    def formula(self, tex: str, x: float, y: float, w: float, h: float,
                size: float = 25, color: str = "ink", label: str | None = None):
        # Keep source/equation notation exactly as supplied; localization is for prose only.
        key = hashlib.sha1(tex.encode("utf-8")).hexdigest()[:12]
        path = self.owner.tmp / f"formula-{key}.png"
        if not path.exists():
            fig = plt.figure(figsize=(max(2.0, w/72), max(0.35, h/72)), dpi=300)
            fig.patch.set_alpha(0.0)
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            ax.text(0.5, 0.5, f"${tex}$", color="#"+hx(color), fontsize=size*300/72,
                    ha="center", va="center")
            fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.035)
            plt.close(fig)
        with Image.open(path) as im:
            iw, ih = im.size
        scale = min(w/iw, h/ih)
        dw, dh = iw*scale, ih*scale
        dx, dy = x+(w-dw)/2, y+(h-dh)/2
        pic = self.slide.shapes.add_picture(str(path), inches(dx), inches(dy), inches(dw), inches(dh))
        try: pic._element._nvXxPr.cNvPr.set("descr", f"LaTeX: {tex}")
        except Exception: pass
        self.pdf.drawImage(str(path), dx, PH-dy-dh, dw, dh, preserveAspectRatio=True, mask="auto")
        if label:
            self.text(x, y+h+2, w, 15, label, 8.5, "muted", italic=True, align="center")
        return pic

    def table(self, x: float, y: float, w: float, headers: list[str],
              rows: list[list[Any]], widths: list[float] | None = None,
              row_h: float = 30, font_size: float = 10.5,
              header_fill: str = "navy", header_size: float | None = None,
              aligns: list[str] | None = None, row_fills: list[str] | None = None,
              cell_colors: list[list[str | None]] | None = None,
              total_h: float | None = None):
        ncol = len(headers)
        if widths is None: widths = [w/ncol]*ncol
        if abs(sum(widths)-w) > 2: raise ValueError(f"Table widths sum {sum(widths)} != {w}")
        head_h = min(34, row_h+2)
        cy = y
        cx = x
        for j, head in enumerate(headers):
            self.rect(cx, cy, widths[j], head_h, header_fill, stroke=header_fill, radius=0, linew=0.25)
            self.text(cx+5, cy+3, widths[j]-10, head_h-6, str(head), header_size or font_size,
                      "white", bold=True, align=(aligns[j] if aligns else "left"), valign="mid")
            cx += widths[j]
        cy += head_h
        for i, row in enumerate(rows):
            cx = x
            rh = row_h
            for j in range(ncol):
                fill = row_fills[i] if row_fills and i < len(row_fills) else ("paper" if i%2 == 0 else "grayp")
                self.rect(cx, cy, widths[j], rh, fill, stroke="line", radius=0, linew=0.35)
                value = "—" if row[j] is None else str(row[j])
                color = (cell_colors[i][j] if cell_colors and i < len(cell_colors) else None) or "ink"
                self.text(cx+5, cy+3, widths[j]-10, rh-6, value, font_size, color,
                          bold=(j == 0), align=(aligns[j] if aligns else "left"), valign="mid")
                cx += widths[j]
            cy += rh
        if total_h and cy-y > total_h:
            self.owner.overflow.append((self.pageno, "table", headers[0], cy-y, total_h))
        return cy

    def note(self):
        try:
            self.slide.notes_slide.notes_text_frame.text = self.notes
        except Exception as exc:
            self.owner.warnings.append(f"speaker notes unavailable on slide {self.pageno}: {exc}")
        self.pdf.showPage()


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(PW/72); self.prs.slide_height = Inches(PH/72)
        self.prs.core_properties.title = "DIRE — comprehensive research and pilot update"
        self.prs.core_properties.subject = "Editable technical presentation; sources and evidence separated"
        self.prs.core_properties.author = "DIRE research project"
        self.prs.core_properties.keywords = "DIRE, diffusion, DDIM, ResNet-50, research pilot"
        self.tmp = Path(tempfile.mkdtemp(prefix="dire-deck-"))
        self.pages: list[Page] = []
        self.overflow: list[tuple] = []
        self.warnings: list[str] = []
        self.pdf = pdfcanvas.Canvas(str(OUT_PDF), pagesize=(PW, PH), pageCompression=1)
        self.pdf.setTitle("DIRE — Comprehensive research and pilot update")
        self.pdf.setAuthor("DIRE research project")

    def page(self, section: str, title: str, source: str, notes: str,
             dark: bool = False, cover: bool = False) -> Page:
        p = Page(self, section, title, source, notes, dark=dark, cover=cover)
        return p

    def save(self):
        self.pdf.save()
        self.prs.save(str(OUT_PPTX))
        return len(self.pages)


def notes_for(source: str, detail: str = "") -> str:
    return (detail + "\n\n" if detail else "") + "Nguồn / provenance: " + source + "\n" + \
        "Tất cả số đo trong deck được ghi nhãn theo provenance; giá trị bị thiếu không được nội suy. " + \
        "Đây là bản trình bày từ artifacts trong checkout, không phải xác nhận live Volume/Modal."


def heading_sub(p: Page, text: str, y: float = 116, color: str = "muted", size: float = 13.2):
    p.text(44, y, 870, 32, text, size, color)


def label_value(p: Page, x: float, y: float, w: float, label: str, value: str,
                accent: str = "teal", value_size: float = 25, fill: str = "paper"):
    p.rect(x, y, w, 85, fill, stroke="line", radius=10)
    p.text(x+15, y+11, w-30, 17, label.upper(), 9, accent, bold=True)
    p.text(x+15, y+32, w-28, 43, value, value_size, "ink", bold=True, valign="mid")


def arrow_box(p: Page, x: float, y: float, w: float, h: float, title: str,
              body: str, fill: str = "bluep", accent: str = "teal"):
    p.rect(x, y, w, h, fill, stroke="line", radius=10)
    p.text(x+7, y+8, w-14, 24, title, 14, accent, bold=True, align="center", valign="mid")
    body_color = "white" if fill == "purple" else "muted"
    p.text(x+10, y+37, w-20, h-44, body, 10.3, body_color, align="center", valign="mid")


def show_equation(p: Page, tex: str, x: float, y: float, w: float, h: float,
                  verbal: str, font_size: float = 24):
    p.rect(x, y, w, h+34, "paper", stroke="line", radius=11)
    p.formula(tex, x+12, y+9, w-24, h, size=font_size)
    p.text(x+14, y+h+5, w-28, 23, verbal, 10.4, "muted", align="center", valign="mid")


# ---------- slide creation ----------
def build_deck() -> Deck:
    d = Deck()
    # 01 — cover
    p = d.page("DIRE · research dossier", "", "", "", dark=True, cover=True)
    p.rect(0, 0, PW, PH, "navy", radius=0)
    # Fine motif: deterministic diffusion trajectory / error field, purely decorative.
    for i in range(8):
        y = 92+i*44
        p.line(610, y, 920, y, "navy2", 0.5)
    for i, color in enumerate(("teal", "mint", "gold")):
        p.circle(650+i*95, 202+(i%2)*22, 12+(i%2)*3, color)
    p.arrow(662, 202, 735, 224, "deepmuted", 1.2)
    p.arrow(758, 224, 835, 202, "deepmuted", 1.2)
    p.text(55, 54, 440, 20, "TECHNICAL DEEP DIVE  ·  RESEARCH UPDATE", 10, "mint", bold=True)
    p.text(55, 112, 520, 112, "DIRE: từ phép tái tạo\ntới bằng chứng", 35, "white", bold=True)
    p.text(58, 238, 500, 60,
           "DIRE · toán học, kiến trúc mô hình,\nđối chiếu bài báo và kết quả pilot trong kho mã", 15, "deepmuted")
    p.badge(58, 325, 174, "PAPER-REPORTED", "navy2", "mint")
    p.badge(243, 325, 186, "REPO-RECORDED", "navy2", "gold")
    p.badge(440, 325, 176, "MISSING ≠ ZERO", "navy2", "white")
    p.text(58, 438, 600, 46,
           "Bản trình bày học thuật · tiếng Việt\nTài liệu nền: DIRE từ số 0 · bản đọc các họ mô hình sinh", 11, "deepmuted")
    p.text(58, 500, 600, 18, "Editable PPTX + PDF  ·  equations typeset from LaTeX source", 9.5, "mint")
    p.note()

    # 02 executive overview
    p = d.page("01 · Tóm tắt", "Báo cáo tách biệt kết quả bài báo, số đo pilot và phần chưa có dữ liệu",
        "handoff.md · report R1+R2 · sổ Table 4 · metadata gallery",
        notes_for("handoff.md; modal-runs/runs/report.json; Table 4 ledger; gallery/*.json",
                  "Các số liệu được giữ theo đúng nguồn và phạm vi; chưa xác minh bản nhị phân checkpoint và trạng thái lưu trữ ngoài repo."))
    heading_sub(p, "Trong checkout có report R1+R2, pilot Table 4 và 64 mẫu gallery; Table 5/Figure 4 chưa có sổ kết quả.")
    label_value(p, 48, 157, 205, "Các mức S đã đo", "5 / 10 / 20", "teal", 22)
    label_value(p, 266, 157, 205, "Mỗi ô kiểm thử", "n = 400", "teal", 23)
    label_value(p, 484, 157, 205, "Mẫu gallery", "64", "teal", 25)
    label_value(p, 702, 157, 205, "Nguồn", "Bài báo / repo", "gold", 18, "goldp")
    p.card(48, 260, 270, 147, "Đã ghi nhận",
           "Báo cáo R1+R2; đánh giá ADM trên tập giữ lại; pilot Table 4; 64 mẫu gallery có metadata.", "teal")
    p.card(345, 260, 270, 147, "Chưa có kết quả",
           "Table 5 mới có giao thức/đầu vào; Figure 4 chưa có kết quả. Không suy diễn S=50 cho pilot.", "gold", "goldp")
    p.card(642, 260, 270, 147, "Giới hạn nguồn",
           "Checkpoint chưa có trong checkout; ảnh gallery là PNG hiển thị, không thay thế map nguồn.", "red", "redp")
    p.text(52, 431, 855, 46,
           "Kết luận: báo cáo nêu rõ đâu là kết quả công bố, đâu là số đo pilot và đâu là câu hỏi chưa được kiểm chứng.",
           14, "ink", bold=True)
    p.note()

    # 03 timeline/evidence discipline
    p = d.page("01 · Cách đọc bằng chứng", "Mốc ghi nhận trong repo nằm sau ngày phiên làm việc",
        "Git metadata · report.json:data.created · developer session date",
        notes_for("git commit 174e8cf (2026-11-25); report.json created=2026-11-17T03:17:09Z; system date=2026-10-14",
                  "Mốc của checkout và report đều sau ngày hiện tại do môi trường khai báo. Vì vậy deck trích chúng như 'repository-recorded', không khẳng định thời điểm thực nghiệm; cần giải quyết timeline provenance trước khi công bố."))
    p.rect(52, 145, 856, 77, "goldp", stroke="gold", radius=12)
    p.text(72, 159, 815, 48,
           "CẢNH BÁO PROVENANCE  ·  Repo commit: 25-11-2026   |   report.created: 17-11-2026   |   ngày phiên: 14-10-2026",
           15, "red", bold=True, align="center", valign="mid")
    steps = [(70, "Paper", "Nguồn công bố\nWang et al., ICCV 2023", "bluep", "teal"),
             (288, "Repo artifact", "Có file + ledger\nnhưng timeline chưa khớp", "goldp", "gold"),
             (506, "Live Volume", "Chưa đối chiếu\ntrong phiên này", "redp", "red"),
             (724, "Claim", "Tạm gọi\nrepo-recorded", "grayp", "muted")]
    for x, title, body, fill, accent in steps: arrow_box(p, x, 275, 165, 116, title, body, fill, accent)
    for x in (241, 459, 677): p.arrow(x, 332, x+37, 332, "muted", 1.4)
    p.text(65, 425, 820, 45,
           "Số đo từ bảng hiện tại được giữ lại và gắn nhãn nguồn. Cho tới khi timeline / Modal Volume được xác nhận, không gọi chúng là kết quả đã kiểm chứng độc lập.",
           13, "ink", bold=True, align="center")
    p.note()

    # 04 project scope and question
    p = d.page("02 · Bài toán", "Câu hỏi trung tâm là khả năng tổng quát hóa qua generator chưa thấy",
        "Wang et al., ICCV 2023 §3 · docs/dire-tu-so-0.pdf §2.3",
        notes_for("Wang et al. (2023); docs/dire-tu-so-0.pdf, pp. 20–22, 41"))
    heading_sub(p, "Train và test có thể khác p_g: nhận diện dấu vết của một generator không đồng nghĩa phát hiện ảnh sinh nói chung.")
    arrow_box(p, 74, 205, 220, 120, "TRAIN", "Ảnh thật + ảnh giả\nnguồn / kiến trúc A", "bluep", "teal")
    p.arrow(302, 265, 390, 265, "teal", 2)
    p.rect(397, 205, 164, 120, "navy", radius=14)
    p.text(411, 229, 136, 74, "DETECTOR\nđã học", 18, "white", bold=True, align="center", valign="mid")
    p.arrow(568, 265, 652, 265, "teal", 2)
    arrow_box(p, 660, 205, 220, 120, "UNSEEN TEST", "Generator B/C\n+ ảnh thật", "goldp", "gold")
    p.card(75, 362, 363, 88, "Shortcut risk", "Mạng có thể học fingerprint generator thay vì dấu hiệu bền vững.", "red", "redp", 14, 11)
    p.card(481, 362, 363, 88, "Research target", "Tìm biểu diễn có ích khi generator test ≠ generator train.", "teal", "greenp", 14, 11)
    p.note()

    # 05 taxonomy
    p = d.page("03 · Các họ mô hình sinh", "Các họ khác nhau ở likelihood, số bước và chất lượng phép nghịch đảo",
        "docs/paper-cac-mo-hinh-sinh.html §§01–07 · docs/dire-tu-so-0.pdf §4",
        notes_for("docs/paper-cac-mo-hinh-sinh.html, §§01–07; citations ở phụ lục"))
    heading_sub(p, "DIRE không hỏi mô hình nào 'mạnh hơn'; nó cần một đường đi–về tái lập được nhưng không phải nghịch đảo hoàn hảo.")
    rows = [
        ["VAE", "encoder → latent → decoder", "xấp xỉ · một lượt", "Có encoder; không phải inverse chính xác", "Không dùng làm I/R của DIRE"],
        ["GAN / StyleGAN", "z → G(z)", "không có inverse tích hợp", "GAN inversion là tối ưu bổ sung", "Không có I/R gốc"],
        ["Normalizing flow", "bijective f", "inverse chính xác (lý tưởng)", "R∘I ≈ identity", "DIRE lý tưởng suy biến về 0"],
        ["VQ / VQ-Diffusion", "token/codebook + decoder", "rời rạc; nhiều bước với VQ-Diffusion", "Có lượng tử hoá / token dynamics", "Cần phân biệt pixel map với token"],
        ["LDM / SD", "diffusion trong latent VAE", "nhiều bước latent", "DDIM có thể inversion xấp xỉ", "Có trong benchmark paper"],
        ["Pixel diffusion / ADM", "denoise U-Net trên pixel", "T bước → S bước", "DDIM inversion + sampling", "Backbone I/R dùng trong DIRE"],
    ]
    p.table(48, 153, 864, ["Họ", "Đường sinh", "Inverse / trajectory", "Hệ quả lý thuyết", "Vị trí với DIRE"],
            rows, [104, 179, 169, 210, 202], 50, 9.4)
    p.text(52, 483, 850, 15, "Autoregressive, EBM và flow-matching được ghi là phạm vi còn thiếu trong tài liệu tổng quan HTML; không suy rộng từ bảng này.", 9, "muted", italic=True)
    p.note()

    # 03a · Generative-model foundations (theory source: supplementary HTML)
    p = d.page("03 · Nền tảng các họ mô hình", "VAE học một phân phối ẩn và một bộ giải mã; ELBO cân bằng tái tạo với độ lệch prior",
        "docs/paper-cac-mo-hinh-sinh.html §02 · Kingma & Welling (2014)",
        notes_for("docs/paper-cac-mo-hinh-sinh.html §02; Kingma & Welling, Auto-Encoding Variational Bayes"))
    show_equation(p, r"\mathcal{L}_{\mathrm{ELBO}}=\mathbb{E}_{q_{\phi}(z\mid x)}[\log p_{\theta}(x\mid z)]-D_{\mathrm{KL}}\!\left(q_{\phi}(z\mid x)\parallel p(z)\right)",
                  87, 145, 786, 72, "Tái tạo tốt hơn ↔ latent posterior gần prior; encoder chỉ xấp xỉ posterior.", 19)
    p.arrow(200, 316, 356, 316, "teal", 1.7)
    p.arrow(592, 316, 749, 316, "teal", 1.7)
    arrow_box(p, 72, 278, 123, 76, "x", "image", "bluep", "teal")
    arrow_box(p, 362, 278, 224, 76, "qφ(z | x)", "encoder → latent distribution", "purple", "white")
    arrow_box(p, 756, 278, 123, 76, "pθ(x | z)", "decoder", "greenp", "teal")
    p.card(107, 390, 350, 84, "Vì sao không dùng làm phép biến đổi DIRE?", "Một lượt encoder/decoder; không có quỹ đạo DDIM được học.", "gold", "goldp", 12.5, 10.5)
    p.card(494, 390, 350, 84, "Vai trò trong so sánh", "VAE là codec không gian ẩn bên trong mô hình khuếch tán tiềm ẩn.", "teal", "bluep", 12.5, 10.0)
    p.note()

    p = d.page("03 · Nền tảng các họ mô hình", "GAN học bộ sinh đối kháng, không có phép nghịch đảo ảnh–tiềm ẩn tường minh",
        "docs/paper-cac-mo-hinh-sinh.html §03 · Goodfellow et al. (2014)",
        notes_for("docs/paper-cac-mo-hinh-sinh.html §03; Goodfellow et al., Generative Adversarial Nets (2014)"))
    show_equation(p, r"\min_G\max_D\;\mathbb{E}_{x\sim p_{data}}\log D(x)+\mathbb{E}_{z\sim p_z}\log\!\left(1-D(G(z))\right)",
                  95, 146, 770, 68, "G biến nhiễu thành ảnh; D phân biệt ảnh thật và ảnh sinh trong lúc huấn luyện.", 19)
    arrow_box(p, 97, 278, 144, 80, "z ~ p(z)", "nhiễu tiềm ẩn", "bluep", "teal")
    p.arrow(244,318,348,318,"teal",1.8)
    arrow_box(p, 357, 278, 201, 80, "G(z)", "bộ sinh", "purple", "white")
    p.arrow(562,318,664,318,"teal",1.8)
    arrow_box(p, 674, 278, 185, 80, "x_fake", "ảnh được sinh", "greenp", "teal")
    p.rect(149, 396, 663, 58, "goldp", stroke="gold", radius=11)
    p.text(167, 406, 627, 39, "GAN cơ bản không có mật độ xác suất hay phép nghịch đảo tích hợp; cần tối ưu riêng để tìm latent.", 12.2, "ink", bold=True, align="center", valign="mid")
    p.note()

    p = d.page("03 · Nền tảng các họ mô hình", "Normalizing flow lý tưởng nghịch đảo chính xác — làm mất tín hiệu residual DIRE",
        "docs/paper-cac-mo-hinh-sinh.html §04 · Dinh et al. (2017)",
        notes_for("docs/paper-cac-mo-hinh-sinh.html §04; change-of-variables density equation; idealized reasoning is not a paper experiment"))
    show_equation(p, r"\log p_X(x)=\log p_Z(f(x))+\log\left|\det J_f(x)\right|",
                  152, 154, 656, 63, "Đổi biến: ánh xạ song ánh giúp tính likelihood chính xác.", 22)
    p.rect(123, 278, 714, 121, "redp", stroke="red", radius=12)
    p.text(147, 291, 666, 96,
           "HỆ QUẢ LÝ TƯỞNG\nChọn I=f và R=f⁻¹ (hoặc ngược lại): R(I(x))=x với mọi x, nên DIRE(x)=0. Đây là suy luận toán học về khả năng nghịch đảo chính xác — không phải kết quả thực nghiệm trong bài DIRE.",
           13, "red", bold=True, align="center", valign="mid")
    p.text(155, 431, 650, 28, "Vì vậy, DIRE cần phép nghịch đảo xấp xỉ — không phải một phép hoàn hảo tùy ý.", 13, "ink", bold=True, align="center")
    p.note()

    p = d.page("03 · Nền tảng các họ mô hình", "VQ-Diffusion và latent diffusion đưa quỹ đạo khuếch tán sang không gian khác",
        "docs/paper-cac-mo-hinh-sinh.html §§05–06 · van den Oord et al. (2017) · Rombach et al. (2022)",
        notes_for("docs/paper-cac-mo-hinh-sinh.html §§05–06; VQ-VAE and latent diffusion papers"))
    show_equation(p, r"z_q=e_{k^*},\qquad k^*=\arg\min_k\|z_e-e_k\|_2",
                  93, 145, 360, 58, "VQ ánh xạ đặc trưng encoder tới mục codebook.", 19)
    show_equation(p, r"z_t=\sqrt{\bar{\alpha}_t}z_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon",
                  507, 145, 360, 58, "LDM khuếch tán liên tục trong không gian ẩn VAE.", 19)
    p.card(73, 273, 385, 135, "VQ-Diffusion", "Khuếch tán rời rạc trên token codebook; sai số lượng tử hóa thuộc về biểu diễn. Mô hình này có trong benchmark generator của bài báo.", "purple", "bluep", 14.5, 11.2)
    p.card(502, 273, 385, 135, "Latent diffusion / SD", "Encoder VAE → quỹ đạo DDIM trong latent → decoder VAE. Bài DIRE vẫn dùng ADM trên pixel làm phép tái tạo tham chiếu.", "teal", "greenp", 14.5, 11.2)
    p.text(126, 439, 707, 26, "Khác biệt không gian mô hình rất quan trọng: generator bị phát hiện không phải backbone tái tạo.", 11.4, "red", bold=True, align="center")
    p.note()

    # 06 conceptual criteria
    p = d.page("03 · Trực giác và giới hạn", "DIRE là sai số của phép vòng lặp xấp xỉ — không phải định lý về ảnh giả",
        "docs/paper-cac-mo-hinh-sinh.html §08 (analysis) · Wang et al. (2023), Eq. 9",
        notes_for("Bài báo định nghĩa DIRE; ba điều kiện C1–C3 là phân tích trong docs/paper-cac-mo-hinh-sinh.html §08, không phải claim của paper"))
    show_equation(p, r"\mathrm{DIRE}(x)=\left\|x-\mathcal{R}(\mathcal{I}(x))\right\|_1",
                  172, 138, 616, 64, "Trị tuyệt đối theo pixel và kênh; bản đồ giữ cấu trúc không gian.", 26)
    for i, (head, body, col) in enumerate([
        ("C1 · Có đường đi–về", "Cùng backbone, schedule và các timestep; inversion xấp xỉ, không stochastic.", "teal"),
        ("C2 · Có hình học hữu ích", "Phép vòng đưa ảnh về vùng mà model đã học — diễn giải, không phải chứng minh.", "purple"),
        ("C3 · Có phân biệt thực nghiệm", "Phân phối residual của ảnh real và generated phải khác nhau; phải đo.", "gold"),
    ]):
        x=70+i*286; p.card(x, 266, 258, 126, head, body, col, "paper", 12.4, 11.1)
    p.rect(115, 419, 730, 48, "redp", stroke="red", radius=10)
    p.text(131, 429, 700, 29, "Giả thuyết cần kiểm chứng: ảnh sinh có thể có residual nhỏ hơn.", 12.7, "red", bold=True, align="center", valign="mid")
    p.note()

    # 07 gaussian forward
    p = d.page("04 · Diffusion mathematics", "Quá trình thuận biến ảnh sạch thành ảnh nhiễu có phân phối đã biết",
        "docs/dire-tu-so-0.pdf §5.1 · Ho et al. (2020), Eqs. 1–4",
        notes_for("Ho, Jain & Abbeel, DDPM (2020); docs/dire-tu-so-0.pdf, pp. 27–28"))
    show_equation(p, r"q(x_t\mid x_{t-1})=\mathcal{N}\!\left(\sqrt{\alpha_t}x_{t-1},\,\beta_t I\right)",
                  122, 145, 716, 62, "Mỗi bước co tín hiệu một ít rồi thêm nhiễu Gauss; α_t = 1−β_t.", 23)
    show_equation(p, r"\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s,\qquad x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)",
                  87, 257, 786, 73, "Dạng đóng: nhảy trực tiếp từ x₀ tới timestep t chỉ với một mẫu nhiễu ε.", 19)
    p.card(86, 383, 355, 88, "T = 1,000", "ADM dùng schedule tuyến tính β_t; tích lũy ᾱ_t làm tín hiệu suy giảm theo thời gian.", "teal", "bluep", 14, 11)
    p.card(468, 383, 405, 88, "Không mất ngẫu nhiên trong định nghĩa", "ε ở forward process giải thích noise target khi train diffusion; DDIM inference chọn đường determinist.", "purple", "paper", 14, 11)
    p.note()

    # 08 epsilon objective
    p = d.page("04 · Diffusion mathematics", "U-Net học dự đoán nhiễu — không dự đoán trực tiếp ảnh sạch",
        "docs/dire-tu-so-0.pdf §5.2 · Ho et al. (2020) · Dhariwal & Nichol (2021)",
        notes_for("docs/dire-tu-so-0.pdf, pp. 27–28; Ho et al. (2020); Dhariwal & Nichol (2021)"))
    show_equation(p, r"\mathcal{L}_{\epsilon}=\mathbb{E}_{x_0,\epsilon,t}\left[\left\|\epsilon-\epsilon_{\theta}(x_t,t)\right\|_2^2\right]",
                  122, 146, 716, 70, "Supervision tự sinh: biết ε vì chính quy trình forward vừa thêm nó.", 24)
    p.arrow(201, 307, 356, 307, "teal", 2)
    arrow_box(p, 69, 270, 126, 77, "x₀ + ε", "noisy input", "bluep", "teal")
    arrow_box(p, 366, 270, 187, 77, "εθ(xₜ,t)", "U-Net output", "greenp", "teal")
    p.arrow(563, 307, 689, 307, "teal", 2)
    arrow_box(p, 697, 270, 191, 77, "MSE", "prediction vs ε", "goldp", "gold")
    p.text(86, 385, 784, 50,
           "ADM dùng epsilon-prediction. Timestep embedding cung cấp t cho mọi khối; loss rút gọn là dạng huấn luyện thực dụng liên quan ELBO.",
           13, "ink", bold=True, align="center")
    p.note()

    # 09 U-Net architecture
    p = d.page("05 · Kiến trúc diffusion / U-Net", "ADM dùng encoder–bottleneck–decoder với skip connections đa tỉ lệ",
        "docs/dire-tu-so-0.pdf §5.3 · guided-diffusion model config",
        notes_for("docs/dire-tu-so-0.pdf, pp. 28–29; configuration recorded in paper/code walkthrough"))
    heading_sub(p, "Timestep embedding đi vào từng stage; attention được đặt tại các bản đồ đặc trưng 32×32, 16×16, 8×8.")
    # Native editable U-shaped stage diagram
    enc = [(74, 210, 114, "256²", "input"), (207, 241, 104, "128²", "down"),
           (329, 272, 96, "64²", "down"), (441, 303, 88, "32²", "down")]
    dec = [(540, 303, 88, "32²", "up"), (644, 272, 96, "64²", "up"),
           (756, 241, 104, "128²", "up"), (872, 210, 54, "256²", "out")]
    for x,y,w,top,bottom in enc+dec:
        fill = "navy" if top=="32²" else "bluep" if bottom in ("input","down") else "greenp"
        p.rect(x,y,w,58,fill,stroke="line",radius=9)
        p.text(x+3,y+7,w-6,20,top,13,"white" if fill=="navy" else "ink",bold=True,align="center")
        p.text(x+3,y+31,w-6,16,bottom,9,"deepmuted" if fill=="navy" else "muted",align="center")
    for a,b in [((188,239),(207,253)),((311,270),(329,284)),((425,301),(441,315)),
                ((529,332),(540,332)),((628,315),(644,284)),((740,284),(756,253)),((860,253),(872,239))]:
        p.arrow(a[0],a[1],b[0],b[1],"teal",1.2)
    # skips
    for y1,y2,x1,x2 in [(211,365,131,582),(242,352,259,690),(273,338,377,795)]:
        p.line(x1,y1,x2,y2,"gold",1.1); p.line(x1,y1,x1+7,y1,"gold",1.1); p.line(x2,y2,x2-7,y2,"gold",1.1)
    p.text(275, 362, 420, 20, "SKIP CONNECTIONS · giữ chi tiết nông qua bottleneck", 9.5, "gold", bold=True, align="center")
    p.rect(69, 397, 818, 70, "paper", stroke="line", radius=11)
    p.text(91, 408, 770, 44,
           "Đi xuống: giảm resolution, tăng khả năng biểu diễn.  Bottleneck: context toàn cục.  Đi lên: khôi phục chi tiết qua skip.  Đây là sơ đồ kiến trúc, không phải tensor visualization.",
           12, "ink", align="center", valign="mid")
    p.note()

    # 10 U-Net blocks
    p = d.page("05 · Kiến trúc diffusion / U-Net", "Residual blocks học biến đổi cục bộ; attention trao đổi thông tin toàn cục",
        "docs/dire-tu-so-0.pdf §5.3 · attention / residual equations",
        notes_for("docs/dire-tu-so-0.pdf, §5.3; Dhariwal & Nichol (2021)"))
    p.rect(63, 151, 405, 289, "paper", stroke="line", radius=12)
    p.text(84, 168, 360, 25, "RESIDUAL BLOCK", 13, "teal", bold=True)
    blocks=[("h", "input feature"),("Norm + SiLU", "stabilize / nonlinearity"),("Conv 3×3", "local spatial mixing"),("+ time embedding", "condition on t"),("Conv 3×3", "refine features")]
    for i,(a,b) in enumerate(blocks):
        y=210+i*38
        p.rect(130,y,270,29,"bluep" if i%2==0 else "grayp",stroke="line",radius=6)
        p.text(139,y+3,251,21,a+"  ·  "+b,10.2,"ink",bold=(i==0),align="center",valign="mid")
        if i<4:p.arrow(265,y+29,265,y+36,"muted",0.8)
    p.arrow(112, 218, 112, 389, "gold", 1.7); p.arrow(112, 389, 128, 389, "gold", 1.7)
    p.text(80, 398, 390, 20, "Residual: y = x + F(x) · đường gradient ngắn", 9.5, "gold", bold=True, align="center")
    p.rect(495, 151, 402, 289, "paper", stroke="line", radius=12)
    p.text(517, 168, 360, 25, "SELF-ATTENTION", 13, "purple", bold=True)
    for i,(lab,x) in enumerate([("Q",545),("K",645),("V",745)]):
        p.rect(x,214,62,42,"grayp",stroke="line",radius=8); p.text(x+4,223,54,22,lab,17,"ink",bold=True,align="center")
    p.text(541, 280, 322, 51, "Q, K, V projections\nacross spatial positions", 12,"muted",align="center")
    p.formula(r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{T}}{\sqrt{d}}\right)V", 521, 340, 354, 40, size=17)
    p.text(517, 401, 359, 21, "Global context, used at selected resolutions.", 10, "muted", align="center")
    p.note()

    # 10a · Hình minh họa trực quan cho convolution, attention và residual path.
    p = d.page("05 · Minh họa các lớp mạng", "Các lớp biến đổi ảnh thành đặc trưng theo không gian",
        "Ảnh tổng hợp · tools/build_report_illustrations.py · minh họa toán tử, không phải activation ADM",
        notes_for("Deterministic synthetic image; Sobel/Laplacian filter responses, illustrative attention weights and a ResNet-style residual block. No model checkpoint was run."))
    cards=[
        (53,"TÍCH CHẬP 3×3","conv_feature_maps.png",(64,198,238,93),
         "Bộ lọc cục bộ làm nổi cạnh dọc, cạnh ngang và chi tiết; mỗi kernel tạo một bản đồ đặc trưng khác nhau."),
        (355,"SELF-ATTENTION","attention_weights.png",(427,188,122,122),
         "Một vị trí truy vấn phân bổ trọng số lên các vị trí khác. Ô vàng là điểm truy vấn; vùng sáng có trọng số lớn hơn."),
        (657,"RESNET BOTTLENECK","resnet_residual_path.png",(665,218,234,55),
         "Nhánh chính biến đổi đặc trưng; nhánh tắt cộng lại trước ReLU để giữ đường truyền thông tin/gradient."),
    ]
    for x,title,filename,(ix,iy,iw,ih),body in cards:
        p.rect(x,151,250,293,"paper",stroke="line",radius=11)
        p.text(x+10,164,230,24,title,10.4,"teal",bold=True,align="center")
        p.image(ILLUSTRATIONS/filename,ix,iy,iw,ih,title)
        p.text(x+14,312,222,105,body,10.5,"ink",align="center",valign="mid")
    p.rect(107,459,746,34,"goldp",stroke="gold",radius=9)
    p.text(119,467,722,19,"Minh họa trên ảnh toy; không phải feature map hay attention score được trích từ ADM/ResNet.",9.7,"ink",bold=True,align="center")
    p.note()

    # 11 ADM config
    p = d.page("05 · Cấu hình ADM", "Cấu hình ADM 256² xác định mô hình tham chiếu cho phép tái tạo",
        "docs/dire-tu-so-0.pdf §5.3 · Table 4 protocol / model SHA recorded in repo",
        notes_for("docs/dire-tu-so-0.pdf, §5.3; Table 4 ledger protocol keys"))
    config=[
        ["Model family", "ADM / guided-diffusion U-Net", "Pixel-space, unconditional"],
        ["Input size", "256 × 256 RGB", "3-channel image tensor"],
        ["Diffusion schedule", "T = 1,000 · linear β", "DIRE paper uses respaced DDIM S=20"],
        ["Width / depth", "num_channels=256 · 2 residual blocks", "ADM config recorded in source"],
        ["Attention", "resolutions 32 / 16 / 8", "64 channels per head"],
        ["Output mode", "learn_sigma=True · class_cond=False", "unconditional bedroom model"],
        ["Weight provenance", "OpenAI ADM checkpoint", "not trained by this project"],
    ]
    p.table(70, 150, 820, ["Setting", "Value", "Interpretation"], config,
            [178, 296, 346], 39, 10.5)
    p.badge(354, 475, 252, "REPOSITORY-CHECKED HASH IN LEDGER", "goldp", "gold")
    p.note()

    # 12 DDPM vs DDIM
    p = d.page("06 · DDIM", "DDIM tạo một quỹ đạo xác định để có thể đi ngược từ ảnh",
        "Ho et al. (2020) · Song, Meng & Ermon (2021) · docs/dire-tu-so-0.pdf §5.4",
        notes_for("Ho et al. (2020); Song et al. (2021); docs/dire-tu-so-0.pdf, pp. 29–30"))
    p.card(65, 157, 395, 148, "DDPM · stochastic reverse", "xₜ₋₁ được lấy mẫu từ phân phối Gaussian mỗi bước.\nƯu điểm: sampling gốc.\nHệ quả cho DIRE: stochastic noise làm residual thay đổi nếu không cố định path.", "purple", "bluep", 17, 12)
    p.card(500, 157, 395, 148, "DDIM · deterministic path", "η = 0 loại thành phần noise bổ sung.\nCùng xₜ và protocol → cùng x₀.\nPhép reverse ODE có thể đi theo chiều tăng timestep.", "teal", "greenp", 17, 12)
    show_equation(p, r"x_{t-1}=\sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0+\sqrt{1-\bar{\alpha}_{t-1}}\,\epsilon_{\theta}(x_t,t)",
                  99, 343, 762, 64, "Dạng rút gọn DDIM khi η=0; không có thành phần nhiễu ngẫu nhiên.", 19)
    p.note()

    # 13 DDIM reconstruct x0 and inverse
    p = d.page("06 · DDIM equations", "Một ước lượng x̂₀ nối noise prediction với sampling và inversion",
        "docs/dire-tu-so-0.pdf §§5.4–5.5 · DDIM Eqs. 5, 8",
        notes_for("Song, Meng & Ermon (2021); docs/dire-tu-so-0.pdf, §§5.4–5.5"))
    show_equation(p, r"\hat{x}_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\,\epsilon_{\theta}(x_t,t)}{\sqrt{\bar{\alpha}_t}}",
                  124, 146, 712, 68, "Khôi phục dự đoán ảnh sạch từ trạng thái nhiễu và ε̂.", 23)
    p.arrow(276, 309, 683, 309, "teal", 2.2)
    p.circle(286,309,24,"bluep","teal",1.3); p.text(265,300,42,20,"xₜ",14,"ink",bold=True,align="center")
    p.circle(670,309,24,"greenp","teal",1.3); p.text(650,300,42,20,"xₜ₋₁",12,"ink",bold=True,align="center")
    p.text(377, 278, 206, 22, "sampling · t giảm", 11, "teal", bold=True, align="center")
    p.arrow(683, 381, 300, 381, "gold", 2.2)
    p.text(380, 389, 210, 22, "inversion · t tăng", 11, "gold", bold=True, align="center")
    p.rect(183, 423, 600, 45, "goldp", stroke="gold", radius=10)
    p.text(198, 432, 570, 26, "Giữ nguyên model, schedule và timestep; thay đổi chúng sẽ đổi residual.", 11.5, "ink", bold=True, align="center", valign="mid")
    p.note()

    # 14 inversion details
    p = d.page("06 · DDIM inversion", "Inversion dùng cùng phương trình theo chiều timestep tăng",
        "docs/dire-tu-so-0.pdf §5.5 · guided_diffusion.ddim_reverse_sample",
        notes_for("Song et al. (2021); docs/dire-tu-so-0.pdf, pp. 29–30; code walkthrough §A"))
    show_equation(p, r"x_{t+1}=\sqrt{\bar{\alpha}_{t+1}}\,\hat{x}_0+\sqrt{1-\bar{\alpha}_{t+1}}\,\hat{\epsilon}_{\theta}(x_t,t)",
                  98, 148, 764, 67, "Đảo ngược x₀ về điểm latent/noisy x_T; deterministic path yêu cầu η=0.", 21)
    for i,(lab,col) in enumerate([("x₀", "teal"),("t₁","mint"),("t₂","mint"),("…","muted"),("x_T","gold")]):
        x=133+i*166; p.circle(x,318,27,"paper",col,1.7); p.text(x-23,308,46,20,lab,14,col,bold=True,align="center")
        if i<4:p.arrow(x+31,318,x+130,318,col,1.5)
    p.text(117, 376, 728, 50,
           "The reverse step is not an algebraically exact inverse after finite-step discretization, clipping, or finite precision. That non-zero reconstruction error is the representation being measured.",
           12, "muted", align="center")
    p.note()

    # 15 · Quỹ đạo DDIM và map DIRE minh họa bằng ảnh tổng hợp.
    p = d.page("06 · Minh họa DDIM", "Từ ảnh sạch qua nhiễu DDIM đến ảnh tái tạo",
        "Minh họa tổng hợp tạo bởi tools/build_report_illustrations.py · seed=3407 · S=20",
        notes_for("Synthetic teaching illustration: deterministic toy DDIM trajectory; the final reconstructed thumbnail is intentionally blurred to make the schematic residual legible. Not ADM/U-Net inference or an experimental measurement."))
    sequence=[
        ("ddim_x0.png", "Ảnh đầu vào x₀", "ảnh tổng hợp"),
        ("ddim_xt500.png", "Đảo · xₜ", "t xấp xỉ 500"),
        ("ddim_xt950.png", "Đầu quỹ đạo", "t xấp xỉ 950"),
        ("ddim_reverse500.png", "Tái tạo · xₜ", "t giảm"),
        ("ddim_xhat0.png", "Ước lượng x̂₀", "DDIM η=0"),
        ("toy_dire_residual.png", "|x₀ − x̂₀|", "map toy minh họa"),
    ]
    left, gap, iw = 43, 14, 132
    for i,(filename,caption,sub) in enumerate(sequence):
        x=left+i*(iw+gap)
        p.rect(x,162,iw,174,"paper",stroke="line",radius=9)
        p.image(ILLUSTRATIONS/filename,x+8,171,iw-16,iw-16,caption)
        p.text(x+6,306,iw-12,21,caption,9.2,"ink",bold=True,align="center",valign="mid")
        p.text(x+6,326,iw-12,14,sub,7.7,"muted",align="center")
        if i<5:
            p.arrow(x+iw+1,238,x+iw+gap-2,238,"teal" if i>=3 else "gold",1.1)
    p.text(71, 365, 818, 29, "Đảo: timestep tăng  →  tái tạo: timestep giảm  →  DIRE = |x₀ − R(I(x₀))|", 14, "ink", bold=True, align="center")
    p.rect(93, 410, 774, 53, "goldp", stroke="gold", radius=10)
    p.text(109, 420, 742, 33, "Minh họa tổng hợp: ảnh tái tạo được làm mờ có chủ ý để residual dễ thấy; không phải ADM hay kết quả đo ảnh thật.", 10.8, "ink", bold=True, align="center", valign="mid")
    p.note()

    # Hình gốc của tác giả: dùng trực tiếp asset từ kho chính thức, không vẽ lại chart.
    p = d.page("06 · Hình gốc từ bài báo", "Sơ đồ gốc của tác giả: đảo DDIM, tái tạo và bản đồ DIRE",
        "Wang et al., ICCV 2023 · kho tác giả ZhendongWang6/DIRE · figs/dire.png",
        notes_for("Author-supplied image asset used without redrawing: https://github.com/ZhendongWang6/DIRE/blob/main/figs/dire.png . The image presents DDIM inversion, reconstruction and DIRE. It is not project pilot data."))
    p.text(80, 117, 800, 19, "Hình nguyên bản từ kho tác giả; nhãn và số phương trình được giữ nguyên.", 9.5, "muted", align="center")
    p.image(ILLUSTRATIONS / "author_dire.png", 185, 137, 590, 361,
            "Hình gốc của tác giả về DDIM inversion, reconstruction và DIRE")
    p.note()

    # 16 DIRE formula
    p = d.page("07 · Định nghĩa DIRE", "DIRE giữ lại bản đồ sai số thay vì ảnh tái tạo",
        "Wang et al. (2023), Eq. 9 · docs/paper-code-walkthrough.md §6",
        notes_for("Wang et al., ICCV 2023, Eq. 9; docs/paper-code-walkthrough.md §6"))
    show_equation(p, r"\mathrm{DIRE}(x_0)=\left|x_0-\mathcal{R}\!\left(\mathcal{I}(x_0)\right)\right|",
                  115, 143, 730, 76, "Absolute error, pixel-wise and channel-wise; output is a 3-channel image.", 29)
    items=[("Input", "x₀", "RGB image", "bluep", "teal"),
           ("Invert", "I(x₀)", "DDIM S steps", "purple", "white"),
           ("Reconstruct", "R(I(x₀))", "DDIM S steps", "teal", "white"),
           ("Residual", "|x₀ − x̂₀|", "DIRE map", "goldp", "gold")]
    for i,(title,eq,sub,fill,accent) in enumerate(items):
        x=64+i*225; arrow_box(p,x,286,178,104,title,eq+"\n"+sub,fill,accent)
        if i<3:p.arrow(x+182,337,x+216,337,"muted",1.3)
    p.text(119, 424, 724, 35, "Lớn hơn không tự động có nghĩa 'fake'; classifier học phân phối map từ dữ liệu có nhãn.", 13, "red", bold=True, align="center")
    p.note()

    # 17 mechanisms / abs ablation
    p = d.page("07 · Ablation paper", "Lấy trị tuyệt đối cải thiện độ ổn định của residual trong Table 6",
        "Wang et al. (2023), Table 6 · docs/dire-tu-so-0.pdf §7.2",
        notes_for("Wang et al., ICCV 2023, Table 6 (paper-reported); docs/dire-tu-so-0.pdf p. 35"))
    heading_sub(p, "Ví dụ minh họa trên detector ADM: Table 6 so sánh raw signed residual với |residual|; đây là paper result, không phải pilot.")
    p.table(104, 159, 752, ["Input", "ADM", "DDPM", "iDDPM", "PNDM", "SD-v1"],
            [["Signed residual", "100 / 100", "99.4 / 100", "100 / 100", "98.2 / 100", "87.0 / 93.0"],
             ["Absolute residual", "100 / 100", "100 / 100", "100 / 100", "99.7 / 100", "99.7 / 100"]],
            [170,116,116,116,116,118], 63, 12)
    p.rect(186, 350, 590, 62, "greenp", stroke="teal", radius=12)
    p.text(204, 362, 554, 37, "Largest listed change: SD-v1 ACC 87.0 → 99.7 (percentage points).", 13, "teal", bold=True, align="center", valign="mid")
    p.text(124, 434, 708, 35, "The result supports the chosen representation; it does not establish a universal mechanism.", 11, "muted", align="center")
    p.note()

    # 18 preprocessing
    p = d.page("08 · Từ pixel tới map", "Ba miền giá trị khác nhau phải được giữ riêng trong pipeline",
        "docs/dire-tu-so-0.pdf §§1.2, 7.3–7.4 · gallery metadata display config",
        notes_for("docs/dire-tu-so-0.pdf; modal-runs/gallery/*.json; paper preprocessing"))
    cols=[("A · ẢNH NGUỒN", "uint8 [0,255]\nRGB", "raw pixels", "bluep","teal"),
          ("B · ADM INPUT", "float [−1,1]\ncenter crop/resize 256", "diffusion domain", "purple","white"),
          ("C · RESIDUAL", "|x − x̂|\nfloat [0,2], 3 channels", "authoritative map array", "greenp","teal"),
          ("D · RESNET INPUT", "resize 256 → crop 224\nImageNet normalize", "detector domain", "goldp","gold")]
    for i,(head,body,desc,fill,acc) in enumerate(cols):
        x=55+i*224; arrow_box(p,x,190,183,125,head,body+"\n"+desc,fill,acc)
        if i<3:p.arrow(x+186,252,x+216,252,"muted",1.1)
    p.rect(73, 356, 815, 80, "redp", stroke="red", radius=11)
    p.text(92, 365, 778, 59,
           "Gallery PNG is display-only: [0,2] → uint8 with gamma 0.5, RGB channels preserved. Do not feed that PNG back as authoritative data or infer source-map encoding from an extension.",
           12.2, "red", bold=True, align="center", valign="mid")
    p.note()

    # 19 ResNet architecture
    p = d.page("09 · Detector / ResNet-50", "ResNet-50 nén bản đồ DIRE thành một logit nhị phân",
        "docs/dire-tu-so-0.pdf §3.3 · ResNet architecture walkthrough",
        notes_for("docs/dire-tu-so-0.pdf pp. 23–25; project detector code"))
    heading_sub(p, "Input detector: RGB DIRE map đã được chuẩn hóa; không phải ảnh RGB thô.")
    stages=[("Input", "224²×3"),("Conv7 / s2", "112²×64"),("MaxPool", "56²×64"),
            ("Layer 1", "56²×256\n3 blocks"),("Layer 2", "28²×512\n4 blocks"),
            ("Layer 3", "14²×1024\n6 blocks"),("Layer 4", "7²×2048\n3 blocks"),
            ("AvgPool", "FC 2048 → 1")]
    start=36; widths=[78,100,90,110,110,110,110,100]
    x=start
    for i,(name,dim) in enumerate(stages):
        w=widths[i]; fill="navy" if i in (0,7) else "bluep" if i%2 else "greenp"
        arrow_box(p,x,229,w,106,name,dim,fill,"white" if fill=="navy" else "teal")
        x+=w+12
        if i<7:p.arrow(x-12,282,x-3,282,"muted",1.0)
    p.text(104, 365, 755, 54,
           "Residual bottleneck counts [3, 4, 6, 3]; final pooled feature vector has 2,048 components. The DIRE checkpoint convention is fc: 2,048 → 1 logit.",
           13, "ink", bold=True, align="center")
    p.badge(349, 445, 260, "50-LAYER RESIDUAL NETWORK", "bluep", "teal")
    p.note()

    # 20 bottleneck
    p = d.page("09 · ResNet-50 block", "Bottleneck block giảm–xử lý–mở rộng kênh và cộng nhánh tắt",
        "docs/dire-tu-so-0.pdf §3.3 · ResNet-50 code walkthrough",
        notes_for("docs/dire-tu-so-0.pdf pp. 23–24; ResNet bottleneck architecture"))
    flow=[("Input", "C channels", "bluep"),("1×1 Conv", "reduce → P", "grayp"),
          ("3×3 Conv", "spatial mix", "teal"),("1×1 Conv", "expand → 4P", "grayp"),
          ("Add + ReLU", "output", "greenp")]
    for i,(t,b,fill) in enumerate(flow):
        x=66+i*174; arrow_box(p,x,230,143,100,t,b,fill,"teal" if fill!="teal" else "white")
        if i<4:p.arrow(x+145,280,x+168,280,"muted",1.1)
    p.arrow(105, 202, 770, 202, "gold", 1.6); p.arrow(770, 202, 770, 225, "gold", 1.6)
    p.text(333, 175, 240, 20, "identity / projection shortcut", 10, "gold", bold=True, align="center")
    show_equation(p, r"y=\mathrm{ReLU}\!\left(F(x)+W_s x\right)", 228, 369, 504, 53,
                  "Projection W_s is used when stride or channel count changes.", 21)
    p.text(125, 445, 710, 24, "Residual path supports gradient flow; it does not itself guarantee generalization.", 11.4, "muted", align="center")
    p.note()

    # 21 one logit BCE
    p = d.page("09 · Detector objective", "DIRE ResNet emits one logit; sigmoid converts it to P(fake)",
        "docs/dire-tu-so-0.pdf §2.2 · Table 4 train.json",
        notes_for("docs/dire-tu-so-0.pdf pp. 20–21; Table 4 train ledger fields"))
    show_equation(p, r"z=f_{\theta}(\mathrm{DIRE}(x)),\qquad P(y=1\mid x)=\sigma(z)=\frac{1}{1+e^{-z}}",
                  88, 145, 784, 61, "Label convention: y=1 denotes generated/fake in the binary detector.", 21)
    show_equation(p, r"\mathcal{L}_{\mathrm{BCE}}=-\left[y\log\sigma(z)+(1-y)\log\left(1-\sigma(z)\right)\right]",
                  102, 242, 756, 64, "Training uses BCEWithLogitsLoss for numerical stability.", 20)
    p.card(82, 356, 360, 82, "Why one logit matters", "The published head shape is [1, 2048]; a 2-logit softmax head is not interchangeable.", "gold", "goldp", 14, 11.2)
    p.card(482, 356, 394, 82, "Why ImageNet initialization", "Transfer learning initializes visual features; pilot protocol records ImageNet init and an independent detector per S.", "teal", "bluep", 14, 11.2)
    p.note()

    # 22 evaluation metrics
    p = d.page("10 · Đánh giá", "ACC chọn một ngưỡng; AP đánh giá thứ hạng trên toàn dải ngưỡng",
        "docs/dire-tu-so-0.pdf §2.5 · score/eval source",
        notes_for("docs/dire-tu-so-0.pdf p. 22; official paper reports ACC and AP"))
    metrics=[("ACC", r"\frac{1}{N}\sum_i\mathbf{1}\!\left[(s_i>\tau)=y_i\right]", "Depends on a fixed threshold τ."),
             ("AP", r"\sum_k (R_k-R_{k-1})P_k", "Area / discrete sum along precision–recall; threshold-free ranking summary."),
             ("AUC", r"\int_0^1 \mathrm{TPR}(u)\,du", "Area under ROC; compare with caution under domain shift.")]
    for i,(name,tex,desc) in enumerate(metrics):
        x=55+i*290; p.rect(x,168,265,200,"paper",stroke="line",radius=12)
        p.text(x+16,182,232,24,name,17,"teal",bold=True,align="center")
        p.formula(tex,x+15,224,235,48,size=18)
        p.text(x+18,282,228,63,desc,11,"muted",align="center",valign="mid")
    p.rect(103, 402, 754, 54, "goldp", stroke="gold", radius=10)
    p.text(120, 412, 720, 34, "Protocol guard: choose epoch and threshold on validation; test is for final evaluation only.", 13, "ink", bold=True, align="center", valign="mid")
    p.note()

    # 23 provenance levels
    p = d.page("11 · Provenance", "R1+R2 means self-computed maps and self-trained detector — not self-generated images",
        "docs/adr-0006-reproduction-levels.md · report.json provenance fields",
        notes_for("docs/adr-0006-reproduction-levels.md; modal-runs/runs/report.json data.note"))
    labels=[("R0", "Re-score", "author map + author detector", "grayp", "muted"),
            ("R1", "Recompute map", "self-computed DIRE", "bluep", "teal"),
            ("R2", "Retrain detector", "self-trained classifier", "greenp", "teal"),
            ("R3", "Regenerate source images", "own generator / data", "goldp", "gold")]
    for i,(tag,head,body,fill,col) in enumerate(labels):
        x=56+i*222; p.rect(x,185,197,145,fill,stroke="line",radius=12)
        p.badge(x+15,199,61,tag,fill,col,10)
        p.text(x+16,235,166,24,head,10.5,"ink",bold=True)
        p.text(x+16,267,167,50,body,11,"muted")
    p.rect(124, 369, 714, 68, "bluep", stroke="teal", radius=11)
    p.text(145, 381, 672, 44,
           "Repo report label: R1+R2. The diffusion checkpoint and source dataset are not project-trained; fake images are marked author-sourced, so do not claim R3.",
           12.5, "ink", bold=True, align="center", valign="mid")
    p.note()

    # 24 Paper Table2
    p = d.page("12 · Paper evidence", "Paper Table 2 reports strong cross-generator results for four DIRE detectors",
        "Wang et al. (2023), Table 2 · src/dire_modal/paper_table2.tsv",
        notes_for("Wang et al., ICCV 2023, Table 2; source TSV has 4 detectors × 8 generators plus published averages"))
    heading_sub(p, "Every cell is shown as ACC / AP (%), reported by the authors. These are not measurements from the project.")
    rows=[]
    for det in PAPER_DETECTORS:
        row=[PAPER_DET_LABELS[det]]
        for gen in PAPER_GENS:
            a,ap=PAPER_MAP[(det,gen)]; row.append(f"{a:.1f} / {ap:.1f}")
        avg=next(r for r in PAPER_ROWS if r["ckpt"]==det and r["generator"]=="avg")
        row.append(f"{float(avg['acc']):.1f} / {float(avg['ap']):.1f}")
        rows.append(row)
    widths=[104]+[88]*8+[54]
    p.table(48, 157, 864, ["Train detector"]+[PAPER_LABELS[g] for g in PAPER_GENS]+["Avg"], rows,
            widths, 50, 9.3, header_size=9.1)
    p.badge(318, 430, 320, "PAPER-REPORTED · ICCV 2023 TABLE 2", "bluep", "teal")
    p.text(132, 468, 695, 20, "Table 2 headline is specific to its dataset, split, reference weights and evaluation protocol.", 9.6, "muted", align="center")
    p.note()

    # 25 paper highlights
    p = d.page("12 · Paper evidence", "Author-reported mean ACC ranges from 97.9% to 99.9% across training generators",
        "Wang et al. (2023), Table 2 · docs/dire-tu-so-0.pdf p. 39",
        notes_for("Wang et al., ICCV 2023, Table 2; all averages below are explicitly paper-reported"))
    means=[]
    for det in PAPER_DETECTORS:
        avg=next(r for r in PAPER_ROWS if r["ckpt"]==det and r["generator"]=="avg")
        means.append((PAPER_DET_LABELS[det],float(avg["acc"]),float(avg["ap"])))
    for i,(name,acc,ap) in enumerate(means):
        y=159+i*63
        p.text(87,y+3,115,25,name,14,"ink",bold=True)
        p.rect(215,y,490,27,"grayp",radius=7)
        p.rect(215,y,490*acc/100,27,"teal",radius=7)
        p.text(719,y+1,158,22,f"ACC {acc:.1f}%  ·  AP {ap:.1f}%",11.6,"ink",bold=True)
    p.rect(92, 431, 784, 52, "goldp", stroke="gold", radius=10)
    p.text(108, 440, 753, 34, "n≈1,000 per test cell; paper provides no CIs. Differences of a few tenths should not be overstated.", 11.4, "ink", bold=True, align="center", valign="mid")
    p.note()

    # 26 Table3 cross dataset
    p = d.page("12 · Paper evidence", "Changing dataset domain can be harder than changing generator family",
        "Wang et al. (2023), Table 3 · docs/dire-tu-so-0.pdf p. 39",
        notes_for("Wang et al., ICCV 2023, Table 3; cells ACC/AP (%) reported in source PDF"))
    tabrows=[["LSUN ADM", "90.2 / 97.9", "97.2 / 99.8", "99.9 / 100"],
             ["LSUN iDDPM", "90.2 / 97.9", "93.7 / 99.3", "99.9 / 100"],
             ["LSUN StyleGAN", "76.9 / 94.4", "89.7 / 99.0", "100 / 100"]]
    p.table(120, 177, 720, ["Train source", "ImageNet ADM", "ImageNet SD-v1", "LSUN StyleGAN"], tabrows,
            [170,183,183,184], 57, 11)
    p.rect(173, 407, 614, 59, "redp", stroke="red", radius=11)
    p.text(193, 418, 574, 37, "Lowest shown cell: ACC 76.9% · cross-domain generalization is a real limitation.", 14, "red", bold=True, align="center", valign="mid")
    p.note()

    # 27 Paper Table4 reference
    p = d.page("12 · Paper evidence", "Paper Table 4 shows an S-dependent trade-off; do not substitute it for pilot data",
        "Wang et al. (2023), Table 4 · docs/dire-tu-so-0.pdf p. 36",
        notes_for("Wang et al., ICCV 2023, Table 4; the paper uses SD-v1 whereas current pilot tests SD-v2"))
    paper_t4=[
        ["S=5", "100 / 100", "100 / 100", "100 / 100", "97.5 / 100", "87.5 / 99.8"],
        ["S=10", "100 / 100", "100 / 100", "100 / 100", "99.4 / 100", "98.2 / 100"],
        ["S=20", "100 / 100", "100 / 100", "100 / 100", "99.7 / 100", "99.7 / 100"],
        ["S=50", "100 / 100", "100 / 100", "100 / 100", "100 / 100", "99.9 / 100"],
    ]
    p.table(55, 155, 850, ["Steps", "ADM", "DDPM", "iDDPM", "PNDM", "SD-v1"], paper_t4,
            [106,148,148,148,148,152], 45, 10.8)
    p.rect(91, 391, 778, 67, "goldp", stroke="gold", radius=10)
    p.text(110, 402, 740, 45, "Paper protocol and full test data; S=20→50 multiplies NFE by 2.5. These rows are paper-reported, not the project's Table 4 ledger.", 12, "ink", bold=True, align="center", valign="mid")
    p.note()

    # 28 Paper Table5
    p = d.page("12 · Paper evidence", "Paper Table 5 supports DIRE over RGB or reconstruction alone",
        "Wang et al. (2023), Table 5 · docs/dire-tu-so-0.pdf p. 36",
        notes_for("Wang et al., ICCV 2023, Table 5; values are paper-reported ACC/AP percentages"))
    paper_t5=[
        ["REC", "57.1 / 57.7", "49.7 / 92.6", "87.1 / 98.7", "46.9 / 57.0"],
        ["RGB", "87.3 / 99.6", "100 / 100", "77.8 / 99.1", "77.4 / 85.8"],
        ["RGB + DIRE", "99.8 / 100", "99.9 / 100", "99.2 / 100", "62.4 / 92.4"],
        ["DIRE", "100 / 100", "100 / 100", "99.7 / 100", "99.7 / 100"],
    ]
    p.table(75, 165, 810, ["Representation", "DDPM", "iDDPM", "PNDM", "SD-v1"], paper_t5,
            [190,155,155,155,155], 49, 10.8)
    p.rect(122, 407, 716, 51, "greenp", stroke="teal", radius=10)
    p.text(140, 416, 680, 32, "RGB + DIRE can underperform DIRE alone: extra features may invite shortcut learning.", 12.4, "teal", bold=True, align="center", valign="mid")
    p.note()

    # 29 Figure4 paper robustness
    p = d.page("12 · Paper evidence", "Paper Figure 4 reports qualitative robustness to blur and JPEG compression",
        "Wang et al. (2023), Figure 4 · docs/dire-tu-so-0.pdf pp. 39–40",
        notes_for("Wang et al., ICCV 2023, Figure 4; PDF does not provide the underlying numeric table"))
    conditions=[("Clean", "reference"),("Blur σ=1", "mild"),("Blur σ=2", "medium"),("Blur σ=3", "strong"),("JPEG q=65", "compressed"),("JPEG q=30", "heavy compression")]
    for i,(a,b) in enumerate(conditions):
        x=53+i*148; arrow_box(p,x,190,130,95,a,b,"bluep" if i==0 else "grayp","teal")
    p.rect(93, 329, 773, 101, "goldp", stroke="gold", radius=12)
    p.text(113, 343, 732, 71,
           "Paper's qualitative claim: DIRE changes little; RGB baselines degrade more. The plot's pixel values are not tabulated in the cited paper copy, so this deck does not invent exact AP points.",
           13, "ink", bold=True, align="center", valign="mid")
    p.note()

    # 30 AEROBLADE
    p = d.page("13 · Họ hàng phương pháp", "AEROBLADE shares the reconstruction-error intuition but uses an autoencoder",
        "Ricker et al. (CVPR 2024) · docs/dire-tu-so-0.pdf §9.8",
        notes_for("Ricker et al., AEROBLADE, CVPR 2024; docs/dire-tu-so-0.pdf pp. 43–44"))
    p.table(107, 176, 746, ["Property", "DIRE", "AEROBLADE"], [
        ["Transform", "ADM diffusion · invert + reconstruct", "LDM autoencoder · encode + decode"],
        ["Domain / score", "Pixel residual |x−x̂|", "Latent reconstruction · LPIPS"],
        ["Decision", "ResNet-50 trained on DIRE maps", "Threshold / training-free"],
        ["Compute", "2S diffusion network evaluations", "One AE reconstruction path"],
        ["Applicability", "Broad generated-image benchmark", "Latent-diffusion images"],
    ], [151,300,295], 48, 10.3)
    p.text(139, 457, 685, 25, "Related work, not a result or feature implemented by this project's prototype.", 11.2, "muted", italic=True, align="center")
    p.note()

    # 32 R1+R2 report
    p = d.page("15 · Project measurements", "R1+R2 report records a broader run — but bookkeeping counters need reconciliation",
        "modal-runs/runs/report.json · runs/eval/gen-adm/eval.json",
        notes_for("modal-runs/runs/report.json; its data.created field is future-dated relative to session 2026-10-14; treat as repository-recorded only"))
    report_cards=[("Map inventory", "5,900", "maps.total/by_split_generator"),
                  ("DDIM setting", "S=20", "steps_mixed=[20]"),
                  ("NFE in report", "234,760", "reported field"),
                  ("Eval scope", "ADM / test", "held-out evaluation")]
    for i,(a,b,c) in enumerate(report_cards):
        label_value(p,52+i*221,158,198,a,b,"gold" if i==3 else "teal",18 if i==3 else 21,"paper")
        p.text(63+i*221, 246, 176, 26, c, 9.6, "muted", align="center")
    p.rect(80, 299, 800, 123, "goldp", stroke="gold", radius=12)
    p.text(99, 311, 761, 92,
           "AUDIT FLAG: report.dire.n = 5,900; n_new = 5,869; n_reused = 2,431. The latter two sum to 8,300, not 5,900. NFE=234,760 equals 5,869×40, not 5,900×40. Keep these fields separate until their definitions / denominator are reconciled.",
           12.2, "ink", bold=True, align="center", valign="mid")
    p.text(104, 438, 755, 35, "Do not recompute, combine or call these counters a reconciled total.", 12, "red", bold=True, align="center")
    p.note()

    # 33 report metrics across generators
    p = d.page("15 · Project measurements", "The R1+R2 artifact records heterogeneous scores across generator-trained runs",
        "modal-runs/runs/report.json:train_by_generator · project warnings",
        notes_for("modal-runs/runs/report.json; artifact identifies R1+R2 and lists warnings on sample counts / comparability"))
    names=[k.removeprefix("gen-") for k in REPORT.get("train_by_generator",{})]
    rows=[]
    for name in names:
        m=REPORT["train_by_generator"]["gen-"+name]
        rows.append([name, f"{m.get('acc',0)*100:.1f}%", f"{m.get('ap',0)*100:.1f}%", f"{m.get('auc',0)*100:.1f}%", str(m.get("n","—")), str(m.get("best_epoch","—"))])
    p.table(73, 150, 814, ["Run tag", "ACC", "AP", "AUC", "n", "Best epoch"], rows,
            [202,122,122,122,106,140], 31, 9.8)
    p.rect(91, 465, 778, 31, "goldp", stroke="gold", radius=8)
    p.text(104, 470, 750, 21, "Repo metrics, not paper Table 2; fake cell n=700 and report warnings limit direct interpretation.", 9.8, "ink", bold=True, align="center", valign="mid")
    p.note()

    # 34 heldout eval
    p = d.page("15 · Project measurements", "ADM held-out eval reports AP=0.911 on the self-computed-map path",
        "modal-runs/runs/eval/gen-adm/eval.json · threshold chosen on val",
        notes_for("modal-runs/runs/eval/gen-adm/eval.json; the artifact itself states this is not comparable to paper Table 2"))
    m=EVAL.get("metrics",{})
    for i,(label,val) in enumerate([("Test n",str(m.get("n","—"))),
                                    ("ACC · val threshold",f"{m.get('acc',0)*100:.1f}%"),
                                    ("AP",f"{m.get('ap',0)*100:.1f}%"),
                                    ("AUC",f"{m.get('auc',0)*100:.1f}%")]):
        label_value(p,58+i*214,158,190,label,val,"teal",22)
    p.rect(75, 277, 810, 111, "redp", stroke="red", radius=12)
    p.text(94, 291, 770, 81,
           "AUDIT FLAG: eval.json lists ACC=78.1% at threshold 0.50029 (chosen on val), but `acc_ci95=[0.47,0.529]` is centered near 50%, inconsistent with that ACC. Do not publish this CI until corrected. At threshold 0.5, ACC is separately 50.0%.",
           12.4, "red", bold=True, align="center", valign="mid")
    p.text(123, 410, 706, 50,
           "This is a project evaluation on self-computed DIRE maps with a from-scratch detector; it is not an absolute Table 2 comparison.",
           12, "ink", bold=True, align="center")
    p.note()

    # 35 · Confusion matrix: counts, not only a single accuracy percentage.
    cm = EVAL.get("metrics", {}).get("confusion", {})
    p = d.page("15 · Project measurements", "Ma trận nhầm lẫn cho thấy số mẫu đúng và sai ở từng lớp",
        "modal-runs/runs/eval/gen-adm/eval.json · test split · threshold chọn trên validation",
        notes_for("modal-runs/runs/eval/gen-adm/eval.json; confusion counts are shown as recorded, with positive class = generated/fake"))
    p.text(111, 155, 710, 25, "Dự đoán của detector", 14, "muted", bold=True, align="center")
    p.text(363, 182, 210, 22, "Dự đoán: ảnh thật", 11, "teal", bold=True, align="center")
    p.text(583, 182, 210, 22, "Dự đoán: ảnh sinh", 11, "red", bold=True, align="center")
    p.text(58, 250, 190, 24, "Thực tế: ảnh thật", 12, "ink", bold=True, align="center", valign="mid")
    p.text(58, 362, 190, 24, "Thực tế: ảnh sinh", 12, "ink", bold=True, align="center", valign="mid")
    matrix_cells = [
        (270, 211, cm.get("tn", 0), "TN · thật → thật", "greenp", "teal"),
        (490, 211, cm.get("fp", 0), "FP · thật → sinh", "redp", "red"),
        (270, 323, cm.get("fn", 0), "FN · sinh → thật", "redp", "red"),
        (490, 323, cm.get("tp", 0), "TP · sinh → sinh", "greenp", "teal"),
    ]
    for x, y, count, label, fill, accent in matrix_cells:
        p.rect(x, y, 210, 100, fill, stroke=accent, radius=12)
        p.text(x+8, y+12, 194, 48, f"{count:,}".replace(",", "."), 28, accent, bold=True, align="center", valign="mid")
        p.text(x+8, y+66, 194, 20, label, 10.4, "ink", bold=True, align="center", valign="mid")
    n = int(EVAL.get("metrics", {}).get("n", sum(int(v) for v in cm.values())))
    correct = int(cm.get("tn", 0)) + int(cm.get("tp", 0))
    p.rect(129, 450, 702, 43, "bluep", stroke="line", radius=9)
    n_label = format(n, ",").replace(",", ".")
    correct_label = format(correct, ",").replace(",", ".")
    acc_label = f"{100*correct/max(n,1):.1f}".replace(".", ",")
    p.text(143, 457, 674, 29,
           f"Đúng {correct_label}/{n_label} mẫu · {acc_label}%  |  Ngưỡng {EVAL.get('threshold_prob', 0.5):.5f} chọn trên validation · lớp dương = ảnh sinh",
           10.6, "ink", bold=True, align="center", valign="mid")
    p.note()

    # 36–37 · Multi-generator examples. Never invent reconstructions from residual maps.
    def generator_examples_slide(section: str, title: str, generators: list[str]):
        p = d.page(section, title,
            "modal-runs/gallery · PNG hiển thị đã lưu cùng metadata nguồn",
            notes_for("modal-runs/gallery; shown originals and maps are committed display PNGs; no reconstruction images or source NPY files are present in this checkout"))
        columns = [(160, "ẢNH GỐC"), (390, "TÁI TẠO  x̂₀"), (620, "DIRE MAP")]
        for x, label in columns:
            p.text(x, 157, 200, 18, label, 9.5, "muted", bold=True, align="center")
        for row, generator in enumerate(generators):
            y = 194 + row*143
            pair = gallery_pair_for(generator)
            generator_label = {"ddpm": "DDPM", "iddpm": "iDDPM", "dalle2": "DALL·E 2",
                               "diff-stylegan": "Diffusion StyleGAN"}.get(generator, generator)
            p.text(42, y+45, 106, 30, generator_label,
                   8.4, "teal", bold=True, align="center", valign="mid")
            if pair:
                p.rect(160, y, 200, 118, "paper", stroke="line", radius=8)
                p.image(pair[0], 166, y+5, 188, 108, f"Ảnh gốc · {generator}")
                p.rect(620, y, 200, 118, "paper", stroke="line", radius=8)
                p.image(pair[1], 626, y+5, 188, 108, f"DIRE map hiển thị · {generator}")
            else:
                for x in (160, 620):
                    p.rect(x, y, 200, 118, "grayp", stroke="line", radius=8)
                    p.text(x+8, y+37, 184, 42, "Chưa có ảnh trong gallery", 10, "muted", align="center", valign="mid")
            p.rect(390, y, 200, 118, "grayp", stroke="gold", radius=8)
            p.text(402, y+24, 176, 67,
                   "Chưa có ảnh tái tạo x̂₀\n(checkpoint / output chưa có trong checkout)",
                   9.5, "muted", align="center", valign="mid")
        p.rect(101, 468, 758, 29, "goldp", stroke="gold", radius=8)
        p.text(111, 473, 738, 19,
               "Không suy dựng reconstruction từ DIRE map: map chỉ lưu |x − x̂₀|, không chứa dấu của sai số.",
               9.5, "ink", bold=True, align="center", valign="mid")
        p.note()

    generator_examples_slide("16 · Ví dụ đa bộ sinh", "Ảnh gốc và DIRE map từ DDPM, iDDPM",
                             ["ddpm", "iddpm"])
    generator_examples_slide("16 · Ví dụ đa bộ sinh", "Ảnh gốc và DIRE map từ DALL·E 2, StyleGAN khuếch tán",
                             ["dalle2", "diff-stylegan"])

    # 38 Table4 design
    p = d.page("17 · Table 4 pilot", "Pilot protocol shares a fixed 200-per-cell subset across S settings",
        "Table 4 ledger · _subset_protocol.json · sampling protocol",
        notes_for("modal-runs/experiments/experiments/table4/table4-lsun-pilot; local repository artifact; live Volume not checked"))
    p.table(85, 154, 790, ["Protocol field", "Repo-recorded value"], [
        ["Subset / seed", "lsun-pilot · 3407"],
        ["Sampling rule", "deterministic per-cell ranking"],
        ["Images / cell", "200 maximum · 10 cells · up to 2,000 rows"],
        ["Train generator", "ADM"],
        ["Test generators", "ADM, DDPM, iDDPM, PNDM, SD-v2"],
        ["Detector initialization", "ImageNet ResNet-50 · seed 3407"],
        ["Training actually recorded", "batch 32 · lr 1e−4 · crop 224 · early stopping"],
    ], [240,550], 37, 10.2)
    p.note()

    # 38 pilot score matrix as compact multiples: only measured S settings.
    p = d.page("17 · Table 4 pilot", "Biểu đồ tách AP và ACC cho ba mức S đã đo",
        "table4.json: per-generator test metrics · ACC threshold=0.5",
        notes_for("Values are repository pilot metrics, percentages; AP threshold-independent, ACC at 0.5. Only measured settings S=5/10/20 are plotted."))
    gens=["adm","ddpm","iddpm","pndm","sdv2"]
    colors={"ap":"teal","acc":"purple"}
    for j,sval in enumerate([5,10,20]):
        x0=54+j*292; w=270
        p.rect(x0,157,w,292,"paper",stroke="line",radius=10)
        p.text(x0+8,168,w-16,22,f"S = {sval}",14,"ink",bold=True,align="center")
        p.line(x0+83,198,x0+83,411,"line",0.6)
        for tick in (85,90,95,100):
            xx=x0+83+(tick-85)/15*112
            p.line(xx,197,xx,410,"line",0.45)
            p.text(xx-11,184,24,12,str(tick),7.5,"muted",align="center")
        for i,gen in enumerate(gens):
            m=s_metrics(sval,gen)
            yy=208+i*39
            p.text(x0+7,yy+5,70,18,gen.upper(),8.3,"ink",bold=True)
            for metric,k,col in (("ap",0,"teal"),("acc",1,"purple")):
                val=100*m[metric]
                bar=max(1,(val-85)/15*112)
                by=yy+k*12
                p.rect(x0+83,by,bar,7,col,radius=3)
            p.text(x0+200,yy-1,65,13,f"{100*m['ap']:.1f} / {100*m['acc']:.1f}",7.6,"ink",align="right")
        p.rect(x0+78,420,12,7,"teal",radius=3)
        p.text(x0+94,416,38,14,"AP",8.1,"muted")
        p.rect(x0+142,420,12,7,"purple",radius=3)
        p.text(x0+158,416,52,14,"ACC (%)",8.1,"muted")
    p.rect(177,463,606,28,"bluep",stroke="line",radius=8)
    p.text(190,469,580,17,"Thang trục 85–100% · AP không phụ thuộc ngưỡng · ACC dùng ngưỡng 0,5",9.4,"ink",bold=True,align="center")
    p.note()

    # 39 compare paper and pilot Table4
    p = d.page("17 · Paper vs pilot", "Paper Table 4 and project Table 4 are not directly comparable experiments",
        "Paper Table 4 · repo Table 4 ledger · different split size, generator and training protocol",
        notes_for("Wang et al. (2023), Table 4; repo ledger. Paper uses SD-v1; pilot uses SD-v2, 200/cell, 32 batch and early-stopped ImageNet initialization."))
    vals=[]
    for gen in ["adm","pndm"]:
        r=[gen.upper()]
        for sval in [5,10,20]:
            a,ap=PAPER_MAP[("lsun_adm",gen)] if sval==20 else (None,None)
            # Paper's S-sweep values come from paper Table 4 values recorded in source PDF.
            pap={(5,"adm"):(100,100),(10,"adm"):(100,100),(20,"adm"):(100,100),
                 (5,"pndm"):(97.5,100),(10,"pndm"):(99.4,100),(20,"pndm"):(99.7,100)}[(sval,gen)]
            cur=s_metrics(sval,gen)
            r.append(f"Paper {pap[0]:.1f}/{pap[1]:.1f}\nRepo {cur['acc']*100:.1f}/{cur['ap']*100:.1f}")
        vals.append(r)
    p.table(94, 178, 772, ["Generator", "S=5 · ACC/AP", "S=10 · ACC/AP", "S=20 · ACC/AP"], vals,
            [142,210,210,210], 73, 11)
    p.rect(110, 410, 740, 58, "redp", stroke="red", radius=10)
    p.text(127, 420, 706, 38, "Different: SD-v1 vs SD-v2; paper full data vs pilot; paper vs 200-per-cell; documented training protocol differs. No delta = reproduction claim.", 11, "red", bold=True, align="center", valign="mid")
    p.note()

    # 40 Table5 state
    p = d.page("18 · Table 5", "Table 5 inputs are staged; this checkout contains no project result ledger",
        "modal-runs/experiments/experiments/table5/table5-lsun-pilot/inputs/_protocol.json",
        notes_for("Table 5 input protocol is present; no table5.json ledger is checked in"))
    reps=[["REC", "reconstruction image"],["RGB", "raw image baseline"],["RGB_DIRE", "six-channel paired input"],["DIRE", "three-channel residual map"]]
    for i,(a,b) in enumerate(reps):
        x=69+i*217; arrow_box(p,x,177,184,100,a,b,"bluep" if i<3 else "greenp","teal")
    p.rect(113, 321, 734, 110, "goldp", stroke="gold", radius=12)
    p.text(135, 335, 688, 81,
           "Repo protocol: S=20 · 40 NFE/image · 256×256 · float16 inputs. Inputs materialized; no Table 5 training/evaluation ledger found. REC needs reconstruction data, not residual .npy alone.",
           13, "ink", bold=True, align="center", valign="mid")
    p.badge(331, 454, 296, "STATUS · INPUTS ONLY · NO METRICS", "goldp", "gold")
    p.note()

    # 41 Table5 paper results and comparison
    p = d.page("18 · Paper vs project Table 5", "The paper's representation ablation is available; this project's Table 5 remains unmeasured",
        "Wang et al. (2023), Table 5 · current repo has protocol only",
        notes_for("Paper values from docs/dire-tu-so-0.pdf p.36; current repo Table 5 protocol lists SD-v2 test data, no output ledger"))
    p.table(74, 163, 812, ["Representation", "Paper SD-v1 ACC/AP", "Pilot status"], [
        ["REC", "46.9 / 57.0", "not measured"],
        ["RGB", "77.4 / 85.8", "not measured"],
        ["RGB_DIRE", "62.4 / 92.4", "not measured"],
        ["DIRE", "99.7 / 100.0", "not measured"],
    ], [200,252,360], 49, 11,
        cell_colors=[["ink","teal","gold"] for _ in range(4)])
    p.text(121, 432, 717, 44, "The source protocol uses SD-v2; that differs from this paper SD-v1 table. These columns are deliberately not combined.", 12, "red", bold=True, align="center")
    p.note()

    # 42 Figure4 current
    p = d.page("19 · Figure 4", "The project has a robustness protocol, but no Figure 4 results ledger",
        "docs/table4-table5-figure4-runbook.md · repository artifact inventory",
        notes_for("Figure 4 runbook; current checkout contains no experiments/figure4 ledger"))
    conditions=[("clean", "baseline"),("blur σ=1", "Gaussian"),("blur σ=2", "Gaussian"),
                ("blur σ=3", "Gaussian"),("JPEG q=65", "compression"),("JPEG q=30", "compression")]
    for i,(a,b) in enumerate(conditions):
        x=52+i*149; arrow_box(p,x,177,133,87,a,b,"grayp" if i else "bluep","teal")
    p.arrow(85, 306, 872, 306, "muted", 1.4)
    p.text(181, 286, 594, 18, "each condition × generator → measured AP shard → result ledger", 10.3, "muted", align="center")
    p.rect(105, 352, 750, 91, "goldp", stroke="gold", radius=12)
    p.text(126, 365, 708, 64,
           "Planned matrix: 30 condition×generator shards (6×5), each comparing 2 project-trained detectors = 60 model-condition metrics. Current output: none found. Runner is not the paper's CNNDetection/PatchForensics baseline reproduction.",
           12.1, "ink", bold=True, align="center", valign="mid")
    p.badge(350, 462, 261, "NOT RUN / NO LEDGER", "goldp", "gold")
    p.note()

    # 43 subset architecture
    p = d.page("20 · Subset / resumption", "One manifest locks sample identity across Table 4, Table 5 and Figure 4",
        "docs/incremental-paper-subsets.md · repo subset protocol manifest",
        notes_for("docs/incremental-paper-subsets.md; subset manifest SHA from Table 4 ledger"))
    p.rect(74, 167, 312, 235, "bluep", stroke="teal", radius=13)
    p.text(99, 183, 262, 25, "SNAPSHOT 200 / CELL", 14, "teal", bold=True, align="center")
    for i in range(4):
        for j in range(4): p.circle(131+j*55,241+i*30,7,"teal" if (i+j)%3 else "mint")
    p.text(108, 374, 242, 19, "up to 2,000 sample rows", 11,"ink",bold=True,align="center")
    p.rect(574, 143, 312, 259, "greenp", stroke="teal", radius=13)
    p.text(599, 158, 262, 25, "NESTED SNAPSHOT 400 / CELL", 13, "teal", bold=True, align="center")
    for i in range(5):
        for j in range(5): p.circle(628+j*42,220+i*27,5.7,"mint" if i<4 and j<4 else "teal")
    p.text(607, 367, 246, 19, "up to 4,000 · contains 200 stage", 10.5,"ink",bold=True,align="center")
    p.arrow(402, 283, 555, 283, "gold", 2)
    p.text(404, 253, 149, 20, "increase cap, keep IDs", 9.4,"gold",bold=True,align="center")
    p.text(104, 426, 747, 38, "Chỉ cập nhật manifest: ảnh nguồn không đổi; seed 3407 · chọn mẫu xác định theo từng ô · số mẫu trong sổ.", 11.7, "ink", bold=True, align="center")
    p.note()

    # 44 experiment matrix
    p = d.page("20 · Ba nghiên cứu", "Ba experiment families share the subset identity but vary different factors",
        "docs/slide-agent-brief.md · experiment runbook and current artifact statuses",
        notes_for("docs/slide-agent-brief.md; docs/table4-table5-figure4-runbook.md; checked-in ledgers"))
    exp=[("Table 4", "Số bước S", "5 · 10 · 20", "3 mức S đã có số đo", "teal"),
         ("Table 5", "Input representation", "REC · RGB · RGB_DIRE · DIRE", "inputs only; no results", "gold"),
         ("Figure 4", "Corruption robustness", "clean · blur · JPEG", "no result ledger", "red")]
    for i,(name,axis,vals,status,accent) in enumerate(exp):
        x=56+i*293; p.rect(x,177,264,237,"paper",stroke="line",radius=12)
        p.text(x+14,192,236,23,name,17,accent,bold=True,align="center")
        p.text(x+17,228,230,20,axis,11,"muted",bold=True,align="center")
        p.text(x+16,267,232,58,vals,13,"ink",bold=True,align="center",valign="mid")
        p.rect(x+17,351,230,37,"goldp" if accent!="red" else "redp",radius=8)
        p.text(x+23,358,218,23,status,10,accent,bold=True,align="center",valign="mid")
    p.note()

    # 47 generator taxonomy project
    p = d.page("22 · Dataset & generators", "Paper benchmark and repository extensions must remain separate",
        "Wang et al. (2023), Table 1 · paper-cac-mo-hinh-sinh.html §§09–10 · report.json",
        notes_for("Paper Table 1 and model-family notes; report JSON includes extra project generator tags"))
    p.rect(58, 157, 402, 260, "bluep", stroke="teal", radius=11)
    p.text(79, 171, 360, 25, "PAPER BENCHMARK", 13, "teal", bold=True)
    p.text(80, 210, 356, 150,
           "LSUN-Bedroom: ADM · DDPM · iDDPM · PNDM · LDM · SD-v1 · SD-v2 · VQ-Diffusion\n\nImageNet: ADM · SD-v1\n\nTrain / test roles and prompt conditions are specified in the paper.",
           12, "ink")
    p.rect(500, 157, 402, 260, "goldp", stroke="gold", radius=11)
    p.text(521, 171, 360, 25, "REPO EXTENSIONS / UNCERTAINTY", 13, "gold", bold=True)
    p.text(522, 210, 356, 175,
           "Report inventory adds DALL·E 2, IF, projected GAN, hybrids and other tags.\n\nSome family assignments are uncertain or model-closed. Names such as diff-projectedgan are not evidence of architecture.",
           12, "ink")
    p.text(105, 444, 750, 31, "Do not merge out-of-paper generators into a paper mean or claim exact source architecture from a folder name.", 11.5, "red", bold=True, align="center")
    p.note()

    # 50 scientific limitations
    p = d.page("25 · Hạn chế & research questions", "High paper scores do not close the mechanism, domain and deployment questions",
        "docs/dire-tu-so-0.pdf §9 · docs/paper-cac-mo-hinh-sinh.html §§08,11",
        notes_for("docs/dire-tu-so-0.pdf pp. 41–44; paper-cac-mo-hinh-sinh.html separates interpretation from paper claim"))
    limits=[("Mechanism", "Euler error, clipping and data-manifold explanations are not isolated experimentally."),
            ("Domain shift", "Table 3 includes 76.9% ACC in a cross-domain cell."),
            ("Robustness", "Paper tests selected blur/JPEG conditions; many transformations remain untested."),
            ("Threat model", "No adaptive attacker or post-processing adversarial optimization."),
            ("Data breadth", "2023 generators/prompts do not cover current closed / transformer / flow models."),
            ("Uncertainty", "No multiple seeds / paper CIs for many comparisons; don't overread tenths of a point.")]
    for i,(head,body) in enumerate(limits):
        col=i%3; row=i//3; p.card(55+col*292, 159+row*137, 264, 111, head, body, "red" if col==0 else "gold", "paper", 13.4, 10.8)
    p.note()

    # 52 conclusion
    p = d.page("27 · Kết luận", "The strongest deliverable today is a transparent evidence system — not a full reproduction",
        "Summary of paper, repository artifacts, known caveats and pilot constraints",
        notes_for("Synthesis; sources are cited on each relevant slide and in appendix"), dark=True)
    p.rect(0, 0, PW, PH, "navy", radius=0)
    p.text(50, 127, 860, 42, "DIRE provides a reproducible representation hypothesis;" , 25, "white", bold=True, align="center")
    p.text(70, 180, 820, 55, "the repository now contains a pilot ledger and gallery, but evidence boundaries still matter.", 18.5, "mint", bold=True, align="center")
    for x, head, body, accent in [
        (93, "Theory", "A deterministic diffusion round trip yields a structured residual.", "mint"),
        (367, "Evidence", "Paper results and repo results must stay in separate lanes.", "gold"),
        (641, "Next", "Resolve dates, live state and metrics before more compute.", "white"),
    ]:
        p.rect(x, 280, 226, 116, "navy2", stroke="deepmuted", radius=10)
        p.text(x+14, 294, 198, 24, head, 14, accent, bold=True, align="center")
        p.text(x+18, 329, 190, 55, body, 11, "white", align="center", valign="mid")
    p.text(120, 443, 720, 32, "DIRE  =  | x − reconstruction(inversion(x)) |", 18, "white", bold=True, align="center")
    p.note()

    # ---------- appendices ----------
    # A1 paper table 2 full matrix as pure ACC/AP pairs (source-backed)
    p = d.page("Appendix A · Paper numbers", "Table 2 source values: complete 4 × 8 DIRE block (ACC / AP, %)",
        "src/dire_modal/paper_table2.tsv · Wang et al. (2023), Table 2",
        notes_for("The TSV stores all author-reported rows plus average; this appendix omits averages to keep generator-level values distinct."))
    r=[]
    for det in PAPER_DETECTORS:
        rr=[PAPER_DET_LABELS[det]]
        for gen in PAPER_GENS:
            a,ap=PAPER_MAP[(det,gen)];rr.append(f"{a:g}/{ap:g}")
        r.append(rr)
    p.table(43, 165, 874, ["Train"]+[PAPER_LABELS[g] for g in PAPER_GENS], r,
            [92]+[97.75]*8, 67, 10.7, header_size=9.1)
    p.text(126, 482, 708, 15, "Citation: Wang et al., ICCV 2023, Table 2. These figures are paper-reported, not repo measurements.", 8.8, "muted", align="center")
    p.note()

    # A2 raw Table 4 pilot AP/AUC/ACC complete 15 cells plus explicit missing
    p = d.page("Appendix B · Pilot data", "Complete Table 4 pilot metrics from the checked-in ledger",
        "table4.json · per-generator test cell metrics; threshold recorded as 0.5",
        notes_for("All values below are from the current repo JSON. They are repository-recorded values; timeline / live Volume remain unverified."))
    r=[]
    for sval in [5,10,20]:
        for gen in ["adm","ddpm","iddpm","pndm","sdv2"]:
            m=s_metrics(sval,gen)
            r.append([f"{sval}",gen.upper(),str(m.get("n")),f"{m['ap']*100:.3f}",f"{m['auc']*100:.3f}",f"{m['acc']*100:.2f}"])
    p.table(67, 145, 826, ["S","Test generator","n","AP %","AUC %","ACC@0.5 %"], r,
            [64,192,75,147,147,201], 17.3, 8.7)
    p.text(122, 491, 716, 14, "Chỉ hiển thị 15 ô đã đo ở S=5/10/20; mỗi ô n=400 (200 ảnh thật + 200 ảnh sinh).", 8.6, "muted", align="center")
    p.note()

    # A3 pilot train metadata
    p = d.page("Appendix B · Pilot protocol", "Independent detector training metadata by S",
        "S5/S10/S20 train.json · manifest protocol hashes",
        notes_for("modal-runs/experiments/experiments/table4/table4-lsun-pilot/S*/subsets/per-cell-000200/model/train.json"))
    r=[]
    for sval in ["5","10","20"]:
        block=T4_RUNS[sval]; t=block["train"]; hp=t["hparams"]
        r.append(["S="+sval,str(t.get("epochs_run")),str(t.get("best_epoch")),f"{t.get('best_val_ap',0):.5f}",f"{t.get('threshold_from_val',0):.4f}",str(hp.get("batch")),str(hp.get("lr")),str(hp.get("crop"))])
    p.table(65, 180, 830, ["Stage","Epochs run","Best epoch","Val AP","Val threshold","Batch","LR","Crop"], r,
            [94,115,112,116,132,85,90,86], 57, 9.5)
    p.rect(111, 399, 738, 63, "goldp", stroke="gold", radius=10)
    p.text(128, 409, 704, 43, "Ledger protocol target says 400 epochs with patience 5; actual run stopped early at 12–15 epochs. Do not present target epochs as executed epochs.", 11.3, "ink", bold=True, align="center", valign="mid")
    p.note()

    # A4 Table4 paper sweep and pilot S=50 difference
    p = d.page("Appendix C · Protocol comparison", "Paper and pilot use different generators and execution scales",
        "Paper Table 4 · repo Table 4 protocol + ledger",
        notes_for("Paper full-data S sweep reports S=5/10/20/50 with SD-v1; repo pilot measured S=5/10/20 with SD-v2 and 200/cell. S=50 is not included in the reproduction sweep."))
    p.table(73, 165, 814, ["Dimension","Paper Table 4","Repo pilot Table 4"], [
        ["Dataset / scale", "Full author protocol; about 1,000 real + 1,000 fake per cell", "200 images/cell subset; n=400 per binary test cell"],
        ["Generator", "ADM, DDPM, iDDPM, PNDM, SD-v1", "ADM, DDPM, iDDPM, PNDM, SD-v2"],
        ["Steps S", "5 / 10 / 20 / 50", "5 / 10 / 20 (sweep tái lập)"],
        ["Detector training", "Paper protocol", "ImageNet init, batch 32, LR 1e−4, early stopped"],
        ["Use", "Paper reference", "Pilot measurement candidate; time provenance unresolved"],
    ], [162,325,327], 54, 9.9)
    p.note()

    # A7 runbook evidence requirements
    p = d.page("Appendix E · Cổng bằng chứng", "A completed scientific cell requires metrics, labels, counts and compatible provenance",
        "docs/dire-tu-so-0.pdf Appendix C · table4/table5/figure4 runbook",
        notes_for("docs/dire-tu-so-0.pdf p. 76; runbook completion checks"))
    gates=[("Before", "R-level · preprocessing · split / threshold plan"),
           ("Per cell", "n_real · n_fake · status · test scores · sample IDs · config"),
           ("After", "ledger complete · no empty class · repeatability checks"),
           ("For claims", "paper vs repo labels · pilot vs full-data · caveat / missingness")]
    for i,(a,b) in enumerate(gates):
        y=157+i*78; p.badge(81,y+7,105,a,"bluep","teal",10)
        p.rect(209,y,668,52,"paper",stroke="line",radius=8)
        p.text(224,y+10,635,32,b,12,"ink",bold=(i==3),valign="mid")
    p.note()

    # A8 references
    p = d.page("Appendix F · Tài liệu tham khảo", "Primary sources and project evidence used in this deck",
        "Bibliography extracted from docs/dire-tu-so-0.pdf and source packet",
        notes_for("Full citation list; internal repository paths are relative to project root"))
    refs=[
        "Wang, Z. et al. DIRE for Diffusion-Generated Image Detection. ICCV 2023. arXiv:2303.09295.",
        "Ho, J., Jain, A., Abbeel, P. Denoising Diffusion Probabilistic Models. NeurIPS 2020. arXiv:2006.11239.",
        "Song, J., Meng, C., Ermon, S. Denoising Diffusion Implicit Models. ICLR 2021. arXiv:2010.02502.",
        "Dhariwal, P., Nichol, A. Diffusion Models Beat GANs on Image Synthesis. NeurIPS 2021. arXiv:2105.05233.",
        "Kingma, D. P., Welling, M. Auto-Encoding Variational Bayes. ICLR 2014. arXiv:1312.6114.",
        "Goodfellow, I. et al. Generative Adversarial Nets. NeurIPS 2014.",
        "Dinh, L. et al. Density Estimation using Real NVP. ICLR 2017. arXiv:1605.08803.",
        "Rombach, R. et al. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022. arXiv:2112.10752.",
        "Gu, S. et al. Vector Quantized Diffusion Model for Text-to-Image Synthesis. CVPR 2022.",
        "Ricker, J. et al. AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error. CVPR 2024. arXiv:2401.17879.",
        "Project sources: docs/dire-tu-so-0.pdf; docs/paper-cac-mo-hinh-sinh.html; docs/paper-code-walkthrough.md; docs/adr-0006-reproduction-levels.md.",
        "Project artifacts: src/dire_modal/paper_table2.tsv; modal-runs/runs/; modal-runs/experiments/; modal-runs/gallery/.",
    ]
    for i,ref in enumerate(refs):
        col=i//6; row=i%6; x=55+col*432; y=151+row*54
        p.text(x,y,410,45,f"{i+1:02d}  {ref}",9.6,"ink")
    p.note()

    # Save
    count=d.save()
    return d


def write_equation_source():
    EQ_SOURCE.write_text(r"""% Source equations for dire-research-complete.pptx / PDF.
% Rendered visually with matplotlib mathtext; project slides remain editable.
\[
q(x_t\mid x_{t-1})=\mathcal{N}\!\left(\sqrt{\alpha_t}x_{t-1},\,\beta_t I\right)
\]
\[
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s,\qquad x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon,
\quad \epsilon\sim\mathcal{N}(0,I)
\]
\[
\mathcal{L}_{\epsilon}=\mathbb{E}_{x_0,\epsilon,t}\left[\left\|\epsilon-\epsilon_{\theta}(x_t,t)\right\|_2^2\right]
\]
\[
\hat{x}_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\,\epsilon_{\theta}(x_t,t)}{\sqrt{\bar{\alpha}_t}}
\]
\[
x_{t-1}=\sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0+\sqrt{1-\bar{\alpha}_{t-1}}\,\epsilon_{\theta}(x_t,t)\quad(\eta=0)
\]
\[
x_{t+1}=\sqrt{\bar{\alpha}_{t+1}}\,\hat{x}_0+\sqrt{1-\bar{\alpha}_{t+1}}\,\hat{\epsilon}_{\theta}(x_t,t)
\]
\[
\mathrm{DIRE}(x_0)=\left|x_0-\mathcal{R}\!\left(\mathcal{I}(x_0)\right)\right|
\]
\[
\mathrm{NFE}_{\mathrm{per\ image}}=S_{\mathrm{invert}}+S_{\mathrm{reconstruct}}=2S
\]
\[
\mathcal{L}_{\mathrm{BCE}}=-\left[y\log\sigma(z)+(1-y)\log\left(1-\sigma(z)\right)\right]
\]
\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d}}\right)V
\]
\[
\mathrm{ELBO}=\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-D_{\mathrm{KL}}(q_\phi(z\mid x)\parallel p(z))
\]
\[
\min_G\max_D\;\mathbb{E}_{x\sim p_{data}}\log D(x)+\mathbb{E}_{z\sim p_z}\log(1-D(G(z)))
\]
\\[
z_q=e_{k^*},\qquad k^*=\arg\min_k\|z_e-e_k\|_2
\\]
\\[
\log p_X(x)=\log p_Z(f(x))+\log\left|\det J_f(x)\right|
\]
\[
\mathrm{ACC}(\tau)=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\!\left[(s_i>\tau)=y_i\right]
\]
\[
\mathrm{AUC}=\int_0^1\mathrm{TPR}(u)\,du
\]
""", encoding="utf-8")


def main():
    write_equation_source()
    deck = build_deck()
    # Inspect presentation text / PDF output consistency and evidence content.
    expected = ["U-Net", "ResNet-50", "DDIM", "Table 4", "S=50", "chưa có"]
    all_text = "\n".join(shape.text for slide in deck.prs.slides for shape in slide.shapes if getattr(shape, "has_text_frame", False))
    missing = [item for item in expected if item.lower() not in all_text.lower()]
    if missing:
        raise SystemExit(f"deck missing expected phrases: {missing}")
    pilot_sections = []
    for slide in deck.prs.slides:
        visible = "\n".join(shape.text for shape in slide.shapes
                              if getattr(shape, "has_text_frame", False))
        if "PILOT TABLE 4" in visible or "PILOT TABLE 4" in _vi_copy(visible):
            pilot_sections.append(visible)
    if not pilot_sections or any("S=50" in text for text in pilot_sections):
        raise SystemExit("pilot sweep must contain only measured S=5/10/20; S=50 belongs to paper reference only")
    if len(deck.prs.slides) < 40:
        raise SystemExit(f"expected expanded deck, got {len(deck.prs.slides)} slides")
    print(f"Built {len(deck.prs.slides)} slides")
    print(f"PPTX: {OUT_PPTX} ({OUT_PPTX.stat().st_size:,} bytes)")
    print(f"PDF:  {OUT_PDF} ({OUT_PDF.stat().st_size:,} bytes)")
    print(f"Equations: {EQ_SOURCE}")
    if deck.overflow:
        print("Potential text overflow (needs visual review):")
        for item in deck.overflow[:30]: print(" ", item)
    if deck.warnings:
        print("Warnings:", deck.warnings)


if __name__ == "__main__":
    main()
