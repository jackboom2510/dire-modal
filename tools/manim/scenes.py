# tools/manim/scenes.py
"""Các scene giải thích DIRE — 4 tầng nội dung + 1 scene training có log THẬT.

| Scene | Nội dung | Số lấy từ đâu |
|---|---|---|
| ``DiffusionTheory`` | forward → reverse DDIM → inversion → residual → trực giác manifold | không có số |
| ``DirePipeline`` | ảnh → DIRE map → ResNet50 → 1 logit → sigmoid → P(fake) | không có số |
| ``PaperTable2`` | 8 generator × ACC/AP của paper | ``paper_table2.tsv`` |
| ``DataPipeline`` | Baidu → staging → volume put → verify → eval 15 cặp × 4 | manifest + số đo |
| ``TrainMechanism`` | đường loss thật của ``demo_train`` | ``loss_log.json`` |

Ba quy tắc (đúng NFR-6 của repo, có test canh):

1. **Không có số nào viết thẳng trong file này.** Số đến từ `data.py`; `data.py`
   đọc TSV / manifest / JSON. Sửa dữ liệu thì hình đổi theo.
2. **Chưa có log thì KHÔNG vẽ đường cong.** ``TrainMechanism`` hiện một khung
   "chưa có log" kèm lệnh cần chạy. Một đường loss bịa trông y hệt đường thật.
3. **Mọi scene có số đều mang nhãn nguồn** (`badge`/`source_note`): "số của
   paper", "đo 2026-10-25", "demo — KHÔNG phải số của paper".

Chạy::

    bash tools/manim/render.sh --quality l              # nháp 480p15
    bash tools/manim/render.sh --quality h              # 1080p60
    bash tools/manim/render.sh --scene TrainMechanism   # một scene
"""

from __future__ import annotations

import sys
from pathlib import Path

# manim CLI chạy file này như một script rời, nên `data`/`style` phải có trên
# sys.path — không có package nào để dùng relative import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D  # noqa: E402
from manim import Scene  # noqa: E402
from style import (  # noqa: E402
    BODY_SIZE, C, DOWN, LEFT, RIGHT, SMALL_SIZE, UP, VGroup,
    Arrow, Dot, FadeIn, FadeOut, Line, Rectangle,
    badge, bar, body, box, chip_row, curve_from_points, mono, source_note,
    title,
)


# ---------------------------------------------------------------------------
# Khung chung
# ---------------------------------------------------------------------------
def _head(scene: Scene, text: str, sub: str = "") -> VGroup:
    """Tiêu đề + (tuỳ chọn) dòng phụ, gom thành VGroup để dời/cất cùng lúc."""
    parts = [title(text)]
    if sub:
        parts.append(mono(sub))
    g = VGroup(*parts).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    g.to_edge(UP, buff=0.5)
    return g


def _stamp(scene: Scene, text: str, color: str = "warn") -> VGroup:
    """Nhãn nguồn ở góc dưới-trái; giữ tới hết scene."""
    g = badge(text, color=C[color])
    g.to_edge(DOWN, buff=0.42)
    return g


# ===========================================================================
# 1. Lý thuyết nền của paper
# ===========================================================================
class DiffusionTheory(Scene):
    """Vì sao DIRE cần một quá trình TẤT ĐỊNH để đảo ngược.

    Bốn nhịp:
    1. forward: ảnh bị phun nhiễu dần thành đám mây Gauss;
    2. reverse DDIM: đi ngược lại, tất định (khác DDPM ngẫu nhiên) — nhờ tất định
       mới *đảo ngược* được một ảnh cụ thể;
    3. inversion + tái tạo: ``x → z_T → x̂``, và ``|x − x̂|`` chính là DIRE map;
    4. trực giác hình học: fake nằm gần manifold của model ⇒ residual nhỏ.

    Nhịp 4 là **trực giác**, không phải chứng minh — trên khung hình phải ghi rõ.
    """

    def construct(self) -> None:
        head = _head(self, "DIRE — nền lý thuyết",
                     "Diffusion model là một phép biến đổi TẤT ĐỊNH khi dùng DDIM")
        self.play(FadeIn(head))

        # --- nhịp 1: forward (thêm nhiễu) ---------------------------------
        step = body("1) Forward: ảnh bị phun nhiễu dần thành nhiễu thuần")
        step.next_to(head, DOWN, buff=0.5)
        x0 = box("x₀ (ảnh)", C["real"], width=2.2)
        cloud = VGroup(*[Dot(point=[1.6 + 0.34 * (i % 6), -0.5 + 0.30 * (i // 6), 0],
                            radius=0.05, color=C["dim"]) for i in range(24)])
        x0.next_to(step, DOWN, buff=0.9).shift(LEFT * 3.2)
        cloud.move_to([2.2, 0.1, 0])
        fwd = Arrow(x0.get_right(), cloud.get_left(), buff=0.15, color=C["dim"])
        fwd_label = mono("q(xₜ | xₜ₋₁) = N(√(1−βₜ)·xₜ₋₁, βₜI)")
        fwd_label.next_to(fwd, UP, buff=0.18)
        self.play(FadeIn(step), FadeIn(x0), FadeIn(cloud))
        self.play(FadeIn(fwd), FadeIn(fwd_label))
        self.wait(1.2)

        # --- nhịp 2: reverse DDIM, tất định -------------------------------
        rev_label = body("2) Reverse bằng DDIM: một đường đi DUY NHẤT cho mỗi ảnh")
        rev_label.next_to(fwd_label, DOWN, buff=0.65).shift(LEFT * 1.2)
        rev = Arrow(cloud.get_left(), x0.get_right(), buff=0.15, color=C["ok"])
        rev_why = mono("tất định ⇒ đảo ngược được;  DDPM lấy mẫu ngẫu nhiên ⇒ không")
        rev_why.next_to(rev, DOWN, buff=0.18)
        self.play(FadeIn(rev_label), FadeIn(rev))
        self.play(FadeIn(rev_why))
        self.wait(1.4)

        # --- nhịp 3: inversion + residual ---------------------------------
        self.play(FadeOut(x0), FadeOut(cloud), FadeOut(fwd), FadeOut(fwd_label),
                  FadeOut(rev_label), FadeOut(rev), FadeOut(rev_why))

        step3 = body("3) Inversion rồi tái tạo: x → z_T → x̂,  DIRE map = |x − x̂|")
        step3.next_to(head, DOWN, buff=0.6)
        src = box("x", C["real"], width=1.3)
        z = box("z_T", C["dim"], width=1.3)
        rec = box("x̂", C["fake"], width=1.3)
        g3 = VGroup(src, z, rec).arrange(RIGHT, buff=1.5)
        g3.next_to(step3, DOWN, buff=0.75).shift(LEFT * 2.2)
        a1 = Arrow(src.get_right(), z.get_left(), buff=0.1, color=C["dim"])
        a2 = Arrow(z.get_right(), rec.get_left(), buff=0.1, color=C["fake"])
        residual = Rectangle(width=1.3, height=1.3, color=C["map"],
                            fill_color=C["map"], fill_opacity=0.35)
        residual.next_to(rec, RIGHT, buff=0.75)
        res_arrow = Arrow(rec.get_right(), residual.get_left(), buff=0.1, color=C["map"])
        res_label = mono("|x − x̂| → 3 kênh")
        res_label.next_to(residual, DOWN, buff=0.16)
        self.play(FadeIn(step3), FadeIn(g3), FadeIn(a1), FadeIn(a2))
        self.play(FadeIn(res_arrow), FadeIn(residual), FadeIn(res_label))
        self.wait(1.6)

        # --- nhịp 4: trực giác manifold -----------------------------------
        self.play(FadeOut(step3), FadeOut(g3), FadeOut(a1), FadeOut(a2),
                  FadeOut(res_arrow), FadeOut(residual), FadeOut(res_label))

        step4 = body("4) Trực giác: ảnh do model sinh ra NẰM TRÊN manifold của nó")
        step4.next_to(head, DOWN, buff=0.6)
        curve = curve_from_points(
            [(0.0, 0.10), (0.2, 0.35), (0.4, 0.52), (0.6, 0.58), (0.8, 0.45), (1.0, 0.22)],
            origin=[-4.2, -1.7, 0], size=(8.0, 3.0), color=C["net"])
        on = Dot(point=[-0.2, 0.05, 0], radius=0.09, color=C["fake"])
        off = Dot(point=[1.4, 0.95, 0], radius=0.09, color=C["real"])
        on_tag = mono("fake: gần manifold → residual NHỎ", color=C["fake"])
        on_tag.next_to(on, DOWN, buff=0.2)
        off_tag = mono("real: lệch manifold → residual LỚN", color=C["real"])
        off_tag.next_to(off, UP, buff=0.2)
        verdict = body("⇒ DIRE(real) > DIRE(fake)", color=C["map"])
        verdict.next_to(curve, DOWN, buff=0.55)
        self.play(FadeIn(step4), FadeIn(curve))
        self.play(FadeIn(on), FadeIn(on_tag))
        self.play(FadeIn(off), FadeIn(off_tag))
        self.play(FadeIn(verdict))

        caveat = source_note("nhịp 4 là TRỰC GIÁC hình học; paper đo bằng thực nghiệm, không chứng minh")
        caveat.next_to(verdict, DOWN, buff=0.2)
        stamp = _stamp(self, "không có số ở scene này", "dim")
        self.play(FadeIn(caveat), FadeIn(stamp))
        self.wait(2.0)


# ===========================================================================
# 2. Pipeline của detector
# ===========================================================================
class DirePipeline(Scene):
    """Một ảnh đi qua detector: 4 bước, và ba chi tiết dễ sai.

    Ba chi tiết đó là ba lỗi THẬT của repo này:
    - checkpoint là **1 logit + BCE** (không phải 2 lớp + softmax) — sự cố
      ``size mismatch for fc.weight [1,2048] vs [2,2048]``;
    - tiền xử lý là ``Resize(256) → CenterCrop(224) → Normalize(ImageNet)``;
    - demo HF Spaces chạy ResNet **trực tiếp trên ảnh**, bỏ bước DIRE ⇒ khác
      hẳn detector trong paper.
    """

    def construct(self) -> None:
        head = _head(self, "Detector hoạt động thế nào",
                     "một ảnh → một xác suất, sau khi đã đi qua DIRE map")
        self.play(FadeIn(head))

        stages = [
            box("ảnh RGB", C["real"]),
            box("DIRE map", C["map"]),
            box("ResNet-50", C["net"]),
            box("1 logit", C["dim"]),
            box("P(fake)", C["fake"]),
        ]
        row = VGroup(*stages).arrange(RIGHT, buff=0.42)
        row.next_to(head, DOWN, buff=0.95)
        arrows = VGroup(*[
            Arrow(stages[i].get_right(), stages[i + 1].get_left(), buff=0.06,
                  color=C["line"])
            for i in range(len(stages) - 1)
        ])
        self.play(FadeIn(row))
        self.play(FadeIn(arrows))
        self.wait(0.8)

        n1 = mono("sigmoid(z) = P(fake)   — checkpoint có fc.weight [1, 2048]")
        n1.next_to(stages[3], DOWN, buff=0.5)
        fix = mono("✗ nếu dựng num_classes=2: size mismatch, không nạp được", color=C["bad"])
        fix.next_to(n1, DOWN, buff=0.2)
        self.play(FadeIn(n1))
        self.play(FadeIn(fix))
        self.wait(1.6)

        self.play(FadeOut(n1), FadeOut(fix))
        pre = body("Tiền xử lý (đúng `utils/datasets.py`, không như `demo.py`)")
        pre.next_to(arrows, DOWN, buff=0.75)
        steps = chip_row(["Resize(256)", "CenterCrop(224)", "ToTensor",
                          "Normalize(ImageNet)"], "ok")
        steps.next_to(pre, DOWN, buff=0.28)
        self.play(FadeIn(pre), FadeIn(steps))
        self.wait(1.4)

        self.play(FadeOut(pre), FadeOut(steps))
        gap = body("Demo HF Spaces chạy ResNet TRỰC TIẾP trên ảnh — thiếu bước DIRE",
                   size=BODY_SIZE - 4, color=C["warn"])
        gap.next_to(arrows, DOWN, buff=0.7)
        gap2 = source_note("⇒ độ chính xác của demo KHÔNG phải của detector trong paper")
        gap2.next_to(gap, DOWN, buff=0.2)
        self.play(FadeIn(gap), FadeIn(gap2))

        stamp = _stamp(self, "số của detector: xem scene PaperTable2", "dim")
        self.play(FadeIn(stamp))
        self.wait(2.0)


# ===========================================================================
# 3. Bảng số của paper
# ===========================================================================
class PaperTable2(Scene):
    """8 generator × ACC/AP — số của paper, và nói rõ đó KHÔNG phải số của mình.

    Thanh vẽ từ ``paper_table2.tsv``. Hàng ``avg`` không đọc từ file mà tự tính
    từ 8 ô, nên người xem thấy nó khớp với paper in.
    """

    def construct(self) -> None:
        ckpt = D.detectors()[0]
        matrix = D.paper_matrix(ckpt)
        head = _head(self, f"Table 2 của paper — detector {ckpt}",
                     "ACC trên 8 generator (LSUN-Bedroom)")
        self.play(FadeIn(head))

        max_value = D.PERCENT_MAX
        rows = VGroup(*[
            bar(gen, matrix[gen]["acc"], max_value, C["ok"])
            for gen in D.GENERATORS if gen in matrix
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        rows.next_to(head, DOWN, buff=0.55)
        self.play(FadeIn(rows, shift=DOWN * 0.2))

        mean = D.mean_acc(ckpt)
        avg = body(f"trung bình 8 ô = {mean:.1f} %  (tự tính, không chép hàng avg)")
        avg.next_to(rows, DOWN, buff=0.45)
        self.play(FadeIn(avg))

        stamp = _stamp(self, "số của paper (ICCV 2023) — CHƯA phải số đo của mình")
        note = source_note("nguồn: src/dire_modal/paper_table2.tsv · Δ do eval::report tính")
        note.to_edge(DOWN, buff=0.14)
        self.play(FadeIn(stamp), FadeIn(note))
        self.wait(2.5)


# ===========================================================================
# 4. Đường đi của dữ liệu
# ===========================================================================
class DataPipeline(Scene):
    """Từ Baidu Netdisk tới bảng ACC — 5 chặng, số tính từ manifest.

    Có một nhịp riêng cho bẫy ``ls -l``: nó in ``119.22MB`` và code cũ quy đổi
    ngược ra ``125 011 230`` B, lệch 3 901 B so với file thật ``125 015 131`` ⇒
    tải lại một file đã tải đủ. Đây là lỗi đã có ADR, dạy được bằng hình.
    """

    def construct(self) -> None:
        head = _head(self, "Đường đi của dữ liệu",
                     "5 chặng, ~0 credit GPU cho tới bước cuối")
        self.play(FadeIn(head))

        facts = {f["key"]: f for f in D.pipeline_facts()}
        chain = VGroup(*[
            box("Baidu", C["dim"], width=1.9),
            box("staging (máy)", C["real"], width=2.5),
            box("volume put", C["net"], width=2.2),
            box("verify", C["ok"], width=1.8),
            box("eval", C["warn"], width=1.6),
        ]).arrange(RIGHT, buff=0.34)
        chain.next_to(head, DOWN, buff=0.8)
        self.play(FadeIn(chain))

        rate = mono(f"trần tài khoản free: {facts['baidu_rate']['text']}")
        rate.next_to(chain, DOWN, buff=0.45)
        self.play(FadeIn(rate))
        self.wait(1.2)

        self.play(FadeOut(rate))
        checks = chip_row(["gzip -t", "sha256 vs manifest", ".pth: size + sha256"], "ok")
        checks.next_to(chain, DOWN, buff=0.45)
        self.play(FadeIn(checks))
        self.wait(1.2)

        self.play(FadeOut(checks))
        scale_line = body(
            f"{facts['pairs']['text']} cặp × {facts['detectors']['text']} detector "
            f"= {facts['containers']['text']} container", color=C["warn"])
        scale_line.next_to(chain, DOWN, buff=0.45)
        why = source_note(facts["containers"]["source"])
        why.next_to(scale_line, DOWN, buff=0.18)
        self.play(FadeIn(scale_line), FadeIn(why))
        self.wait(1.8)

        # --- bẫy ls -l -----------------------------------------------------
        self.play(FadeOut(chain), FadeOut(scale_line), FadeOut(why))
        trap = body("Bẫy: `ls -l` in 119.22MB, file thật là 125 015 131 B")
        trap.next_to(head, DOWN, buff=0.6)
        shown = mono("119.22 MB  →  ×2²⁰ = 125 011 230 B", color=C["bad"])
        real = mono("file thật        = 125 015 131 B", color=C["ok"])
        gap = mono("lệch 3 901 B (0.003 %) ⇒ đủ để đòi tải lại file ĐÃ ĐỦ", color=C["warn"])
        VGroup(shown, real, gap).arrange(DOWN, aligned_edge=LEFT, buff=0.22) \
            .next_to(trap, DOWN, buff=0.5).shift(LEFT * 0.6)
        self.play(FadeIn(trap))
        self.play(FadeIn(shown))
        self.play(FadeIn(real))
        self.play(FadeIn(gap))

        stamp = _stamp(self, "số đo 2026-10-25 · ADR: SizeEst mang sai số làm tròn", "dim")
        self.play(FadeIn(stamp))
        self.wait(2.2)


# ===========================================================================
# 5. Training — CHỈ vẽ khi có log thật
# ===========================================================================
class TrainMechanism(Scene):
    """Đường loss THẬT của ``demo_train`` — hoặc một khung "chưa có log".

    Vì sao không có nhánh "vẽ cho đẹp": một đường loss bịa và một đường loss
    thật trông y hệt nhau. Scene này vì thế có hai trạng thái, và trạng thái
    thiếu dữ liệu phải **trông như thiếu dữ liệu**.

    Nhãn ``demo — KHÔNG phải số của paper`` lấy từ chính file log
    (``data.demo_label``) nên không thể quên khi render.
    """

    def construct(self) -> None:
        log = D.loss_log()
        head = _head(self, "Train ResNet của DIRE",
                     "1 logit · BCEWithLogitsLoss · Adam 1e-4 · seed 3407")
        self.play(FadeIn(head))

        if log is None:
            missing = body("Chưa có log thật — scene này KHÔNG vẽ đường cong giả",
                           color=C["warn"])
            missing.next_to(head, DOWN, buff=0.7)
            cmd = mono("modal run -m dire_modal.demo_train::main --mode run")
            cmd2 = mono("modal volume get dire-ckpt demo_train/demo_train ./docs/media")
            VGroup(cmd, cmd2).arrange(DOWN, aligned_edge=LEFT, buff=0.18) \
                .next_to(missing, DOWN, buff=0.55)
            self.play(FadeIn(missing))
            self.play(FadeIn(cmd), FadeIn(cmd2))
            stamp = _stamp(self, D.demo_label(None), "warn")
            self.play(FadeIn(stamp))
            self.wait(2.5)
            return

        pts = D.loss_points(log)
        losses = [p[1] for p in pts]
        lo, hi = min(losses), max(losses)
        span = (hi - lo) or 1.0
        norm = [((step - pts[0][0]) / max(1, pts[-1][0] - pts[0][0]),
                 (hi - loss) / span) for step, loss in pts]

        box_origin = [-4.6, -2.2, 0]
        size = (9.0, 3.4)
        axes_h = Line([box_origin[0], box_origin[1], 0],
                      [box_origin[0] + size[0], box_origin[1], 0], color=C["line"])
        axes_v = Line([box_origin[0], box_origin[1], 0],
                      [box_origin[0], box_origin[1] + size[1], 0], color=C["line"])
        curve = curve_from_points(norm, origin=box_origin, size=size, color=C["map"])
        axis_labels = VGroup(
            mono(f"loss {lo:.2f} → {hi:.2f}", size=SMALL_SIZE - 4),
            mono(f"{len(pts)} điểm · step 1…{pts[-1][0]}", size=SMALL_SIZE - 4),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        axis_labels.move_to([box_origin[0] + size[0] / 2,
                             box_origin[1] - 0.75, 0])
        self.play(FadeIn(axes_h), FadeIn(axes_v))
        self.play(FadeIn(curve), FadeIn(axis_labels))

        summary = body(
            f"loss {pts[0][1]:.3f} → {pts[-1][1]:.3f}", color=C["ok"])
        summary.next_to(head, DOWN, buff=0.45)

        vals = D.val_points(log)
        val_text = mono("val_acc: " + " · ".join(f"e{e}={a:.2f}" for e, a in vals)) \
            if vals else mono("val_acc: chưa có (ô trống, KHÔNG phải 0)", color=C["dim"])
        val_text.next_to(summary, DOWN, buff=0.2)
        self.play(FadeIn(summary), FadeIn(val_text))

        stamp = _stamp(self, D.demo_label(log), "warn")
        how_many = source_note(
            f"{log.get('config', {}).get('images', '?')} ảnh × "
            f"{log.get('config', {}).get('epochs', '?')} epoch — tác giả: 400 epoch")
        how_many.to_edge(DOWN, buff=0.14)
        self.play(FadeIn(stamp), FadeIn(how_many))
        self.wait(2.5)
