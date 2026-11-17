"""Lệnh chạy ở máy nhà — không cần GPU, phần lớn không cần torch.

::

    python -m dire_zero info       # lịch, mốc, NFE, bảng tiền
    python -m dire_zero keys       # đối chiếu 566 key với fixture
    python -m dire_zero dataset    # KIỂM cây dữ liệu trên ổ (rò rỉ, phân bố, giấy phép)
    python -m dire_zero ingest     # tar.gz → PNG 256² + manifest sha256
    python -m dire_zero selftest   # chạy trọn 4 bước trên U-Net 32×32

`dataset` và `ingest` là hai lệnh đáng chạy **trước khi** thuê GPU: chúng trả lời
"dữ liệu trên ổ F của tôi có dùng được không" bằng vài giây CPU, thay vì bằng
một hoá đơn GPU và một bảng kết quả vô nghĩa.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from . import budget, dataset as ds, schedule as sch


# --------------------------------------------------------------------------- #
# info / keys / selftest
# --------------------------------------------------------------------------- #

def cmd_info(args: argparse.Namespace) -> int:
    ts = sch.timesteps(args.train_steps, args.steps)
    gamma = sch.alphas_cumprod(sch.linear_betas(args.train_steps))
    print(f"T={args.train_steps}  S={args.steps}  mốc: {ts[0]} … {ts[-1]}  (số mốc: {len(ts)})")
    print(f"NFE/ảnh = 2S = {sch.nfe_per_image(args.steps)}")
    for t in ts:
        print(f"  t={t:4d}  ᾱ={gamma[t]:.6g}  √(1−ᾱ)={max(0.0, 1 - gamma[t]) ** 0.5:.6g}")
    print("\nChi phí theo sec/step (mỗi ảnh 256², NFE = 2S):")
    print(f"{'ảnh':>6} {'s/step':>7} {'GPU-h':>8} {'USD A10G':>10}")
    for n in (500, 1000, 2000):
        for sec in (0.05, 0.35, 1.0, 2.5):
            est = budget.Estimate(n, 2 * args.steps, sec)
            print(f"{n:6d} {sec:7.2f} {est.gpu_h:8.3f} {est.usd:10.2f}")
    return 0


def cmd_keys(args: argparse.Namespace) -> int:
    fixture = Path(args.fixture)
    if not fixture.exists():
        print(f"không thấy {fixture}", file=sys.stderr)
        return 3
    if importlib.util.find_spec("torch") is None:
        print("cần torch để dựng mô hình (pip install torch)", file=sys.stderr)
        return 3
    import torch  # noqa: PLC0415
    from .checkpoint import check_bijection
    from .unet import ZeroUNet

    doc = json.loads(fixture.read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = ZeroUNet()
    mapping = check_bijection(model, doc["keys"])
    print(f"OK: {len(mapping)} tensor, {model.n_params:,} tham số — khớp {fixture.name}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """gen → dire → train trên U-Net 32×32, 8 ảnh."""
    if importlib.util.find_spec("torch") is None:
        print("cần torch (pip install torch)", file=sys.stderr)
        return 3
    from .detector import ResNet, ResNetDetector
    from .diffusion import ModelCounter
    from .pipeline import RunMeta, compute_maps, generate, train
    from .unet import UNetConfig, ZeroUNet

    cfg = UNetConfig(image_size=32, model_channels=32, num_res_blocks=1,
                     attention_ds=(4,), channel_mult=(1, 2), num_heads=2, head_channels=-1)
    model = ZeroUNet(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    f = ModelCounter(model)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        g = generate(f, n=16, out_dir=tmp / "gen", meta=RunMeta("0" * 64, steps=3), seed_base=1, size=32)
        print(f"gen: {g['n']} ảnh, NFE={g['nfe_total']}, {g['seconds']}s")

        items = [(str(tmp / "gen" / nm), 1, "train" if nm in g["split"]["train"] else "test")
                 for nm in g["names"]]
        d = compute_maps(f, items, tmp / "maps", RunMeta("0" * 64, steps=3), limit=16, size=32)
        print(f"dire: {d['n']} map, NFE={d['nfe_total']}, thang {d['scale']}")

        maps = [{"name": x["name"], "path": str(tmp / "maps" / x["name"]),
                 "label": x["label"], "split": x["split"]} for x in d["items"]]
        tr = train(lambda: ResNetDetector(ResNet(blocks=(1, 1))), maps, tmp / "train",
                   epochs=1, bs=4, crop=28, device="cpu", patience=99)
        print(f"train: {tr['split']} · test n={tr['test']['n']} acc={tr['test']['acc']:.3f}")
    print("SELFTEST OK")
    return 0


# --------------------------------------------------------------------------- #
# Dataset riêng
# --------------------------------------------------------------------------- #

def _print_scan(scan: ds.Scan) -> None:
    rows = scan.matrix()
    print(f"Cây dữ liệu: {scan.root}   (bố cục: {scan.layout})")
    for dataset, splits in rows.items():
        print(f"\n  [{dataset}]")
        for split, gens in splits.items():
            print(f"    {split:5s}  " + "  ".join(gens))
    for note in scan.notes:
        print(f"\n  [bỏ qua] {note}")


def _cost_lines(n_real: int, n_fake: int, gen_steps: int, dire_steps: int, sec: float | None) -> list[str]:
    """Chi phí cho **đúng** số ảnh đang có. Không có sec/step ⇒ không báo tiền."""
    need_gen = 0 if n_fake else n_real
    nfe_gen = need_gen * gen_steps
    nfe_dire = (n_real + max(n_fake, need_gen)) * 2 * dire_steps
    out = [
        f"  ảnh thật {n_real}, ảnh giả {n_fake or f'(chưa có → tự sinh {need_gen})'}",
        f"  NFE sinh (S={gen_steps}): {nfe_gen:,}   NFE DIRE (S={dire_steps}, 2 lượt/ảnh): {nfe_dire:,}",
    ]
    if sec is None:
        out += [
            "  GPU-h: không tính — chưa ĐO sec/step.",
            "  Tiền: KHÔNG báo. Khoảng thực tế 0,05–2,5 s/step ⇒ chênh 50×. Đo trước:",
            "        modal run -m dire_zero.modal_app::bench",
        ]
    else:
        est = budget.Estimate(n_real + max(n_fake, need_gen), 2 * dire_steps, sec)
        gen_gpu_h = nfe_gen * sec / 3600
        out += [
            f"  GPU-h: ~{gen_gpu_h + est.gpu_h:.2f}",
            f"  Tiền DIRE: {est.usd:.2f} USD (khoảng {est.band()[0]:.2f}–{est.band()[1]:.2f}) @ A10G",
        ]
    return out


def cmd_dataset(args: argparse.Namespace) -> int:
    """Kiểm cây dữ liệu trước khi tiêu GPU. Exit 3 nếu có gì chặn."""
    try:
        root = ds.resolve_root(args.root)
        scan = ds.scan_tree(root, allow_published=args.allow_published)
    except (FileNotFoundError, ds.IntegrityError) as exc:
        print(f"[CHẶN] {exc}", file=sys.stderr)
        return 3

    _print_scan(scan)

    available = scan.generators(args.dataset) if args.dataset else scan.generators()
    if args.dataset and args.dataset not in scan.matrix():
        print(f"\n[CHẶN] không có dataset {args.dataset!r}. Có: {sorted(scan.matrix())}", file=sys.stderr)
        return 3

    items = ds.select(
        scan, dataset=args.dataset, splits=tuple(args.splits),
        generators=tuple(args.generators) if args.generators else None,
        include_real=not args.no_real,
    )
    if not items:
        print("\n[CHẶN] bộ lọc không chọn archive nào", file=sys.stderr)
        return 3

    # Đếm ảnh **chỉ trong phần sẽ dùng** — cây 12 GB, đếm hết là vài phút vô ích.
    print(f"\nĐang đếm {len(items)} archive (bung gzip, không ghi ra đĩa)…")

    def _tick(it: ds.Item, info: dict) -> None:
        mb = info["bytes"] / 1e6
        print(f"  {it.split}/{it.dataset}/{it.source.name}: {info['n_images']} ảnh / {mb:.1f} MB")

    _, count_warn = ds.count_items(items, progress=_tick)
    if args.limit:
        for it in items:
            if it.n_images is not None:
                it.n_images = min(it.n_images, args.limit)
    n_real = sum(it.n_images or 0 for it in items if it.label == 0)
    n_fake = sum(it.n_images or 0 for it in items if it.label == 1)
    print(f"\nDùng: {len(items)} archive · {n_real} ảnh thật · {n_fake} ảnh giả"
          + (f" (cắt {args.limit}/archive)" if args.limit else ""))
    for w in count_warn + scan.warnings:
        print(f"  [cảnh báo] {w}")

    # ---- cổng 1: giấy phép ---------------------------------------------
    if not args.license:
        print(
            "[CHẶN] thiếu --license. Dữ liệu riêng vẫn cần khai giấy phép: repo có đường\n"
            "  mirror công khai (HF/Kaggle) kèm quy tắc attribution + takedown, và data_card\n"
            "  ghi thẳng giá trị này. Ví dụ: --license research-only · private · CC-BY-4.0",
            file=sys.stderr,
        )
        return 3

    # ---- cổng 2: phân bố ------------------------------------------------
    same_dist = (args.dataset or "").lower() in ("lsun_bedroom", "lsun-bedroom", "lsun")
    if not same_dist and not args.ack_ood:
        print(
            f"[CHẶN] dataset {args.dataset!r} KHÔNG phải lsun_bedroom, mà ADM 256 vô điều kiện\n"
            "  được train trên LSUN-Bedroom. DIRE dựa trên: mô hình tái tạo tốt ảnh THẬT (đã học\n"
            "  phân bố đó) và kém ảnh SINH. Với ảnh thật ngoài phân bố, cả hai lớp đều bị tái tạo\n"
            "  kém ⇒ khoảng cách DIRE co lại ⇒ ACC về mức đoán mò. Không phải lỗi code.\n"
            "  Hiểu rồi và vẫn muốn chạy: thêm --ack-ood (data_card sẽ ghi comparable_to_paper=false).",
            file=sys.stderr,
        )
        return 3

    # ---- nhãn mức tái lập ----------------------------------------------
    fake_src = args.fake_source
    if fake_src == "auto":
        fake_src = "author" if n_fake else "gen"
    label = {"gen": "R1+R2+R3", "author": "R1+R2", "published": "R0"}[fake_src]
    print(
        f"\nNhãn tối đa được nói: {label}"
        + ("   (ảnh giả tự sinh ⇒ cả 3 thành phần của bạn)" if fake_src == "gen"
           else "   (ảnh giả của tác giả ⇒ KHÔNG được nói R3)" if fake_src == "author"
           else "   (dùng map phát hành ⇒ chỉ là chấm lại)")
    )

    print("\nKế hoạch:")
    for line in _cost_lines(n_real, n_fake, args.gen_steps, args.dire_steps, args.sec_per_step):
        print(line)

    if args.manifest:
        ds.write_manifest_tsv(args.manifest, [it for it in items if it.member is None] or items)
        print(f"\n  danh sách archive → {args.manifest}")
    if args.card:
        card = ds.DataCard(
            root=str(root), layout=scan.layout, dataset=args.dataset or "all",
            license=args.license, splits=list(args.splits),
            generators=available, n_real=n_real, n_fake=n_fake, seed=args.seed,
            fake_source=fake_src, label=label, comparable_to_paper=same_dist,
            warnings=scan.warnings + scan.notes, created=ds.now(),
        )
        ds.write_data_card(args.card, card)
        print(f"  data_card → {args.card}")

    print("\nBước tiếp:")
    print("  python -m dire_zero ingest --root <cùng đường dẫn> --out <thư mục ra>")
    return 0


def _ingest_warnings(stats: dict) -> list[str]:
    """Đường dự phòng đã dùng ⇒ vào thẳng ``data_card.warnings``.

    Không có bước này thì ``data_card.json`` của một dataset thiếu 3 % trông
    y hệt dataset đủ — và đó là kiểu sai lệch tệ nhất, vì nó đi theo dữ liệu
    lên Volume, vào báo cáo, vào con số cuối cùng.
    """
    out: list[str] = []
    for rec in stats.get("partial_archives", []):
        out.append(
            f"[dự phòng] archive đọc dở {rec['key']}: {rec['read']} ảnh, "
            f"đọc bằng {rec['mode']} ({rec['error']})"
        )
    if stats.get("n_fallbacks"):
        out.append(f"[dự phòng] {stats['n_fallbacks']} ảnh đọc ở chế độ chấp nhận ảnh cụt")
    if stats.get("n_rejects"):
        out.append(f"[dự phòng] {stats['n_rejects']} ảnh không dùng được (xem stats['rejects'])")
    if stats.get("reused"):
        out.append(f"[resume] {stats['reused']} ảnh đã có từ lần chạy trước, không ghi lại")
    return out


def _print_fallbacks(stats: dict) -> None:
    """In phần dự phòng — chỉ in khi có, và in đủ để sửa được."""
    partial = stats.get("partial_archives") or []
    if partial:
        print(f"\n[dự phòng] {len(partial)} archive đọc dở:")
        for rec in partial[:5]:
            print(f"  ! {rec['key']}: {rec['read']} ảnh · {rec['mode']} · {rec['error']}")
        if len(partial) > 5:
            print(f"  … và {len(partial) - 5} archive khác")
    if stats.get("n_fallbacks"):
        print(f"[dự phòng] {stats['n_fallbacks']} ảnh đọc ở chế độ ảnh cụt (vài dòng cuối khác gốc):")
        for line in (stats.get("fallbacks") or [])[:3]:
            print(f"  ~ {line}")
    if stats.get("n_rejects"):
        print(f"[dự phòng] {stats['n_rejects']} ảnh bỏ (không giải mã được / không ghi được):")
        for rec in (stats.get("rejects") or [])[:3]:
            print(f"  - {rec['split']}/{rec['generator']}:{rec['member']} — {rec['reason']}")
    if stats.get("reused"):
        print(f"[resume] {stats['reused']} ảnh đã có sẵn, dùng lại (không ghi lại)")


def cmd_ingest(args: argparse.Namespace) -> int:
    """tar.gz → PNG 256² + manifest sha256, rồi in lệnh đẩy lên Volume."""
    import datetime as _dt

    if importlib.util.find_spec("PIL") is None:
        print("cần Pillow (pip install pillow)", file=sys.stderr)
        return 3
    try:
        root = ds.resolve_root(args.root)
        scan = ds.scan_tree(root, allow_published=args.allow_published)
        items = ds.select(
            scan, dataset=args.dataset, splits=tuple(args.splits),
            generators=tuple(args.generators) if args.generators else None,
            include_real=not args.no_real,
        )
    except (FileNotFoundError, ds.IntegrityError) as exc:
        print(f"[CHẶN] {exc}", file=sys.stderr)
        return 3
    if not items:
        print("[CHẶN] bộ lọc không chọn archive nào", file=sys.stderr)
        return 3

    out = Path(args.out)
    print(f"Đang đổ {len(items)} archive → {out} (cắt tối đa {args.limit} ảnh/archive)…")

    try:
        made, stats = ds.ingest(
            items, out, size=args.size, limit_per_archive=args.limit,
            on_error="strict" if args.strict else "record",
            resume=not args.no_resume,
        )
    except ds.IntegrityError as exc:
        print(f"[CHẶN] {exc}", file=sys.stderr)
        return 3

    manifest = Path(args.manifest or (out / "manifest.tsv"))
    cardpath = Path(args.card or (out / "data_card.json"))
    ds.write_manifest_tsv(manifest, made)

    n_real = sum(1 for m in made if m.label == 0)
    n_fake = sum(1 for m in made if m.label == 1)
    fake_src = args.fake_source
    if fake_src == "auto":
        fake_src = "author" if n_fake else "gen"
    label = {"gen": "R1+R2+R3", "author": "R1+R2", "published": "R0"}[fake_src]

    card = ds.DataCard(
        root=str(root), layout=scan.layout, dataset=args.dataset or "all",
        license=args.license or "private", splits=list(args.splits),
        generators=scan.generators(args.dataset), n_real=n_real, n_fake=n_fake,
        seed=args.seed, fake_source=fake_src, label=label,
        comparable_to_paper=(args.dataset or "").lower() in ("lsun_bedroom", "lsun-bedroom", "lsun"),
        manifest_sha256=ds.manifest_sha(made),
        warnings=scan.warnings + scan.notes + _ingest_warnings(stats), created=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    ds.write_data_card(cardpath, card)

    ok, bad = ds.verify_manifest(manifest)
    for key, n in sorted(stats["per_archive"].items()):
        print(f"  {key}: {n} ảnh")
    print(f"\n[xong] {stats['written']} ảnh mới · {stats['reused']} ảnh dùng lại"
          f" · thật {n_real} · giả {n_fake} · bỏ {stats['n_rejects']} ảnh hỏng")
    print(f"  manifest: {manifest}\n  data_card: {cardpath}")
    print(f"  kiểm lại sha256: {ok} khớp, {len(bad)} lệch")
    for b in bad[:5]:
        print(f"    ! {b}")
    print(f"  nhãn tối đa được nói: {label}")

    _print_fallbacks(stats)

    if stats["partial_archives"]:
        print("\nĐẩy lên Volume — **CHỈ SAU KHI** tải lại archive hỏng và chạy lại:")
    else:
        print("\nĐẩy lên Volume:")
    print(f"  modal volume put dire-raw {out} zero/data -r")
    print("  modal volume put dire-raw " + str(manifest) + " zero/data/manifest.tsv")
    print("  modal volume put dire-raw " + str(cardpath) + " zero/data/data_card.json")
    print("\nRồi:  modal run -m dire_zero.modal_app::checkdata")
    if stats["partial_archives"]:
        print("\n[KHÔNG ĐẦY ĐỦ] có archive đọc dở ⇒ kế hoạch NFE/chi phí tính trên số ảnh"
              " thực có sẽ sai. Tải lại archive đó rồi chạy lại (resume sẽ không ghi trùng).",
              file=sys.stderr)
        return 3
    return 0 if not bad else 3


def cmd_peek(args: argparse.Namespace) -> int:
    """Xem **bên trong** tar: đuôi file + vài tên mẫu. Không giải nén.

    Đây là lệnh đáng chạy trước khi viết bất cứ pipeline nào cho ``dire/``,
    ``recons/`` hay ``checkpoints/``: nó trả lời "bên trong là .npy hay .png"
    bằng dữ liệu, thay vì bằng phỏng đoán.
    """
    from pathlib import Path as _P

    targets: list[_P] = []
    if args.tar:
        targets = [_P(t) for t in args.tar]
    if args.root:
        root = ds.resolve_root(args.root)
        targets += sorted(root.rglob("*.tar.gz")) if args.recursive else sorted(root.glob("*.tar.gz"))
    targets = [t for t in targets if t.is_file()]
    if not targets:
        print("[CHẶN] không có file .tar.gz nào để xem", file=sys.stderr)
        return 3

    for t in targets:
        try:
            info = ds.tar_inventory(t, n_sample=args.sample)
        except Exception as exc:   # tar hỏng, nén lạ, member rác — đều là chuyện thật
            print(f"{t.name}: [lỗi] {exc}")
            continue
        print(f"{t.name}  ·  {info['n_members']} member  ·  {info['bytes'] / 1e6:.1f} MB")
        print("  đuôi: " + "  ".join(f"{k}:{v}" for k, v in info["by_ext"].items()))
        for nm in info["sample"]:
            print(f"  vd: {nm}")
    return 0


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dire_zero", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("info", help="lịch nhiễu, mốc, NFE, bảng tiền")
    a.add_argument("--steps", type=int, default=20)
    a.add_argument("--train-steps", type=int, default=sch.T_TRAIN)
    a.set_defaults(func=cmd_info)

    b = sub.add_parser("keys", help="đối chiếu bảng ánh xạ với fixture 566 key")
    b.add_argument("--fixture", default="tests/fixtures/adm_lsun_state_keys.json")
    b.set_defaults(func=cmd_keys)

    c = sub.add_parser("selftest", help="chạy trọn 4 bước trên mô hình tí hon")
    c.set_defaults(func=cmd_selftest)

    f = sub.add_parser("peek", help="xem bên trong tar (đuôi file, tên mẫu) — không giải nén")
    f.add_argument("--root", default=None, help="thư mục chứa .tar.gz")
    f.add_argument("--tar", nargs="+", default=None, help="một hoặc nhiều file .tar.gz")
    f.add_argument("--recursive", action="store_true")
    f.add_argument("--sample", type=int, default=3)
    f.set_defaults(func=cmd_peek)

    for name, func, helptext in (
        ("dataset", cmd_dataset, "kiểm cây dữ liệu (tar.gz) trước khi chạy GPU"),
        ("ingest", cmd_ingest, "tar.gz → PNG 256² + manifest sha256"),
    ):
        d = sub.add_parser(name, help=helptext)
        d.add_argument("--root", required=True,
                       help="F:/du-lieu  ·  F:\\\\du-lieu  ·  /f/du-lieu (tự lùi vào DiffusionForensics/)")
        d.add_argument("--dataset", default="lsun_bedroom", help="lsun_bedroom · imagenet · celebahq")
        d.add_argument("--splits", nargs="+", default=["train", "test"])
        d.add_argument("--generators", nargs="+", default=None,
                       help="chỉ lấy các generator này (mặc định: tất cả trừ real)")
        d.add_argument("--no-real", action="store_true", help="bỏ lớp thật")
        d.add_argument("--license", default=None, help="research-only · private · CC-BY-4.0")
        d.add_argument("--fake-source", default="auto", choices=["auto", "gen", "author", "published"],
                       help="ảnh giả lấy từ đâu — quyết định NHÃN (gen ⇒ R1+R2+R3, author ⇒ R1+R2)")
        d.add_argument("--seed", type=int, default=3407)
        d.add_argument("--allow-published", action="store_true",
                       help="cho phép đọc dire/ — chỉ để chấm lại R0, KHÔNG cho from-scratch")
        d.add_argument("--manifest", default=None)
        d.add_argument("--card", default=None)
        d.set_defaults(func=func)
        if name == "dataset":
            d.add_argument("--limit", type=int, default=None,
                           help="tính kế hoạch với tối đa bao nhiêu ảnh MỖI archive")
            d.add_argument("--ack-ood", action="store_true")
            d.add_argument("--gen-steps", type=int, default=50)
            d.add_argument("--dire-steps", type=int, default=20)
            d.add_argument("--sec-per-step", type=float, default=None)
        else:
            d.add_argument("--out", required=True)
            d.add_argument("--size", type=int, default=256)
            d.add_argument("--limit", type=int, default=None, help="tối đa bao nhiêu ảnh MỖI archive")
            d.add_argument("--strict", action="store_true",
                           help="archive hỏng ⇒ dừng hẳn (mặc định: giữ phần đọc được + ghi chú)")
            d.add_argument("--no-resume", action="store_true",
                           help="ghi lại cả PNG đã có (mặc định: bỏ qua file đã đúng)")

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
