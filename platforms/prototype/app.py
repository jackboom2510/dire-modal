"""Standalone, read-only Streamlit research dashboard for DIRE pilot evidence."""
from __future__ import annotations

import streamlit as st

from data import (
    EVAL_PATH, GALLERY, REPORT_PATH, TABLE4_PATH, gallery_samples, paper_table2,
    pipeline_status, read_json, table4, table4_rows, table5_status, figure4_status,
)

st.set_page_config(page_title="DIRE Research Prototype", page_icon="◈", layout="wide")
st.markdown("""
<style>
:root { color-scheme: dark; }
.block-container { max-width: 1440px; padding-top: 2rem; }
.hero { padding: 1.5rem 1.7rem; border: 1px solid #263a4a; border-radius: 18px;
        background: linear-gradient(120deg,#10202c,#142a32 65%,#1b2834); margin-bottom: 1rem; }
.hero-kicker { color:#8ed9c0; font-size:.75rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
.hero h1 { margin:.35rem 0; letter-spacing:-.04em; }
.hero p { color:#c0cdd3; max-width:850px; margin-bottom:0; }
.evidence { border-left:3px solid #e6b85c; background:#211f19; padding:.8rem 1rem; border-radius:5px; }
.small-muted { color:#9eabb4; font-size:.88rem; }
</style>
<div class="hero">
  <div class="hero-kicker">Research workspace · evidence-first</div>
  <h1>DIRE / Experimental dashboard</h1>
  <p>Recorded paper references and pilot measurements, kept visibly separate.
  This MVP is read-only: it does not launch jobs, write to Modal, or claim an
  uploaded RGB image has been classified.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Pilot hard cap: **$30 total**, including spend already incurred and any live job. "
    "Current spend/headroom is not verified by this local dashboard; no new GPU "
    "run or on-demand action is available here.", icon="⚠️",
)

paper_tab, pilot_tab, gallery_tab, upload_tab, pipeline_tab = st.tabs([
    "Paper reference · Table 2", "Recorded pilot", "DIRE gallery", "Image sandbox", "Pipeline & provenance",
])

with paper_tab:
    st.subheader("Table 2 · paper-reported reference")
    st.markdown(
        "These are values reported by Wang et al., ICCV 2023, Table 2—not results "
        "measured by this project. The source TSV is `src/dire_modal/paper_table2.tsv`."
    )
    rows = paper_table2()
    if rows:
        rows = [r for r in rows if r.get("generator") != "avg"]
        detector = st.selectbox("Paper detector / training generator", sorted({r["ckpt"] for r in rows}))
        selected = [r for r in rows if r["ckpt"] == detector]
        st.dataframe(
            [{"Detector": r["ckpt"], "Evaluation generator": r["generator"],
              "ACC (%)": float(r["acc"]), "AP (%)": float(r["ap"]),
              "Provenance": "Paper-reported · ICCV 2023 Table 2"} for r in selected],
            hide_index=True, width="stretch",
        )
        st.caption("Paper averages are omitted from the row view; values are not project measurements.")
    else:
        st.info("Paper reference TSV is not available in this checkout.")
    st.info(
        "The pilot uses LSUN-Bedroom with SD-v2 among its test generators; this is "
        "not an equivalent reproduction of paper Table 2's SD-v1 condition. Do not "
        "merge these measurements or compute a reproduction delta from this view."
    )

with pilot_tab:
    st.subheader("Recorded project measurements")
    ledger = table4()
    report = read_json(REPORT_PATH)
    eval_result = read_json(EVAL_PATH)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gallery samples", len(gallery_samples()))
    c2.metric("Table 4 S values recorded", ", ".join(sorted(ledger.get("by_steps", {}), key=int)) if ledger else "—")
    c3.metric("Table 4 missing S", ", ".join(map(str, ledger.get("missing_steps", []))) if ledger else "—")
    c4.metric("Table 5 / Figure 4", "inputs only / not run")

    if report:
        dire = report.get("dire", {})
        st.markdown("#### R1+R2 aggregate report (separate run; not paper Table 2)")
        a, b, c, d = st.columns(4)
        a.metric("DIRE maps recorded", f"{dire.get('n', '—'):,}" if isinstance(dire.get('n'), int) else "—")
        b.metric("DDIM steps", str(dire.get("steps", "—")))
        b.metric("NFE total", f"{dire.get('nfe_total', 0):,}" if dire.get("nfe_total") else "—")
        c.metric("New / reused maps", f"{dire.get('n_new', '—'):,} / {dire.get('n_reused', '—'):,}" if isinstance(dire.get('n_new'), int) else "—")
        d.metric("Manifest SHA-256", str(report.get("data", {}).get("manifest_sha256", "—"))[:12] + "…")
        st.caption("Source: modal-runs/runs/report.json · label: " + str(report.get("label", "unspecified")))
        st.dataframe([
            {"Generator": name.removeprefix("gen-"), "ACC": m.get("acc"), "AP": m.get("ap"), "AUC": m.get("auc"), "n": m.get("n"), "Evidence": "Recorded R1+R2 run"}
            for name, m in report.get("train_by_generator", {}).items()
        ], hide_index=True, width="stretch")
    else:
        st.info("R1+R2 aggregate report not found.")

    if eval_result:
        metrics = eval_result.get("metrics", {})
        st.markdown("#### ADM evaluation · held-out project run")
        st.write(
            f"n={metrics.get('n', '—')} · ACC={metrics.get('acc', '—')} · "
            f"AP={metrics.get('ap', '—')} · AUC={metrics.get('auc', '—')} · "
            f"threshold source={eval_result.get('threshold_source', '—')}"
        )
        st.caption(eval_result.get("honesty", ""))

    st.markdown("#### Table 4 pilot · S sweep")
    if ledger:
        st.caption(
            f"Pilot subset `{ledger.get('subset_id')}` · {ledger.get('per_cell')} images/cell · "
            f"seed {ledger.get('sample_seed')} · test generators: {', '.join(ledger.get('test_generators', []))}. "
            "Metrics below are read directly from the ledger; ACC uses its recorded 0.5 threshold."
        )
        rows = table4_rows(ledger)
        st.dataframe(rows, hide_index=True, width="stretch")
        completed = [r for r in rows if r["Status"] == "ran" and isinstance(r["AP"], (int, float))]
        if completed:
            import pandas as pd
            chosen = st.multiselect("Generators in recorded AP chart", sorted({r["Generator"] for r in completed}), default=sorted({r["Generator"] for r in completed}))
            chart_rows = [r for r in completed if r["Generator"] in chosen]
            chart = pd.DataFrame(chart_rows).pivot(index="S", columns="Generator", values="AP").sort_index()
            st.line_chart(chart, y_label="Average Precision (test)")
            st.caption("Recorded test AP only; missing S=50 remains missing and is never interpolated.")
    else:
        st.info("No Table 4 ledger is present. Inputs alone are not a completed experiment.")

    t5 = table5_status()
    f4 = figure4_status()
    st.markdown("#### Other requested pilot outputs")
    st.dataframe([
        {"Output": "Table 5", "Status": "Inputs materialized; result ledger absent" if t5["protocol"] and not t5["ledger_count"] else "See evidence files", "Evidence": str(t5["protocol"].get("representations", [])) if t5["protocol"] else "No protocol found"},
        {"Output": "Figure 4", "Status": "Not run / no ledger found" if not f4["ledger_count"] else "Result ledger found", "Evidence": "No result ledger in checked-in pilot artifacts"},
    ], hide_index=True, width="stretch")

with gallery_tab:
    st.subheader("Recorded DIRE maps")
    samples = gallery_samples()
    if not samples:
        st.info("No complete gallery sample (metadata + original + map) is available.")
    else:
        options = [s["meta"].get("name", "unnamed") for s in samples]
        picked = st.selectbox("Choose a verified gallery record", options)
        sample = samples[options.index(picked)]
        meta = sample["meta"]
        left, right = st.columns(2)
        left.image(str(sample["orig"]), caption="Original RGB image", width="stretch")
        right.image(str(sample["diff"]), caption="Recorded DIRE map (display PNG)", width="stretch")
        st.caption(
            f"{meta.get('generator')} · {meta.get('split')} · label {meta.get('label')} · "
            f"S={meta.get('S')} · map source: {meta.get('map_source')}"
        )
        st.caption("Display PNG is illustrative; authoritative DIRE values are the stored map array on the source Volume, not this PNG.")
        with st.expander("Sample provenance"):
            st.json(meta)
        st.caption(f"Gallery path: `{GALLERY.relative_to(GALLERY.parents[1])}/` · metadata and image hashes are shown above.")

with upload_tab:
    st.subheader("Image sandbox · preview only")
    st.markdown(
        "Upload is currently a **local preview only**. No image is sent to Modal or "
        "saved by this app. DIRE computation and prediction are intentionally disabled "
        "until a compatible, provenance-verified diffusion checkpoint, detector "
        "checkpoint, preprocessing contract, and cost-guarded execution path are available."
    )
    image = st.file_uploader("Choose an image (PNG/JPEG/WebP; max 10 MiB)", type=["png", "jpg", "jpeg", "webp"], max_upload_size=10)
    if image is not None:
        try:
            from PIL import Image
            preview = Image.open(image)
            preview.verify()
            image.seek(0)
            preview = Image.open(image).convert("RGB")
            st.image(preview, caption=f"Preview · {preview.width}×{preview.height} · not classified", width="stretch")
            st.success("Validated for display only. No DIRE map or class score was computed.")
        except Exception as exc:  # malformed uploads should not interrupt the dashboard
            st.error(f"This file could not be decoded as an image ({type(exc).__name__}).")
    st.button("Compute DIRE / predict", disabled=True, help="No verified compatible local checkpoint or cost-guarded runner is available in this MVP.")
    st.caption("No checkpoint is bundled in this repo. Ledger SHA-256 identifiers alone do not mean checkpoint files are available for inference.")

with pipeline_tab:
    st.subheader("Pipeline and evidence state")
    st.markdown(
        "**Conceptual path:** input image → DDIM inversion → reconstruction → "
        "pixel-wise DIRE map → ResNet-50 trained on DIRE maps → score. The app "
        "does not execute this path on uploaded images yet."
    )
    st.code("RGB input → diffusion DDIM inversion → reconstruction x̂₀ → |x − x̂₀| DIRE → ResNet-50 detector → P(fake)", language="text")
    st.dataframe(pipeline_status(), hide_index=True, width="stretch")
    if ledger:
        with st.expander("Table 4 protocol and checkpoint identifiers"):
            st.json({
                "subset": {k: ledger.get(k) for k in ("subset_id", "per_cell", "sample_seed", "subset_status")},
                "steps": {s: {"manifest_sha256": r.get("protocol", {}).get("manifest_sha256"),
                              "diffusion_checkpoint_sha256": r.get("protocol", {}).get("adm_ckpt_sha256"),
                              "detector_sha256_recorded": r.get("detector_sha256"),
                              "epochs_run": r.get("train", {}).get("epochs_run"),
                              "init": r.get("train", {}).get("init")}
                          for s, r in ledger.get("by_steps", {}).items()},
                "note": "Identifiers are copied from experiment ledgers; checkpoint binaries are not bundled here.",
            })
    st.markdown("#### DIRE in brief")
    st.markdown(
        "DIRE (Diffusion Reconstruction Error) uses a pretrained diffusion model to "
        "invert an image to a noisy latent and reconstruct it. The absolute pixel "
        "difference between input and reconstruction forms a three-channel map; a "
        "detector trained on these maps—not raw RGB—estimates whether the image is generated."
    )
    st.caption("This panel explains the method and recorded evidence; it does not certify a full-paper reproduction.")

st.divider()
st.caption("Private research prototype · read-only data paths · no Modal calls · no persisted uploads · no automatic job launch")
