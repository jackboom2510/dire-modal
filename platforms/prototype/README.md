# DIRE research prototype (private, read-only MVP)

A separate Streamlit dashboard for reviewing checked-in paper references, Modal
pilot ledgers, and gallery assets. This is not the public HF Space and does not
launch work on Modal.

## Run locally

From the repository root:

```bash
python -m pip install -r platforms/prototype/requirements.txt
streamlit run platforms/prototype/app.py --server.address 0.0.0.0
```

The dashboard reads only fixed paths under this repository:

- `src/dire_modal/paper_table2.tsv` (paper-reported Table 2 reference)
- `modal-runs/runs/` (recorded R1+R2 report and evaluation)
- `modal-runs/experiments/` (pilot ledgers and protocols)
- `modal-runs/gallery/` (metadata plus original/map display PNGs)

No Modal API or credentials are required. Uploaded files are decoded only for a
browser-facing preview; they are not transmitted or persisted. DIRE computation
and prediction remain disabled: checkpoint hashes in a ledger do not establish
that compatible checkpoint files are locally available. The upload is therefore
not presented as a detector demo.

## Evidence and safety semantics

- Paper-reported values appear in a separate tab and are labelled as such.
- Project measurements are read from their ledgers; absent cells remain `not run`.
- Table 4's per-generator ACC uses the recorded fixed 0.5 threshold; AP/AUC are
  displayed alongside it. The chart is a view of recorded AP only, without
  interpolation or estimates.
- Table 4 is a 200-images-per-cell pilot. It is not a full-paper reproduction.
- Current GPU spend/headroom cannot be verified by this local app. The $30 hard
  cap remains a project constraint; no on-demand controls are enabled.
- The gallery PNG is display-only. Source `.npy` DIRE maps on the Modal Volume
  are authoritative.

## Current capability boundary

Included: DIRE explanation and pipeline, paper Table 2 reference, recorded
R1+R2 report/evaluation, interactive view of Table 4's recorded S sweep, explicit
Table 5/Figure 4 missing/partial status, metadata-backed gallery, and a local
image preview.

Not included: image-to-DIRE execution, detector inference, checkpoint selection,
training, Modal calls, generator integration, or editable slides/PDF. Before
adding inference, verify compatible diffusion + detector binaries and their
preprocessing contracts; before enabling remote runs, independently verify live
jobs and budget headroom within the all-in $30 cap.
