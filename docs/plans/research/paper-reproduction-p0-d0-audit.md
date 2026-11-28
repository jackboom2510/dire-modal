# Paper-reproduction track — preliminary P0/D0 audit

- **Snapshot:** 2026-10-14 (UTC)
- **Repo commit audited:** `a11205902480c2dfbc62687970fd0c52b60dfb11`
- **Track:** paper reproduction only. No CelebA-HQ, ImageNet-extension, or other research-extension result is used as paper-reproduction evidence.
- **Status:** P0 partial; D0 blocked; no experiment authorized or launched.

This is a repository/source audit, not an experimental result and not a claim that the paper protocol is fully resolved. It records what can be checked from the current checkout and what still needs authoritative paper/data evidence. Topic-by-topic provisional mismatches are tracked in [`paper-protocol-discrepancies.md`](paper-protocol-discrepancies.md); those rows are unresolved, not approved settings.

## Source and provenance inventory

| Item | Local/source identity | Audit state | Remaining issue |
|---|---|---|---|
| Paper | DIRE for Diffusion-Generated Image Detection, ICCV 2023; cited as arXiv:2303.09295 in the plan and upstream README | The local reading HTML `docs/paper-tung-trang.html` labels itself arXiv:2303.09295v1. A local PDF exists at `docs/dire-tu-so-0.pdf` (SHA-256 `9ce67b56bed55fa20dbda6da8e875fb7f5ec3b0ee31f45985befe417efe17aaf`); another bundled copy is `platforms/site/dire-research-complete.pdf` (SHA-256 `a58074b096882e0f91cfedb02cc5d62f1ebdb3646a3547c08fc87f697a4d5a18`) | The local HTML label does not establish PDF origin/version or supplementary material identity. Verify against an authoritative arXiv revision and record the reference checksum before locking table cells. |
| Authors' source | `https://github.com/ZhendongWang6/DIRE`, checkout fetched read-only for audit at commit `1f89fff15f0e478307cdea9abdfe06a52fe7838a` (`main`/HEAD at audit); deterministic `git archive` TAR SHA-256 `c83aed5ff0dee6f0934f2a89ea643f346e2bf552973ca01bd2c85d68307b124f` | Commit-pinned source inspected. This is the code snapshot reviewed, not proof the authors intended it to supersede paper text. | Review commit history/tags and inspect paper-vs-code disagreements with a second reviewer. |
| This repository | `a11205902480c2dfbc62687970fd0c52b60dfb11` | Pin recorded. | None for identity; protocol configs still need versioned lock. |
| Public checkpoints | The upstream source README points to author-hosted release links; no author ADM or detector checkpoint is present in this checkout. | Not locally available or hash-verified. | Obtain the exact checkpoint used per experiment from a permitted source, preserve source URL/terms, and record SHA-256 and mapping. |
| Dataset release | No raw paper dataset/archive is present in the checkout. | D0 cannot pass from repository evidence. | Inventory operator-provided storage/archive bytes, members, IDs, splits, counts, hashes, provenance and license; do not infer missing rows as zero. |

## Source-code observations that must be reconciled with the paper

These are findings from the pinned source commit, not choices already accepted as the reproduction protocol:

- `utils/config.py`: default ResNet-50; batch size 64; load size 256; crop 224; up to 400 epochs; Adam, `lr=1e-4`, `beta1=0.9`; ImageNet-pretrained initialization; seed 3407; random resize interpolation configured as bilinear; train random crop and horizontal flip; ImageNet normalization. Defaults may be overridden by experiment configs, so the actual per-experiment files/commands must be recovered.
- `utils/datasets.py` and `utils/eval.py`: validation/test resize and center-crop under the default flags; evaluation reports raw-prevalence sklearn AP and accuracy at a strict `score > 0.5` comparison. This differs from any plan that assumes a threshold convention or balanced-prior AP without checking the paper's table definition.
- `train.py` and `utils/earlystop.py`: validation reports AP and ACC, but early stopping/checkpoint updates are driven by ACC (with patience and learning-rate adjustment), not by AP. Do not silently substitute validation-AP checkpoint selection in a claimed reproduction.
- `guided-diffusion/compute_dire.sh` uses DDIM 20-step respacing, unconditional ADM configuration, 256-pixel images, and FP16 model inference in the example. `compute_dire.py` applies the same `real_step` argument to reverse and sampling loops and saves the absolute residual as an 8-bit visualization. Exact paper representation/range and source/reconstruction/map protocol still require paper and released-artifact comparison; PNG output alone is not a reference-quality numeric map.
- Upstream code and runbooks are implementation evidence, not proof of data provenance, license, complete released counts, or a completed experiment.

## Existing local experiment artifacts

- The checkout contains Table 4 pilot/subset metadata, including `modal-runs/experiments/experiments/table4/table4-lsun-pilot/subsets/per-cell-000200/table4.json`; it is explicitly marked `pilot`, and `missing_steps` includes 50. This is not a full Table 4 result.
- The inspected Table 4 run directories contain protocol/input metadata, not a complete full-data Table 4 ledger. Table 5 has protocol metadata but no `table5.json` result. No `figure4.json` result was found.
- These artifacts are not promoted to `ran`/`audited` paper claims by this audit. Any pilot rows remain separate from paper-table results.

## P0/D0 gate ledger

| Gate item | State | Evidence / exit work |
|---|---|---|
| P0: pin this repo and authors' source | Partial | Repo commit, upstream source commit, and deterministic source-archive checksum recorded above. Preserve the source archive itself and review history/tags. |
| P0: pin paper and supplementary version | Blocked | Local PDFs are hashed but version/provenance not established. Identify authoritative version and compare full protocol/table/supplement. |
| P0: public checkpoints/configs and licenses | Blocked | Exact experiment checkpoint lineage, bytes, terms and config variants not present/verified. |
| P0: machine-readable protocol contract for M0–M7 | Blocked | Existing plans/runbooks are plans with known caveats; create versioned YAML and paper/source discrepancy register, independently reviewed. |
| D0: split × domain × generator × artifact inventory | Blocked | No source dataset mounted/available in this checkout; populate the schema-only `manifests/paper_data_matrix.tsv` from the actual operator-held artifacts, with counts/hash/license/source/status and explicit null reasons. The header-only state is not an inventory. |
| D0: IDs, split independence, aliases, provenance | Blocked | Requires archive/member inventory and exact ID/content checks; do not infer from basenames or prior CelebA-HQ reports. |
| D0: license/sharing and storage decision | Blocked | Verify terms per source and intended storage/runtime before mirroring or compute. |
| Experiment selection / compute authorization | Not selected | No first paper-reproduction experiment is locked. No GPU or data-processing job was launched. Select a bounded M0/Mx only after P0/D0 rows and the intended evidence tier are approved. |

## Stop conditions and next action

Do not launch paper-reproduction GPU work, create cloud resources, or convert a pilot into a paper result while P0/D0 are open. The next useful step is to prepare the authoritative paper-version comparison and fill the data-matrix inventory from the actual paper dataset archives or a user-provided inventory manifest. Then resolve every missing matrix cell with source, license, count/hash and a disposition (recreate exactly, request source, or mark unreproducible). Keep all results from CelebA-HQ-SDv2 under ADR-0007's separate research-extension track.

## Stage 10 handoff (separate track; user-supplied evidence)

The archived Stage 10 JSON is `docs/reports/celebahq-stage10-evaluation.json`, byte-identical to the user attachment (SHA-256 `191b40ad405fbe064f9a0863d48f3f644db597d0475d003b869166290502b833`). Its claims are not independently verified against Kaggle artifacts. It reports `status=pass`, inference and image reads, no training, no mounted-input write, and no test-based checkpoint selection. Primary macro balanced-prior AP mean across seeds is `0.9993401250` (population SD `0.0004547671`); family mean APs are DALL·E 2 `0.9989824727`, IF `0.9992378134`, Midjourney `0.9999318802`, StarGAN `0.9992083337`. Mean ACC@0.5 varies by family and is lower (DALL·E 2 `0.9211`, IF `0.8228`, Midjourney `0.9694`, StarGAN `0.7253`). These metrics do not open or tune the paper-reproduction track. The original report contains six bootstrap upper endpoints slightly above 1 from floating-point roundoff; the archived source is unchanged and those endpoints must not be presented as valid probabilities without correction/qualification.
