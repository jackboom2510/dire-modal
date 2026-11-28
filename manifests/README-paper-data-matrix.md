# Paper-reproduction data matrix scaffold

`paper_data_matrix.tsv` currently contains only its schema header. It has **zero inventory rows**, which means the matrix is not yet inventoried; it does not mean any expected/observed count is zero. Populate rows only after P0 identifies the complete paper/supplementary matrix and D0 inventories real archives/members. Use `null` plus an explicit reason in `notes` for unknown values; record source, checksum and license provenance rather than inferring them.

Current gate state: P0 partial, D0 blocked. See [`../docs/plans/research/paper-reproduction-p0-d0-audit.md`](../docs/plans/research/paper-reproduction-p0-d0-audit.md). No experiment or compute is authorized by this scaffold.
