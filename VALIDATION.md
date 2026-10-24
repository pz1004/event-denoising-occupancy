# Validation record — 25 October 2026

This public release contains implementation code, experiment runners, frozen
protocols and recorded results. Development tests, numerical audit tools and
figure assets are maintained outside the public package. The validation record
below distinguishes historical checks of the larger package from validation of
this simplified release.

## Historical validation of the larger package

Before simplification, a larger package was checked in temporary directories
containing only its manifest-listed files, with no manuscript files, dataset
symlinks or private-checkout import paths. These results describe that earlier
package; they are not commands or bundled tests available in this release.

| Historical check | Recorded result |
|---|---|
| Combined implementation, evidence-regression and release-integrity tests | **363 passed, 1 explicitly skipped** |
| Separate upstream E-RAFT voxel-conversion integration | **1 passed** against pinned upstream source |
| Supplied-evidence and figure audits | **15 passed**: 13 numerical/provenance and 2 figure audits |
| Synthetic three-arm filtering example | Exact quotas, stable arrival ties and original-event output passed |
| Experiment command entry points | All **9** documented `--help` entry points passed |
| Data-free figure regeneration | All **5** generators passed, producing **7** figures; both figure audits passed afterward |
| Figure font and rendering inspection | All **8** PDFs had embedded fonts and no Type 3 fonts; 7 vector charts and one intentionally rasterized event illustration |
| Immutable scientific files in the larger package | **249** implementation/test/protocol/evidence/report files matched their source bytes |
| EDmamba import-compatibility patch | Applied cleanly to the pinned upstream originals |

The one historical skip required external E-RAFT source. The comparison then
passed separately against actual upstream commit
`c58ce0524ea0ebfa9849991caafb547f44fe9bfd` and an independent trilinear reference.
It required no event dataset, checkpoint inference or GPU. These historical
checks did not install all third-party environments from scratch.

## Current release verification

The simplified release uses `tools/run_frozen_study.py` to select the protocols
relocated to `protocols/`. Scientific source files, protocol contents and the
principal phase79/80/81 result artifacts retain their original bytes. The
external validation process checked this preservation separately from the public
file checksums.

**37 launcher checks passed** using validation tooling kept outside this release.
The checks cover all three study choices, exact argument forwarding and protocol
path resolution, rejection of missing or altered protocols before runner import,
and rejection of a runner imported from a different checkout. They also verify
restoration of protocol globals, arguments, module aliases and import paths after
normal return, an explicit exit or an exception. All three runner help commands
passed, and the real rotation synthetic preflight passed in an independent copy
without the excluded folders. Its expected entry-point dispatcher aliases were
present, and kernel signatures stayed stable.

All four protocol files retain their original hashes. Scientific implementation
and recorded-evidence bytes are unchanged. The reduced file inventory, source
provenance mappings and local documentation links were checked separately; the
regenerated checksum list was verified from another independent copy. The
documented synthetic quick-start command was also exercised without external data.

Users can check supplied file integrity with `sha256sum -c SHA256SUMS` and run
the data-free rotation example with
`python tools/run_frozen_study.py rotation --stage synthetic`.
Neither operation reproduces the recorded benchmark outcomes from raw data.

## Historical verification environment

- Python 3.13.9
- NumPy 2.3.5; SciPy 1.16.3; Numba 0.62.1
- PyTorch 2.10.0+cu128, with synthetic checks on CPU
- pytest 8.4.2; Matplotlib and Poppler for the historical figure inspection

This is the historical release-check environment, not a claim that all recorded
experiments used these versions. The experiment-specific recipes preserve their
distinct environments and known setup gaps.

## What was not rerun

No raw-data benchmark, frozen downstream study, complete application timing,
learned-model training or checkpoint inference was rerun while preparing the
release. GPU extensions were not rebuilt from an empty machine. Historical
aggregate checks do not independently reproduce absent raw data.

Frozen evidence hashes and the full payload inventory are in
[PROVENANCE.md](PROVENANCE.md), [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json) and
[SHA256SUMS](SHA256SUMS).
