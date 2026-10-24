# Release provenance

Public research release of the RSF implementation, experiment runners, and recorded results. External datasets and pretrained model checkpoints must be obtained from their original providers.

The associated study is **Occupancy versus Multiplicity in Training-Free Event
Denoising: Representation, Horizon, and Application Cost**, prepared for submission
by Sooyoung Jang and Kyuseung Han. The repository is
[pz1004/event-denoising-occupancy](https://github.com/pz1004/event-denoising-occupancy).
Repository preparation does not establish journal acceptance.

## What is preserved

The scientific implementation modules, frozen protocols and retained numerical
results preserve their original bytes. The broad module set includes historical
research helpers needed by reused readers and baselines; the README identifies
the supported entry points. Historical helpers are not additional validated
configurations.

These principal artifacts retain their original SHA-256 digests:

| Artifact | SHA-256 |
|---|---|
| phase79_tstar_cost_quality.json | `e8d9bf8fc950a65b3f416016b5b21e316f275424e64ced05c79c3c3b3415c197` |
| phase80_frozen_dsec_downstream.json | `290590ad21333abc643c0f39550cecf4f2cf459016ce5ed1e25a6eb531cfb594` |
| phase81_frozen_rotation_application.json | `f4f4af9b96074411c658c0e45aa98ad3db4e081f27d61ce40bdd0dad524eb19c` |

All are under `native3d_ssm/experiments/analytic_contrast/`. The four files in
`protocols/` are the unchanged frozen experimental protocols. Some original
experiment metadata records their former paths under `plan/`; those strings
remain historical provenance, not the public execution paths.

## Public-package adaptations

Public documentation supplies acquisition instructions and executable commands.
The public launcher, `tools/run_frozen_study.py`, selects the relocated protocol
files for phases 79–81 while leaving their scientific source files and protocol
contents unchanged. Baseline setup helpers adapt path discovery and optional
imports without changing model mathematics. [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json)
records copied and adapted files and their original source digests.

Some immutable records contain historical absolute paths or machine-library
locations. These are provenance metadata, not requirements to recreate the
authors' home directory. Follow [DATA.md](DATA.md) and the environment recipes
for current setup. Historical model-run reports retain their numerical findings;
the setup instructions distinguish compatibility changes from model changes.

## Integrity and validation boundaries

[RELEASE_MANIFEST.json](RELEASE_MANIFEST.json) records the size and digest of each
payload file. [SHA256SUMS](SHA256SUMS) additionally binds that manifest. Both omit
themselves as appropriate to avoid recursive hashes. Git metadata, local
environments, rerun outputs, externally acquired inputs and runtime compilation
caches are not release payload. Verify the supplied bytes from the release root:

```bash
sha256sum -c SHA256SUMS
```

Checksums establish file integrity, not scientific correctness. The validation
record in [VALIDATION.md](VALIDATION.md) distinguishes this release from the
larger development package previously used for numerical and regression audits.
Those validation tools are maintained outside this public package. Fresh timings
depend on hardware, load and environment; preparing this release did not rerun
raw-data experiments or regenerate recorded timing measurements.

Manuscript sources, manuscript/supplement PDFs, figure assets, development tests
and audit tooling are excluded. Raw event recordings, third-party model weights
and external repositories are also excluded. Acquisition and licensing
requirements remain separate from the repository's MIT grant.
