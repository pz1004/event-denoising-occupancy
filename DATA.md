# Datasets and baselines

Neither the datasets nor the learned baselines are vendored in this repository. Both are
redistributed by their original authors under their own terms, and the datasets are far too large
for git. This file records what is needed and where each result came from.
See [environment recipes](environments/README.md) and [upstream revisions and
compatibility changes](reproduction/UPSTREAM.md) before rerunning external models.

## Datasets

| Name | Resolution | Labels | Used for |
|---|---|---|---|
| **DND21** | 346×260 | per-event | Dense synthetic noise, 10 conditions, 3,393 blocks. Track-A development |
| **DVSCLEAN** | 1280×720 | per-event | Sparse simulated noise, 10 conditions, 315 blocks. Track-A development |
| **ED24** | 346×260 | per-event | Current manuscript: 100 scenes × 21 attenuator-voltage levels (1.5–3.5 V), 2,100 conditions; prior-contact evaluation |
| **E-MLB** | 346×260 | none | 8 cells (Daylight/Night × ND1/4/16/64), 48 scenes per lighting condition × 3 reps; structural evaluation |
| **LED** | 1280×720 | per-event partition | 66 released test scenes; recorded raw, denoised, and noise events |
| **BUAA_Campus** | 346×260 | per-event | 2 base scenes × 3 simulated noise rates; first 3 million events per sequence |

DND21 and ED24 are linked by the [EDmamba release](https://github.com/gongzihang/EDmamba)
and [EDformer release](https://github.com/NJUVISION/EDformer). The ED24 row describes
the in-range evaluation subset, not all historical experiment artifacts. E-MLB is
from [Ding et al., IEEE TMM](https://doi.org/10.1109/TMM.2023.3260638); its reference
software is [cuke-emlb](https://github.com/KugaMaxx/cuke-emlb). BUAA_Campus is
released with [EBF](https://github.com/shicy17/EBF),
[Guo et al., IEEE TCSVT](https://doi.org/10.1109/TCSVT.2025.3568604).

| Input | Original-author acquisition page or download |
|---|---|
| DND21 and ED24 | [EDmamba dataset instructions](https://github.com/gongzihang/EDmamba#dataset-preparation), with authors' OneDrive/Baidu links |
| DVSCLEAN | [AEDNet authors' release and dataset download instructions](https://github.com/Fanghuachen/AEDNet) |
| E-MLB | [cuke-emlb dataset instructions](https://github.com/KugaMaxx/cuke-emlb#dataset), [D-END](https://drive.google.com/file/d/1ZatTSewmb-j6RsrJxMWEQIE3Sm1yraK-/view), [N-END](https://drive.google.com/file/d/17ZDhuYdtHui9nqJAfiYYX27omPY7Rpl9/view) |
| LED | [LED authors' release](https://github.com/Yee-Sing/led), linked from the [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Duan_LED_A_Large-scale_Real-world_Paired_Dataset_for_Event_Camera_Denoising_CVPR_2024_paper.html) |
| BUAA_Campus | [EBF release](https://github.com/shicy17/EBF), [authors' download](https://drive.google.com/drive/folders/1NiswzR7yJ9z_nxOyc7dXW76b0hXZdViz) (upstream states academic use only) |
| DSEC | [official DSEC dataset](https://dsec.ifi.uzh.ch/), exact files/hashes below |
| DAVIS Event-Camera Dataset | [official dataset page](https://www.ifi.uzh.ch/en/rpg/software_datasets/davis_datasets.html), exact recordings below |

Dataset access conditions and redistribution rights remain with their providers.
The public package redistributes only repository-authored outputs and provenance,
not event recordings or third-party model weights.

The DND21, DVSCLEAN, ED24, and E-MLB runners resolve data relative to the repository root.
Either place the datasets there or symlink them:

```bash
ln -s /path/to/ECCV2024_datasets ECCV2024_datasets   # contains AUC_test/ (DND21) and ED24/
ln -s /path/to/DVSCLEAN          DVSCLEAN            # contains simulated_data/
ln -s /path/to/E-MLB             E-MLB               # contains D-END/ and N-END/
```

The LED and BUAA runners default to sibling directories instead:
`../LED/LED_Test_upload` and `../BUAA_Campus`. Override these with `--root` in phase54/62
or the dataset-specific arguments shown by each experiment's `--help`.

LED labels are reconstructed from the released `raw_event`, `denoised_event`, and
`noise_event` folders. Phase54 verifies their exact multiset partition and records
ambiguous keys and skipped chunks. BUAA's polarity field contains invalid values in four
of six sequences; phase62 documents the label-free random-polarity repair, and current
paper tables use the repaired stream. These preprocessing choices are part of the protocol.

### Selection history

The operand, radius, and fixed operating point `r = 0.28` were selected on labelled
development data before E-MLB evaluation. ED24 was checked during earlier development.
E-MLB and LED were initially outside development, but their results informed the later
decision to retire the global-density gate. They therefore do not independently confirm
the final ungated configuration. Preserve this history when interpreting old artifacts or
extending the experiments; a new confirmation needs untouched data.

## Learned baselines

We reproduced both locally rather than quoting published tables. Neither is included here; obtain
them from their authors.

| Baseline | Used for | Our reproduction |
|---|---|---|
| **EDformer** | Track-B structural metric | mean **1.0069** vs 1.0059 published (+0.0010); Raw anchor matches to 3 dp |
| **EDmamba** | Track-A AUC | 8 of 10 cells within 0.0003 of the paper |

Runners are in `reproduction/`. Measured outputs are in `reproduction/results/`.

**Two cautions carried from our audit, both verifiable from the released artifacts:**

1. **EDmamba's structural row is not regenerable from its released code** — `compute_average_esr`
   is commented out and defined nowhere, and its E-MLB reader sets `labels = polarity` and covers
   13 scenes at one ND rather than 48 × 4. We therefore do **not** cite that row, and neither
   should you.
2. **The EDformer checkpoint count differs from the reported efficiency table.** The table reports 49.80K;
   counting the released checkpoint gives **584,019 tensor elements, of which 46 are BatchNorm
   buffers → 583,973 trainable**. The gap is exactly the two SparseConvNet weight tensors
   (264,384 each, 90.5 % of the model), which a counter that does not traverse those custom
   modules misses. The GFLOPs figure comes from the same counter and is therefore a **lower
   bound** — we use it unchanged, making our reported ratio conservative.

### EDformer environment

EDformer needs its own environment and does not coexist with the filter's:
Python 3.10, torch 2.2.1+cu118, numpy 1.26.4, **dv-processing 1.7.9** (the authors' pin — 2.0.3
renames `doEveryNumberOfEvents` and breaks their metrics), CUDA 11.8, plus SparseConvNet and
pytorch3d built from source. Run inference at `--batch-blocks 1`: batching is *not* numerically
equivalent, because float32 reduction order shifts logits by up to 5.2e-06 against a threshold
that sits in a dense part of the score distribution.

## Metric reference

Our ESR implementation (`native3d_ssm/phase5_emlb_mesr.py`, reused by `phase9`/`phase10`) was
verified **bit-exact** against the official `cuke-emlb` implementation during development. That
check requires `cuke-emlb`, which is not vendored here, so this historical metric comparison is not a bundled public command. Obtain `cuke-emlb` from its authors for an independent metric comparison.

MESR is the mean ESR over 30,000-event slices. Absolute MESR moves with event caps and repetition
subsets (capped vs uncapped Raw differs by 0.0151), so **always compare Δ-over-Raw**, which is
stable to 0.0004.

## Baseline reproduction reports

[EDMAMBA_REPRODUCIBILITY_REPORT.md](reproduction/EDMAMBA_REPRODUCIBILITY_REPORT.md)
preserves the original audit, including historical environment paths and limitations.
Use [the public setup recipes](environments/README.md) for portable commands.
The import-only compatibility patch mentioned by the audit is included in
[reproduction/patches](reproduction/patches/edmamba-import-compatibility.patch),
with revisions, scope and license information in [UPSTREAM.md](reproduction/UPSTREAM.md).
No third-party source checkout or checkpoint is vendored.

## Frozen downstream validation inputs (22 October 2026)

The follow-up protocol is in
[`protocol_frozen_dsec_downstream_20261022.md`](protocols/protocol_frozen_dsec_downstream_20261022.md).
It fixes three DSEC public-flow training recordings: `thun_00_a`,
`zurich_city_02_a`, and `zurich_city_11_a`. A workspace audit found no previous
DSEC outcomes, and the author confirmed no known external prior use for tuning
this denoiser. DSEC's release-split name does not imply training in this study:
the frozen E-RAFT checkpoint was trained on MVSEC outdoor_day_2. These recordings
provide new downstream task evidence, not per-event sensor-noise labels.

Official inputs are under
`https://download.ifi.uzh.ch/rpg/DSEC/train/<sequence>/` with each remote filename
prefixed by `<sequence>_`. Save them locally without that prefix: `events_left.zip`
(including the rectification map), `optical_flow_forward_event.zip`,
`optical_flow_forward_timestamps.txt`, `image_timestamps.txt`, and `calibration.zip`.
Place each recording's files in
`../downstream/assets/dsec_frozen_20261022/<sequence>/`, or pass `--data-root`.
The [download manifest](native3d_ssm/experiments/analytic_contrast/phase80_dsec_download_manifest.json)
records all exact URLs, byte counts and SHA-256 hashes. Copy it to the data root as
`download_manifest.json` after verifying the downloaded files; the result artifact
also embeds this manifest.
The full recordings contain 440,377,112 events; the released forward-flow files
contain 336 pairs before the protocol's coverage checks. The experiment filters
all raw events before rectification and evaluates every usable annotated pair.

Use the [official E-RAFT source](https://github.com/uzh-rpg/E-RAFT) at revision
`c58ce0524ea0ebfa9849991caafb547f44fe9bfd`, checked out as `../external/E-RAFT`.
Place the authors' [MVSEC 20-Hz checkpoint](https://download.ifi.uzh.ch/rpg/ERAFT/checkpoints/mvsec_20.tar)
at `../downstream/assets/eraft/mvsec_20.tar`. Its expected SHA-256 is
`d9845c0b85a1e5b2d5b664f27706e9a86270afcff42cb16457aeb9c2cbdb9ba2`.
Use an environment with PyTorch/CUDA, NumPy, Numba, scikit-learn, h5py,
`hdf5plugin` for the compressed event arrays, `pypng` for lossless 16-bit RGB PNG decoding,
and the E-RAFT source dependencies. No DSEC-trained checkpoint is used.

```bash
python tools/run_frozen_study.py dsec --stage download
python tools/run_frozen_study.py dsec --stage prepare
python tools/run_frozen_study.py dsec --stage evaluate \
  --output /path/to/reproduction-phase80.json
```

The public launcher locates the unchanged protocols under `protocols/`; the pipeline checks the frozen protocol and checkpoint digests. Do not edit the
protocol or tune on these recordings and continue to describe them as untouched.
The public data and third-party model remain outside this repository.

## Frozen application-level rotation inputs (23 October 2026)

The [rotation cost–quality protocol](protocols/protocol_frozen_rotation_application_20261023.md)
selects all three static-scene DAVIS240C rotation recordings from the
[Event-Camera Dataset](https://www.ifi.uzh.ch/en/rpg/software_datasets/davis_datasets.html):
`shapes_rotation`, `poster_rotation`, and `boxes_rotation`. The workspace audit and
author confirmation established no known prior use in developing this denoiser.
`dynamic_rotation` was excluded prospectively because its moving person violates
the static-scene motion model. The selected hand-held recordings still contain
residual translation; the application estimates rotation rather than full camera pose.

Download the official `<sequence>.zip` files from
`https://files.ifi.uzh.ch/rpg/datasets/davis/` into
`../downstream/assets/davis_rotation_frozen_20261023/`, or pass `--data-root`.
The pipeline uses `events.txt`, `calib.txt` and `groundtruth.txt` from each ZIP.
It records archive, event-text, canonical-payload, calibration and mocap hashes.
Images and IMU signals are not estimator inputs. No checkpoint or training is used.
The estimator is a fixed-time adapter of published rotational contrast maximization,
not a reproduction of a complete CMax-SLAM or ESMD system.

```bash
python tools/run_frozen_study.py rotation --stage download
python tools/run_frozen_study.py rotation --stage prepare
python tools/run_frozen_study.py rotation --stage evaluate \
  --output /path/to/reproduction-phase81.json
```

The runner needs NumPy, SciPy, Numba and threadpoolctl. Run without competing
experiment workloads for comparable timing. It processes the entire original stream
continuously, estimates every prescribed 20-ms aperture at 10-Hz cadence, and
measures three complete interleaved repetitions per arm. Raw source loading,
compilation and external-target metrics are outside the timer; filtering, copied
payloads, preparation and optimization are inside. The input archives and canonical
arrays remain outside the repository. The result artifact contains the full provenance
and all per-update estimates, coverage exclusions and repeated timing results.

The [pre-outcome timestamp clarification](protocols/protocol_rotation_timestamp_adapter_20261023.md)
records the source-format audit. All 378,165,371 events differ from the common
nearest-microsecond clock by at most one nanosecond; the original ZIP/text hashes
and canonical payload hashes are distinct. Both frozen protocol digests are checked.

The completed study used Python 3.12.12, NumPy 1.26.4, SciPy 1.17.1 and Numba 0.64
on an Intel Core i7-12700 under WSL2. These recorded versions describe the frozen
run; the repository's general dependency minimums are not an exact environment
lock. NumPy/SciPy BLAS pools report one thread and the exercised Numba kernels are
serial. An imported but unused scikit-learn OpenMP runtime reports 20 threads;
the artifact preserves the full inventory. Use the recorded environment when
reproducing exact random masks, floating-point outputs and solver behavior.

The [complete result artifact](native3d_ssm/experiments/analytic_contrast/phase81_frozen_rotation_application.json)
has SHA-256 `f4f4af9b96074411c658c0e45aa98ad3db4e081f27d61ce40bdd0dad524eb19c`.
It accounts for all 378,165,371 source events and 45 trials, with 1,794 application
outputs per arm and 1,760 shared eligible mocap targets. No result was excluded for
poor quality or failure to converge. The initial interrupted signature-logging
attempt is preserved in the artifact's provenance and the external data directory;
its incomplete timing and unscored partial estimates are not used in the results.
