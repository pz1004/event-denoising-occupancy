# Occupancy versus multiplicity in training-free event denoising

Research code and reproducibility artifacts for **Rank-Support Filter (RSF)** and the study *Occupancy versus Multiplicity in Training-Free Event Denoising: Representation, Horizon, and Application Cost*. External datasets and pretrained model checkpoints must be obtained from their original providers.

## Overview

RSF replaces local event multiplicity with binary occupancy. For each event it counts occupied pixels of the same polarity in a center-excluded 5 × 5 neighborhood, giving an integer support score from 0 to 24. Larger support is retained first; equal scores preserve input order. The selected score has no global-density term (`beta = 0`) and uses no trained weights.

The research contribution is a controlled account of representation, temporal horizon and complete application cost. Binary support and histogram selection are established primitives. Matched experiments distinguish the information removed by occupancy from changes in decay, directional compression and event admission. Application studies test whether filtering savings translate into useful downstream results.

| Configuration | Admission and selection | Validated scope |
|---|---|---|
| Fixed count | `N = 10,240`, `r = 0.28`; keep `floor(r × N)` events | Integer occupancy versus exact-count filtering, including input, retention and ordered output |
| Fixed time | `T* = 24,302 µs`, `r = 4/5`; keep exactly `floor(4 × n / 5)` events in each nonempty window, including the terminal window | Complete filtering on LED; frozen optical-flow and rotation applications |
| Adaptive fixed count | Separately evaluated floating-point routing and Otsu retention | Retrospective filtering evidence; separate costs |
| Causal scoring | Trailing-window support in a timestamp/polarity map | Scoring only; causal emission policy remains unvalidated |

Whole-block/window occupancy uses lookahead and buffering. Output consists of original events in input order. The five-bit bound applies to the support score, and timestamp removal applies to occupancy state; neither bounds the complete system. No combined adaptive time-window configuration is validated. This is a research implementation, not a live acquisition service.

## Repository layout

```text
native3d_ssm/                 Filter implementations and experiment runners
  experiments/               Recorded numerical results and provenance
reproduction/                Learned-baseline runners and measured results
protocols/                   Four unchanged frozen scientific protocols
tools/run_frozen_study.py     Launcher for fixed-time, DSEC and rotation studies
environments/                General and experiment-specific setup guidance
DATA.md                      External acquisition, directory layout and dataset roles
RELEASE_MANIFEST.json        Payload sizes and SHA-256 digests
SHA256SUMS                   Release file checksums
LICENSE                      License for repository-authored material
```

Module and artifact `phase` numbers are stable identifiers, not an execution order. The [implementation and evidence map](native3d_ssm/README.md) groups the retained material by representation, temporal horizon and application cost. RSF-G denotes the gated global control; RSF-M denotes a decayed-magnitude control at 250 µs, distinct from exact integer multiplicity. Match result rows to their documented score definitions when comparing these controls.

The four frozen protocols and recorded numerical results retain their original bytes. Source hashes, protocol hashes and acquisition metadata remain in the evidence artifacts. Absolute paths and original `plan/` paths inside those records identify the recorded environment; they are not required local directory layouts. The launcher resolves the protocols from `protocols/`. [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json) lists payload sizes and digests, and [SHA256SUMS](SHA256SUMS) also covers that manifest.

## Installation and quick start

Run commands from the release root. Use Python 3.12 or 3.13 for the general installation and the recorded environments for numerical reproduction. PyTorch is required by package exports even though RSF itself has no learned weights.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements.txt
sha256sum -c SHA256SUMS
python tools/run_frozen_study.py rotation --stage synthetic
```

The synthetic example exercises the rotation adapter without acquiring recordings or running a benchmark. The checksum command verifies the supplied file bytes; it does not establish scientific correctness. See [environments/README.md](environments/README.md) for the separate recorded CPU, rotation, DSEC and learned-baseline environments. The studies used different Python, NumPy, PyTorch and CUDA versions; one general installation does not reproduce every recorded timing or floating-point result.

## Results and their limits

The following values describe distinct experiments and must not be pooled into a single benchmark score.

| Study | Result | Interpretation |
|---|---|---|
| Complete fixed-time LED filtering | Equal-scene balanced accuracy: RSF **0.8414**, exact count **0.8449**, EBF **0.8080**. Complete throughput: **37.15**, **20.66**, **7.90** million events/s, respectively | Quality covers all 66 full sources; throughput is the median of nine runs on one LED timing source. RSF is +0.0335 versus EBF and −0.0035 versus exact count; rates are 4.70× and 1.80× those baselines |
| Frozen DSEC optical flow | Equal-recording mean endpoint error: raw **6.7190 px**, RSF **7.0279 px**, exact count **7.1750 px**, EBF **6.8051 px**, random deletion **6.7484 px** | Negative primary downstream result: RSF improves one of three recordings and worsens two |
| Frozen DAVIS rotation | Raw → RSF: **0.9343° → 0.8510°** mean error and **183.28 s → 173.76 s** complete time | Meets the frozen aggregate joint criterion; the improvement is driven by `boxes_rotation`, while `shapes_rotation` and `poster_rotation` worsen |
| Rotation controls | Exact count **0.8433° / 175.24 s**; EBF **0.8292° / 203.29 s**; random deletion **0.9601° / 172.79 s** | Exact count and EBF are more accurate than RSF; filtering-rate savings alone do not predict accuracy or complete application time |

LED quality includes all **111,502,674** original events in the fixed-time study. The common-target diagnostic evaluates **111,134,720** eligible events while preserving original scoring context; its matched 80% balanced accuracies are **0.8417** for RSF and **0.8078** for EBF. That retrospective diagnostic is separate from the full-source cost study.

DSEC covers all **336** forward-flow pairs from three previously unused recordings, with a frozen cross-domain E-RAFT checkpoint. Each recording's endpoint error uses valid-pixel weighting, followed by an equal-recording mean. Rotation covers **378,165,371** events, **1,760** target-covered outputs and **1,794** timed outputs per arm. Unsuccessful solver outputs remain in the evaluation. Complete times sum recording-median totals; stage breakdowns select the actual median-total-time trial within each recording and arm, breaking ties by repeat index.

DND21 and DVSCLEAN selected the development configuration; ED24 had prior contact. E-MLB and LED informed the selection of the support-only score and therefore do not independently confirm that selection. The recorded-noise comparisons remain retrospective and use lookahead and buffering. E-MLB's MESR measures structural concentration, not event classification accuracy. BUAA results use the documented label-free polarity repair; ten repair seeds are an imputation sensitivity study, not ten independent datasets. DSEC and DAVIS provide frozen downstream evidence, not independent per-event sensor-noise labels or complete SLAM validation.

## Reproducing the experiments

Obtain inputs from their original authors using [DATA.md](DATA.md). Datasets, checkpoints, external repositories and their licenses remain separate from this release. Keep the frozen protocol files and scientific implementations unchanged when reproducing the recorded evidence. A changed configuration or new dataset constitutes a new experiment.

Use **fresh output paths** and an otherwise idle machine for timed experiments. The commands below use the documented default dataset layout; each runner's `--help` lists available path overrides. The output directory is local and ignored by Git. The launcher resolves the frozen protocols without editing the scientific runner sources.

```bash
mkdir -p reruns

# Matched-horizon controls, including exact integer counts.
python -m native3d_ssm.phase69_operand_isolation --output reruns/phase69.json
python -m native3d_ssm.phase70_operand_isolation_led --output reruns/phase70.json

# Complete fixed-count filtering; all quality, timing and resource results.
python -m native3d_ssm.phase75_integer_count_cost --stage all --output reruns/phase75.json

# Common-target sensitivity, fixed operating points and repair-seed sensitivity.
python -m native3d_ssm.phase74_common_event_led --output reruns/phase74.json
python -m native3d_ssm.phase77_led_operating_points --output reruns/phase77.json
python -m native3d_ssm.phase78_buaa_repair_sensitivity --output reruns/phase78.json

# Complete fixed-time filtering; default stage alone would omit timings.
python tools/run_frozen_study.py tstar --stage all --output reruns/phase79.json
```

The DSEC pipeline requires the pinned E-RAFT source and verified checkpoint **before its download stage**, because the runner checks the checkpoint at startup. With those prerequisites and the DSEC environment in place:

```bash
python tools/run_frozen_study.py dsec --stage download
python tools/run_frozen_study.py dsec --stage prepare
python tools/run_frozen_study.py dsec --stage evaluate --output reruns/phase80.json
```

In the recorded rotation environment, the synthetic preflight precedes the full recording study:

```bash
python tools/run_frozen_study.py rotation --stage preflight
python tools/run_frozen_study.py rotation --stage download
python tools/run_frozen_study.py rotation --stage prepare
python tools/run_frozen_study.py rotation --stage evaluate --output reruns/phase81.json
```

The download/prepare stages write external input caches, not shipped result artifacts. Timing is hardware- and environment-dependent. Dataset runs may be lengthy and require substantial storage. For runners without an output option, use a disposable copy so that supplied evidence remains untouched. Learned-baseline setup, including the EDmamba compatibility patch, is documented in [reproduction/UPSTREAM.md](reproduction/UPSTREAM.md).

## Citation

If you use this work, please cite the manuscript:

S. Jang and K. Han, “Occupancy versus Multiplicity in Training-Free Event Denoising: Representation, Horizon, and Application Cost,” manuscript prepared for submission to *IEEE Transactions on Circuits and Systems for Video Technology*, 2026.

```bibtex
@unpublished{jang2026occupancy,
  author = {Jang, Sooyoung and Han, Kyuseung},
  title = {{Occupancy versus Multiplicity in Training-Free Event Denoising:
            Representation, Horizon, and Application Cost}},
  year = {2026},
  note = {Manuscript prepared for submission to IEEE Transactions on
          Circuits and Systems for Video Technology},
  url = {https://github.com/pz1004/event-denoising-occupancy}
}
```

For reproducible use of the implementation, also record the repository commit or release used. Cite the original datasets and baseline methods when using their data or results; [DATA.md](DATA.md) and [reproduction/UPSTREAM.md](reproduction/UPSTREAM.md) identify those sources.

## License

Repository-authored code, evidence and documentation are covered by [LICENSE](LICENSE). External datasets, learned models, checkpoints and reference implementations retain their original terms; they are not redistributed by this release.
