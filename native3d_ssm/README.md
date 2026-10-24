# native3d_ssm

Implementation and experiment archive for the occupancy-versus-multiplicity study.
The manuscript is not included in this public package. Historical module and artifact names
are retained for provenance; some predate the adoption of support-only RSF.

## Canonical score

**RSF** counts occupied same-polarity taps in a radius-2 ring over each 10,240-event block.
The count `S` lies in `0..24`, and the noise score is `s = -log1p(S)`. Fixed retention keeps
the lowest `floor(0.28 * N)` scores, equivalently the highest counts, with stable ordering
for ties. There is no global term: `beta = 0`.

The block-local count reads future events and requires a complete block. The original
forward/reverse floating-point extractor and the two-bit-plane implementation produce the
same operand. The five-bit bound concerns the score; buffering and selection need
additional storage.

**RSF-M** replaces the count with the decayed ring magnitude at 250 µs, still with
`beta = 0`. **RSF-G** adds the gated global term and is a non-adopted control. Old artifacts
may call RSF-G `adopted`, `support`, or `log_support|float`; current manuscript tables name
their actual source arms explicitly. Do not substitute a gated row for RSF.

The streaming count in `causal_occupancy.py` uses EBF's timestamp/polarity map and a trailing
time window. It has no block buffer, but keeps that map's memory cost. A forward-only
block score is a different ablation: it resets at block boundaries and retains a block.

## Implementation modules

| Module | Role |
|---|---|
| `core.py` | Original floating-point event-state extractor; forward/reverse features |
| `evaluate_adaptive_beta.py` | Historical experiment plus reused dataset readers and `ring_support` |
| `matched_horizon.py` | Raw directional amplitudes, one log per direction, exact integer flat counts |
| `phase75_integer_count_cost.py` | Matched integer block filters, ordered emission and resource replay |
| `phase79_tstar_cost_quality.py` | Complete time-window occupancy/count/EBF filtering; exact quotas and ordered original payloads |
| `phase81_frozen_rotation_application.py` | Exact rotational contrast-maximization adapter, frozen DAVIS cohort, complete application timing and external motion errors |
| `phase52_bitmap_profile.py` | Bit-plane construction, occupancy readout, exactness and profiling |
| `causal_occupancy.py` | Causal trailing-window count and reference implementation |
| `ebf.py` | EBF implementation used for matched comparisons |
| `retention.py` | Separately evaluated adaptive retention and historical alternatives |
| `data.py` | Labelled text and raw AEDAT readers |

`retention.gated_retention` routes on a floating-point correlation, and its Otsu branch
uses floating-point arithmetic. Fixed retention does not need that route. There is no
standalone production denoiser executable here. Phase79 supplies a tested research
API for complete time-window filtering and copied ordered outputs; it is not a
live acquisition service.

## Current revision experiments

Run modules from the repository root. Data-dependent runs need the datasets in
[DATA.md](../DATA.md). Supply a separate `--output` path when the runner supports it. Phase76 and
`audit_matched_horizon_results` use fixed output paths; invoke these only in a
throwaway copy if the shipped artifacts must remain unchanged.

| Module | Evidence |
|---|---|
| `phase51_ed24_scene_clustered` | Scene-clustered ED24 contrasts; `--left` / `--right` select arms |
| `phase52_bitmap_profile` | Bit-plane exactness and operand-kernel timings |
| `phase54_led` | LED score arms and fixed/adaptive retention metrics |
| `phase55_emlb_ungated` | Ungated E-MLB comparison |
| `phase62_buaa_campus` | BUAA polarity audit and repaired-stream controls |
| `phase63_gap_predictors` | Descriptive occupancy/gap associations; explicit `--arm` |
| `phase66_time_budget` | LED fixed-duration intervention and forward-only block ablation |
| `phase67_matched_causal_led` | Causal occupancy versus EBF on LED |
| `phase68_matched_causal_labelled` | Causal comparison on development and BUAA data |
| `phase69_operand_isolation` | Matched-horizon controls on development and BUAA data |
| `phase70_operand_isolation_led` | Matched-horizon controls on LED |
| `phase71_end_to_end_cost` | Historical float-layout memory accounting and separate causal-kernel timing |
| `phase75_integer_count_cost` | Matched integer count/occupancy quality, complete block-processing timing, memory and FIFO replay |
| `phase76_led_equal_scene_statistics` | Equal-scene LED point estimates, intervals and tests |
| `phase74_common_event_led` | Pairwise and six-partition common-event LED sensitivity; equal-scene event-level AUC |
| `phase77_led_operating_points` | Matched LED retained-event budgets, eleven-point curves and paired scene contrasts |
| `phase78_buaa_repair_sensitivity` | Ten BUAA repair seeds, five paired arms, both AUC aggregations and subgroup summaries |

Artifacts named `phase72_*` are produced by the revised phase51/54/62 modules, and
`phase73_gap_predictors_beta0.json` by phase63 with the ungated arm. They are output
versions, not missing modules. Current per-cell values and their aggregation metadata remain available in the stored artifacts.

The phase71 memory totals are a ledger for specified layouts, not measured peak process
memory. The phase52 and phase71 timings price operand/causal kernels rather than a complete
packed filter with input, retention, and output. No hardware synthesis is included.

Phase69/70 schema 2 correct the earlier double-log error and use exact integer flat
counts; obsolete `*.legacy_mislabelled.json` evidence is excluded from this public package. All 26 development/BUAA
units and 66 LED scenes under both partitions were rerun. BUAA uses the main benchmark
polarity-repair seed 20260825. `audit_matched_horizon_results` checks per-unit contrasts,
valid single-log anchors, precision and tie sensitivity.

Phase75 times common fixed-N/fixed-r input, score, retention and output work. Its packed
layout estimates are separate from implemented array capacities. Quality covers all 66
LED scenes; timing and FIFO latency/queue observations cover specified DND21/LED prefixes.
Phase76 aligns LED contrast points, intervals and tests on equal scene weights.

Phase74 preserves phase66's source streams, windows and scoring context, applying shared
eligibility masks only to AUC evaluation. It reproduces all 1,584 native per-scene anchors,
records interval membership and source/mask/code hashes, and uses 200,000 paired scene
bootstrap and sign-flip draws. The primary comparison is fixed N versus the pre-declared
24,302 us window; the all-six-partition sweep is exploratory. Run it with the LED dataset
available and a separate `--output` path to preserve the shipped artifact.

Phase77 preserves phase74's full scoring contexts, selecting exactly
`floor(r * n_target)` common targets per original window using decimal-rational arithmetic
and arrival-order ties. The eleven retention fractions, primary `r=.80` and reference
`r=.28` were recorded before the run. All 66 scenes and 111,134,720 common targets are
included. At `r=.80`, RSF minus EBF balanced accuracy is +.0338 at T* but −.1609 at
fixed N. This is offline common-target selection, not a timed full-source implementation.
The descriptive sweep does not select an optimum; original AUC and r=.28 anchors are checked.

Phase78 repeats repair with seeds 20260825–20260834 on all six first-3M-event BUAA sources.
All five arms see identical input per sequence/seed and use the common 2,990,080-event
complete-block prefix. The two clean sequences reuse scores only after independently
reloading and hashing each seed's input. The artifact retains per-sequence and per-seed
metrics, paired contrasts, and clean/corrupt groups. Ranges and sample SD describe repair
sensitivity, not independent-observation inference. Both AUC aggregations preserve the
five-arm benchmark ordering at all ten seeds; no individual-sequence contrast reverses.

```bash
python3 -m native3d_ssm.phase77_led_operating_points --output /tmp/rsf-led-operating-points.json
python3 -m native3d_ssm.phase78_buaa_repair_sensitivity --output /tmp/rsf-buaa-repair-seeds.json
```

Phase79 processes every original event in T*=24,302-us windows at r=4/5, including
sparse windows and terminal tails. EBF history remains continuous; its cutoff uses
partition and stable scanning. Exact-count widths adapt safely to window size.
The full LED quality study and the fixed DND21/LED timing prefixes use the same
configuration. Timing includes input validation, setup, window discovery, copying,
scoring, selection and emission; explicit-array ledgers and external FIFO queues
are separate from process RSS. The implementation is frozen for phase80 inputs.

Phase80 evaluates raw, occupancy, exact count, EBF and one frozen random-deletion
control on all usable forward-flow pairs in three previously unused DSEC recordings.
It uses the unchanged MVSEC E-RAFT checkpoint and full-resolution DSEC preprocessing.
No flow labels enter filtering. Exact quotas apply to filter windows, while model
intervals can contain different retained counts and normalize retained time endpoints.
The primary EPE contrast is RSF minus raw, summarized equally over recordings.
See the prospective protocols in `../protocols/` and acquisition instructions in [DATA.md](../DATA.md). Use the public launcher for phases 79–81, as shown below.

Phase81 holds the same denoisers fixed on all three static-scene DAVIS rotation
recordings, unused for this denoiser before the prospective protocol. It estimates
20-ms relative rotations at 10 Hz and charges complete filtering, input preparation
and optimization across the full source stream. Mocap supplies external targets;
all 1,794 estimates execute and 1,760 satisfy target coverage. Three interleaved
repetitions per arm give 45 full trials with identical predictions. RSF's
equal-recording mean error is 0.8510° versus 0.9343° raw, and summed median time is
173.76 versus 183.28 s. The aggregate criterion passes, driven by boxes; exact
counts are more accurate at 175.24 s. All solver budget stops remain in quality.
Neither the cost nor the motion result is a full SLAM or live-deadline claim.

## Running the frozen studies

Install the appropriate environment from [the setup guide](../environments/README.md).
Run these commands from the repository root, using fresh output paths:

```bash
mkdir -p reruns
python tools/run_frozen_study.py tstar --stage all --output reruns/phase79.json
python tools/run_frozen_study.py dsec --stage evaluate --output reruns/phase80.json
python tools/run_frozen_study.py rotation --stage evaluate --output reruns/phase81.json
```

The DSEC and rotation commands assume their inputs have been downloaded and
prepared following [DATA.md](../DATA.md). DSEC also requires the pinned E-RAFT
source and checkpoint before any stage. The launcher locates the unchanged frozen
protocol files under `protocols/` without editing scientific source files.
The package initializer imports historical learned models, making PyTorch an
import dependency even for the nonlearned occupancy modules. Use
`python tools/run_frozen_study.py rotation --stage synthetic` for a data-free
rotation example and `sha256sum -c SHA256SUMS` to verify the release file bytes.

## Historical code and selection history

`model.py`, `advanced_model.py`, `robust.py`, and the training/evaluation modules preserve
an earlier learned design. `analytic.py` and `filter_analytic_aedat.py` implement an older
global/local contrast, not the adopted RSF score. Gate experiments, multi-timescale studies,
and fixed-point studies remain because earlier papers and the supplement cite them.
A historical docstring calling its score canonical or adopted is not the current definition.

The operand, radius, and fixed retention were selected on DND21/DVSCLEAN. ED24 has prior
development contact; E-MLB and LED informed the later decision to retire the gate. The
new phase80 study adds previously unused downstream task evidence for the frozen
T*/r=.80 configuration. Its three recordings have flow targets, not per-event noise
labels; it does not establish universal transfer or a live deployment deadline.
