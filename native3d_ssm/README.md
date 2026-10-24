# Implementation and evidence map

This directory contains the filter implementations and numerical evidence for the
occupancy-versus-multiplicity study. Module and artifact numbers are stable identifiers;
the sections below group them by scientific purpose.

## Score and configurations

**RSF** counts occupied same-polarity taps in a center-excluded radius-2 neighborhood.
The support count `S` lies in `0..24`; the floating-point noise score `-log1p(S)` preserves
the same ranking. Fixed-count filtering uses `N = 10,240`, `r = 0.28`; fixed-time filtering
uses `T* = 24,302 µs`, `r = 4/5`. Each block or nonempty window retains its exact floored
quota, including the terminal window in the fixed-time configuration. Equal scores
preserve input order, and output consists of original event payloads. There is no global
term in RSF (`beta = 0`).

Whole-block/window occupancy reads future events and requires buffering. Forward/reverse
floating-point extraction and two-bit-plane construction produce the same occupancy
operand. The five-bit bound concerns the score; buffering and selection require additional
storage. Timestamp removal applies to occupancy state, not the complete pipeline.

**RSF-M** uses decayed ring magnitude at 250 µs with `beta = 0`; it is distinct from exact
integer multiplicity. **RSF-G** adds the gated global term. Some stored control rows use
`adopted`, `support`, or `log_support|float` for RSF-G; identify each row from its score
and gate configuration, not its short label. A gated row must not be substituted for RSF.

| Module | Implementation role |
|---|---|
| `core.py` | Floating-point forward/reverse event-state extractor |
| `evaluate_adaptive_beta.py` | Reused dataset readers, `ring_support`, and gate controls |
| `matched_horizon.py` | Directional amplitudes, one log per direction, exact integer flat counts |
| `phase52_bitmap_profile.py` | Bit-plane construction, occupancy readout and exactness |
| `phase75_integer_count_cost.py` | Integer fixed-count filters, ordered emission and resource replay |
| `phase79_tstar_cost_quality.py` | Complete fixed-time occupancy/count/EBF filtering and exact quotas |
| `causal_occupancy.py` | Trailing-window occupancy and its reference implementation |
| `ebf.py` | EBF implementation for matched comparisons |
| `retention.py` | Adaptive retention and retention controls |
| `data.py` | Labelled text and raw AEDAT readers |

`retention.gated_retention` uses floating-point correlation routing and an Otsu branch.
Those costs are separate from fixed retention. Causal occupancy uses EBF's timestamp/polarity
map without a scoring block; its emission policy remains unvalidated. A forward-only
block score is a different ablation, which resets at block boundaries and retains a block.
No combined adaptive time-window configuration is validated. The complete fixed-time
filter is a research API, not a live acquisition service.

## Representation and score-selection evidence

| Modules / artifact identifiers | Question or measurement |
|---|---|
| `phase3`, `phase7`, `phase8`, `phase11`, `phase14` | Gate-key and operand selection, radius attribution and score quantization |
| `phase0`, `phase17`, `phase44` | Occupancy, decay invariance and class-conditional support properties |
| `phase5b`, `phase15`, `phase16`, `phase18`, `phase19`, `phase23`–`phase29`, `phase50` | Causal-score, retention, density, missingness, random-deletion and statistical diagnostics |
| `phase51_ed24_scene_clustered`, `phase54_led`, `phase55_emlb_ungated`, `phase62_buaa_campus` | Dataset-specific score comparisons and scene/source summaries |
| `phase63_gap_predictors` | Descriptive associations between occupancy and comparator gaps |
| `phase78_buaa_repair_sensitivity` | Polarity-repair sensitivity with paired inputs and five arms |

Artifacts under `experiments/analytic_contrast/` preserve per-cell values and aggregation
metadata. Output names need not match the generator's number: `phase72_*` is produced by
phase51/54/62, and `phase73_gap_predictors_beta0.json` by phase63 with the ungated arm.
The corresponding source files are present.

DND21 and DVSCLEAN selected the operand, radius and fixed retention. ED24 had development
contact. E-MLB and LED informed the selection of the support-only score, so they do not
independently confirm that selection. E-MLB's MESR describes structural concentration,
not event classification accuracy. Learned-baseline runners and their results are under
`reproduction/`; [UPSTREAM.md](../reproduction/UPSTREAM.md) specifies their inputs and
aggregation restrictions. Shared model modules remain dependencies of package imports
and baseline readers; their presence does not define additional validated RSF configurations.

For the ED24 in-range paired comparison, supply the retained per-block inputs explicitly
instead of the pairing helper's default paths:

```bash
mkdir -p reruns
python -m native3d_ssm.phase35_ed24_paired \
  --ours native3d_ssm/experiments/analytic_contrast/phase36_ed24_inrange_ours.json \
  --theirs native3d_ssm/experiments/analytic_contrast/phase37_ed24_inrange_edformer.json \
  --output reruns/ed24-inrange-paired.json
```

The BUAA sensitivity study uses seeds 20260825–20260834 on six first-3M-event sources.
All five arms share the same input per sequence/seed and the common 2,990,080-event
complete-block prefix. The two clean sequences reuse scores only after independent
input reloads and hash checks. Both AUC aggregations retain the five-arm ordering for
all ten seeds, and no individual-sequence contrast reverses. Ranges and sample standard
deviations describe imputation sensitivity, not ten independent datasets.

## Temporal horizon and retention comparisons

| Module | Evidence |
|---|---|
| `phase66_time_budget` | Fixed-duration intervention and forward-only block ablation |
| `phase67_matched_causal_led`, `phase68_matched_causal_labelled` | Trailing-window occupancy versus EBF |
| `phase69_operand_isolation`, `phase70_operand_isolation_led` | Matched-horizon representation controls with exact integer counts |
| `phase74_common_event_led` | Pairwise and six-partition common-event sensitivity |
| `phase76_led_equal_scene_statistics` | Equal-scene contrast points, intervals and tests |
| `phase77_led_operating_points` | Matched retained-event budgets and descriptive retention curves |

Phase69/70 schema 2 uses one logarithm per direction and exact integer flat counts.
It covers all 26 development/BUAA units and all 66 LED scenes under both partitions.
BUAA uses repair seed 20260825. `audit_matched_horizon_results` verifies per-unit
contrasts, single-log anchors, precision and tie sensitivity.

Phase74 preserves source streams, windows and scoring context, applying shared eligibility
masks only to evaluation. It reproduces all 1,584 native per-scene anchors, records
source/mask/code hashes and interval membership, and uses 200,000 paired scene bootstrap
and sign-flip draws. The primary comparison is fixed N versus the predeclared 24,302-µs
window; the all-six-partition sweep is exploratory.

Phase77 selects exactly `floor(r * n_target)` common targets per original window using
decimal-rational arithmetic and arrival-order ties. Its eleven retention fractions,
primary `r = .80` and reference `r = .28` were specified before the run. All 66 scenes
and 111,134,720 common targets are included. At `r = .80`, RSF minus EBF balanced accuracy
is +.0338 at T* but −.1609 at fixed N. This is retrospective common-target selection,
separate from timed full-source filtering. The descriptive sweep does not select an optimum.
Original AUC and `r = .28` anchors are checked.

## Filtering and complete application cost

| Module | Evidence and boundary |
|---|---|
| `phase52_bitmap_profile` | Operand-kernel timings and exactness; not complete filter timing |
| `phase71_end_to_end_cost` | Float-layout storage ledger and causal-kernel timings |
| `phase75_integer_count_cost` | Complete fixed-count quality, timing, array capacities and FIFO replay |
| `phase79_tstar_cost_quality` | Complete fixed-time filtering, full-source LED quality and timing |
| `phase80_frozen_dsec_downstream` | Frozen five-arm E-RAFT optical-flow comparison |
| `phase81_frozen_rotation_application` | Frozen five-arm rotation estimation and complete application timing |

Storage ledgers describe specified layouts; they are not measured peak process memory.
No hardware synthesis is included. Phase75 times common fixed-N/fixed-r input, scoring,
retention and output. Its packed-layout estimates are separate from implemented array
capacities. Quality covers all 66 LED scenes; timing and FIFO observations use specified
DND21/LED prefixes.

Phase79 processes every original event, including sparse windows and terminal tails.
EBF history remains continuous; its cutoff uses partition and stable scanning, and
exact-count widths adapt to window size. The full LED quality study and fixed DND21/LED
timing prefixes share the same configuration. Timing includes input validation, setup,
window discovery, copying, scoring, selection and emission. Explicit-array ledgers and
external FIFO queues remain separate from process RSS. The LED quality cohort contains
111,502,674 events across 66 sources; reported LED throughput is the median of nine runs
on one timing source.

Phase80 uses all 336 usable forward-flow pairs in three DSEC recordings unused for this
denoiser before the frozen protocol. The unchanged MVSEC E-RAFT checkpoint and full-resolution
preprocessing are fixed; flow labels do not enter filtering. Filter windows have exact
quotas, while model intervals may contain different retained counts and normalize retained
time endpoints. Each recording's EPE uses valid-pixel weighting, followed by an equal-recording
mean. The negative primary result is 6.7190 px for raw events versus 7.0279 px for RSF;
RSF improves one recording and worsens two.

Phase81 fixes the denoisers on three previously unused static-scene DAVIS rotation
recordings. It estimates 20-ms relative rotations at 10 Hz and charges complete filtering,
input preparation and optimization. Mocap supplies external targets; all 1,794 estimates
execute and 1,760 satisfy target coverage, including unsuccessful solver outputs.
Three interleaved repetitions per arm give 45 trials with identical predictions.
RSF's equal-recording mean error is 0.8510° versus 0.9343° raw; summed recording-median
times are 173.76 versus 183.28 s. The aggregate joint criterion passes, driven by
`boxes_rotation`, while `shapes_rotation` and `poster_rotation` worsen. Exact counts are
more accurate at 175.24 s. Stage breakdowns select each recording/arm's actual median-total-time
trial, breaking ties by repeat index. Neither application study validates independent
per-event recorded-noise labels, complete SLAM or live-deadline operation.

## Running studies

Run modules from the repository root with datasets from [DATA.md](../DATA.md) and the
appropriate [environment](../environments/README.md). Use fresh output paths; run modules
with fixed output paths, including phase76 and `audit_matched_horizon_results`, only in a
disposable copy. The root [README](../README.md#reproducing-the-experiments) lists filtering
and sensitivity commands.

```bash
mkdir -p reruns
python tools/run_frozen_study.py tstar --stage all --output reruns/phase79.json
python tools/run_frozen_study.py dsec --stage evaluate --output reruns/phase80.json
python tools/run_frozen_study.py rotation --stage evaluate --output reruns/phase81.json
```

DSEC and rotation evaluation require downloaded and prepared inputs. DSEC also requires
the pinned E-RAFT source and checkpoint before any stage. The launcher resolves the four
unchanged protocols under `protocols/`; source and protocol hashes remain in the recorded
results. Recorded absolute paths identify the experiment environment, not required local
paths. Package exports import learned-model modules, so PyTorch is required even for
nonlearned occupancy modules. Use `python tools/run_frozen_study.py rotation --stage synthetic`
for a data-free example and `sha256sum -c SHA256SUMS` to verify release file integrity.
