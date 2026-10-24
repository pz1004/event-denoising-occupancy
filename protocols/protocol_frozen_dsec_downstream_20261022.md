# Frozen downstream validation on previously unused DSEC recordings

Recorded 22 October 2026 before evaluation of any DSEC downstream task outcome.
This is an internal prospective specification, not an external preregistration.
The cohort was chosen from official download metadata for raw-event/flow-label
availability, manageable size, and distinct named recordings. No task outcome,
filter retention statistic or visual inspection selected these recordings.

## Prior-use audit and frozen cohort

The existing repository and parent-workspace experiments used DVS Gesture test
subjects and MVSEC outdoor_day_1, including prior r=.80 runs. Neither is treated
as untouched here. A local audit found DSEC mentioned only as a proposed extension
and in upstream code, without downloaded recordings or completed outcome files.
The user additionally confirmed no known use of these three recordings outside
the workspace for developing or tuning this denoiser.

The fixed cohort is the public DSEC flow training-set recordings thun_00_a,
zurich_city_02_a and zurich_city_11_a. The training designation refers to DSEC's
release split; these recordings are not used for training or selecting the model
or denoiser in this study. Metadata lists respectively 41, 64 and 231 forward
flow pairs, before deterministic source-coverage checks. Evaluate all usable
forward pairs under the same rule; record every exclusion and its reason.
The unit of descriptive comparison is the named recording. These are three
recordings, not 336 independent acquisition units, and acquisition-session
independence is not assumed without additional metadata.

Use official raw left-camera events, rectification maps, forward flow targets and
target timestamps. Store exact download URLs, file sizes and SHA-256 digests.
Do not inject synthetic noise. Natural raw events may contain sensor noise;
task labels do not provide event-wise noise ground truth.

## Frozen downstream model and preprocessing

Use the existing released E-RAFT MVSEC 20-Hz checkpoint, SHA-256
`d9845c0b85a1e5b2d5b664f27706e9a86270afcff42cb16457aeb9c2cbdb9ba2`.
Its documented training source is MVSEC outdoor_day_2, not DSEC. This is
cross-domain evaluation of both a frozen downstream model and a frozen denoiser;
no DSEC model checkpoint, training, fine-tuning, model selection or calibration
is permitted. Upstream E-RAFT revision:
`c58ce0524ea0ebfa9849991caafb547f44fe9bfd`.

Use the official DSEC standard (no warm-start) evaluation representation: native
640x480 rectified coordinates, 15 temporal bins, 100-ms event intervals, and
normalization over nonzero voxels. Feed the preceding and current intervals for
each forward-flow pair using the upstream time convention. Verify this contract
against the upstream loader and independent synthetic voxel checks before result
runs. Preserve event timestamps as integer microseconds until the standard voxel
conversion. Record endpoint normalization and rectification validity rules. Use
identical targets, valid-pixel masks, time intervals and model preprocessing for
all conditions; never infer validity from retained events or prediction quality.
Full-resolution memory feasibility may be tested with synthetic tensors before
outcome access. Any required departure from this specification must be recorded
before the first task outcome and must not be chosen using outcome quality.

## Frozen event filtering

Use phase79's tested implementation and its protocol:
[complete T* protocol](protocol_tstar_fullstream_20261022.md).
T=24,302 us, r=4/5, radius 2, beta=0. Filter the original unrectified event stream
continuously from its first event with first-event-anchored scheduled windows,
including every nonempty low-count and terminal window. Filter before cropping,
rectification or downstream voxelization. Keep floor(4*n/5), break score ties by
arrival index and preserve original payloads and output order. EBF state persists
across all windows. No chunk/file/flow-label boundary resets the filter.

Five fixed conditions:
1. Raw unfiltered events (100% retention).
2. Adopted RSF occupancy.
3. Exact whole-window integer multiplicity.
4. EBF, radius 2 and tau=32ms, using the same windowed retention policy.
5. Uniform random retention with the identical per-window integer quota, fixed
   seed 20261022 and deterministic independent sequence streams in cohort order.

Random retention is a deletion control, not a denoiser. No event labels or flow
labels enter any filter. A T* window may read future events within that window;
its output requires window closure. The result is offline blockwise preprocessing,
not a causal live optical-flow system. Save source and selected-index hashes,
window counts, quotas, retention, processing times and source-code digests.

## Endpoints and reporting

Primary contrast: RSF minus raw EPE; negative favors RSF. EBF, exact multiplicity
and random retention are named secondary paired comparisons.

Primary endpoint: mean endpoint error (EPE) over valid flow pixels within each
recording, then an equal-weight mean over the three recordings. Report every
per-recording EPE and paired RSF differences against raw, EBF, exact multiplicity
and random retention. Secondary endpoints: valid-pixel percentages with EPE>1px
and EPE>3px, frame/pixel counts, actual retained counts, and separated filter/model
processing times. No per-frame or per-pixel significance test will be presented as
independent-recording inference. With three recordings, emphasize descriptive
paired results rather than an unstable population confidence interval.

The practical question is whether the fixed filter improves downstream EPE beyond
raw input and matched random deletion, and how it compares with the same-budget
baselines. No post-hoc noninferiority margin, successful-subset selection or best
retention is allowed. Preserve all results, including degradation. This study can
support limited new-recording downstream transfer; it cannot establish independent
per-event denoising accuracy, universal transfer, real-time deployment or a new
noise-label benchmark.
