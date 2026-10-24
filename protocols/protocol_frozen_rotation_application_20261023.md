# Frozen application-level rotation cost–quality study

23 October 2026. This internal prospective specification precedes download or
inspection of the new recordings and all task outcomes. It is not an external
preregistration. The previous frozen DSEC study is retained unchanged, including
its negative primary result; it motivated seeking an application whose computation
scales with the retained event population, not selection of favorable recordings.

## Cohort and provenance

Use all three static-scene DAVIS240C rotation recordings listed by the official
Event-Camera Dataset: shapes_rotation, poster_rotation and boxes_rotation, in that
order. Their metadata identify increasing-speed, predominantly rotational camera
motion. The published sequences are hand-held and contain residual translation;
this is a limitation of the rotation model, not exact pure-rotation ground truth.
Exclude dynamic_rotation before outcomes because its moving person violates the
static-scene model; no result on it selects this cohort.

A repository/parent-workspace audit found no previous use of these named recordings
in denoiser development, and the user confirmed no known external prior use on
23 October 2026. These are previously unused data for this denoiser, not data
unknown to the research literature. Use the official events, camera calibration
and motion-capture camera poses. No synthetic noise is added. Save exact download
URLs, byte counts and SHA-256 hashes. Do not inspect task-quality outcomes to
choose a prefix, speed regime, subset, algorithm setting or recording.

Official source: https://www.ifi.uzh.ch/en/rpg/software_datasets/davis_datasets.html
Files: https://files.ifi.uzh.ch/rpg/datasets/davis/<sequence>.zip
Published dataset reference: https://arxiv.org/abs/1610.08336

## Application and common temporal contract

The application estimates camera angular motion at a fixed 10-Hz update cadence,
using a 20-ms event aperture for each update. For original first timestamp t_first,
use every interval [t_first + k*100000, t_first + k*100000 + 20000) microseconds
whose end is no later than the last source timestamp, for integers k>=0. This
covers the full recording timeline with a fixed cadence, not every possible
20-ms window or every event as an estimator input. All original events are still
processed by a continuously operating denoiser, including gaps and terminal tails.

This short acquisition aperture makes the constant-angular-velocity approximation
more appropriate than the 100-ms output cadence; cadence is fixed for bounded
experimental compute before outcomes, not presented as a validated control-loop
requirement. The application is a motion-estimation frontend, not a complete SLAM
system or a state-of-the-art CMax benchmark reproduction.

All arms have identical task intervals, output cadence, absolute time references,
calibration, solver and external targets. Use t_event - t_interval_start in seconds;
never renormalize by retained-event endpoints. Do not cap input events or increase
a filtered arm's aperture to replace discarded events. Include low-count and
constant-objective intervals in the output/quality record. An incomplete terminal
application aperture is omitted by the same source-coverage rule in every arm.

## Frozen denoising arms

1. Raw, unfiltered input (100% retention).
2. RSF occupancy, radius2, beta0.
3. Exact whole-window integer multiplicity.
4. EBF, radius2, tau32000us.
5. Uniform random deletion, with one fixed realization.

Use the phase79 implementation unchanged: first-source-event-anchored half-open
T*=24302us windows, exact quota floor(4*n/5), stable arrival ties, copied original
payloads and ordered output, every nonempty sparse/tail window included. EBF state
is continuous and updates on rejected original events too. No task, file, chunk
or timing-repeat subinterval resets a filter inside a recording. Reset complete
state at a new recording or a fresh full timing repetition.

Random deletion uses NumPy SeedSequence(20261023).spawn(3), assigning one child
stream to each recording in the stated cohort order and resetting it identically
for each repeat. It uses exact per-window quotas and
ordered original payloads; use an efficient membership-mask scan rather than a
full sort of random scores. The random control is one realization, not an expected
performance over seeds. All four filtered arms match source-window counts, not
necessarily counts inside overlapping application apertures.

No target pose, angular velocity, event-noise label or fitted threshold enters any
filter. Within-window lookahead remains necessary. This is offline buffered
preprocessing and does not establish a live emission deadline.

## Fixed motion estimator

Use a transparent fixed-time adapter of rotational contrast maximization, following
Gallego et al. and the CMax-SLAM frontend's bilinear accumulation/variance approach.
It is not a new motion algorithm or a reproduction of the original fixed-event
packetization. Related joint denoising/motion methods such as ESMD already exist.

- Native240x180 image. Read released fx,fy,cx,cy,k1,k2,p1,p2,k3. Build a lookup
  table of undistorted normalized camera bearings using the Brown–Conrady model
  and a fixed20-step Newton inverse with the exact2x2 distortion Jacobian,
  initialized by distorted normalized coordinates; verify reprojection synthetically.
- Camera-to-world convention R_WC(t)=R_WC(t0) Exp([omega]x*(t-t0)). Backwarp a
  calibrated ray to t0 with the exact positive Rodrigues rotation. Verify all
  three axes, nonidentity initial poses and quaternion sign equivalence on
  synthetic data before any task run.
- Project into the same240x180 pinhole image, use polarity-free bilinear splats,
  discard nonpositive-depth rays, and give zero contribution to out-of-bounds
  splat neighbors. No crop, spatial rescaling or target-dependent mask.
- Gaussian blur sigma1pixel, truncate4 (radius4), constant-zero border. Normalize the IWE by the
  estimator's original input event count and multiply by the image pixel count;
  maximize its variance. The normalization is independent of candidate motion
  and prevents raw/filtered event counts changing the scale of stopping criteria.
- Exact analytic Rodrigues/projection/bilinear derivatives, verified against
  independent central differences; the zero-motion series uses stable Taylor
  limits. Keep the full derivative of border-adjacent contributions.
- L-BFGS-B with each omega component bounded to[-35,+35]rad/s, maxiter20,
  maxfun60, ftol1e-9, gtol1e-5. Use the same implementation and tolerances for
  every arm. The package's line-search evaluation semantics apply; record actual
  function/gradient counts and any budget overshoot. No post-outcome budget change.
- Initialize zero at each recording. Each arm subsequently uses only its own
  previous finite estimate. Never use raw-arm estimates or ground truth to
  initialize another arm. There is no fitted time lag or trajectory alignment.
- Fewer than two events: emit the previous estimate with fallback status-1;
  all input timestamps identical: hold previous with status-3. A finite solver
  result is emitted even when convergence fails; preserve status and count failures.
  A nonfinite optimizer result or objective holds the previous finite estimate
  with status-2. All fallbacks remain in quality and are explicitly counted.
  Invalid source/calibration inputs are fatal validation errors, not outcome exclusions.

Source references:
https://rpg.ifi.uzh.ch/docs/CVPR18_Gallego.pdf
https://github.com/tub-rip/cmax_slam
https://openaccess.thecvf.com/content/ICCV2025/html/Shiba_Simultaneous_Motion_And_Noise_Estimation_with_Event_Cameras_ICCV_2025_paper.html

## Independent task quality

Released mocap poses map camera coordinates to world coordinates and already
include the dataset's hand-eye calibration. Normalize quaternions and use SO(3)
SLERP at each fixed interval endpoint. Primary per-output error is the geodesic
angle, in degrees, between Exp(estimated_omega*0.02) and R_WC(t0)^T R_WC(t1).
No IMU signal, optimized IWE contrast or event concentration metric is a target.

Produce application estimates regardless of mocap availability. Evaluate quality
where both endpoints lie within the released pose range and their interpolation
brackets are at most20ms apart. Record every target-coverage exclusion and its
reason; it cannot change application execution, warm starts or timing. No exclusion
based on event count, optimized objective, convergence, estimated error or motion
magnitude is permitted.

Primary quality is mean relative-rotation angle per recording, then an equal-weight
mean over the three recordings. Also report each recording's RMSE, median and p95
angle, angular-rate-equivalent error (angle/0.02), convergence/fallback counts,
optimizer iterations/evaluations and all per-update estimates/errors. Three named
recordings do not justify frame-level population inference or prove independent
acquisition sessions. No population confidence interval or significance claim.

## Complete application cost

The primary timer starts with one resident original event array and released
calibration parameters, before source validation, filter initialization, lookup
construction, selection or task-input preparation. It stops only after both the
final motion estimate has been written to its output array and all original source
events have been consumed, including the trailing gap/tail after the last aperture. Include full-source continuous
filtering, original-payload copying/ordered output, task-window collection, bearing
lookup, objective/gradient work, optimizer overhead and estimate emission. Raw is
an identity adapter and may use views; do not add unnecessary copying to slow it.
Filtered event arrays or indices must not be cached across primary timing repeats.
Exclude file download/load, JIT compilation, external-target interpolation, metric
calculation, diagnostic hashing and final JSON serialization, and list these
exclusions explicitly. Stage timers are descriptive decompositions of total cost.

Load one full recording into RAM before its runs. Warm JIT kernels and the estimator
only on fixed synthetic data, including both uint16 and uint32 filter paths.
Record kernel signatures before and after timed runs to detect late compilation.
Stop other experiment workloads during timing. Run
three complete repetitions per arm and recording, cyclically interleaving arm order
from raw/occupancy/multiplicity/EBF/random, left-shifted by
(recording_index + repetition_index) mod5 for zero-based indices. This interleaving
is not a fully position-balanced design. Reset all filter/optimizer/random state
for every full repeat. Record the order, environment, all three raw wall times,
median and range. Quality uses the first repetition; require repeated retained
counts and numerical motion predictions to agree, rather than selecting a best
quality or fastest-only run. Timing repeats are not additional quality observations.

Primary aggregate cost is the sum of each recording's median full-path wall time.
Report raw and every filtered baseline, speed ratio, total/retained events,
application-input events, number of updates and objective evaluations. Lower event
count alone is not a runtime result. Do not infer energy, hardware area, peak RSS,
a live deadline, complete SLAM throughput or a universal speedup from these timings.

## Joint interpretation fixed before outcomes

Primary contrast: RSF versus raw on external rotation error and complete processing
time. Describe an observed Pareto improvement only if equal-recording mean error
is no worse and aggregate median complete time is lower. This is descriptive,
not a statistical noninferiority claim. There is no post-hoc allowed quality loss,
new retention optimum, favorable speed regime or successful-subset selection.

Secondary comparisons are RSF against random, exact counts and EBF using the same
endpoints. A score-specific utility claim additionally requires better mean motion
quality than the matched random deletion control; otherwise report deletion effects
or an explicit tradeoff. Report per-recording reversals and all negative outcomes.
No result from this task may overwrite or explain away the frozen DSEC degradation.
Preserve both studies and their different application assumptions.
