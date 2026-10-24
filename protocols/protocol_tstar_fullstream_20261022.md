# Frozen full-stream T* cost–quality protocol

Recorded 22 October 2026 before the phase79 full experiment. This is an internal,
prospective analysis specification, not an external preregistration. LED has already
informed development; this experiment does not make LED independent validation.
The downstream cohort and its additional protocol are recorded separately before any
new downstream outcomes are evaluated.

## Configuration and targets

- Adopted ungated RSF occupancy, radius 2, two polarity planes, no global term.
- T = 24,302 microseconds; r = 4/5 exactly. These constants are inherited from
  earlier work, not optimized in this study.
- Consecutive half-open time windows anchored at the first input event. Process
  every event in every nonempty window, including low-count windows and the final
  partial window. No labels, common-event masks or baseline-dependent exclusions
  enter filtering. Empty windows produce no output.
- Scheduled right-edge window closure is used for delay accounting, including the
  final window. An idle-stream timer is assumed by the replay; no live timer or
  transmission deadline is claimed.
- Keep floor(4*n/5) events in each window; highest signal score wins, ties are
  resolved by arrival order, and retained payloads are copied in arrival order.
- Compare occupancy, exact whole-window integer multiplicity, and EBF with its
  frozen radius-2 / 32-ms score. EBF updates on every original event and carries
  its last-timestamp/last-polarity state across all windows in a sequence. It uses
  the same windowed retention policy; this is not its released threshold policy.
- Multiplicity uses a counter width sufficient for the actual window population;
  uint16 must not overflow on windows longer than 65,535 events. Storage accounts
  for the actual implementation and any separately stated packed design.

## Complete cost and quality

Quality covers all 66 released LED test sequences on their original full input
streams with aligned released labels. AUC is noise-positive; filtering metrics
are signal-positive. Primary operating metric is equal-sequence balanced accuracy,
with signal recall, noise rejection, precision, F1, retained count and event AUC
reported alongside it. Pair occupancy against EBF and exact multiplicity. Record
all per-sequence results; no sequence or window is removed because of a poor score
or class composition. AUC is undefined when a target population has one class and
such cases must be counted rather than silently converted to a value.

For each paired primary contrast use 200,000 sequence bootstrap resamples for
percentile 95% intervals, with seed 20261022; no multiple-comparison or independent
acquisition-session guarantee is implied. Report the sequence-independence
assumption. The prior common-target phase77 study remains a separate endpoint.

Timing uses the lexicographically first LED sequence and first development DND21
condition, each truncated to at most the first four million events, fixed without
reading results. Run nine interleaved repeats after warmup, serially without other
CPU/GPU experiments. Report median and dispersion, host/runtime/affinity and
single-threaded kernel details. Complete processing includes boundary discovery,
input buffering/copying, map construction or state updates, scoring, retention,
and ordered output copying. Exclude file decoding, JIT compilation and evaluation
metrics; identify any remaining harness overhead. Score-only timings, if reported,
are secondary. Comparators use the same event layout and output contract.

Report actual working-array capacities (including selector temporary arrays and
EBF state), window-size distributions and observed maximum. Packed designs, if
included, are estimates rather than measured process memory. Distinguish online
working state from complete input/output arrays held by the evaluation harness.
Use a serialized FIFO replay with scheduled closures and measured window service
times to report emitted-event delay and extra queue occupancy. Report exclusions
such as output transmission; this is a replay, not measured live sensor latency.

## Integrity and decision rules

Require independent small-stream score/selection tests, timestamp-boundary and
arrival-tie tests, tail/low-count coverage, EBF state continuity, >65,535-event
multiplicity correctness and exact rational retention. Check source/output counts,
window quotas, ordered payload equality and reference scores on deterministic
subsets. Save protocol/code/source digests, all results and environment metadata.
Do not select a favorable retention, duration, subset or repeat after evaluation.
Report unfavorable outcomes and maintain the distinction between the old
common-target comparison and this complete full-source configuration.
