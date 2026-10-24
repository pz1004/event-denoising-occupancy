# Pre-outcome timestamp-adapter clarification

23 October 2026 (Asia/Seoul). This clarification supplements, without replacing,
`protocol_frozen_rotation_application_20261023.md`, SHA-256
`fe0ff7cdff47d07404aa8b13e083fe79d5205f19e896629c98ef56c89a2ecb64`.
It is fixed after source-format inspection and before any real-data motion estimate
or application-quality outcome. The recording cohort, filter parameters, physical
intervals, estimator, repetition schedule, endpoints and interpretation remain as
specified in that original protocol.

The official events.txt files print seconds to nine decimal places. Some values
have a one-nanosecond offset from an integer microsecond (for example 0.000011001).
The original input adapter rounded to integer microseconds for the unchanged phase79
filter, but rejected floating residuals strictly greater than 0.001 microseconds;
floating-point representation made that test reject a one-nanosecond offset.
The rejection occurred during input preparation, before any task outcome.

A complete source-only audit read every event in all three archives and verified:

| Recording | Events | First canonical us | Last canonical us | Maximum exact residual |
|---|---:|---:|---:|---:|
| shapes_rotation | 23,126,288 | 0 | 59,798,386 | 1 ns |
| poster_rotation | 169,350,136 | 0 | 59,799,647 | 1 ns |
| boxes_rotation | 185,688,947 | 16,667 | 59,818,451 | 1 ns |

Total: 378,165,371 events. The maximum floating residual in every recording was
0.0010000020265579224 microseconds. Coordinates and polarities are integer-valued;
all polarity values are 0 or 1. Calibration inversion passed. The audit inspected
source encoding and calibration only, with no motion fitting or error evaluation.

The final adapter uses nearest-integer microsecond timestamps for all five arms,
including raw, before the original-source array becomes the timed input. It must
verify an exact integer-nanosecond residual no greater than 1 ns and a floating
microsecond residual no greater than 0.00101 us. Larger deviations are fatal input
validation errors; no event is removed to satisfy this bound. The original archive
and event-text hashes are retained separately from the canonical integer-payload
hash. Maximum residuals, this clarification's digest and the original protocol's
digest accompany the result artifact and implementation freeze.

Thus preserved payloads mean original coordinates/polarities and the common
canonical microsecond clock; they do not claim bit-exact preservation of the
nine-decimal source timestamp text. Source-window anchoring and application
apertures use the same canonical clock in every arm. Motion-capture timestamps
remain at their released precision and are interpolated at those common physical
endpoints. The at-most-one-nanosecond quantization is disclosed in the supplement.
No noise is injected, no time offset is fitted, and no scientific setting is
changed or chosen from a task outcome.
