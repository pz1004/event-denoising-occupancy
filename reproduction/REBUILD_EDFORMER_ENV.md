# Rebuilding the EDformer inference environment

The portable commands are in [environments/README.md](../environments/README.md#edformer-baseline).
They retain Python 3.10, torch 2.2.1+cu118, NumPy 1.26.4, dv-processing 1.7.9,
SparseConvNet 0.2, and PyTorch3D 0.7.5. Use a CUDA 11.8 toolkit, a compatible C++
compiler, and Python development headers to build the two CUDA extensions.
Source, checkpoint and interpreter placement are explicit there; no original
workstation path is required. The SparseConvNet source revision was not recorded.

The complete E-MLB cell reproduction uses `--batch-blocks 1 --reps 1 --max-events 0`.
The runner checkpoints after each cell. Compare against the shipped
`results/emlb_edformer_mesr.json`; D-END/ND00 has `edformer_mesr = 0.9520324576641584`.
A fresh result that differs from the stored values is environment-sensitive evidence,
not permission to silently replace the reported baseline.

The recorded sequence protocols differ: RSF's original E-MLB rows average up to
144 sequences per cell (repetitions 1–3), while this EDformer reproduction has
48 per cell (repetition 1). Only repetition 1 can be sequence-paired. Structural
scores can be undefined on short filtered streams; paired analysis must use the
finite intersection and report its denominator. These qualifications are retained
in the evidence and public checks.
