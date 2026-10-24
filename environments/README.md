# Environments and independent setup

Run commands from the public repository root. The root `requirements.txt` is a
general execution environment; it is not an exact lock of every historical
experiment. Different recorded studies used different Python/NumPy stacks.
Do not install these profiles together. CPU timing depends on the processor,
compiler, memory and workload as well as package versions.

## General setup and synthetic example

Use Python 3.12 or 3.13 in a fresh virtual environment. Install a suitable PyTorch
wheel first, then the root requirements. CPU PyTorch is sufficient for the
synthetic rotation example; RSF itself has no trained weights. The preserved
package initializer imports historical learned-model classes, which require torch.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements.txt
python tools/run_frozen_study.py rotation --stage synthetic
```

The synthetic command exercises the rotation adapter without downloading data or
running a real recording. The public launcher selects the frozen protocols in
`protocols/`; it preserves the scientific runner sources and their hash guards.
Use the launcher for the fixed-time, DSEC and rotation studies as shown in the
[root README](../README.md).

## Fixed-count and fixed-time filtering costs

The phase75/79 artifacts record Python 3.13.9, NumPy 2.3.5, Numba 0.62.1 and
llvmlite 0.45.1. Use `requirements-recorded-cpu.txt` in a separate Python 3.13.9
environment and install PyTorch for package imports. Additional pins in that file
come from the retained environment inspected during release preparation; they
are not a recovered complete lock from the measurement date.

```bash
python -m pip install -r environments/requirements-recorded-cpu.txt
```

Run complete filtering timings on a quiet machine. Warm compilation is separate
from measured processing. Use fresh output paths so the released evidence is not
overwritten. A fresh run is new evidence and its source/runtime hashes can differ
from historical artifacts.

## Frozen rotation application

Use Python 3.12.12 and `requirements-recorded-rotation.txt`, plus a PyTorch wheel
for the preserved package imports. The stored phase81 environment establishes
NumPy 1.26.4 and SciPy 1.17.1; the retained environment also has Numba 0.64.0 and
llvmlite 0.46.0. Ancillary package pins are a release-time inventory. The runner
records thread pools and controls exercised BLAS pools during timed evaluation.
The three exact protocol files used by phase79/81 are included under `protocols/`.

## Frozen DSEC/E-RAFT application

Use Python 3.12.12. The recorded GPU run used PyTorch 2.10.0+cu128 and NumPy 1.26.4.
Install the CUDA wheel appropriate to that profile and the additional input readers:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
python -m pip install -r environments/requirements-dsec.txt
mkdir -p ../external
git clone https://github.com/uzh-rpg/E-RAFT.git ../external/E-RAFT
git -C ../external/E-RAFT checkout c58ce0524ea0ebfa9849991caafb547f44fe9bfd
```

The frozen adapter loads `model.eraft` and `utils.dsec_utils` from that checkout.
It does not require the original upstream Python 3.7 environment. Put the official
MVSEC 20-Hz checkpoint at `../downstream/assets/eraft/mvsec_20.tar`; the exact URL
and SHA-256 are in [DATA.md](../DATA.md). Dataset paths can use `--data-root`.
The upstream source and checkpoint paths intentionally retain their recorded
sibling-directory layout. This avoids changing frozen source bytes.

## EDformer baseline

Use a separate Python 3.10 environment, PyTorch 2.2.1+cu118, CUDA toolkit 11.8,
NumPy 1.26.4, dv-processing 1.7.9, SparseConvNet 0.2 and PyTorch3D 0.7.5. CUDA
drivers alone do not provide the `nvcc` compiler needed for source builds.

```bash
git clone https://github.com/NJUVISION/EDformer.git EDformer
git -C EDformer checkout 820ff16b3bc50694e163f054fbb52cf3505acfff
python3.10 -m venv ../.venv-edformer-cu118
../.venv-edformer-cu118/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.2.1+cu118
../.venv-edformer-cu118/bin/python -m pip install -r environments/requirements-edformer.txt
```

Build [SparseConvNet](https://github.com/facebookresearch/SparseConvNet) and
[PyTorch3D v0.7.5](https://github.com/facebookresearch/pytorch3d/tree/v0.7.5) with
that interpreter and matching CUDA toolkit; set `TORCH_CUDA_ARCH_LIST` to the GPU
architecture actually used. The original SparseConvNet source revision was not
recorded, so the version 0.2 and compiled environment cannot constitute a complete
source lock. Obtain the released `pretrained_model.pth` from the EDformer authors.

The E-MLB reproduction uses batch size one; larger inference batches change
floating reduction order and are not the recorded protocol:

```bash
../.venv-edformer-cu118/bin/python reproduction/reproduce_edformer_emlb.py --batch-blocks 1 --reps 1 --max-events 0 --output reproduced-results/emlb-edformer.json
```

The public `tools/run_edformer.py` uses `EDformer/` under this checkout by default.
It accepts `--edformer-dir`, `--model`, or `RSF_EDFORMER_ROOT`. The precomputation
helper accepts `RSF_EDFORMER_PYTHON` for a nondefault interpreter. Precompute phase65
cache units with that helper before running the unchanged historical phase65
module, whose direct fallback still expects `../.venv-edformer-cu118/bin/python`.
The standard sibling environment above satisfies that fallback.

## EDmamba baseline

Use Python 3.8.20 and the recorded PyTorch 1.12.1/CUDA 11.6 stack separately from
the modern filter environment. [The upstream manifest](../reproduction/UPSTREAM.md)
records source revisions and the three import-only compatibility changes.

```bash
git clone https://github.com/gongzihang/EDmamba.git EDmamba
git -C EDmamba checkout 302f7586c083aa0bd5f7befa13011a1b6c1e7022
git -C EDmamba apply --check ../reproduction/patches/edmamba-import-compatibility.patch
git -C EDmamba apply ../reproduction/patches/edmamba-import-compatibility.patch
conda env create -f environments/environment-edmamba.yml
conda activate rsf-edmamba
python -m pip install torch-scatter==2.1.0 torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
```

Complete the upstream PointOps/Mamba installation. The original runners explicitly
add `EDmamba/libs/pointops/build/lib.linux-x86_64-cpython-38` to the import path;
use the supplied compatible binary or build PointOps in that Python environment.
Install causal-conv1d 1.1.3 and mamba-ssm 1.1.4 from source as described upstream
or use the release wheels recorded below. Their tags use cu118 despite the
PyTorch CUDA 11.6 runtime in this historical environment; this is the recorded
combination, not a general compatibility promise.

| Wheel from upstream release assets | SHA-256 |
|---|---|
| `causal_conv1d-1.1.3+cu118torch1.12cxx11abiFALSE-cp38-cp38-linux_x86_64.whl` | `af8e92445d13a61eecacbb4f0e7b7c997a932a431f134e3e998f78dd25fec70b` |
| `mamba_ssm-1.1.4+cu118torch1.12cxx11abiFALSE-cp38-cp38-linux_x86_64.whl` | `341ef5d0720e9ff01aaf77e13272def831123973942f4e41f616bd2893baec35` |

Download from the [causal-conv1d v1.1.3 release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.1.3)
and [Mamba v1.1.4 release](https://github.com/state-spaces/mamba/releases/tag/v1.1.4),
verify the hashes, and install the wheels in the EDmamba environment. Model
weights remain external at `EDmamba/exp/ED24/EDmamba/model/model_best.pth` unless
overridden using a runner's `--checkpoint` option. The dataset converters and
training details missing from upstream are documented in the reproduction report;
this release does not claim to recover them.

This recipe captures known dependencies and local compatibility changes. It has
not rebuilt every third-party GPU extension from an empty machine during public
release preparation. The recorded numerical results and provenance remain available without those GPU environments.
