# EDmamba Reproducibility Report

Date: 2026-07-19 (Asia/Seoul)  
Workspace: `/home/pz1004/Workspaces/dvs-noise-filter`  
Location: moved into this repository at `reproduction/` on 2026-08-26, because
`paper-tcsvt/checks/verify_v3_numbers.py` reads four figures out of it and a
clone could not reach it at the workspace root. Content is unchanged apart from
the two link targets noted below.  
EDmamba commit: `302f7586c083aa0bd5f7befa13011a1b6c1e7022` (`main`, upstream initial commit)

**Dataset update (2026-07-21):** a separate `DVSCLEAN` directory was later
added at the workspace root and has now been evaluated. See
`DVSCLEAN_REPRODUCIBILITY_REPORT.md` (not vendored; see `DATA.md`) for
the direct-HDF5 EDmamba-versus-Native3D comparison. Statements below about an
absent DVSCLEAN dataset describe the original 2026-07-19
`ECCV2024_datasets`-only audit; E-MLB remains unavailable.

## 1. Executive summary

I audited the bundled EDmamba paper (arXiv:2505.05391v3; the PDF is not vendored, see `DATA.md`), the complete checked-in implementation, the official checkpoint, and the 25 GB `ECCV2024_datasets` download. I then reconstructed the legacy CUDA environment, verified a strict checkpoint load, ran a real forward/backward/optimizer step, and evaluated all ten DND21 table cells reported by the paper (two scenes at 1, 3, 5, 7, and 10 Hz/pixel).

The checkpoint reproduction is strong but not perfect:

- The model has **88,978 trainable parameters**, matching the paper's 88.98K claim.
- The full test evaluated **3,393 blocks / 34,744,320 labeled events**.
- **8 of 10** reported AUC cells are within **0.0003 absolute AUC** of the paper.
- Mean/median absolute error over all ten cells is **0.000491 / 0.000146**.
- The 5 Hz ablation-reference pair is reproduced as **0.995480 / 0.973510**, versus **0.9955 / 0.9734** in the paper.
- The two persistent discrepancies are at 10 Hz: Hotel-bar is lower by 0.002386 and Driving is higher by 0.001466. A fresh-process rerun ruled out carried random state as the explanation.
- A full paper-equivalent training run was not attempted because the paper used 8 x RTX A6000 GPUs and global batch 128, whereas this host has one 6 GB GTX 1660 SUPER. A single reduced training step completed successfully with finite gradients.

This is therefore a **successful checkpoint/evaluation reproduction for 8/10 DND21 results and a successful training-path smoke test**, not a from-scratch training reproduction.

## 2. Important conceptual note

EDmamba operates on event-camera data, but it is **not a spiking neural network**. The released network contains no spiking neurons, membrane-state dynamics, spike reset, temporal unrolling of neuron states, or surrogate-gradient training. It is a conventional differentiable point/event classifier built from sparse convolutions, MLPs, and Mamba-style state-space layers. Its biological connection is the asynchronous event sensor, not the learning architecture.

## 3. What the paper and code implement

### 3.1 Input and task

Each raw event is `(timestamp, x, y, polarity, label)`. Inside each 10,240-event block, timestamp is normalized to `z in [0, 1]`, polarity is mapped from `{0, 1}` to `{-1, +1}`, and the model receives a four-channel feature `(x, y, z, p)`. The output head classifies each retained event as class 0 (signal/non-noise in the labeled files) or class 1 (noise).

### 3.2 Released architecture

The implementation in `EDmamba/pointcept/models/EDmamba/` contains:

1. **Coarse Feature Extraction (CFE)**
   - One Conv1d branch processes `(x, y, z)`.
   - A second Conv1d branch processes `(x, y, p)`.
   - Their 8-channel features are concatenated, projected back to 8 channels, batch-normalized, activated, and dropped out.

2. **Two-stage U-Net-style point backbone**
   - Encoder depths `[2, 4]`, channels `[8, 16]`.
   - Serialized pooling by factor 2 between encoder stages.
   - One decoder stage of depth 2 and channel width 8, with serialized unpooling and a skip connection.

3. **Parallel spatial and temporal SSM paths in each block**
   - The spatial path serializes points using Z-order, transposed Z-order, Hilbert order, or transposed Hilbert order, depending on block index.
   - The temporal path sorts by normalized time in ascending or descending order, again selected by block index.
   - Both use the custom `SimpleAwareStateSpaceModel` with state size 16, expansion factor 2, and depthwise Conv1d kernel 5.
   - Spatial and temporal outputs are concatenated and fused with a 1x1 Conv1d, followed by a residual MLP.

4. **Binary event head**
   - Linear `8 -> 2` classifier.
   - Cross-entropy objective.

The code reaches bidirectional/multi-order coverage across successive blocks. It does not run both temporal directions inside every individual block, which is a nuance relative to the paper's prose description of bidirectional Mamba blocks.

## 4. Artifact and dataset audit

### 4.1 Repository artifacts

| Artifact | Finding |
|---|---|
| EDmamba source | One upstream commit, `302f758`; no preprocessing program is included |
| EDmamba checkpoint | `EDmamba/exp/ED24/EDmamba/model/model_best.pth`, tracked by Git |
| Checkpoint SHA-256 | `b61d502f165f41ef2b4836ba27fb1e63adbc247f0627f0219d76e33aa761aae2` |
| Checkpoint contents | epoch 13, best metric 0.8879312271, model/optimizer/scheduler states |
| Checkpoint tensor state | 335 tensors/buffers, 89,062 stored elements including BN buffers; 88,978 trainable parameters after model construction |
| Dataset `pretrained_model.pth` | An EDformer checkpoint, not EDmamba (keys are transformer/LXformer/SCformer/GXformer); it was not used |
| EDformer checkpoint SHA-256 | `ad88928d9e0e3a3c99bcc9d34f028d96f8794869396b4a904c8332ba7805734e` |

### 4.2 ED24 raw data

The download contains:

- 100 scene directories.
- 4,100 CSVs, or 41 files per scene (`0.0` through `4.0` in steps of `0.1`).
- 789,047,143 labeled events.
- 74,932 complete non-overlapping 10,240-event blocks when each file remainder is dropped.
- 21,743,463 events in per-file remainders.

This reveals a material paper/release difference. The paper describes ED24 as 21 illumination levels (1.5 through 3.5), which are 2,100 files in this download. Those 2,100 files contain 42,640 complete blocks. The released checkpoint scheduler has `total_steps = 117050`, which exactly agrees with using **all 74,932 blocks from all 41 levels**:

```text
74,932 blocks x config loop 4 = 299,728 samples
ceil(299,728 / 8 GPUs) = 37,466 samples/GPU
floor(37,466 / batch 16 per GPU) = 2,341 steps per trainer epoch
2,341 x 50 trainer epochs = 117,050 scheduler steps
```

Using only the paper's 21 levels would yield 66,600 scheduler steps, not 117,050. The official checkpoint was therefore trained with the expanded 41-level release data under the checked-in configuration.

### 4.3 DND21/AUC data

`ECCV2024_datasets/AUC_test` contains labeled Hotel-bar (`mix_result.txt`) and Driving (`driving_mix_result.txt`) streams at every integer rate from 1 to 10 Hz/pixel. I evaluated the five rates reported by EDmamba. Each stream was split into complete consecutive 10,240-event blocks; incomplete tail blocks were dropped, matching the README's fixed-size NPZ description and the training scheduler evidence.

Across the ten reported streams:

- Raw labeled events: 34,795,491.
- Evaluated events: 34,744,320.
- Dropped tail events: 51,171.
- Complete blocks: 3,393.

At the time of this audit, the `ECCV2024_datasets` bundle did **not** contain
the paper's DVSCLEAN or E-MLB evaluation datasets. A separate root-level
DVSCLEAN download was added on 2026-07-21 and was subsequently evaluated, but
the released repository still lacks the raw-HDF5-to-NPZ converter needed to
reconstruct EDmamba's exact Table 1 preprocessing. E-MLB remains unavailable,
so Table 2's MESR results cannot be reproduced here.

## 5. Environment reconstruction

### 5.1 Hardware and system

| Component | Reproduction host | Paper |
|---|---:|---:|
| GPU | NVIDIA GeForce GTX 1660 SUPER, compute 7.5 | 8 x NVIDIA RTX A6000 |
| GPU memory | 6,144 MiB | 48 GiB per A6000 |
| CPU | Intel Core i7-12700, 20 logical CPUs | not stated |
| RAM | 31 GiB | not stated |
| OS | Ubuntu under WSL2, Linux 6.6.87.2 | not stated |
| NVIDIA driver | 591.86 | not stated |

### 5.2 Core software

The pre-existing Conda environment `/home/pz1004/anaconda3/envs/pointcept` already matched most of the README:

| Package | Version |
|---|---:|
| Python | 3.8.20 |
| PyTorch / torchvision | 1.12.1 / 0.13.1 |
| PyTorch CUDA | 11.6 |
| spconv | 2.3.6 (`spconv-cu116`) |
| torch-scatter / torch-geometric | 2.1.0 / 2.6.1 |
| mamba-ssm | 1.1.4 |
| causal-conv1d | 1.1.3 |
| NumPy / SciPy / pandas | 1.24.4 / 1.10.1 / 2.0.3 |
| scikit-learn | 1.3.2 |
| Numba | 0.58.1 |
| timm | 1.0.11 |
| transformers / Triton | 4.38.2 / 2.0.0 |

There is no `nvcc` CUDA compiler on this host. Instead of the README's forced source builds, I installed official prebuilt GitHub release wheels for Mamba 1.1.4 and causal-conv1d 1.1.3. Their SHA-256 values were:

- causal-conv1d wheel: `af8e92445d13a61eecacbb4f0e7b7c997a932a431f134e3e998f78dd25fec70b`
- mamba-ssm wheel: `341ef5d0720e9ff01aaf77e13272def831123973942f4e41f616bd2893baec35`

The checked-in Python 3.8 pointops binary was loaded from `EDmamba/libs/pointops/build/lib.linux-x86_64-cpython-38` through `PYTHONPATH`.

WSL's `libcuda.so` directory was not in the dynamic loader path, so runs used:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
```

The legacy CUDA 11.6 combination occasionally aborted during interpreter destructor cleanup after otherwise successful EDmamba runs. The reproduction runner writes results atomically and supports `--hard-exit` to bypass only that final cleanup. This does not bypass model execution or result computation.

## 6. Release fixes needed to run

No model math, model configuration, checkpoint, or dataset file was changed. I made three import-only compatibility fixes:

1. `EDmamba/pointcept/models/losses/misc.py`
   - Made `pytorch3d` and `faiss` optional because EDmamba's configured cross-entropy loss does not use them.
2. `EDmamba/pointcept/engines/test.py`
   - Made `dv_processing` optional because the AUC tester never uses it.
3. `EDmamba/pointcept/datasets/denoise_emlb.py`
   - Made `dv_processing` optional so importing the dataset package does not require E-MLB support during DND21 evaluation.

I also installed Numba 0.58.1, which `transform.py` imports but the README does not list.

The raw-to-NPZ converter described by the README is absent. I added `reproduction/reproduce_edmamba_dnd21.py`, which streams the raw labeled files and applies the released loader/test transforms directly, avoiding a large redundant NPZ copy.

## 7. Exact evaluation protocol

For every labeled scene/rate stream:

1. Read consecutive blocks of exactly 10,240 events.
2. Drop the final incomplete block.
3. Normalize block timestamps to `[0, 1]` exactly as `denoise_ed24.py` does.
4. Map polarity to `{-1, +1}`.
5. Apply `GridSample_V3(grid_size=0.05, mode="train", return_inverse=True)` exactly as the test config does. This randomly retains one event per `(x, y, time-bin, polarity)` voxel while retaining an inverse mapping.
6. Apply exhaustive test partitioning with `GridSample_V3(grid_size=0.1, mode="test")`.
7. Strict-load the official EDmamba checkpoint and run softmax event classification.
8. Map sampled predictions back to all original events with the first grid sampler's inverse.
9. Compute ROC AUC independently for every 10,240-event block.
10. Report the unweighted mean of block AUCs, matching `pointcept/engines/test.py` lines 317-318 and 473.

I additionally computed one event-global ROC AUC per whole stream. This is a useful statistical cross-check, but it is not the aggregation used by the released tester or the primary reproduced number below.

Seed 1024 was used. Re-running 10 Hz alone in a new process changed Hotel-bar by only +0.000021 and Driving by +0.000006, demonstrating that the 10 Hz discrepancy is not caused by carrying random sampler state across rates.

## 8. DND21 reproducibility results

Primary metric: released tester's unweighted mean of per-block AUC.

| Noise | Scene | Blocks | Events | Paper AUC | Reproduced AUC | Delta | Event-global AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 Hz | Hotel-bar | 87 | 890,880 | 0.9963 | **0.996415** | +0.000115 | 0.996522 |
| 1 Hz | Driving | 300 | 3,072,000 | 0.9694 | **0.969654** | +0.000254 | 0.974288 |
| 3 Hz | Hotel-bar | 157 | 1,607,680 | 0.9949 | **0.995078** | +0.000178 | 0.995382 |
| 3 Hz | Driving | 368 | 3,768,320 | 0.9710 | **0.971136** | +0.000136 | 0.975705 |
| 5 Hz | Hotel-bar | 229 | 2,344,960 | 0.9955 | **0.995480** | -0.000020 | 0.995825 |
| 5 Hz | Driving | 437 | 4,474,880 | 0.9734 | **0.973510** | +0.000110 | 0.977044 |
| 7 Hz | Hotel-bar | 299 | 3,061,760 | 0.9939 | **0.994057** | +0.000157 | 0.994560 |
| 7 Hz | Driving | 504 | 5,160,960 | 0.9689 | **0.968989** | +0.000089 | 0.973889 |
| 10 Hz | Hotel-bar | 405 | 4,147,200 | 0.9916 | **0.989214** | -0.002386 | 0.990266 |
| 10 Hz | Driving | 607 | 6,215,680 | 0.9636 | **0.965066** | +0.001466 | 0.970896 |

Aggregate comparison:

- Mean absolute error: 0.000491.
- Median absolute error: 0.000146.
- Maximum absolute error: 0.002386.
- Cells within 0.0003: 8/10.

The full process took 356.60 seconds end to end. Timed model inference totaled 280.74 seconds; per-stream pipeline wall time totaled 318.12 seconds. Maximum process RSS was 4,426,008 KiB. These timings are not comparable to the paper's A6000 latency benchmark because the GPU, block size, framework launch pattern, and measurement scope differ.

### Interpretation of the 10 Hz discrepancy

The discrepancy is deterministic with the public artifacts:

- Combined-run 10 Hz: 0.9892144259 / 0.9650656278.
- Fresh-process 10 Hz: 0.9892356281 / 0.9650714808.
- Fresh minus combined: +0.0000212 / +0.0000059.

Since the same inferred protocol reproduces all eight 1-7 Hz cells very closely, the likely remaining causes are a difference between the authors' 10 Hz evaluation files and the distributed files, or a different unreleased checkpoint/evaluation snapshot for those table entries. The repository hardcodes only `5hz` in `denoise_auc.py`, and it contains no per-rate commands, processed test manifest, or recorded prediction files with which to resolve this. I did not tune sampling, truncate streams, or select a different checkpoint to force agreement.

## 9. Training-path reproduction

### 9.1 Successful optimizer-step smoke test

I ran the checked-in training augmentation stack and one complete forward/backward/AdamW update from a real ED24 block (`Car_01_2.0.csv`):

| Measurement | Result |
|---|---:|
| Raw events | 10,240 |
| Events after grid sampling and 80% sphere crop | 6,744 |
| Class counts `[signal, noise]` | `[1,984, 4,760]` |
| Cross-entropy loss | 1.3182964 |
| Gradients | all finite |
| Step wall time | 2.091 s |
| Peak allocated CUDA memory | 135.88 MiB |

This verifies model construction, sparse convolution, serialization, custom selective-scan forward/backward, the loss, and optimizer update on the local GPU.

### 9.2 Why full training was not run

The paper used 8 x RTX A6000, global batch 128, and a 50-epoch statement. This machine has one 6 GB GTX 1660 SUPER. More importantly, the checked-in runtime is not a literal 50 full-data-pass configuration:

- `epoch = 200`
- `eval_epoch = 50`
- config parsing sets `data.train.loop = epoch // eval_epoch = 4`
- the trainer runs `eval_epoch = 50` outer epochs

The result is 50 trainer epochs over a four-times-looped dataset, or 200 effective data passes, with 117,050 distributed optimizer steps. Reproducing that schedule faithfully requires the multi-GPU setup and the missing canonical preprocessing/split manifest.

## 10. Paper/code/release discrepancies affecting reproducibility

1. **Dataset level count**
   - Paper: 21 ED24 noise/illumination levels.
   - Download and README tree examples: 41 levels (`00` through `40`).
   - Checkpoint scheduler: proves all 41 levels were used.

2. **Epoch semantics**
   - Paper: trained for 50 epochs.
   - Code: 50 trainer epochs with a dataset loop of 4, equivalent to 200 data passes.

3. **Weight decay**
   - Paper: `5e-2`.
   - Config/checkpoint optimizer: `5e-3`.

4. **Training/validation overlap**
   - `denoise_ed24.py` maps training to both `TRAIN` and `VAL`, while validation uses `VAL` again. The validation set is therefore included in training.

5. **Missing preprocessing and split manifest**
   - The README describes NPZ output but provides no converter and no exact TRAIN/VAL scene list.

6. **Hardcoded test rate**
   - The public AUC dataset class tests only `5hz`; reproducing the full table requires code or data-root changes.

7. **AUC aggregation is underspecified in the paper**
   - The released tester averages per-block ROC AUCs rather than constructing one ROC over the full scene.

8. **Undocumented dependencies/imports**
   - Numba is required but absent from the README.
   - `pytorch3d`, `faiss`, and `dv_processing` were imported unconditionally for paths not used by EDmamba+DND21.
   - Mamba's runtime dependencies (`transformers`, Triton) are not explicitly installed in the README.

9. **Checkpoint epoch**
   - The only official EDmamba checkpoint is the best snapshot marked epoch 13, not a final epoch-50 artifact.

10. **Unavailable benchmarks in the original dataset bundle**
    - DVSCLEAN and E-MLB are referenced by the config/paper but were absent
      from `ECCV2024_datasets`. DVSCLEAN was later supplied separately and
      evaluated under a documented direct-HDF5 protocol; E-MLB is still absent.

## 11. Reproduction commands

The main command used was:

```bash
cd /home/pz1004/Workspaces/dvs-noise-filter
LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" \
  /home/pz1004/anaconda3/envs/pointcept/bin/python -u \
  reproduction/reproduce_edmamba_dnd21.py \
  --rates 1hz 3hz 5hz 7hz 10hz \
  --output reproduction/results/dnd21_auc_full.json \
  --hard-exit
```

For a quick two-block-per-scene validation:

```bash
cd /home/pz1004/Workspaces/dvs-noise-filter
LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" \
  /home/pz1004/anaconda3/envs/pointcept/bin/python -u \
  reproduction/reproduce_edmamba_dnd21.py \
  --rates 5hz --max-blocks 2 \
  --output reproduction/results/dnd21_auc_smoke.json \
  --hard-exit
```

On a normal native Linux CUDA setup that exits cleanly, omit `--hard-exit` and the WSL-specific library path.

## 12. Generated evidence

| File | Purpose |
|---|---|
| `reproduction/reproduce_edmamba_dnd21.py` | Streaming, protocol-faithful DND21 evaluation runner |
| `reproduction/results/dnd21_auc_full.json` | Machine-readable full metrics and timing |
| `reproduction/results/dnd21_auc_full.log` | Full progress/output log |
| `reproduction/results/dnd21_auc_10hz_fresh_seed.json` | Fresh-process 10 Hz diagnostic |
| `reproduction/results/train_step_smoke.log` | Forward/backward/optimizer smoke evidence |

## 13. Bottom line

The EDmamba architecture and official checkpoint are executable, the parameter claim is exact, the training path is numerically healthy, and most DND21 AUC claims reproduce to roughly the fourth decimal using the supplied ECCV2024 data. Full from-scratch reproducibility is currently limited by hardware and, more fundamentally, by release incompleteness: missing preprocessing/split code, configuration-paper differences, use of 41 rather than the stated 21 ED24 levels, an unavailable E-MLB dataset, no released DVSCLEAN HDF5 converter, and no recorded 10 Hz predictions or final training checkpoint.

The highest-value additions from the authors would be: the exact CSV-to-NPZ converter and split manifest; a frozen Conda/Docker environment; per-rate test configs; final and best checkpoints with training logs; and hashes for the precise DND21, DVSCLEAN, and E-MLB evaluation artifacts used in the tables.
