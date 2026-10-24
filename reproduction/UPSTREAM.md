# External implementations and evaluation requirements

The repository supplies experiment runners and numerical results. Obtain external
model implementations, checkpoints, datasets and compiled extensions separately.
The pinned source revisions below identify the reference checkouts; the E-RAFT
revision is also recorded in the frozen downstream artifact.

| External source | Revision | License at this revision |
|---|---|---|
| [EDmamba](https://github.com/gongzihang/EDmamba) | `302f7586c083aa0bd5f7befa13011a1b6c1e7022` | No root license file found; consult the authors' terms |
| [EDformer](https://github.com/NJUVISION/EDformer) | `820ff16b3bc50694e163f054fbb52cf3505acfff` | No root license file found; consult the authors' terms |
| [E-RAFT](https://github.com/uzh-rpg/E-RAFT) | `c58ce0524ea0ebfa9849991caafb547f44fe9bfd` | MIT |
| [cuke-emlb](https://github.com/KugaMaxx/cuke-emlb) | `105a1c36c368f10ff63b4a6f862992ae8842628d` | MIT |
| [EBF](https://github.com/shicy17/EBF) | `e2fa7760c841dc056bfde9fa1d70125a14275d2b` | GNU GPL v3 license text |

[upstream_manifest.json](upstream_manifest.json) records these revisions and
hashes of the available license files. The repository's MIT grant does not
relicense upstream code, patch context, model weights or datasets. EBF is a
baseline and dataset source; its upstream implementation is not copied here.

## Checkpoint identities

| Model | Expected location | SHA-256 |
|---|---|---|
| EDmamba | `EDmamba/exp/ED24/EDmamba/model/model_best.pth` | `b61d502f165f41ef2b4836ba27fb1e63adbc247f0627f0219d76e33aa761aae2` |
| EDformer | `EDformer/pretrained_model.pth` | `ad88928d9e0e3a3c99bcc9d34f028d96f8794869396b4a904c8332ba7805734e` |
| E-RAFT, MVSEC 20 Hz | `../downstream/assets/eraft/mvsec_20.tar` | `d9845c0b85a1e5b2d5b664f27706e9a86270afcff42cb16457aeb9c2cbdb9ba2` |

Use the providers linked in [DATA.md](../DATA.md) and the
[environment recipes](../environments/README.md). The EDmamba checkpoint is the
released epoch-13 best snapshot, with 88,978 trainable parameters. EDformer's
checkpoint has 583,973 trainable parameters and 46 BatchNorm buffer elements;
its two SparseConvNet weight tensors must be included in parameter counts.
The DSEC study uses the MVSEC-trained E-RAFT checkpoint, not a DSEC-trained model.

## EDmamba compatibility patch

[patches/edmamba-import-compatibility.patch](patches/edmamba-import-compatibility.patch)
makes `dv_processing` optional in the E-MLB dataset loader and AUC engine, and
makes PyTorch3D/FAISS optional in the loss module. The configured cross-entropy
inference path does not use these imports. The patch does not change model
arithmetic, configuration, checkpoint contents or dataset values.

Apply it once to the pinned clean checkout using the
[environment commands](../environments/README.md#edmamba-baseline). The patch
SHA-256 is `81922208bdf4b9b74582a0c136da4d5a061422b26755de4e0d38affa11937106`.
EDformer and E-RAFT use their pinned upstream sources without tracked patches.

## Labelled EDmamba evaluation

The DND21 runner uses seed 1024 and consecutive 10,240-event blocks, dropping the
incomplete final block of each stream. It normalizes time within each block to
`[0, 1]`, maps polarity to `{-1, +1}`, uses the released first grid sampler at
0.05 and exhaustive test partitioning at 0.1, strict-loads the checkpoint, and
maps sampled predictions back to original events through the inverse mapping.
The primary result is the unweighted mean of block ROC AUCs, not event-global
ROC AUC. The ten DND21 streams contribute 3,393 complete blocks and 34,744,320
evaluated events; 51,171 tail events are omitted.

The supplied checkpoint reproduces eight of ten published DND21 cells within
0.0003 AUC. Hotel-bar and Driving at 10 Hz differ by −0.002386 and +0.001466,
respectively. The retained fresh-process diagnostic does not eliminate these
differences. Checkpoint evaluation does not establish from-scratch training
reproducibility: upstream lacks the canonical raw-to-NPZ converter and exact
training/validation split manifest.

DVSCLEAN uses the documented direct-HDF5 protocol, 10,240-event blocks, seed 1024,
per-condition block-mean AUC, and an equal mean over ten conditions. Native
1280 × 720 coordinates and scaling to the ED24 coordinate range are separate
controls recorded in their respective artifacts. Upstream does not provide the
HDF5-to-NPZ preprocessing needed to claim equivalence to its published DVSCLEAN
benchmark. The EDmamba structural E-MLB row is not used: its released metric path
is incomplete. EDformer supplies the structural comparison.

## Structural EDformer evaluation

The runner uses the released 4,096-event blocks, recording-wide timestamp
normalization, unnormalized pixel coordinates, seed 230086 and retention rule
`sigmoid(f) < 0.005`. MESR uses 30,000-event slices. Run with
`--batch-blocks 1 --reps 1 --max-events 0`; larger batches alter floating-point
reduction order near the retention threshold. Group results by ND level and
retain the matching raw-stream metric.

The EDformer results cover repetition 1, with 48 sequences per cell. RSF cell
summaries may cover repetitions 1–3, so sequence-paired comparisons use only
repetition 1 and the finite intersection of defined scores, with its denominator
reported. Absolute MESR depends on stream caps and repetition subsets; compare
methods on the same inputs. MESR measures structural concentration, not per-event
classification accuracy. These restrictions and the source-build requirements
are detailed in [DATA.md](../DATA.md) and the
[EDformer setup](../environments/README.md#edformer-baseline).
