# External implementations and compatibility provenance

The public package contains our runners and measured evidence, not the external
model implementations, checkpoints, datasets, or compiled extensions. The
revisions below were read from the local checkouts used by this workspace on
2026-10-25. The E-RAFT revision is also recorded in the frozen result artifact.

| External source | Revision | License found at this revision |
|---|---|---|
| [EDmamba](https://github.com/gongzihang/EDmamba) | `302f7586c083aa0bd5f7befa13011a1b6c1e7022` | No root license file found; consult the authors' terms |
| [EDformer](https://github.com/NJUVISION/EDformer) | `820ff16b3bc50694e163f054fbb52cf3505acfff` | No root license file found; consult the authors' terms |
| [E-RAFT](https://github.com/uzh-rpg/E-RAFT) | `c58ce0524ea0ebfa9849991caafb547f44fe9bfd` | MIT |
| [cuke-emlb](https://github.com/KugaMaxx/cuke-emlb) | `105a1c36c368f10ff63b4a6f862992ae8842628d` | MIT |
| [EBF](https://github.com/shicy17/EBF) | `e2fa7760c841dc056bfde9fa1d70125a14275d2b` | GNU GPL v3 license text |

[upstream_manifest.json](upstream_manifest.json) records these revisions and
hashes of the license files that exist. The repository's MIT grant does not
relicense upstream code, patch context, model weights, or datasets. EBF is cited
as a baseline and dataset source; its upstream implementation is not copied here.

## EDmamba compatibility patch

[patches/edmamba-import-compatibility.patch](patches/edmamba-import-compatibility.patch)
is the actual local Git diff against the revision above, restricted to the three
files documented by the original reproduction audit. It makes `dv_processing`
optional in the E-MLB dataset loader and AUC engine, and makes PyTorch3D/FAISS
optional in the loss module. The configured cross-entropy inference path does
not use these imports. It does not change model arithmetic, configuration,
checkpoint contents, or dataset values. The two trailing-newline differences
are preserved from the original local diff.

Apply it once to a clean checkout using the commands in
[the environment recipes](../environments/README.md). The patch hash is
`81922208bdf4b9b74582a0c136da4d5a061422b26755de4e0d38affa11937106`.
EDformer and E-RAFT had no tracked local modifications at this audit.

## Scope of reproducibility

The bundled EDmamba audit is a historical record and retains the original host
paths and then-current limitations. Use the portable paths and current dataset
instructions in [DATA.md](../DATA.md), not its historical shell examples.
The reported trained checkpoints are external inputs; training from scratch and
unreleased upstream preprocessing are outside the supported reproduction claim.
