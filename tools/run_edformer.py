"""Score an event stream with the released EDformer checkpoint.

Runs under the pinned `.venv-edformer-cu118` interpreter, because `sparseconvnet` and
`pytorch3d` are installed in that environment (see `environments/README.md`). It is
invoked as a subprocess so the calling phase can stay on the project's own interpreter.

The protocol is copied from the released `EDformer/eval_auc.py`, not reinvented:

  * timestamps are min-max normalised over the WHOLE stream before slicing, which is what
    that script does and which makes the normalisation depend on the file boundary;
  * the stream is cut into `seq_len = 4096` event slices and the incomplete tail dropped;
  * the per-event output is `sigmoid(logit)`, which its own AUC call feeds straight to
    `roc_curve(label, prediction)` with DND21's label convention, where 1 is noise. So a
    LARGER value means more noise-like, the same orientation this paper uses throughout.

Input  : .npz with `events` [N,4] int64 as (t_us, x, y, p) and `width`, `height`.
Output : .npy float32 of length `4096 * (N // 4096)`, aligned to the leading events.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

SEQ_LEN = 4096
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDFORMER_ROOT = Path(os.environ.get("RSF_EDFORMER_ROOT", str(PROJECT_ROOT / "EDformer")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=EDFORMER_ROOT / "pretrained_model.pth")
    parser.add_argument("--edformer-dir", type=Path,
                        default=EDFORMER_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.edformer_dir))
    from model import EDformer  # noqa: E402  (path must be set first)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    payload = np.load(args.input)
    events = payload["events"].astype(np.float64)
    height, width = int(payload["height"]), int(payload["width"])

    device = torch.device("cuda:0")
    model = EDformer(height=height, width=width).to(device)
    model.load_state_dict(torch.load(str(args.model), map_location=device))
    model.eval()

    t = events[:, 0]
    span = t.max() - t.min()
    normalised = (t - t.min()) / span if span > 0 else np.zeros_like(t)
    xytp = np.column_stack((normalised, events[:, 1], events[:, 2], events[:, 3]))

    n_slices = len(xytp) // SEQ_LEN
    if n_slices == 0:
        raise SystemExit(f"{args.input}: fewer than {SEQ_LEN} events")
    xytp = xytp[: n_slices * SEQ_LEN].reshape(n_slices, SEQ_LEN, 4)

    scores = np.empty((n_slices, SEQ_LEN), dtype=np.float32)
    for i in range(n_slices):
        batch = torch.tensor(xytp[i], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            scores[i] = torch.sigmoid(model(batch)).squeeze().cpu().numpy()
        if (i + 1) % 50 == 0 or i + 1 == n_slices:
            print(f"  slice {i + 1}/{n_slices}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, scores.reshape(-1))
    print(f"wrote {args.output} ({n_slices * SEQ_LEN} events)")


if __name__ == "__main__":
    main()
