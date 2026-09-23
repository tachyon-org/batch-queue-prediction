"""One loader for the evaluation data and, crucially, the exact splits the harness
trains on.

    from eval.dataset import load_experiment
    d = load_experiment("e2")
    X, y = d.Xsub, d.wait_log
    te = d.test("temporal")            # indices, matching the harness exactly
"""

import json
import os

import numpy as np

from eval.helper import temporal_masks, terminal_time
from eval.paths import DATA_ROOT
from eval.splits import build_splits

# 2025-07-01 00:00 UTC, the deployment cutoff the temporal protocol simulates.
DEFAULT_CUTOFF = "2025-07-01"
# 2025-01-01: timestamps below this are unset sentinels, not real dates.
WINDOW_FLOOR = 1735689600
# The random split is drawn with this fixed generator so it is identical in the
# harness and in any notebook; it is deliberately NOT tied to the model seed.
SPLIT_RNG_SEED = 0

_SUBMIT_TIME_EXPERIMENTS = ("e2",)


def cutoff_epoch(cutoff=DEFAULT_CUTOFF):
    """ISO date or epoch seconds -> epoch seconds, pinned to UTC."""
    import datetime as dt

    v = str(cutoff).strip()
    if v.isdigit():
        return int(v)
    return int(dt.datetime.strptime(v, "%Y-%m-%d")
               .replace(tzinfo=dt.timezone.utc).timestamp())


class ExperimentData:
    """Matrices, targets and the harness's splits for one experiment."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def train(self, split="temporal"):
        return self.splits[split][0]

    def test(self, split="temporal"):
        return self.splits[split][1]

    @property
    def X(self):
        """The feature matrix this experiment actually uses."""
        return self.Xsub if self.experiment in _SUBMIT_TIME_EXPERIMENTS else self.Xmatch

    def out_of_time(self):
        """The shared out-of-time test slice, scored by both arms, trained on by
        neither. Empty under the legacy design."""
        return getattr(self, "oot", np.array([], dtype=np.int64))

    def __repr__(self):
        s = "  ".join(f"{k}: train {len(a):,} / test {len(b):,}"
                      for k, (a, b) in self.splits.items())
        n_oot = len(self.out_of_time())
        return (f"<ExperimentData {self.experiment} @ {self.cutoff}  {s}"
                + (f"  oot: {n_oot:,}" if n_oot else "") + ">")


def load_experiment(experiment="e2", cutoff=DEFAULT_CUTOFF, data_root=None,
                    mmap=True, verbose=True):
    """Loads the data and reproduces the harness's splits for `experiment`.
    """
    root = data_root or DATA_ROOT
    mm = "r" if mmap else None

    def _load(name):
        return np.load(os.path.join(root, name), mmap_mode=mm)

    targets = np.load(os.path.join(root, "targets_and_masks.npz"))
    schema = {}
    meta_path = os.path.join(root, "schema_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            schema = json.load(f)

    qs = targets["qs"]
    cut = cutoff_epoch(cutoff)

    # Wait time is observable at execution start; failure and attribution only at
    # termination. E2 therefore gets an earlier observation time than E1/E3.
    label_src = targets["jst"] if experiment in _SUBMIT_TIME_EXPERIMENTS else targets["comp"]
    tau, _tau_src = terminal_time(
        label_src,
        job_start=targets["jst"],
        wall_clock=targets["wall"] if "wall" in targets.files else None,
        qdate=qs,
        floor=WINDOW_FLOOR,
    )
    tr_t, te_t, _ = temporal_masks(qs, cut, label_time=tau, verbose=verbose)
    tri_t, tei_t = np.where(tr_t)[0], np.where(te_t)[0]

    # Both arms come from eval/splits.py, which feat-engineering.ipynb also calls,
    # so the notebook that writes the matrices and this loader cannot disagree.
    _sp = build_splits(tri_t, tei_t, order=tau, n_rows=len(qs), verbose=verbose)
    oot = _sp.pop("oot")
    splits = _sp

    ran = None
    ran_path = os.path.join(root, "ran.npy")
    if os.path.exists(ran_path):
        ran = _load("ran.npy").astype(bool)
    elif "ran" in targets.files:
        ran = targets["ran"].astype(bool)
    if ran is not None and experiment in ("e1", "e3"):
        # Never-ran jobs have no run-time outcome to predict, but stay in the
        # matrices because they occupied the queue.
        splits = {k: (a[ran[a]], b[ran[b]]) for k, (a, b) in splits.items()}

    wait_sv = targets["wait_sv"]
    d = ExperimentData(
        experiment=experiment, cutoff=cutoff, cutoff_epoch=cut, root=root,
        Xmatch=_load("Xmatch.npy"), Xsub=_load("Xsub.npy"),
        failed=targets["failed"], hw=targets["hw"], fault_type=targets["fault_type"],
        wait_sv=wait_sv, wait_log=np.log1p(np.maximum(wait_sv, 0)), oot=oot,
        qs=qs, jst=targets["jst"], comp=targets["comp"], ran=ran,
        splits=splits, schema=schema,
        xmatch_cols=schema.get("XMATCH_COLS"), xsub_cols=schema.get("XSUB_COLS"),
        ncat_match=schema.get("NCAT_MATCH"), ncat_sub=schema.get("NCAT_SUB"),
    )
    if verbose:
        print(d)
    return d
