"""The one definition of the evaluation splits.

`eval/dataset.py` and `feat-engineering.ipynb` both call `build_splits`, so the
notebook that writes the matrices and the loader that reads them cannot disagree.

The window is divided by label-observation time into three parts:

    pre-cutoff        everything before the cutoff                  (Feb - Jun)
    protocol period   post-cutoff, before the out-of-time slice     (July)
    out-of-time       the latest OOT_FRACTION of the post-cutoff    (August)

Each protocol partitions `pre-cutoff + protocol period` into its own train and test,
with the same row counts, so the arms differ only in HOW the pool is partitioned:

    temporal    train = pre-cutoff            test = protocol period (chronological)
    random      train = a random draw of the pool, same size
                test  = a random remainder, same size as temporal's test

The random arm therefore trains on rows from the protocol period -- data the temporal
arm cannot have -- which is the leak being measured, and each arm's own test set is
what that protocol would report.

The out-of-time slice is then scored by BOTH models. Neither trained on any of it and
nothing is selected on it, so it is a second test set rather than a validation set: it
shows whether the random protocol's advantage survives into a period later than
anything either model saw, or evaporates.

"legacy" restores the pre-2026-09-23 behaviour, where the random arm permuted the
whole window and the two arms were scored on different rows.

The arms keep the names "random" and "temporal" so result files, plotting code and
the paper's terminology carry over unchanged.
"""

import os

import numpy as np

SPLIT_RNG_SEED = 42

SPLIT_DESIGN = os.environ.get("FIFE_SPLIT_DESIGN", "shared-oot")

# Latest share of the post-cutoff population held out as the common out-of-time test
# set. 0.41 is August on the Feb-Aug window; the remainder (July) is the protocol
# period that the two arms partition between them.
OOT_FRACTION = float(os.environ.get("FIFE_OOT_FRACTION", "0.41"))


def build_splits(tri_t, tei_t, order, n_rows=None, design=None, oot_fraction=None,
                 seed=SPLIT_RNG_SEED, verbose=True):
    """Build both protocol arms plus the shared out-of-time slice.

    `tri_t` / `tei_t` are index arrays for the pre- and post-cutoff populations, as
    produced by `temporal_masks` on label-observation time. `order` is the per-row time
    the post-cutoff rows are sorted by. `n_rows` is the height of the full matrix,
    needed only by the legacy design.

    Returns {"random": (train, test), "temporal": (train, test), "oot": idx}.
    """
    design = design or SPLIT_DESIGN
    frac = OOT_FRACTION if oot_fraction is None else float(oot_fraction)
    rng = np.random.default_rng(seed)
    tri_t = np.asarray(tri_t)
    tei_t = np.asarray(tei_t)

    if design == "legacy":
        perm = rng.permutation(np.arange(int(n_rows)))
        rte = np.sort(perm[: len(tei_t)])
        rtr = np.sort(perm[len(tei_t):])
        if verbose:
            print("[legacy split] random arm permutes the whole window; the arms are "
                  "scored on DIFFERENT rows and there is no out-of-time slice",
                  flush=True)
        return {"random": (rtr, rte), "temporal": (tri_t, tei_t),
                "oot": np.array([], dtype=np.int64)}

    if design != "shared-oot":
        raise ValueError(f"unknown split design {design!r}; "
                         f"expected 'shared-oot' or 'legacy'")

    # Rank the post-cutoff rows chronologically; unknown times sort last.
    ot = np.asarray(order, dtype=np.float64)[tei_t]
    ot = np.where(np.isfinite(ot), ot, np.inf)
    rank = np.argsort(ot, kind="stable")

    n_oot = int(round(frac * len(tei_t)))
    if not 0 < n_oot < len(tei_t):
        raise ValueError(f"oot_fraction={frac} yields {n_oot} rows from {len(tei_t)}")
    cut = len(tei_t) - n_oot

    proto = np.sort(tei_t[rank[:cut]])      # earlier post-cutoff: the protocol period
    oot = np.sort(tei_t[rank[cut:]])        # latest: scored by both arms

    # Temporal arm: the chronological partition of the pool.
    tr_temporal, te_temporal = tri_t, proto

    # Random arm: the same pool partitioned at random, sized to match exactly, so the
    # arms differ only in how rows were assigned.
    pool = np.concatenate([tri_t, proto])
    shuffled = rng.permutation(pool)
    tr_random = np.sort(shuffled[: len(tr_temporal)])
    te_random = np.sort(shuffled[len(tr_temporal): len(tr_temporal) + len(te_temporal)])

    if verbose:
        import datetime as _dt

        def _d(x):
            return (_dt.datetime.fromtimestamp(float(x), _dt.timezone.utc)
                    .strftime("%Y-%m-%d") if np.isfinite(x) else "?")

        post = float(np.isin(tr_random, proto).mean()) * 100
        print(f"[shared-oot split] pool {len(pool):,} | temporal train "
              f"{len(tr_temporal):,} / test {len(te_temporal):,} (through "
              f"{_d(ot[rank[cut - 1]])}) | random same sizes, {post:.1f}% of its "
              f"training rows from the protocol period", flush=True)
        print(f"[shared-oot split] out-of-time test {len(oot):,} rows from "
              f"{_d(ot[rank[cut]])}, scored by both arms, trained on by neither",
              flush=True)

    return {"random": (tr_random, te_random),
            "temporal": (tr_temporal, te_temporal),
            "oot": oot}


def describe(splits, qs=None, cutoff_epoch=None):
    """Rows, date range and post-cutoff share per arm, for a run log or a notebook."""
    out = []
    for name, val in splits.items():
        parts = (("test", val),) if name == "oot" else zip(("train", "test"), val)
        for part, idx in parts:
            row = {"split": name, "part": part, "n": int(len(idx))}
            if qs is not None and len(idx):
                v = np.asarray(qs)[idx]
                v = v[np.isfinite(v) & (v > 0)]
                if len(v):
                    row["first"] = float(v.min())
                    row["last"] = float(v.max())
                    if cutoff_epoch is not None:
                        row["pct_post_cutoff"] = float((v >= cutoff_epoch).mean()) * 100
            out.append(row)
    return out
