"""The one definition of the evaluation splits.

`eval/dataset.py` and `feat-engineering.ipynb` both call `build_splits`, so the
notebook that writes the matrices and the loader that reads them cannot disagree.

"8020" (the default since 2026-10-06; prototyped step by step in
experiment-setup.ipynb, section B):

    cutoff       a fixed UTC midnight (--cutoff, default 2025-07-10): the midnight
                 nearest the point where 80% of E1/E3 labels have been observed
    window       every job whose label is observed before the cutoff
    test (OOT)   every job observed after it, MINUS straddlers: jobs queued before
                 the cutoff are purged (their prediction would have been made before
                 it, by a model that could not see the last pre-cutoff outcomes)
    R            a random RANDOM_TEST_FRACTION of the window, drawn over all rows so a
                 job is on the same side in every experiment

    random      train = window minus R     test = R, and the OOT test set
    temporal    train = the same rows      test = the OOT test set

Validation is carved out of the shared training rows by `validation_split`: the
temporal arm validates on jobs SUBMITTED on or after a date V (`val_start_date`), and
the random arm on a random draw of exactly as many jobs, so both arms fit and validate
on equal-sized sets.

The window is divided by label-observation time into a training window and a later
out-of-time slice:

    window            everything before the out-of-time slice      (Feb - Jul)
    out-of-time       the latest OOT_FRACTION of the post-cutoff    (August)

"matched-window" (the default since 2026-10-01). Both arms train on the SAME rows: the
window minus a random RANDOM_TEST_FRACTION held out as the random arm's own test set.

    random      train = window minus R     test = R (random) and the OOT slice
    temporal    train = the same rows      test = the OOT slice only

The arms then differ only in how each carves its validation slice out of those rows
(`holdout_split`: a random 10% vs the most recent 10%), which sets the threshold for
every model and the epoch count for the neural ones. "Random, Random" against
"Random, OOT" is the optimism of random evaluation on one model; "Random, OOT" against
"Temporal, OOT" is the effect of selecting on recent data, with training data held
fixed. The temporal arm's test set IS the OOT slice, so its protocol metrics and its
`oot` block describe the same rows.

"shared-oot" (2026-09-23 to 2026-10-01). The arms partition the window differently:
temporal trains on the pre-cutoff rows and tests on the protocol period (July); the
random arm draws the same sizes at random, so it trains on July rows the temporal arm
cannot see. Both are also scored on the OOT slice -- but with unequal training windows,
so an OOT difference mixes protocol with recency.

"legacy" restores the pre-2026-09-23 behaviour, where the random arm permuted the
whole window and the two arms were scored on different rows.

The arms keep the names "random" and "temporal" so result files, plotting code and
the paper's terminology carry over unchanged.
"""

import os

import numpy as np

SPLIT_RNG_SEED = 42

SPLIT_DESIGN = os.environ.get("FIFE_SPLIT_DESIGN", "8020")

# 8020 and matched-window: share of the training window held out at random as the
# random arm's own test set.
RANDOM_TEST_FRACTION = float(os.environ.get("FIFE_RANDOM_TEST_FRACTION", "0.10"))

# Latest share of the post-cutoff population held out as the common out-of-time test
# set. 0.41 is August on the Feb-Aug window; the remainder (July) is the protocol
# period that the two arms partition between them.
OOT_FRACTION = float(os.environ.get("FIFE_OOT_FRACTION", "0.41"))

# 8020: share of the training rows (by submit time) that sets the validation date V.
VAL_FRAC = float(os.environ.get("FIFE_VAL_FRAC", "0.10"))


def build_splits(tri_t, tei_t, order, n_rows=None, design=None, oot_fraction=None,
                 seed=SPLIT_RNG_SEED, verbose=True, qs=None, cutoff=None):
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

    if design == "8020":
        if qs is None or cutoff is None or n_rows is None:
            raise ValueError("the 8020 design needs qs, cutoff and n_rows")
        qs_a = np.asarray(qs, dtype=np.float64)
        # Drawn first from the seeded generator, over every row, exactly as
        # experiment-setup.ipynb step 5 does -- so R is the same set of jobs.
        in_r = rng.random(int(n_rows)) < RANDOM_TEST_FRACTION
        window = np.sort(tri_t)
        te_random = window[in_r[window]]
        train = window[~in_r[window]]
        post = np.sort(tei_t)
        straddle = qs_a[post] < cutoff
        oot = post[~straddle]
        if verbose:
            print(f"[8020 split] window {len(window):,} | both arms train on the same "
                  f"{len(train):,} rows | random test R {len(te_random):,} "
                  f"({RANDOM_TEST_FRACTION:.0%} of the window)", flush=True)
            print(f"[8020 split] test {len(post):,} -> {len(oot):,} after purging "
                  f"{int(straddle.sum()):,} straddlers queued before the cutoff", flush=True)
        return {"random": (train, te_random), "temporal": (train, oot), "oot": oot}

    if design not in ("matched-window", "shared-oot"):
        raise ValueError(f"unknown split design {design!r}; "
                         f"expected '8020', 'matched-window', 'shared-oot' or 'legacy'")

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

    if design == "matched-window":
        window = np.concatenate([tri_t, proto])
        shuffled = rng.permutation(window)
        n_te = int(round(RANDOM_TEST_FRACTION * len(window)))
        te_random = np.sort(shuffled[:n_te])
        train = np.sort(shuffled[n_te:])
        if verbose:
            import datetime as _dt
            _ot = np.asarray(order, dtype=np.float64)
            _d = lambda x: (_dt.datetime.fromtimestamp(float(x), _dt.timezone.utc)
                            .strftime("%Y-%m-%d") if np.isfinite(x) else "?")
            print(f"[matched-window split] window {len(window):,} through "
                  f"{_d(np.nanmax(_ot[window]))} | both arms train on the same "
                  f"{len(train):,} rows | random test {len(te_random):,} "
                  f"({RANDOM_TEST_FRACTION:.0%} of the window, drawn at random)",
                  flush=True)
            print(f"[matched-window split] out-of-time test {len(oot):,} rows from "
                  f"{_d(np.nanmin(_ot[oot]))}: the temporal arm's only test set, and "
                  f"the random arm's second", flush=True)
        return {"random": (train, te_random), "temporal": (train, oot), "oot": oot}

    # shared-oot. Temporal arm: the chronological partition of the pool.
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


def val_start_date(qs, train_idx, val_frac=None):
    """The validation date V: the UTC midnight nearest the point where (1 - val_frac)
    of `train_idx` had been submitted (experiment-setup.ipynb step 6). The harness
    computes it once from the E1/E3 training rows and uses it for every experiment."""
    val_frac = VAL_FRAC if val_frac is None else float(val_frac)
    q = np.asarray(qs, dtype=np.float64)[np.asarray(train_idx)]
    q = q[np.isfinite(q)]
    qv = float(np.quantile(q, 1 - val_frac))
    day = 86400.0
    return min((np.floor(qv / day) * day, np.ceil(qv / day) * day),
               key=lambda c: abs((q < c).mean() - (1 - val_frac)))


def validation_split(tri, split, qs, val_start, seed=SPLIT_RNG_SEED):
    """(fit, validation) index arrays carved out of the training rows `tri`.

    Temporal arm: validation = jobs submitted on or after `val_start`; every job
    submitted before it trains, including jobs still running at V. Random arm: a
    random draw of exactly as many jobs, so both arms fit and validate on the same
    number of rows. The draw uses the split seed, not the model seed, so every model
    seed sees the same partition.
    """
    tri = np.sort(np.asarray(tri))
    late = np.asarray(qs, dtype=np.float64)[tri] >= val_start
    if split == "temporal":
        return tri[~late], tri[late]
    pick = np.zeros(len(tri), dtype=bool)
    pick[np.random.default_rng(seed).choice(len(tri), int(late.sum()), replace=False)] = True
    return tri[~pick], tri[pick]


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
