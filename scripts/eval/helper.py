import argparse
import os
import glob
import json
import warnings
import sys
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    accuracy_score, roc_auc_score, average_precision_score, r2_score,
    matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
from eval.paths import DATA_ROOT, RESULTS_DIR, all_model_dirs as _all_model_dirs

MODEL_DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
    "lightgbm": "LightGBM",
    "lgb": "LightGBM",
    "catboost": "CatBoost",
    "cat": "CatBoost",
    "mlp": "MLP",
    "tabnet": "TabNet",
    "saint": "SAINT",
    "ft": "FT-Transformer",
    "ft_transformer": "FT-Transformer",
    "tsmixer": "TSMixer",
    "tabr": "TabR",
    "hierarchical": "Hierarchical",
}

# Attribution is not the same measurement in each family by default: trees report
# split-based importance, TabNet attention mass, and the remaining neural models the
# score drop under permutation. All are normalised to sum to 1, but a 5-point move in
# gain share and in permutation share are different phenomena, which is why the
# heatmaps keep the families in separate row blocks. load_saved_importances(...,
# tree_importance="permutation") puts the trees on the neural models' measure.
MODEL_FAMILIES = {
    "tree": ["xgboost", "xgb", "lightgbm", "lgb", "catboost", "cat"],
    "neural": ["mlp", "tabnet", "saint", "ft", "ft_transformer", "tsmixer", "tabr",
               "hierarchical"],
}

def model_family(name):
    """'tree', 'neural', or None for a model this module does not classify."""
    low = str(name).lower()
    for fam, members in MODEL_FAMILIES.items():
        if low in members:
            return fam
    return None


PREFERRED_MODEL_ORDER = [
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
    "tabnet",
    "saint",
    "ft",
    "ft_transformer",
    "tsmixer",
    "tabr",
    "hierarchical"
]

# ---- Memory Helpers ----
def _empty_gpu():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _nfeat(parts):
    return int(sum(1 if getattr(p, "ndim", 1) == 1 else p.shape[1] for p in parts))


def _get_slice(parts, idxs):
    """Zero-disk row extraction across list of memory arrays."""
    idxs = np.asarray(idxs)
    if len(parts) == 1:
        return parts[0][idxs]
    cols = [p[idxs] if p.ndim == 2 else p[idxs, None] for p in parts]
    return np.hstack(cols)


# ---- Model Metric & Thresholding Helpers ----
THR_GRID = np.linspace(0.05, 0.95, 91)


def pick_thr(y_true, y_prob, target=1):
    """Find the threshold on P(y=1) that maximizes F1 for class `target`.

    A prediction is positive when `y_prob >= thr`, so raising the threshold trades
    positive-class recall for precision and does the reverse for the negative class.
    The two classes therefore peak at different cuts. `target=0` tunes for the
    negative class's own F1, which is the operating point to use when that class
    names a task of its own rather than "everything else".
    """
    best_f1, best_thr = 0.0, 0.5
    for thr in THR_GRID:
        preds = (y_prob >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, preds, average=None, labels=[0, 1], zero_division=0
        )
        if f1[target] > best_f1:
            best_f1, best_thr = f1[target], thr
    return float(best_thr)


def cls_metrics(y_true, y_prob, thr=0.5):
    """Compute standard binary classification evaluation metrics."""
    preds = (y_prob >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "threshold": float(thr),
    }

def cls_metrics_per_class(
    y_true, y_prob, thr=0.5, thr_neg=None, pos_name="hardware", neg_name="payload"
):
    """Binary metrics split out per class, for tasks where both classes are of interest.

    E3 asks two questions of one classifier -- "did we catch the hardware faults?"
    and "did we catch the payload faults?" -- and a single positive-class row only
    answers the first.

    `thr` is the cut tuned for the positive class; `thr_neg`, when given, is the cut
    tuned separately for the negative class. Each class block then reports its
    threshold-dependent metrics at its *own* operating point, since the two tasks
    peak at different cuts. Left as None, both classes are scored at `thr` and the
    two blocks describe one shared decision rule.

    The threshold-free metrics -- ROC AUC and each class's PR AUC -- are unaffected
    by either choice, and are the fairer basis for comparing models.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = int(len(y_true))

    def _prfs(t):
        # index 0 -> negative class (neg_name), index 1 -> positive class (pos_name)
        return precision_recall_fscore_support(
            y_true, (y_prob >= t).astype(np.int8), average=None, labels=[0, 1],
            zero_division=0,
        )

    pr, rc, f1, sup = _prfs(thr)
    if thr_neg is None:
        thr_neg = thr
        pr_n, rc_n, f1_n = pr, rc, f1
    else:
        pr_n, rc_n, f1_n, _ = _prfs(thr_neg)

    # ROC AUC is invariant under swapping which class is "positive", so one value
    # describes the ranking for both. Average precision is not -- the negative
    # class needs its own labels and scores inverted.
    roc = float(roc_auc_score(y_true, y_prob))
    ap_pos = float(average_precision_score(y_true, y_prob))
    ap_neg = float(average_precision_score(1 - y_true, 1.0 - y_prob))

    per_class = {
        pos_name: {
            "threshold": float(thr),
            "pr_auc": ap_pos,
            "precision": float(pr[1]),
            "recall": float(rc[1]),
            "f1": float(f1[1]),
            "support": int(sup[1]),
            "prevalence": float(sup[1] / n) if n else 0.0,
        },
        neg_name: {
            "threshold": float(thr_neg),
            "pr_auc": ap_neg,
            "precision": float(pr_n[0]),
            "recall": float(rc_n[0]),
            "f1": float(f1_n[0]),
            "support": int(sup[0]),
            "prevalence": float(sup[0] / n) if n else 0.0,
        },
    }

    return {
        "roc_auc": roc,
        # The remaining shared metrics describe the single decision rule at `thr`,
        # the only one of the two cuts a deployed classifier could actually run.
        "accuracy": float(accuracy_score(y_true, (y_prob >= thr).astype(np.int8))),
        # MCC uses all four confusion-matrix cells and is symmetric between the two
        # classes, so one value scores both tasks and a trivial majority-class
        # classifier gets 0 rather than the ~0.9 accuracy flatters it with.
        "mcc": float(matthews_corrcoef(y_true, (y_prob >= thr).astype(np.int8))),
        "balanced_accuracy": float((rc[0] + rc[1]) / 2.0),
        "macro_f1": float((f1[0] + f1[1]) / 2.0),
        # Each task at its own best cut. Not reachable by one rule, so read it as a
        # per-task ceiling rather than a deployable operating point.
        "macro_f1_tuned": float((f1_n[0] + f1[1]) / 2.0),
        "threshold": float(thr),
        "n_test": n,
        "positive_class": pos_name,
        **per_class,
        # Flat aliases for the positive class, matching the pre-split schema.
        "pr_auc": ap_pos,
        "precision": float(pr[1]),
        "recall": float(rc[1]),
        "f1": float(f1[1]),
    }


# ---- Wait-time regimes and reference predictors ----
#
# Fitted by `fgmix` in data-analysis.ipynb: a 3-component lognormal mixture on
# FermiGrid workflow jobs (pilots excluded, waits capped at 1 d), n = 47,794,160.
# k = 3 sits at the BIC elbow -- the raw BIC argmin is 7, but the sweep flattens
# after 3 and KS stops moving (0.0081 -> 0.0069). Medians are in seconds.
WAIT_MIX = {
    "site": "FermiGrid",
    "k": 3,
    "cap_s": 86_400.0,
    "weights": [0.167, 0.450, 0.383],
    "medians_s": [28.0, 779.0, 8988.0],
    "sds_log": [1.566, 1.073, 1.064],
    "ks": 0.0081,
    "n": 47_794_160,
}

# Error-reporting strata. The first three edges come from the fitted mixture, cut
# where one component overtakes the next (106 s and 2,857 s, rounded to 2 min and
# 45 min); k = 3 components give exactly these three regions:
#   inst -- matched into an already-idle pilot slot
#   turn -- waiting for a slot to turn over
#   prov -- waiting for new pilot provisioning
#
# `park` is NOT a fourth regime. It is everything beyond the 1 d cap applied to the
# fit's population (2.46% of jobs), so the fit says nothing about it -- the observed
# distribution runs smoothly through the cap (no pile-up at 86,400 s; 42,417 jobs in
# the following hour). It is reported separately rather than folded into `prov`
# because its errors are ~20x larger (MAE 229,689 s vs 10,920 s), so pooling would
# let ~2% of jobs dominate the regime that matters.
WAIT_REGIMES = (
    ("inst", 0.0, 120.0),
    ("turn", 120.0, 2_700.0),
    ("prov", 2_700.0, 86_400.0),
    ("park", 86_400.0, np.inf),
)


def mixture_cdf_s(t, mix=None):
    """Lognormal-mixture CDF evaluated at wait times `t` (seconds)."""
    from scipy import stats as _st

    mix = mix or WAIT_MIX
    t = np.clip(np.atleast_1d(np.asarray(t, dtype=np.float64)), 1e-9, None)
    lt = np.log(t)
    return sum(
        w * _st.norm.cdf(lt, np.log(m), s)
        for w, m, s in zip(mix["weights"], mix["medians_s"], mix["sds_log"])
    )


def mixture_quantile_s(q, mix=None, lo=1e-3, hi=1e7, n=200_000):
    """Inverts `mixture_cdf_s` on a log grid; returns wait times in seconds."""
    mix = mix or WAIT_MIX
    grid = np.logspace(np.log10(lo), np.log10(hi), n)
    return np.interp(np.asarray(q, dtype=np.float64), mixture_cdf_s(grid, mix), grid)






def best_constant_log(y_true_log, metric="within2x"):
    """The single best constant prediction under `metric`, as a log1p value.

    This is the floor every E2 model has to clear. It matters because the wait
    distribution is wide but unimodal in log space: a constant already scores
    within2x ~ 0.25, so a model at 0.28 has bought almost nothing from its
    features. `metric` is "within2x" (maximised) or "mae_log1p" (minimised, where
    the answer is just the median).

    For a constant prediction c (raw seconds) the within-2x set is an interval in
    the true wait -- c <= 2T+1 and T <= 2c+1 means T in [(c-1)/2, 2c+1] -- so the
    search is a pair of searchsorted lookups against the sorted labels rather than
    a full pass per grid point. That difference matters: the training slice here
    runs to tens of millions of rows.
    """
    y = np.asarray(y_true_log, dtype=np.float64)
    if metric == "mae_log1p":
        return float(np.median(y))
    wt = np.sort(np.expm1(y))
    grid_s = np.logspace(0, 5, 2000)
    lo = np.searchsorted(wt, (grid_s - 1.0) / 2.0, side="left")
    hi = np.searchsorted(wt, 2.0 * grid_s + 1.0, side="right")
    return float(np.log1p(grid_s[int(np.argmax(hi - lo))]))


def wait_reference_metrics(y_true_log_train, y_true_log_test):
    """Scores the two no-feature reference predictors on the test slice.

    Both are fitted on `y_true_log_train` only, so they are admissible baselines
    rather than oracles:
      - `constant`   : the best single number under within-2x
      - `mixture_med`: the median of the fitted lognormal mixture (`WAIT_MIX`)
    Report model metrics against these, not against zero -- an R2_log of 0.27 reads
    very differently once the reader knows a constant already reaches within2x 0.25.
    """
    yte = np.asarray(y_true_log_test, dtype=np.float64)
    out = {}
    c = best_constant_log(y_true_log_train)
    out["constant"] = {"pred_s": float(np.expm1(c)),
                       **reg_metrics(yte, np.full_like(yte, c))}
    m = float(np.log1p(mixture_quantile_s(0.5)))
    out["mixture_med"] = {"pred_s": float(np.expm1(m)),
                          **reg_metrics(yte, np.full_like(yte, m))}
    return out


def skill_score(model_value, reference_value, higher_is_better=True):
    """Fraction of the reference predictor's headroom that the model closes.

    1.0 is a perfect model, 0.0 is no better than the reference, negative is worse.
    For error metrics (`higher_is_better=False`) this is 1 - model/reference.
    """
    m, r = float(model_value), float(reference_value)
    if not higher_is_better:
        return float("nan") if r == 0 else 1.0 - m / r
    return float("nan") if r >= 1.0 else (m - r) / (1.0 - r)


def reg_metrics(y_true_log, pred_log, legacy_bins=True):
    """Computes honest wait time regression metrics on both log and original second scales.

    Error is broken out by queueing regime (`WAIT_REGIMES`), whose boundaries come
    from the fitted mixture's component crossovers rather than from round numbers.
    `legacy_bins` additionally emits the older <10m / 10m-2h / >2h keys so entries
    appended to results/wait_time_results.json before this change stay comparable --
    the new keys are named separately so no existing key silently changes meaning.
    """
    wt_true = np.expm1(y_true_log)
    wt_pred = np.clip(np.expm1(pred_log), 0, None)
    ae = np.abs(wt_pred - wt_true)

    # sMAPE on the raw second scale, symmetric so a 10x over- and
    # under-prediction cost the same. Jobs where both true and predicted wait
    # round to ~0 have no meaningful relative error, so they score 0 rather
    # than dividing by a vanishing denominator.
    denom = (np.abs(wt_true) + np.abs(wt_pred)) / 2.0
    nonzero = denom > 1e-8
    smape_vals = np.zeros_like(wt_true, dtype=np.float64)
    if np.any(nonzero):
        smape_vals[nonzero] = np.abs(wt_true[nonzero] - wt_pred[nonzero]) / denom[nonzero]

    out = {
        "r2_log": float(r2_score(y_true_log, pred_log)),
        # Scale-free error in log1p space -- the space the models train in, so
        # this is the loss they actually optimize rather than a raw-second
        # figure dominated by the longest queues.
        "mae_log1p": float(np.mean(np.abs(y_true_log - pred_log))),
        "smape_pct": float(100.0 * np.mean(smape_vals)),
        "median_ae_s": float(np.median(ae)),
        "within2x": float(
            np.mean((wt_pred <= 2 * wt_true + 1) & (wt_true <= 2 * wt_pred + 1))
        ),
        "mae_raw_s": float(np.mean(ae)),
    }

    # Per-regime error. Mean AND median AE per stratum: waits are heavily
    # right-skewed inside every stratum too, so mean AE can look far worse than
    # the error a typical job in that stratum actually sees.
    for name, lo, hi in WAIT_REGIMES:
        b = (wt_true >= lo) & (wt_true < hi)
        out[f"n_{name}"] = int(b.sum())
        out[f"mae_{name}"] = float(ae[b].mean()) if b.any() else np.nan
        out[f"median_ae_{name}"] = float(np.median(ae[b])) if b.any() else np.nan
        out[f"within2x_{name}"] = float(
            np.mean((wt_pred[b] <= 2 * wt_true[b] + 1) & (wt_true[b] <= 2 * wt_pred[b] + 1))
        ) if b.any() else np.nan

    if legacy_bins:
        b_short = wt_true < 600
        b_med = (wt_true >= 600) & (wt_true < 7200)
        b_long = wt_true >= 7200
        out.update({
            "mae_10m": float(ae[b_short].mean()) if b_short.any() else np.nan,
            "mae_2h": float(ae[b_med].mean()) if b_med.any() else np.nan,
            "mae_long": float(ae[b_long].mean()) if b_long.any() else np.nan,
            "median_ae_10m": float(np.median(ae[b_short])) if b_short.any() else np.nan,
            "median_ae_2h": float(np.median(ae[b_med])) if b_med.any() else np.nan,
            "median_ae_long": float(np.median(ae[b_long])) if b_long.any() else np.nan,
        })
    return out


# ---- Reproducibility ----
DEFAULT_SEED = 42


def set_global_seed(seed=DEFAULT_SEED, deterministic=False):
    """Seed every RNG a fit touches, so a run is reproducible from its seed alone.

    Seeds Python, NumPy and Torch (CPU and all CUDA devices). Call this immediately
    before constructing a model: the neural trainers draw their weight init, dropout
    masks and batch shuffling from the global Torch RNG, so seeding here covers them
    without threading a seed through every architecture.

    `deterministic` additionally pins cuDNN to deterministic kernels. That makes a
    single seed reproduce bit-for-bit, but it disables kernel autotuning and can cost
    a large fraction of training throughput -- so it is off by default. Seed-to-seed
    variance, which is what a spread across seeds measures, does not need it.
    """
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return int(seed)


# ---- Test-set prediction persistence ----
# The cascade, bootstrap intervals and seed-variance reporting all need the raw
# test-set scores. Re-deriving them by reloading models is fragile -- the neural
# trainers persist bare state_dicts, so reloading means reconstructing each
# architecture -- and it is wasted compute. Every fit writes its scores once here.
from eval.paths import PRED_ROOT as PRED_DIR


def pred_path(exp_tag, lib, split, seed=DEFAULT_SEED, pred_dir=None):
    d = pred_dir if pred_dir is not None else PRED_DIR
    return os.path.join(d, f"{exp_tag}_{lib}_{split}_seed{seed}.npz")


def save_predictions(exp_tag, lib, split, idx, probs, seed=DEFAULT_SEED, pred_dir=None,
                     **extra):
    """Persist test-set scores. `idx` are row indices into the full dataset."""
    path = pred_path(exp_tag, lib, split, seed, pred_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        idx=np.asarray(idx, dtype=np.int64),
        probs=np.asarray(probs, dtype=np.float32),
        **extra,
    )
    print(f"[{lib}] Saved {len(probs):,} test predictions to {path}", flush=True)
    return path


def load_predictions(exp_tag, lib, split, seed=DEFAULT_SEED, pred_dir=None,
                     calibrated=True):
    """Returns (idx, probs) as written by `save_predictions`.

    Prefers the calibrated scores when the run stored them, since composing two
    uncalibrated stages reorders the cascade. Pass calibrated=False for the raw
    model output.
    """
    path = pred_path(exp_tag, lib, split, seed, pred_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved predictions at {path}. Re-run that experiment so the fit "
            f"writes its test scores."
        )
    z = np.load(path)
    key = "probs_cal" if (calibrated and "probs_cal" in z.files) else "probs"
    return z["idx"], z[key]


def align_predictions(idx_a, probs_a, idx_b, probs_b):
    """Align two score vectors onto their shared rows, preserving idx_a's order.

    The two cascade stages score different populations -- E1 covers every test job,
    E3 only the ones it was asked about -- so composing them means intersecting on
    row index rather than assuming the arrays line up positionally.
    """
    idx_a = np.asarray(idx_a)
    idx_b = np.asarray(idx_b)
    common = np.intersect1d(idx_a, idx_b, assume_unique=True)
    return (
        common,
        np.asarray(probs_a)[np.searchsorted(idx_a, common)],
        np.asarray(probs_b)[np.searchsorted(idx_b, common)],
    )


def cascade_metrics(y_hw, p_fail, p_hw_cond, thr_fail=None, thr_attr=None):
    """Match-time hardware-fault detection as a two-stage cascade.

    Scores every test job, not only the ones already known to have failed, so the
    numbers describe a population a deployed system can actually assemble. The
    composed score is P(failure) * P(hardware | failure); with calibrated stages
    that is P(hardware), and it is a ranking score either way.

    Reports three things a reviewer will ask for separately:

    * `soft` -- threshold-free quality of the composed score over all test jobs.
    * `stage1` -- what the failure detector costs the pipeline. Its recall on true
      hardware faults is a hard ceiling on cascade recall: a hardware fault whose
      job is not flagged as failing can never be attributed. This is where
      first-stage error propagation becomes visible.
    * `hard` -- the deployable rule, gate on P(failure) then attribute, at the
      supplied operating points.

    `p_fail_only` is the ablation: ranking by the failure score alone. If the
    composed score does not beat it, the attribution stage is adding nothing at
    match time, whatever its conditional metrics look like.
    """
    y_hw = np.asarray(y_hw).astype(np.int8)
    p_fail = np.asarray(p_fail, dtype=np.float64)
    p_hw_cond = np.asarray(p_hw_cond, dtype=np.float64)
    n = int(len(y_hw))
    score = p_fail * p_hw_cond

    def _rank(s):
        return {
            "pr_auc": float(average_precision_score(y_hw, s)),
            "roc_auc": float(roc_auc_score(y_hw, s)),
        }

    out = {
        "n_test": n,
        "n_hardware": int(y_hw.sum()),
        # A random ranker scores exactly this AP, so it is the number every
        # reported AP has to be read against.
        "prevalence": float(y_hw.mean()),
        "soft": _rank(score),
        "p_fail_only": _rank(p_fail),
        "p_attr_only": _rank(p_hw_cond),
    }

    if thr_fail is not None:
        gate = p_fail >= thr_fail
        caught = int((gate & (y_hw == 1)).sum())
        out["stage1"] = {
            "threshold": float(thr_fail),
            "flagged": int(gate.sum()),
            "flagged_frac": float(gate.mean()),
            # Ceiling on anything the second stage can recover.
            "hardware_recall_ceiling": float(caught / max(int(y_hw.sum()), 1)),
        }
        if thr_attr is not None:
            pred = (gate & (p_hw_cond >= thr_attr)).astype(np.int8)
            pr, rc, f1, _ = precision_recall_fscore_support(
                y_hw, pred, average=None, labels=[0, 1], zero_division=0
            )
            out["hard"] = {
                "threshold_fail": float(thr_fail),
                "threshold_attr": float(thr_attr),
                "precision": float(pr[1]),
                "recall": float(rc[1]),
                "f1": float(f1[1]),
                "specificity": float(rc[0]),
                "mcc": float(matthews_corrcoef(y_hw, pred)),
                "alerts": int(pred.sum()),
                "alert_frac": float(pred.mean()),
            }
    return out

def aggregate_seeds(runs, keys=None):
    """Mean, std and range across repeated fits that differ only by seed.

    `runs` is a list of metric dicts, one per seed. Nested per-class blocks are
    flattened to "block.metric" so hardware/payload sub-metrics aggregate too.
    Reporting std alongside the mean is what lets a reader tell an ordering that
    reflects a real difference from one that is seed noise.
    """
    def _flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(_flat(v, f"{prefix}{k}."))
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f"{prefix}{k}"] = float(v)
        return out

    flat = [_flat(r) for r in runs]
    if not flat:
        return {}
    if keys is None:
        keys = sorted(set().union(*(f.keys() for f in flat)))
    agg = {"n_seeds": len(flat)}
    for k in keys:
        vals = np.array([f[k] for f in flat if k in f], dtype=np.float64)
        if not len(vals):
            continue
        agg[k] = {
            "mean": float(vals.mean()),
            # Sample std (ddof=1): these are a sample of seeds, not the population.
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n": int(len(vals)),
        }
    return agg


RESULT_NAMES = {"e1": "protocol_eval", "e2": "wait_time", "e3": "fault_attr"}


def load_results(name, results_dir=None):
    """{model: {"<split>__seed<n>": metrics, ...}} for one experiment, gathered from
    results/<name>/<model>.json. `name` is the results name ("protocol_eval",
    "wait_time", "fault_attr") or the experiment ("e1", "e2", "e3")."""
    import glob as _glob
    name = RESULT_NAMES.get(name, name)
    folder = os.path.join(results_dir if results_dir is not None else RESULTS_DIR, name)
    out = {}
    for path in sorted(_glob.glob(os.path.join(folder, "*.json"))):
        with open(path) as fh:
            out[os.path.splitext(os.path.basename(path))[0]] = json.load(fh)
    return out


def collect_seed_runs(exp_name, lib, split, results_dir=None):
    """Gather every seed's metrics for one model/split out of a results JSON.

    Repeated runs are stored under "<split>" (the default seed) and
    "<split>__seed<N>" (the rest), so this pulls them back together for
    `aggregate_seeds`.
    """
    entry = load_results(exp_name, results_dir).get(lib, {})
    runs = [v for k, v in entry.items() if k == split or k.startswith(f"{split}__seed")]
    return runs

# ---- Temporal partitioning on label-observation time ----
def terminal_time(completion, job_start=None, wall_clock=None, qdate=None, floor=None):
    """When each job's outcome became observable, as an epoch-seconds array.

    Splitting on submission time cannot simulate a model deployed at the cutoff: a job
    submitted in June that terminates in July carries a label nobody could have known.
    The observation time is the job's terminal event, resolved in order of how directly
    it is recorded:

      1. CompletionDate, where present (86% of jobs);
      2. JobStartDate + RemoteWallClockTime, for jobs that ran but were removed rather
         than completing (2.6%);
      3. QDate, for jobs removed before ever starting (11.2%).

    Case 3 is a LOWER BOUND, not an observation -- HTCondor does not record a removal
    timestamp for a job that never ran. Those jobs land on the training side by their
    queue time, so a job queued shortly before the cutoff and removed shortly after is
    still mis-assigned. The exposure is bounded and should be reported: quote the count
    of fallback-assigned training jobs queued within the removal window of the cutoff.
    """
    out = np.asarray(completion, dtype=np.float64).copy()
    ok = np.isfinite(out) & (out > 0)
    if floor is not None:
        ok &= out >= floor
    src = np.where(ok, 0, -1)

    if job_start is not None and wall_clock is not None:
        js = np.asarray(job_start, dtype=np.float64)
        wc = np.asarray(wall_clock, dtype=np.float64)
        can = (~ok) & np.isfinite(js) & (js > 0) & np.isfinite(wc) & (wc > 0)
        if floor is not None:
            can &= js >= floor
        out[can] = js[can] + wc[can]; src[can] = 1; ok |= can

    if qdate is not None:
        q = np.asarray(qdate, dtype=np.float64)
        can = (~ok) & np.isfinite(q) & (q > 0)
        out[can] = q[can]; src[can] = 2; ok |= can

    out[~ok] = np.nan
    return out, src


def temporal_masks(qs, cutoff, label_time=None, verbose=True):
    """Partition on when each label became observable, not on when the job was queued.

    Training is every job whose outcome was already known at the cutoff; the test set
    is everything still outstanding, which includes boundary jobs -- queued before the
    cutoff, terminating after it. Those are predictions a model deployed at the cutoff
    would still have had open, so they belong on the evaluation side rather than being
    discarded. No training label can post-date the cutoff by construction.

    Falls back to splitting on `qs` when no `label_time` is supplied.
    """
    qs = np.asarray(qs, dtype=np.float64)
    if label_time is None:
        train, test = qs < cutoff, qs >= cutoff
        stats = {"n_train": int(train.sum()), "n_test": int(test.sum()), "basis": "qdate"}
    else:
        lt = np.asarray(label_time, dtype=np.float64)
        usable = np.isfinite(lt)
        # Unresolvable terminal time: fall back to queue time so no job is dropped.
        eff = np.where(usable, lt, qs)
        train, test = eff < cutoff, eff >= cutoff
        stats = {
            "n_train": int(train.sum()), "n_test": int(test.sum()), "basis": "label_time",
            "n_moved_to_test": int((train | test).sum() and ((qs < cutoff) & test).sum()),
            "n_unresolved": int((~usable).sum()),
        }
    assert not (train & test).any() and (train | test).all(), "partition must cover all rows once"
    if verbose:
        extra = ""
        if stats["basis"] == "label_time":
            extra = (f" | {stats['n_moved_to_test']:,} boundary jobs moved to test"
                     f" | {stats['n_unresolved']:,} rows fell back to queue time")
        print(f"[temporal split] train {stats['n_train']:,} | test {stats['n_test']:,}{extra}",
              flush=True)
    return train, test, stats


def entity_memorization(y_true, probs, seen, entity_name="entity", min_n=1000):
    """Compare performance on test jobs from seen vs. unseen entities.

    Both subsets come from the same test period and the same trained model, so
    chronology, drift and training data are all held constant and the remaining gap
    is attributable to entity familiarity rather than to the four things that move
    at once between a random and a temporal split.

    Leads on ROC AUC deliberately: the two subsets typically differ sharply in class
    prevalence, and average precision moves with prevalence while ROC AUC does not.
    AP is reported alongside as a lift over each subset's own base rate, which is the
    only way it can be compared across subsets at all.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    probs = np.asarray(probs, dtype=np.float64)
    seen = np.asarray(seen, dtype=bool)

    out = {"entity": entity_name}
    for name, m in (("seen", seen), ("unseen", ~seen)):
        n = int(m.sum())
        blk = {"n": n, "frac": float(m.mean())}
        if n >= min_n and len(np.unique(y_true[m])) == 2:
            prev = float(y_true[m].mean())
            ap = float(average_precision_score(y_true[m], probs[m]))
            blk.update({
                "prevalence": prev,
                "roc_auc": float(roc_auc_score(y_true[m], probs[m])),
                "pr_auc": ap,
                "pr_auc_lift": ap / prev if prev > 0 else float("nan"),
            })
        else:
            blk["skipped"] = "too few rows or single-class subset"
        out[name] = blk

    if "roc_auc" in out["seen"] and "roc_auc" in out["unseen"]:
        out["delta_roc_auc"] = out["seen"]["roc_auc"] - out["unseen"]["roc_auc"]
        out["delta_pr_auc_lift"] = out["seen"]["pr_auc_lift"] - out["unseen"]["pr_auc_lift"]
    return out


# Resolved against XMATCH_COLS / XSUB_COLS by name, so a feature-layout change
# surfaces as a missing entity rather than a silently wrong column.
COLD_START_ENTITIES = (
    ("group", "Group"),
    ("user", "Owner"),
    ("campaign", "CampaignName"),
    ("campaign_stage", "CampaignStageName"),
    ("site", "MatchSite"),
)


def entity_columns(X, cols, wanted=COLD_START_ENTITIES):
    """Entity code vectors from a feature matrix, by column name. Entities absent
    from the matrix (MatchSite is not known at submit time) are skipped, not faked."""
    idx = {c: i for i, c in enumerate(cols)}
    out = {}
    for label, name in wanted:
        if name in idx:
            out[label] = np.asarray(X[:, idx[name]]).ravel()
    return out


def subgroup_metrics(y_true, probs, codes, kind="bin", min_n=1000, top_k=12):
    """Per-subgroup metrics for the largest `top_k` levels; smaller levels pool into
    "other". Matters here because FermiGrid runs ~88% of jobs, so an aggregate score
    can look strong while every small site fails."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    codes = np.asarray(codes)
    lv, cnt = np.unique(codes, return_counts=True)
    order = np.argsort(cnt)[::-1]
    keep = [lv[i] for i in order[:top_k] if cnt[i] >= min_n]

    rows = {}
    covered = np.zeros(len(codes), dtype=bool)
    for level in keep:
        m = codes == level
        covered |= m
        rows[str(int(level))] = _subgroup_block(y_true[m], probs[m], kind)
    if (~covered).any():
        rows["other"] = _subgroup_block(y_true[~covered], probs[~covered], kind)
    return rows


def _subgroup_block(y, p, kind):
    """One subgroup's metrics; classification and regression handled separately."""
    n = int(len(y))
    blk = {"n": n}
    if n == 0:
        return blk
    if kind == "reg":
        blk.update(reg_metrics(y, p, legacy_bins=False))
        return blk
    blk["prevalence"] = float(np.mean(y))
    if len(np.unique(y)) < 2:
        blk["skipped"] = "single-class subgroup"
        return blk
    ap = float(average_precision_score(y, p))
    blk.update({
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": ap,
        # AP moves with prevalence, so across subgroups with different base rates
        # only the lift over each subgroup's own base rate is comparable.
        "pr_auc_lift": ap / blk["prevalence"] if blk["prevalence"] > 0 else float("nan"),
    })
    return blk


def cold_start_report(y_true, probs, X, cols, train_idx, test_idx, kind="bin",
                      entities=COLD_START_ENTITIES, min_n=1000, top_k=12,
                      verbose=True):
    """Cold-start and subgroup behaviour for one fitted model on one split.

    Both are measured on the same test period with the same fitted model, so unlike
    the random-vs-temporal gap neither is confounded by drift. Post-hoc: call it with
    predictions loaded from disk (`load_predictions`).

    Returns {entity_label: {"cold_start": ..., "subgroups": ...}}.
    """
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    ent_all = entity_columns(X, cols, entities)
    out = {}
    for label, codes_all in ent_all.items():
        tr_codes = codes_all[train_idx]
        te_codes = codes_all[test_idx]
        seen_levels = set(np.unique(tr_codes).tolist())
        seen = np.fromiter((c in seen_levels for c in te_codes), dtype=bool,
                           count=len(te_codes))
        block = {
            "n_levels_train": int(len(seen_levels)),
            "n_levels_test": int(len(np.unique(te_codes))),
            "n_levels_unseen": int(len(set(np.unique(te_codes).tolist()) - seen_levels)),
        }
        if kind == "reg":
            block["cold_start"] = _cold_start_reg(y_true, probs, seen, label, min_n)
        else:
            block["cold_start"] = entity_memorization(y_true, probs, seen,
                                                      entity_name=label, min_n=min_n)
        block["subgroups"] = subgroup_metrics(y_true, probs, te_codes, kind=kind,
                                              min_n=min_n, top_k=top_k)
        out[label] = block
        if verbose:
            cs = block["cold_start"]
            u = cs.get("unseen", {})
            s = cs.get("seen", {})
            head = (f"  {label:15s} levels tr/te/unseen "
                    f"{block['n_levels_train']}/{block['n_levels_test']}/"
                    f"{block['n_levels_unseen']}")
            if kind == "reg":
                print(f"{head}  seen R2 {s.get('r2_log', float('nan')):+.3f} "
                      f"(n={s.get('n', 0):,})  unseen R2 "
                      f"{u.get('r2_log', float('nan')):+.3f} (n={u.get('n', 0):,})",
                      flush=True)
            else:
                print(f"{head}  seen AUC {s.get('roc_auc', float('nan')):.3f} "
                      f"(n={s.get('n', 0):,})  unseen AUC "
                      f"{u.get('roc_auc', float('nan')):.3f} (n={u.get('n', 0):,})  "
                      f"delta {cs.get('delta_roc_auc', float('nan')):+.3f}", flush=True)
    return out


def _cold_start_reg(y_true, pred, seen, entity_name, min_n):
    """Regression counterpart of `entity_memorization` for the wait-time task."""
    y_true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    seen = np.asarray(seen, dtype=bool)
    out = {"entity": entity_name}
    for name, m in (("seen", seen), ("unseen", ~seen)):
        n = int(m.sum())
        blk = {"n": n, "frac": float(m.mean())}
        if n >= min_n:
            blk.update(reg_metrics(y_true[m], pred[m], legacy_bins=False))
        else:
            blk["skipped"] = "too few rows"
        out[name] = blk
    for k in ("r2_log", "mae_log1p", "within2x"):
        if k in out["seen"] and k in out["unseen"]:
            out[f"delta_{k}"] = out["seen"][k] - out["unseen"][k]
    return out

# ---- Probability calibration ----
def fit_calibrator(p_val, y_val, method="isotonic", clip=1e-6):
    """Fit a calibrator on the held-out slice and return a callable score -> probability.

    Both cascade stages are fitted with `scale_pos_weight`, which deliberately distorts
    their output away from the true posterior. That is harmless for a single stage --
    ranking metrics are invariant to any monotone transform -- but it breaks the
    composition: the product of two monotonically distorted scores is NOT a monotone
    function of the product of the true probabilities, so an uncalibrated cascade
    reorders jobs relative to a calibrated one. Calibrating each stage makes
    P(failure) * P(hardware | failure) an actual probability rather than a heuristic.

    Fitted on the same held-out slice used for threshold selection, which the model was
    not fitted on and which carries the natural (unweighted) class balance.

    'isotonic' is a monotone step fit -- flexible, needs a few thousand positives.
    'platt' is a one-parameter sigmoid on the logit -- stiffer, safer when positives
    are scarce.
    """
    p_val = np.clip(np.asarray(p_val, dtype=np.float64), clip, 1 - clip)
    y_val = np.asarray(y_val).astype(np.int8)
    if len(np.unique(y_val)) < 2:
        return lambda p: np.asarray(p, dtype=np.float64)

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(p_val, y_val)
        return lambda p: ir.predict(np.clip(np.asarray(p, dtype=np.float64), clip, 1 - clip))

    if method == "platt":
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e10, solver="lbfgs")
        lr.fit(np.log(p_val / (1 - p_val)).reshape(-1, 1), y_val)
        def _apply(p):
            p = np.clip(np.asarray(p, dtype=np.float64), clip, 1 - clip)
            return lr.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
        return _apply

    raise ValueError(f"unknown calibration method: {method!r}")


def calibration_report(y_true, p_raw, p_cal, n_bins=10):
    """Brier score and expected calibration error, before and after.

    Brier and ECE both move with calibration; ROC AUC does not, because a monotone
    map cannot change the ranking. Seeing AUC hold constant while Brier falls is the
    confirmation that the calibrator fixed the probabilities without disturbing the
    ordering.
    """
    y = np.asarray(y_true).astype(np.float64)

    def _ece(p):
        p = np.asarray(p, dtype=np.float64)
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        tot = 0.0
        for b in range(n_bins):
            m = idx == b
            if m.any():
                tot += m.mean() * abs(p[m].mean() - y[m].mean())
        return float(tot)

    return {
        "brier_raw": float(np.mean((np.asarray(p_raw) - y) ** 2)),
        "brier_cal": float(np.mean((np.asarray(p_cal) - y) ** 2)),
        "ece_raw": _ece(p_raw),
        "ece_cal": _ece(p_cal),
        "mean_pred_raw": float(np.mean(p_raw)),
        "mean_pred_cal": float(np.mean(p_cal)),
        "base_rate": float(y.mean()),
    }


def thr_sample(a, max_n=500_000, seed=42):
    """Subsample index array for fast threshold calculation if large."""
    a = np.asarray(a)
    if len(a) > max_n:
        return np.random.default_rng(seed).choice(a, max_n, replace=False)
    return a


# Fixed epoch budget for every neural trainer. Nothing selects an epoch count from
# data, so there is no validation set for E2 and no refit: each model trains this many
# epochs on its arm's full training window and is scored once. Uniform across models
# and across split protocols, which is what makes the two arms comparable.
NEURAL_EPOCHS = int(os.environ.get("FIFE_NEURAL_EPOCHS", "5"))

# Categorical embeddings for the neural models, OFF by default. Measured on 10.7M
# rows, embedding the code columns costs R2_log on BOTH split arms while reaching
# LOWER training loss -- memorising entity identity that does not recur after the
# cutoff:
#     arm       no cat    low-cardinality only    all columns
#     random    +0.764    +0.288                  +0.442
#     temporal  +0.177    -11.647                 -0.029
# Every saved model in Pre-Trained-Models/neural-models/old_082026 also has
# cat_idxs=[]. Set FIFE_NEURAL_CAT=1 to re-enable across all neural trainers.
NEURAL_CAT = os.environ.get("FIFE_NEURAL_CAT", "0") not in ("0", "false", "False")

VAL_FRAC = 0.10


def holdout_split(tri, frac=VAL_FRAC, order=None, seed=42):
    """Carve a threshold-selection slice out of the training indices.

    The operating point has to be chosen on predictions the model has not fit, or
    it inherits however much that model memorized its training rows -- a bias that
    lands unevenly across model families and so distorts cross-model comparison.
    The returned index arrays are disjoint: fit on the first, pick the cut on the
    second.

    `order` is a per-row sort key (queue-start time, for this dataset). Given one,
    the slice is the most recent `frac` of the training window, so the validation
    rows sit between the fitting rows and the test period exactly as the temporal
    protocol intends. Without it the slice is drawn at random, which is the
    matching choice for the random split protocol.
    """
    tri = np.asarray(tri)
    n_val = int(round(frac * len(tri)))
    if n_val < 1 or n_val >= len(tri):
        raise ValueError(f"frac={frac} yields {n_val} validation rows from {len(tri)}")

    if order is None:
        cut = np.random.default_rng(seed).permutation(len(tri))
    else:
        # argpartition, not a full sort -- only the boundary position matters.
        cut = np.argpartition(np.asarray(order)[tri], len(tri) - n_val)

    # Sorted so downstream mmap slicing stays sequential.
    return np.sort(tri[cut[:-n_val]]), np.sort(tri[cut[-n_val:]])


def log_result(exp_name, model, split, **kwargs):
    pass  # Placeholder for custom logging callbacks


class MLPAdapter(nn.Module):
    """Wraps train.mlp.MLP safely with categorical bounds checking."""

    def __init__(self, model, n_cat, cards_f):
        super().__init__()
        self.model = model
        self.n_cat = n_cat
        self.cards_f = cards_f

    def forward(self, x):
        if self.n_cat > 0:
            # Clamp categorical columns to valid embedding bounds [0, max_cardinality - 1]
            xc = x[:, : self.n_cat].long()
            for col_i, card in enumerate(self.cards_f):
                xc[:, col_i] = torch.clamp(xc[:, col_i], min=0, max=card - 1)
            xn = x[:, self.n_cat :].float()
        else:
            xc = torch.zeros((x.shape[0], 0), dtype=torch.long, device=x.device)
            xn = x.float()
        return self.model(xc, xn)

def _predict_pytorch_model(net, X_mat, batch_size, device):
    """Executes robust inference by dynamically probing the model's required forward pass signature."""
    net = net.to(device)
    net.eval()

    probe = torch.tensor(X_mat[:2], dtype=torch.float32, device=device)

    # Determine categorical feature count
    n_cat = 0
    if hasattr(net, "cards_f") and net.cards_f is not None:
        n_cat = len(net.cards_f)
    elif hasattr(net, "embs") and isinstance(net.embs, torch.nn.ModuleList):
        n_cat = len(net.embs)

    forward_fn = None

    # Probe 1: Split x_cat and x_num
    if n_cat > 0:
        try:
            x_c = probe[:, :n_cat].long()
            x_n = probe[:, n_cat:]
            out = net(x_c, x_n)
            if isinstance(out, (torch.Tensor, tuple)):
                forward_fn = lambda b: net(b[:, :n_cat].long(), b[:, n_cat:])
        except Exception:
            pass

    # Probe 2: Single matrix input
    if forward_fn is None:
        try:
            out = net(probe)
            if isinstance(out, (torch.Tensor, tuple)):
                forward_fn = lambda b: net(b)
        except Exception:
            pass

    # Probe 3: Explicit empty x_cat with x_num
    if forward_fn is None:
        try:
            x_c = torch.empty((2, 0), dtype=torch.long, device=device)
            out = net(x_c, probe)
            forward_fn = lambda b: net(
                torch.empty((len(b), 0), dtype=torch.long, device=device), b
            )
        except Exception:
            pass

    if forward_fn is None:
        forward_fn = lambda b: net(b)

    probs = []
    with torch.no_grad():
        for start in range(0, len(X_mat), batch_size):
            bx = torch.tensor(
                X_mat[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            out = forward_fn(bx)

            if isinstance(out, tuple):
                out = out[0]
            if hasattr(out, "logits"):
                out = out.logits

            if out.ndim > 1 and out.shape[1] > 1:
                p = torch.softmax(out, dim=1)[:, 1]
            else:
                p = torch.sigmoid(out.squeeze())

            probs.append(p.cpu().numpy())

    return np.concatenate(probs)

def predict_proba_model(m_lower, path, X_data, batch_size=32768, device=None):
    """Predicts class probabilities across tree-based, TabNet, and PyTorch models."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_mat = np.asarray(X_data, dtype=np.float32)

    # 1. Tree Models
    if m_lower in ("xgboost", "xgb"):
        import xgboost as xgb

        try:
            bst = xgb.Booster()
            bst.load_model(path)
            probs = bst.predict(xgb.DMatrix(X_mat))
        except Exception:
            import joblib

            model = joblib.load(path)
            probs = (
                model.predict_proba(X_mat)[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(X_mat)
            )
        return np.asarray(probs, dtype=np.float32)

    elif m_lower in ("lightgbm", "lgb"):
        import lightgbm as lgb

        try:
            bst = lgb.Booster(model_file=path)
            probs = bst.predict(X_mat)
        except Exception:
            import joblib

            model = joblib.load(path)
            probs = (
                model.predict_proba(X_mat)[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(X_mat)
            )
        return np.asarray(probs, dtype=np.float32)

    elif m_lower in ("catboost", "cat"):
        from catboost import CatBoostClassifier

        cb = CatBoostClassifier()
        cb.load_model(path)
        return np.asarray(cb.predict_proba(X_mat)[:, 1], dtype=np.float32)

    # 2. TabNet
    elif m_lower == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier

        clf = TabNetClassifier()
        clf.load_model(path)
        return np.asarray(clf.predict_proba(X_mat)[:, 1], dtype=np.float32)

    # 3. PyTorch Models
    elif path.endswith((".pt", ".pth")):
        loaded_obj = torch.load(path, map_location=device)

        if isinstance(loaded_obj, torch.nn.Module):
            net = loaded_obj
        elif isinstance(loaded_obj, dict) and isinstance(
            loaded_obj.get("model"), torch.nn.Module
        ):
            net = loaded_obj["model"]
        else:
            state_dict = (
                loaded_obj.get("state_dict")
                or loaded_obj.get("model_state_dict")
                or loaded_obj
                if isinstance(loaded_obj, dict)
                else loaded_obj
            )

            # Strip key prefixes if present
            if isinstance(state_dict, dict):
                cleaned_sd = {}
                for k, v in state_dict.items():
                    new_k = k
                    for prefix in [
                        "module.",
                        "model.",
                        "_orig_mod.",
                        "net.",
                        "mlp.",
                    ]:
                        if new_k.startswith(prefix):
                            new_k = new_k[len(prefix) :]
                    cleaned_sd[new_k] = v
                state_dict = cleaned_sd

            net = _instantiate_pytorch_model(
                m_lower, X_mat.shape[1], state_dict=state_dict
            )

        if net is None:
            raise ValueError(f"Could not instantiate PyTorch model for {m_lower}")

        return _predict_pytorch_model(net, X_mat, batch_size, device)

    raise ValueError(f"Unrecognized model path format: {path}")


def evaluate_protocol_models(
    X_all,
    y_all,
    rtr,
    rte,
    tri_t,
    tei_t,
    model_dirs=None,
    preferred_models=PREFERRED_MODEL_ORDER,
    batch_size=32768,
    exp_tag="bin_e1",
    output_path="results/protocol_eval_results.json",
):
    """Evaluates models across Random and Temporal splits using thr_r statically calibrated on the Random split validation set."""
    if tree_importance not in ("gain", "permutation"):
        raise ValueError(f"tree_importance must be 'gain' or 'permutation', got {tree_importance!r}")
    aliases = {
        "xgboost": ["xgboost", "xgb"],
        "lightgbm": ["lightgbm", "lgb"],
        "catboost": ["catboost", "cat"],
        "mlp": ["mlp"],
        "tabnet": ["tabnet"],
        "saint": ["saint"],
        "ft": ["ft", "ft_transformer"],
        "tsmixer": ["tsmixer"],
        "tabr": ["tabr"],
    }

    results = {}
    trs_r = thr_sample(rtr)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in preferred_models:
            m_lower = m.lower()
            search_names = aliases.get(m_lower, [m_lower])
            matched_paths = {}

            for split in ["random", "temporal"]:
                for d in (model_dirs or _all_model_dirs()):
                    if not os.path.exists(d):
                        continue
                    for fname in sorted(os.listdir(d)):
                        fn_lower = fname.lower()
                        has_alias = any(s in fn_lower for s in search_names)
                        has_split = split in fn_lower
                        has_tag = exp_tag is None or exp_tag.lower() in fn_lower

                        if (
                            has_alias
                            and has_split
                            and has_tag
                            and fn_lower.endswith(
                                (
                                    ".zip",
                                    ".pt",
                                    ".pth",
                                    ".json",
                                    ".txt",
                                    ".bin",
                                    ".cbm",
                                )
                            )
                        ):
                            matched_paths[split] = os.path.join(d, fname)
                            break
                    if split in matched_paths:
                        break

            if "random" in matched_paths and "temporal" in matched_paths:
                try:
                    print(
                        f"\n[Evaluating Fixed Threshold] {m.upper()} (Tag: {exp_tag})"
                    )

                    # 1. Calibrate threshold statically on Random split validation subsample
                    p_trs_r = predict_proba_model(
                        m_lower,
                        matched_paths["random"],
                        X_all[trs_r],
                        batch_size=batch_size,
                    )
                    thr_r = pick_thr(y_all[trs_r], p_trs_r)

                    # 2. Evaluate Random test set
                    p_te_r = predict_proba_model(
                        m_lower,
                        matched_paths["random"],
                        X_all[rte],
                        batch_size=batch_size,
                    )
                    m_rnd = cls_metrics(y_all[rte], p_te_r, thr=thr_r)

                    # 3. Evaluate Temporal test set using thr_r
                    p_te_t = predict_proba_model(
                        m_lower,
                        matched_paths["temporal"],
                        X_all[tei_t],
                        batch_size=batch_size,
                    )
                    m_tmp = cls_metrics(y_all[tei_t], p_te_t, thr=thr_r)

                    results[m_lower] = {"random": m_rnd, "temporal": m_tmp}

                    print(
                        f"  Random   | ROC: {m_rnd['roc_auc']:.3f} | PR: {m_rnd['pr_auc']:.3f} | P: {m_rnd['precision']:.3f} | R: {m_rnd['recall']:.3f} | F1: {m_rnd['f1']:.3f} @ thr={thr_r:.4f}"
                    )
                    print(
                        f"  Temporal | ROC: {m_tmp['roc_auc']:.3f} | PR: {m_tmp['pr_auc']:.3f} | P: {m_tmp['precision']:.3f} | R: {m_tmp['recall']:.3f} | F1: {m_tmp['f1']:.3f} @ thr={thr_r:.4f} (fixed)"
                    )

                except Exception as e:
                    print(f"[Error] Failed evaluating model {m}: {e}")

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] Complete evaluation results saved to {output_path}")

    return results


def compute_permutation_importance(
    model, X_eval, y_eval, device="cuda", batch_size=32768, sample_size=None, kind="bin",
    predict_fn=None, seed=0,
):
    """Permutation feature importance: the drop in ROC-AUC for a classifier
    (kind="bin"), or in R^2 on the model's own target scale -- log1p wait for E2 --
    for a regressor (kind="reg").

    `model` is a PyTorch module (sigmoid applied for kind="bin"). Pass `predict_fn`
    instead -- a function from a float32 matrix to probabilities (bin) or raw
    predictions (reg) -- for any other model, e.g. the trees (see _tree_predict_fn).

    The shuffles come from `seed`, so every model scored on the same X_eval sees the
    same permutation of each feature. Uses all of X_eval unless sample_size is smaller.
    Negative drops are clipped to 0 and the result is normalised to sum to 1.
    """
    rng = np.random.default_rng(seed)
    if predict_fn is None:
        if hasattr(model, "eval"):
            model.eval()

        def predict_fn(X_data):
            preds = []
            for i in range(0, len(X_data), batch_size):
                bx = torch.from_numpy(X_data[i : i + batch_size]).float().to(device)
                with torch.no_grad():
                    out = model(bx)
                    if hasattr(out, "logits"):
                        out = out.logits
                    if out.ndim > 1:
                        out = out.squeeze(-1)
                    out = out.float() if kind == "reg" else torch.sigmoid(out.float())
                    preds.append(out.cpu().numpy())
            return np.concatenate(preds)

    if sample_size is not None and len(X_eval) > sample_size:
        idx = rng.choice(len(X_eval), size=sample_size, replace=False)
        X_sub, y_sub = X_eval[idx], y_eval[idx]
    else:
        X_sub, y_sub = X_eval.copy(), y_eval.copy()
    X_sub = np.ascontiguousarray(X_sub, dtype=np.float32)

    _score = r2_score if kind == "reg" else roc_auc_score
    base = _score(y_sub, predict_fn(X_sub))

    n_features = X_sub.shape[1]
    importance_scores = np.zeros(n_features)
    for f_idx in range(n_features):
        X_perm = X_sub.copy()
        X_perm[:, f_idx] = X_perm[rng.permutation(len(X_perm)), f_idx]
        importance_scores[f_idx] = max(0.0, base - _score(y_sub, predict_fn(X_perm)))

    total_imp = importance_scores.sum()
    if total_imp > 0:
        importance_scores = importance_scores / total_imp

    return importance_scores


def _tree_predict_fn(model_name, file_path, kind="bin"):
    """A predict function for a saved tree model, for compute_permutation_importance:
    probabilities for kind="bin", raw predictions (log1p wait for E2) for kind="reg".
    Categorical columns are handled as at training: XGBoost reads the feature types
    stored in the model, LightGBM its stored categorical features, and CatBoost gets
    the integer-coded frame it was fitted on."""
    m_name = model_name.lower()
    if m_name in ("xgboost", "xgb"):
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(file_path)
        bst.set_param({"device": "cpu"})
        ft = bst.feature_types
        cat = bool(ft) and "c" in ft
        return lambda X: bst.predict(xgb.DMatrix(X, feature_types=ft, enable_categorical=cat))
    if m_name in ("lightgbm", "lgb"):
        import lightgbm as lgb
        bst = lgb.Booster(model_file=file_path)
        return lambda X: bst.predict(X)
    if m_name in ("catboost", "cat"):
        from catboost import CatBoostClassifier, CatBoostRegressor
        from train.tree import _cb_frame
        cb = CatBoostRegressor() if kind == "reg" else CatBoostClassifier()
        cb.load_model(file_path)
        cat_idx = list(cb.get_cat_feature_indices())

        def _predict(X):
            frame = _cb_frame(X, len(cat_idx), cat_idx) if cat_idx else None
            data = frame if frame is not None else X
            return cb.predict(data) if kind == "reg" else cb.predict_proba(data)[:, 1]
        return _predict
    raise ValueError(f"Unsupported tree model identifier: {model_name}")


def extract_tree_importance(model_name, file_path, n_feats, kind="bin"):
    """Loads saved decision tree models and extracts normalized feature gain importances
    (summing to 1.0). kind="reg" loads the E2 regressors."""
    m_name = model_name.lower()

    if m_name in ("lightgbm", "lgb"):
        import lightgbm as lgb

        bst = lgb.Booster(model_file=file_path)
        imp = bst.feature_importance(importance_type="gain")

    elif m_name in ("catboost", "cat"):
        from catboost import CatBoostClassifier, CatBoostRegressor

        cb = CatBoostRegressor() if kind == "reg" else CatBoostClassifier()
        cb.load_model(file_path)
        imp = cb.get_feature_importance()

    elif m_name in ("xgboost", "xgb"):
        import xgboost as xgb

        m = xgb.XGBRegressor() if kind == "reg" else xgb.XGBClassifier()
        m.load_model(file_path)
        imp = m.feature_importances_

    else:
        raise ValueError(f"Unsupported tree model identifier: {model_name}")

    imp = np.nan_to_num(np.asarray(imp, dtype=np.float32))

    if len(imp) < n_feats:
        imp = np.pad(imp, (0, n_feats - len(imp)))
    elif len(imp) > n_feats:
        imp = imp[:n_feats]

    total = np.sum(imp)
    return (imp / total) if total > 0 else imp

def _instantiate_pytorch_model(m_name, n_feats, state_dict=None):
    """Dynamically instantiates PyTorch models inspecting state_dict shapes."""
    m_name = m_name.lower()

    if m_name == "mlp":
        from train.mlp import MLP

        if state_dict is not None:
            emb_keys = sorted(
                [
                    k
                    for k in state_dict
                    if k.startswith("embs.") and k.endswith(".weight")
                ],
                key=lambda k: int(k.split(".")[1]),
            )
            cards_f = [state_dict[k].shape[0] for k in emb_keys]
            sum_emb_dim = sum(state_dict[k].shape[1] for k in emb_keys)

            # Filter exclusively for 2D weights (nn.Linear) and exclude 1D weights (nn.BatchNorm1d)
            linear_keys = sorted(
                [
                    k for k, v in state_dict.items()
                    if k.startswith("body.") and k.endswith(".weight") and v.ndim == 2
                ],
                key=lambda k: int(k.split(".")[1])
            )

            first_linear_in = state_dict[linear_keys[0]].shape[1]
            n_num = first_linear_in - sum_emb_dim
            out_dim = state_dict["head.weight"].shape[0]

            # Reconstruct hidden layer dimensions from Linear output features
            hidden = [state_dict[k].shape[0] for k in linear_keys]

            net = MLP(cards_f=cards_f, n_num=n_num, out=out_dim, hidden=tuple(hidden))
            net.load_state_dict(state_dict)
            return MLPAdapter(net, len(cards_f), cards_f)
        else:
            net = MLP(cards_f=[], n_num=n_feats)
            return MLPAdapter(net, 0, [])

    elif m_name in ("ft", "ft_transformer"):
        from train.ft import FTTransformer

        d_token = 32
        depth = 2
        heads = 4
        if isinstance(state_dict, dict):
            if "feature_embedder.weight" in state_dict:
                d_token = state_dict["feature_embedder.weight"].shape[-1]
            transformer_keys = [k for k in state_dict if k.startswith("transformer.layers.")]
            if transformer_keys:
                depth = max([int(k.split(".")[2]) for k in transformer_keys]) + 1

        return FTTransformer(num_features=n_feats, d_token=d_token, depth=depth, heads=heads)

    elif m_name == "saint":
        try:
            from train.saint import SAINT

            d_token = 32
            depth = 2
            heads = 4
            cat_dims = []
            n_num = n_feats

            if isinstance(state_dict, dict):
                if "num_embed" in state_dict:
                    n_num = state_dict["num_embed"].shape[0]
                    d_token = state_dict["num_embed"].shape[1]

                layer_indices = [
                    int(k.split(".")[1])
                    for k in state_dict.keys()
                    if k.startswith("layers.") and k.split(".")[1].isdigit()
                ]
                if layer_indices:
                    depth = max(layer_indices) + 1

            return SAINT(
                n_num=n_num,
                cat_dims=cat_dims,
                d_token=d_token,
                depth=depth,
                heads=heads,
            )
        except Exception as e:
            print(f"[Error Instantiating SAINT] {e}")
            return None

    elif m_name == "tsmixer":
        # Takes the raw (rows, features) matrix like FT-Transformer; width and depth
        # are read off the saved weights.
        from train.tsmixer import TSMixer

        d_model, depth = 32, 3
        if isinstance(state_dict, dict):
            if "feature_proj.weight" in state_dict:
                d_model = state_dict["feature_proj.weight"].shape[0]
            blocks = {int(k.split(".")[1]) for k in state_dict
                      if k.startswith("blocks.") and k.split(".")[1].isdigit()}
            if blocks:
                depth = max(blocks) + 1
        return TSMixer(num_features=n_feats, d_model=d_model, depth=depth)

    return None

def load_saved_importances(
    model_dirs=None,
    models=(
        "xgboost",
        "lightgbm",
        "catboost",
        "mlp",
        "tabnet",
        "saint",
        "ft",
        "tabr",
        "tsmixer",
    ),
    splits=("random", "temporal"),
    n_feats=44,
    X_eval=None,
    y_eval=None,
    exp_tag="bin_e1",
    device=None,
    kind="bin",
    tree_importance="gain",
):
    """Scans directories, loads saved Tree, TabNet, and PyTorch DL models, and returns.

    `tree_importance` is "gain" (each library's own split-based importance, read from
    the saved model) or "permutation" (the same permutation importance as the neural
    models, on the same X_eval / y_eval with the same shuffles). The choice is recorded
    in imp_dict["__method__"] so the heatmap notes describe it.

    `kind="reg"` is for the E2 wait-time regressors: trees load as regressors, TabNet as
    a TabNetRegressor, and permutation importance is the drop in R^2 on `y_eval`, which
    must then be on the models' target scale (log1p wait seconds).

    an imp_dict keyed by (model, split) with normalized importances.

    `device=None` resolves to CUDA-if-present at CALL time. It must not be a default
    argument expression: Python evaluates those at import, so a `torch.cuda` probe
    there ran every time this module was imported or reloaded -- including from
    analysis notebooks that never touch a model -- and blocked outright whenever the
    driver was wedged. The other loaders here already take `device=None`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if tree_importance not in ("gain", "permutation"):
        raise ValueError(f"tree_importance must be 'gain' or 'permutation', got {tree_importance!r}")
    aliases = {
        "xgboost": ["xgboost", "xgb"],
        "lightgbm": ["lightgbm", "lgb"],
        "catboost": ["catboost", "cat"],
        "mlp": ["mlp"],
        "tabnet": ["tabnet"],
        "saint": ["saint"],
        "ft": ["ft", "ft_transformer"],
        "tabr": ["tabr"],
        "tsmixer": ["tsmixer"],
    }
    imp_dict = {"__method__": {"tree": tree_importance}}

    for m in models:
        m_lower = m.lower()
        search_names = aliases.get(m_lower, [m_lower])

        for split in splits:
            matched_path = None

            for d in (model_dirs or _all_model_dirs()):
                if not os.path.exists(d):
                    continue
                for fname in os.listdir(d):
                    fn_lower = fname.lower()
                    has_model_alias = any(s in fn_lower for s in search_names)
                    has_split = split in fn_lower
                    has_exp_tag = exp_tag is None or exp_tag.lower() in fn_lower

                    if has_model_alias and has_split and has_exp_tag:
                        matched_path = os.path.join(d, fname)
                        break
                if matched_path:
                    break

            if not matched_path:
                print(f"[Missing] No saved model found for {m}/{split}")
                continue

            # 1. Decision Trees
            if m_lower in ("xgboost", "xgb", "lightgbm", "lgb", "catboost", "cat"):
                try:
                    if tree_importance == "permutation":
                        if X_eval is None or y_eval is None:
                            print(f"[Skipping Permutation Imp] {m:10s} ({split:8s}): Pass X_eval and y_eval.")
                            continue
                        imp_dict[(m, split)] = compute_permutation_importance(
                            None, X_eval, y_eval, kind=kind,
                            predict_fn=_tree_predict_fn(m_lower, matched_path, kind=kind))
                        print(f"[Computed Permutation Imp] {m:10s} ({split:8s}) <- {matched_path}")
                    else:
                        imp_dict[(m, split)] = extract_tree_importance(
                            m_lower, matched_path, n_feats, kind=kind
                        )
                        print(f"[Loaded Tree Model] {m:10s} ({split:8s}) <- {matched_path}")
                except Exception as e:
                    print(f"[Error] Failed tree importance for {m}/{split} from {matched_path}: {e}")

            elif m_lower == "tabnet":
                try:
                    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

                    clf = TabNetRegressor() if kind == "reg" else TabNetClassifier()
                    clf.load_model(matched_path)

                    if X_eval is not None:
                        X_mat = (
                            X_eval.values
                            if hasattr(X_eval, "values")
                            else np.asarray(X_eval)
                        ).astype(np.float32)

                        # TabNet explain computes attention mask importance across X_eval
                        M_explain, _ = clf.explain(X_mat)
                        imp = M_explain.sum(axis=0)
                        imp = np.nan_to_num(imp)
                        total = np.sum(imp)
                        imp_dict[(m, split)] = (imp / total) if total > 0 else imp
                        print(f"[Loaded TabNet Model] {m:10s} ({split:8s}) computed feature importances successfully.")
                    else:
                        print(f"[Warning] Pass X_eval to compute TabNet feature importances: {m}/{split}")
                except Exception as e:
                    print(f"[Error] Failed loading TabNet model {m}/{split}: {e}")

            # 3. PyTorch Deep Learning Models
            elif matched_path.endswith((".pt", ".pth")):
                if X_eval is None or y_eval is None:
                    print(f"[Skipping Permutation Imp] {m:10s} ({split:8s}): Pass X_eval and y_eval.")
                    continue

                try:
                    state_dict = torch.load(matched_path, map_location=device)
                    net = _instantiate_pytorch_model(
                        m_lower, n_feats, state_dict=state_dict
                    )

                    if net is None:
                        print(f"[Warning] Could not auto-instantiate class for {m}. Skipping.")
                        continue

                    net = net.to(device)
                    if m_lower != "mlp":
                        net.load_state_dict(state_dict)

                    imp = compute_permutation_importance(
                        net, X_eval, y_eval, device=device, kind=kind
                    )
                    imp_dict[(m, split)] = imp
                    print(f"[Computed Permutation Imp] {m:10s} ({split:8s}) on {device}")
                except Exception as e:
                    print(f"[Error] Failed permutation importance for {m}/{split}: {e}")

    return imp_dict

def predict_reg_model(path, X_data, batch_size=32768, device=None):
    """Loads and predicts regression targets across Tree models, TabNet, and PyTorch architectures."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    fname = os.path.basename(path).lower()
    X_mat = np.asarray(X_data, dtype=np.float32)

    # 1. Tree Models
    if "xgboost" in fname or "xgb" in fname:
        import xgboost as xgb
        try:
            model = xgb.XGBRegressor()
            model.load_model(path)
            return model.predict(X_mat)
        except Exception:
            bst = xgb.Booster()
            bst.load_model(path)
            return bst.predict(xgb.DMatrix(X_mat))

    elif "lightgbm" in fname or "lgb" in fname:
        import lightgbm as lgb
        model = lgb.Booster(model_file=path)
        return model.predict(X_mat)

    elif "catboost" in fname or "cb" in fname:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(path)
        return model.predict(X_mat)

    # 2. TabNet
    elif "tabnet" in fname or fname.endswith(".zip"):
        from pytorch_tabnet.tab_model import TabNetRegressor
        model = TabNetRegressor()
        model.load_model(path)
        return model.predict(X_mat).squeeze()

    # 3. PyTorch Deep Learning Models (.pt / .pth)
    elif path.endswith((".pt", ".pth")):
        m_lower = None
        for key in ("mlp", "saint", "ft_transformer", "ft", "tsmixer", "tabr"):
            if key in fname:
                m_lower = key
                break

        if m_lower is None:
            raise ValueError(f"Could not infer model architecture type from filename: '{fname}'")

        loaded_obj = torch.load(path, map_location=device)

        # Extract or infer categorical vs continuous column split
        cards_f = loaded_obj.get("cards_f", None) if isinstance(loaded_obj, dict) else None
        n_cat = len(cards_f) if cards_f is not None else 0

        # Extract scaling statistics or fallback to feature standardization
        mean = loaded_obj.get("mean", None) if isinstance(loaded_obj, dict) else None
        std = loaded_obj.get("std", None) if isinstance(loaded_obj, dict) else None

        X_cat = X_mat[:, :n_cat].astype(np.int64)
        X_num = X_mat[:, n_cat:].astype(np.float32)

        if mean is not None and std is not None:
            mean_np = np.asarray(mean, dtype=np.float32)
            std_np = np.asarray(std, dtype=np.float32)
            std_np = np.where(std_np == 0, 1.0, std_np)
            X_num_scaled = np.nan_to_num((X_num - mean_np) / std_np)
        else:
            # Fallback scaling for raw state_dict checkpoints missing training stats
            num_mean = np.mean(X_num, axis=0)
            num_std = np.std(X_num, axis=0)
            num_std = np.where(num_std == 0, 1.0, num_std)
            X_num_scaled = np.nan_to_num((X_num - num_mean) / num_std)

        X_mat_proc = np.hstack([X_cat, X_num_scaled]).astype(np.float32)

        if isinstance(loaded_obj, torch.nn.Module):
            net = loaded_obj.to(device)
        else:
            state_dict = (
                loaded_obj["state_dict"]
                if isinstance(loaded_obj, dict) and "state_dict" in loaded_obj
                else loaded_obj
            )
            if isinstance(state_dict, dict):
                state_dict = {
                    k.replace("module.", "").replace("model.", ""): v
                    for k, v in state_dict.items()
                }

            net = _instantiate_pytorch_model(
                m_lower, X_mat_proc.shape[1], state_dict=state_dict
            )
            if net is None:
                raise ValueError(
                    f"Could not instantiate PyTorch model for '{fname}' (identifier: '{m_lower}')"
                )

            net = net.to(device)
            if isinstance(state_dict, dict) and m_lower != "mlp":
                try:
                    net.load_state_dict(state_dict, strict=True)
                except Exception:
                    net.load_state_dict(state_dict, strict=False)

        net.eval()

        preds = []
        with torch.no_grad():
            for start in range(0, len(X_mat_proc), batch_size):
                bx = torch.tensor(
                    X_mat_proc[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                try:
                    out = net(bx)
                except TypeError:
                    if n_cat > 0:
                        xc = bx[:, :n_cat].long()
                        xn = bx[:, n_cat:].float()
                        out = net(xc, xn)
                    else:
                        xc = torch.zeros((len(bx), 0), dtype=torch.long, device=device)
                        out = net(xc, bx)

                if isinstance(out, tuple):
                    out = out[0]
                if hasattr(out, "logits"):
                    out = out.logits

                preds.append(out.squeeze().cpu().numpy())

        return np.concatenate(preds).squeeze()

    # 4. Joblib / Pickle / Generic
    else:
        import joblib
        model = joblib.load(path)
        if hasattr(model, "predict"):
            import inspect
            sig = inspect.signature(model.predict)
            if "n_cat" in sig.parameters:
                n_cat = globals().get("NCAT_MATCH", 0)
                return model.predict(X_mat, n_cat=n_cat).squeeze()
            return model.predict(X_mat).squeeze()

    raise ValueError(f"Unrecognized model file format: {fname}")


def evaluate_wait_models(
    model_paths, X_test, y_test_raw, is_log_pred=True,
    campaign_ids=None, min_campaign_n=100, campaign_group_cols=None,
):
    """Evaluates saved regression models already on disk (no retraining --
    loads each cached model and scores it on X_test/y_test_raw). Reuses
    `reg_metrics` so results match the same schema `eval/harness.py` writes
    to `results/wait_time_results.json`, plus a per-campaign error breakdown
    when `campaign_ids` (row-aligned with X_test) is given. `campaign_group_cols`
    (optional {name: row-aligned array}) is forwarded to `campaign_breakdown`
    to also tag each campaign with the majority value of another column
    (e.g. for coloring a campaign-level plot by some coarser category).

    Returns (df_eval, results_by_model, campaign_by_model):
      - df_eval: flat summary table, one row per model, sorted by r2_log.
      - results_by_model: {raw_key: reg_metrics(...) dict} -- same shape as
        wait_time_results.json's per-model/per-split entries.
      - campaign_by_model: {raw_key: {campaign_id: {...}}} or {} if
        campaign_ids wasn't provided.
    """
    y_true = np.asarray(y_test_raw, dtype=np.float64).ravel()
    rows = []
    results_by_model = {}
    campaign_by_model = {}

    for path in model_paths:
        model_name = os.path.basename(path)
        raw_key = model_name.split("_")[0].lower()

        try:
            preds = predict_reg_model(path, X_test)
            preds = np.asarray(preds, dtype=np.float64).ravel()

            # reg_metrics expects both true and predicted values already in
            # log1p space; clip to prevent np.expm1 overflow downstream.
            pred_log = np.clip(preds, -20.0, 20.0) if is_log_pred else np.log1p(np.maximum(0, preds))
            y_true_clean = np.maximum(0, y_true)
            y_true_log = np.log1p(y_true_clean)

            valid_mask = np.isfinite(y_true_log) & np.isfinite(pred_log)
            if not np.any(valid_mask):
                print(f"[SKIP] {model_name}: No valid finite predictions found.")
                continue

            mm = reg_metrics(y_true_log[valid_mask], pred_log[valid_mask])
            results_by_model[raw_key] = mm
            rows.append({"Model": raw_key, "Model File": model_name, **mm})

            if campaign_ids is not None:
                y_pred = np.maximum(0, np.expm1(pred_log[valid_mask]))
                group_cols = {
                    name: np.asarray(arr)[valid_mask]
                    for name, arr in (campaign_group_cols or {}).items()
                }
                campaign_by_model[raw_key] = campaign_breakdown(
                    y_true_clean[valid_mask],
                    y_pred,
                    np.asarray(campaign_ids)[valid_mask],
                    min_n=min_campaign_n,
                    group_cols=group_cols,
                )

            print(f"[OK] {model_name}")
        except Exception as e:
            print(f"[FAILED] {model_name}: {e}")

    df_eval = pd.DataFrame(rows)
    if not df_eval.empty:
        df_eval = df_eval.sort_values("r2_log", ascending=False).reset_index(drop=True)
    return df_eval, results_by_model, campaign_by_model

# ---- Visualizations ----
def _klabel(k):
    """Converts key identifiers like ('ft', 'random') or 'ft' into clean display labels."""
    if isinstance(k, tuple):
        model_part = MODEL_DISPLAY_NAMES.get(str(k[0]).lower(), str(k[0]))
        rest = [str(x) for x in k[1:]]
        return " ".join([model_part] + rest)
    return MODEL_DISPLAY_NAMES.get(str(k).lower(), str(k))

def _clean_and_normalize(arr):
    """Clamps negative permutation noise to 0 and normalizes vector to sum to 100%."""
    a = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    a = np.maximum(0.0, a)  # Clamp negative noise to 0
    total = np.sum(a)
    return (a / total * 100.0) if total > 0 else a

def feature_split_imp_heatmap(
    imp_dict: dict,
    feats: list[str],
    n_cat: int = None,  # Pass NCAT_MATCH or NCAT_SUB from meta
    cards: dict | list = None,  # Pass cards from meta
    cat_cols: list[str] = None,  # Explicit list of categorical feature names
    cardinality_threshold: int = 50,  # Threshold if cards is a dict/list
    split_key: str = "temporal",
    models: list[str] = None,
    title: str = None,
    exp: str = None,
    cmap=None,
    min_importance_threshold: float = 0.1,
    output_prefix: str = "feature_split_imp_heatmap",
):
    """Renders importance heatmaps separating Categorical vs. Non-Categorical features

    using schema metadata (n_cat, cards, or cat_cols).
    """
    all_keys = list(imp_dict.keys())
    available_models = (
        set(k[0] for k in all_keys if isinstance(k, tuple))
        if models is None
        else set(models)
    )

    preferred_order = globals().get("PREFERRED_MODEL_ORDER", [])
    display_names = globals().get("MODEL_DISPLAY_NAMES", {})

    # Order models according to preferred_order and available models
    ordered_models = []
    for pref in preferred_order:
        for m in available_models:
            if m.lower() == pref and m not in ordered_models:
                if (m, split_key) in imp_dict:
                    ordered_models.append(m)

    for m in sorted(available_models):
        if m not in ordered_models and (m, split_key) in imp_dict:
            ordered_models.append(m)

    def _norm_rows(sk):
        """Each model's importances on split `sk`, scaled by that row's maximum."""
        out = {}
        for m in ordered_models:
            val = imp_dict.get((m, sk))
            if isinstance(val, np.ndarray):
                series = pd.Series(val, index=feats[: len(val)])
            elif isinstance(val, (pd.Series, dict)):
                series = pd.Series(val).reindex(feats)
            else:
                continue
            series = series.reindex(feats).fillna(0.0)
            max_val = np.nanmax(np.abs(series.values))
            if max_val > 0:
                series = series / max_val
            out[display_names.get(m.lower(), m)] = series
        return out

    rows = _norm_rows(split_key)
    if not rows:
        print(
            f"[skip] {title}: no valid model data found for split '{split_key}'"
        )
        return

    df = pd.DataFrame(rows).T
    df.columns = feats

    # Which features appear, and in what order, is decided from EVERY split in
    # imp_dict, so the random and temporal figures share one x axis and can be read
    # column by column.
    _splits = sorted({k[1] for k in imp_dict if isinstance(k, tuple)})
    col_score = pd.concat(
        [pd.DataFrame(_norm_rows(sk)).T.abs().max(axis=0) for sk in _splits
         if _norm_rows(sk)], axis=1).max(axis=1).reindex(feats).fillna(0.0)

    # --- Step 1: Detect Categorical Features using Metadata ---
    detected_cat = set()

    if cat_cols is not None:
        detected_cat = set(cat_cols)
    elif n_cat is not None and isinstance(n_cat, int):
        # The first n_cat columns in feats are categorical
        detected_cat = set(feats[:n_cat])
    elif cards is not None:
        if isinstance(cards, dict):
            for col in feats:
                if col in cards and isinstance(cards[col], (int, float)):
                    if cards[col] <= cardinality_threshold or cards[col] > 1:
                        detected_cat.add(col)
        elif isinstance(cards, (list, tuple, np.ndarray)):
            for i, card_val in enumerate(cards):
                if i < len(feats):
                    detected_cat.add(feats[i])
    else:
        # Keyword-based fallback
        cat_keywords = [
            "group",
            "owner",
            "campaign",
            "site",
            "user",
            "type",
            "id",
            "status",
            "code",
            "name",
            "queue",
            "vo",
            "role",
        ]
        for col in feats:
            if any(kw in col.lower() for kw in cat_keywords):
                detected_cat.add(col)

    cat_feats = [c for c in feats if c in detected_cat]
    non_cat_feats = [c for c in feats if c not in detected_cat]

    df_cat = df[cat_feats] if cat_feats else pd.DataFrame(index=df.index)
    df_non_cat = (
        df[non_cat_feats] if non_cat_feats else pd.DataFrame(index=df.index)
    )

    # --- Step 2: Apply Threshold & Sorting, on the all-splits score ---
    def _select(frame):
        if frame.empty:
            return frame
        keep = [c for c in frame.columns if col_score[c] >= min_importance_threshold]
        return frame[sorted(keep, key=lambda c: -col_score[c])]

    df_cat, df_non_cat = _select(df_cat), _select(df_non_cat)

    if df_cat.empty and df_non_cat.empty:
        print(f"[skip] {title}: all features fell below importance threshold")
        return

    # --- Step 3: Render, trees and neural models in separate row blocks ---
    if cmap is None:
        from matplotlib.colors import LinearSegmentedColormap
        from eval.style import COLORS
        cmap = LinearSegmentedColormap.from_list("imp", ["#FFFFFF", COLORS["primary"]])
    _disp = {display_names.get(m.lower(), m): model_family(m) for m in ordered_models}
    _title = title or f"Normalized Feature Importance, {split_key.capitalize()} Split"
    _family_block_heatmap(
        df_cat, df_non_cat, _disp, cmap=cmap, vmin=0.0, vmax=1.0, center=None,
        cbar_ticks=[0.0, 0.5, 1.0], cbar_ticklabels=["0", "0.5", "1"],
        cbar_label="Importance within row, $I / I_{\\max}$", title=_title,
        note="Each row is divided by its largest importance.\n" + _attribution_note(imp_dict),
        row_labels=_row_labels(imp_dict),
        output_prefix=(f"{exp}_" if exp else "") + f"{output_prefix}_{split_key}")
    return df_cat, df_non_cat


_SHIFT_FONT = 14   # the bar charts' FONT in prediction-analysis.ipynb


def _shift_rc(base=_SHIFT_FONT):
    """The bar charts' rcParams, so heatmaps and bar charts share typeface and sizes."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": "Nimbus Sans",
        "font.size": base,
        "axes.labelsize": base + 1,
        "axes.titlesize": base + 1,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "axes.grid": False,
    }


def _shift_cmap():
    """Diverging map in the paper's split colours: delta = temporal - random, so a
    feature that gains share under the temporal split takes the temporal colour."""
    from matplotlib.colors import LinearSegmentedColormap
    from eval.style import COLORS
    return LinearSegmentedColormap.from_list(
        "shift", [COLORS["secondary"], "#FFFFFF", COLORS["primary"]])


def _tree_method(imp_dict):
    """How the tree rows were computed: "gain" or "permutation" (load_saved_importances)."""
    return (imp_dict.get("__method__") or {}).get("tree", "gain")


def _row_labels(imp_dict):
    tree = "Trees\n(permutation)" if _tree_method(imp_dict) == "permutation" else "Trees\n(gain)"
    return {"tree": tree, "neural": "Neural\n(permutation)"}


def _attribution_note(imp_dict):
    """How each family's importances are computed -- method only, nothing about results."""
    perm = ("permutation importance (the score drop when one feature's values are "
            "shuffled) on a held-out sample of out-of-time jobs")
    if _tree_method(imp_dict) == "permutation":
        body = (f"All models except TabNet: {perm}, with the same shuffles for every model. "
                "TabNet: attention-mask mass on the same jobs.")
    else:
        body = ("Tree models: split-based importance read from the trained model "
                "(XGBoost: average gain; LightGBM: total gain; CatBoost: "
                f"PredictionValuesChange). Neural models: {perm}; TabNet: attention-mask "
                "mass on the same jobs.")
    return body + " Each model's importances are normalised to sum to 1."


def _family_block_heatmap(df_cat, df_non_cat, row_family, *, cmap, vmin, vmax, center,
                          cbar_ticks, cbar_ticklabels, cbar_label, title, note,
                          output_prefix, row_labels):
    """One table with rows grouped by model family (trees / neural) and columns by
    categorical / continuous, a gap between the groups in both directions, and one
    shared colorbar. `row_family` maps each row label to "tree", "neural" or None.
    Saves output/<output_prefix>.pdf/.png and returns the combined frame."""
    df_all = pd.concat([df_cat, df_non_cat], axis=1)
    col_groups = [(t, list(d.columns)) for t, d in
                  (("Categorical Features", df_cat), ("Continuous Features", df_non_cat))
                  if not d.empty]
    row_groups = []
    for fam in ("tree", "neural", None):
        members = [r for r in df_all.index if row_family.get(r) == fam]
        if members:
            row_groups.append((fam, members))
    n_cols = sum(len(c) for _, c in col_groups)
    n_rows = sum(len(r) for _, r in row_groups)

    with plt.rc_context(_shift_rc()):
        fig, axes = plt.subplots(
            len(row_groups), len(col_groups) + 1,
            figsize=(3.2 + 0.5 * n_cols, 1.8 + 0.5 * n_rows + 0.3 * len(row_groups)),
            squeeze=False, layout="constrained",
            gridspec_kw={
                # The trailing narrow column holds the shared colorbar.
                "width_ratios": [len(c) for _, c in col_groups] + [0.35],
                "height_ratios": [len(r) for _, r in row_groups],
            },
        )
        fig.get_layout_engine().set(wspace=0.04, hspace=0.08)

        for ri, (fam, rnames) in enumerate(row_groups):
            is_last = ri == len(row_groups) - 1
            for ci, (ctitle, cnames) in enumerate(col_groups):
                ax = axes[ri, ci]
                sns.heatmap(
                    df_all.loc[rnames, cnames], cmap=cmap, center=center,
                    vmin=vmin, vmax=vmax, annot=True, fmt=".2f",
                    annot_kws={"size": _SHIFT_FONT - 3}, cbar=False,
                    linewidths=0.5, linecolor="white",
                    xticklabels=is_last, yticklabels=(ci == 0), ax=ax,
                )
                ax.tick_params(length=0)
                ax.set_xlabel("")
                ax.set_ylabel("")
                if ri == 0:
                    ax.set_title(ctitle, pad=8, fontweight="bold")
                if is_last:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                if ci == 0:
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                    ax.set_ylabel(row_labels.get(fam, "Other"),
                                  fontweight="bold", labelpad=10)
            axes[ri, -1].axis("off")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
        cbar = fig.colorbar(sm, ax=axes[:, -1], fraction=1.0, aspect=25)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)
        cbar.set_label(cbar_label, labelpad=8)
        cbar.outline.set_visible(False)

        fig.suptitle(title, fontweight="bold", fontsize=_SHIFT_FONT + 3)
        fig.text(0.0, -0.01, note, ha="left", va="top", fontsize=_SHIFT_FONT - 2,
                 color="0.30", transform=fig.transFigure, wrap=True)

        figures_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, f"{output_prefix}.pdf"), bbox_inches="tight")
        #plt.savefig(os.path.join(figures_dir, f"{output_prefix}.png"),
                    # bbox_inches="tight", dpi=300)
        plt.show()
    return df_all


def imp_diff_heatmap(
    imp_dict: dict,
    feats: list[str],
    n_cat: int = None,  # Pass NCAT_MATCH or NCAT_SUB from metadata
    cards: dict | list = None,  # Pass cards from metadata
    cat_cols: list[str] = None,  # Explicit list of categorical feature names
    cardinality_threshold: int = 50,
    models: list[str] = None,
    title: str = "Feature Gain Shift: Random vs. Temporal Split Protocol",
    min_importance_threshold: float = 0.1,
    output_prefix: str = "imp_diff_heatmap",
    exp: str = None,
):
    """Renders the (Temporal - Random) feature importance shift as one table: rows
    grouped by model family (trees / neural), columns split into categorical and
    continuous features, matching feature_split_imp_heatmap column for column.

    Each cell is the temporal cell minus the random cell of the two per-split
    heatmaps (each split's row scaled by its own largest importance), on a fixed
    +/-1 scale. `exp` ("e1", "e2", "e3") prefixes the saved file name.
    """
    # Helper function to convert arrays/dicts to aligned pandas Series
    def _clean_and_normalize(val):
        if isinstance(val, np.ndarray):
            series = pd.Series(val, index=feats[: len(val)])
        elif isinstance(val, (pd.Series, dict)):
            series = pd.Series(val).reindex(feats)
        else:
            series = pd.Series(0.0, index=feats)
        return series.fillna(0.0)

    all_keys = list(imp_dict.keys())
    available_models = (
        set(k[0] for k in all_keys if isinstance(k, tuple))
        if models is None
        else set(models)
    )

    preferred_order = globals().get("PREFERRED_MODEL_ORDER", [])
    display_names = globals().get("MODEL_DISPLAY_NAMES", {})

    ordered_models = []
    for pref in preferred_order:
        for m in available_models:
            if m.lower() == pref and m not in ordered_models:
                if (m, "random") in imp_dict and (m, "temporal") in imp_dict:
                    ordered_models.append(m)

    for m in sorted(available_models):
        if (
            m not in ordered_models
            and (m, "random") in imp_dict
            and (m, "temporal") in imp_dict
        ):
            ordered_models.append(m)

    def _by_max(s):
        """Divided by the row's largest |importance| -- the scaling the per-split
        heatmaps (feature_split_imp_heatmap) display."""
        mx = np.nanmax(np.abs(s.values))
        return s / mx if mx > 0 else s

    rows, score = {}, {}
    for m in ordered_models:
        imp_rand = _clean_and_normalize(imp_dict[(m, "random")])
        imp_temp = _clean_and_normalize(imp_dict[(m, "temporal")])
        disp_name = display_names.get(m.lower(), m)

        # The difference of the two per-split heatmaps, cell for cell: each split
        # scaled by its own largest importance, then subtracted, with no further
        # rescaling. Dividing the difference by its own row maximum (the earlier
        # form) stretched a sub-1-point shift to +/-1 whenever a row's shifts were
        # all small, and squeezed every other cell when one shift was large.
        r, t = _by_max(imp_rand), _by_max(imp_temp)
        rows[disp_name] = t - r
        score[disp_name] = np.maximum(r.abs(), t.abs())

    if not rows:
        print(
            f"[skip] {title}: need both (random & temporal) importances for target models"
        )
        return

    df = pd.DataFrame(rows).T
    df.columns = feats

    # --- Step 1: Detect Categorical Features using Metadata or Heuristics ---
    detected_cat = set()

    if cat_cols is not None:
        detected_cat = set(cat_cols)
    elif n_cat is not None and isinstance(n_cat, int):
        # The first n_cat columns in feats are categorical
        detected_cat = set(feats[:n_cat])
    elif cards is not None:
        if isinstance(cards, dict):
            for col in feats:
                if col in cards and isinstance(cards[col], (int, float)):
                    if cards[col] <= cardinality_threshold or cards[col] > 1:
                        detected_cat.add(col)
        elif isinstance(cards, (list, tuple, np.ndarray)):
            for i, card_val in enumerate(cards):
                if i < len(feats):
                    detected_cat.add(feats[i])
    else:
        # Keyword-based fallback
        cat_keywords = [
            "group",
            "owner",
            "campaign",
            "site",
            "user",
            "type",
            "id",
            "status",
            "code",
            "name",
            "queue",
            "vo",
            "role",
        ]
        for col in feats:
            if any(kw in col.lower() for kw in cat_keywords):
                detected_cat.add(col)

    cat_feats = [c for c in feats if c in detected_cat]
    non_cat_feats = [c for c in feats if c not in detected_cat]

    df_cat = df[cat_feats] if cat_feats else pd.DataFrame(index=df.index)
    df_non_cat = (
        df[non_cat_feats] if non_cat_feats else pd.DataFrame(index=df.index)
    )

    # --- Step 2: Apply Threshold & Sorting ---
    # Same columns, in the same order, as the per-split heatmaps: features whose
    # importance reaches the threshold under either split, most important first.
    # Selecting on |delta| instead would drop exactly the stable features those
    # heatmaps show, and the figures could no longer be read column by column.
    col_score = pd.DataFrame(score).T.max(axis=0).reindex(feats).fillna(0.0)

    def _select(frame):
        if frame.empty:
            return frame
        keep = [c for c in frame.columns if col_score[c] >= min_importance_threshold]
        return frame[sorted(keep, key=lambda c: -col_score[c])]

    df_cat, df_non_cat = _select(df_cat), _select(df_non_cat)

    if df_cat.empty and df_non_cat.empty:
        print(
            f"[skip] {title}: all features fell below importance threshold ({min_importance_threshold})"
        )
        return

    # --- Step 3: Render ---
    # Normalised form: the per-split heatmaps' cells, temporal minus random, on a fixed
    # +/-1 scale. The family blocks remain because split gain and permutation
    # importance are still not commensurate (see the note under the figure).
    _disp = {display_names.get(m.lower(), m): model_family(m) for m in ordered_models}
    return _family_block_heatmap(
        df_cat, df_non_cat, _disp, cmap=_shift_cmap(), vmin=-1.0, vmax=1.0, center=0,
        cbar_ticks=[-1.0, 0.0, 1.0], cbar_ticklabels=["\u22121\nrandom", "0", "+1\ntemporal"],
        cbar_label="\u0394 importance within row, $I / I_{\\max}$", title=title,
        note=("\u0394 = temporal \u2212 random, cell for cell, of the two per-split "
              "heatmaps, each row divided by its largest importance.\n"
              + _attribution_note(imp_dict)),
        row_labels=_row_labels(imp_dict),
        output_prefix=f"{exp}_{output_prefix}" if exp else output_prefix)


# ---- Feature-matrix standardisation ----------------------------------------
# The "(std)" columns are scaled when the matrix is built and the sin/cos encodings
# are bounded by construction, but the trailing rates and log running-counts are
# written raw -- log_idle_queue_depth lands around mean 10.5 / sd 1.2 beside features
# at mean 0 / sd 1. Boosted trees do not care (the transform is monotonic); the neural
# models cannot fit inputs spanning two orders of magnitude.
#
# feat-engineering.ipynb applies this when a matrix is built. These helpers apply it
# in place to a matrix that already exists, so an 18 GB rebuild is not needed.

_STD_SKIP_MARKERS = ("(std)",)
_STD_SKIP_PREFIXES = ("sin_", "cos_")

_STD_MATRICES = {
    "Xmatch": {"ncat": "NCAT_MATCH", "cols": "XMATCH_COLS",
               "split": "e1e3__temporal__train"},
    "Xsub": {"ncat": "NCAT_SUB", "cols": "XSUB_COLS",
             "split": "e2__temporal__train"},
}


def standardize_targets(cols, first):
    """Indices of the numeric columns in `cols` that still need scaling."""
    out = []
    for j in range(first, len(cols)):
        name = cols[j]
        if any(m in name for m in _STD_SKIP_MARKERS):
            continue
        if any(name.startswith(p) for p in _STD_SKIP_PREFIXES):
            continue
        out.append(j)
    return out

# ---- Paper tables -----------------------------------------------------------------

_E3_TABLE_MODELS = [
    ("xgboost", r"XGBoost~\cite{chen_xgboost_2016}"),
    ("lightgbm", r"LightGBM~\cite{ke_lightgbm}"),
    ("catboost", r"CatBoost~\cite{catboost}"),
    ("mlp", "MLP"),
    ("ft", r"FT-Transformer~\cite{ft_transformer}"),
    ("tabnet", r"TabNet~\cite{arik_tabnet_2021}"),
    ("saint", r"SAINT~\cite{saint}"),
    ("tsmixer", r"TSMixer~\cite{chen_tsmixer_2023}"),
]
# ROC-AUC once -- it is identical with either class as positive -- then PR-AUC per
# class, which is what separates models that catch hardware faults from payload ones.
_E3_TABLE_COLUMNS = [("roc_auc", r"\textbf{ROC-AUC}"),
                     ("hardware.pr_auc", r"\textbf{Hardware PR-AUC}"),
                     ("payload.pr_auc", r"\textbf{Payload PR-AUC}")]


def e3_results_table(results_path=None, out_path=None):
    """LaTeX table of E3 (fault attribution) results, Random vs Temporal per column.

    Mean +/- sample SD over the seeded runs (`<split>__seed<n>`); a plain `<split>`
    key is an older single run and is ignored. Best value per column in bold. Writes
    `out_path` when given and returns the LaTeX either way.
    """
    results = load_results("fault_attr", results_path)
    splits = ("random", "temporal")

    def _get(run, path):
        for p in path.split("."):
            run = run[p]
        return float(run)

    stats = {}
    for mk, _ in _E3_TABLE_MODELS:
        for split in splits:
            runs = [v for k, v in results[mk].items() if k.startswith(split + "__seed")]
            if len(runs) < 2:
                raise ValueError(f"{mk}/{split}: {len(runs)} seeded run(s); need >= 2")
            for key, _ in _E3_TABLE_COLUMNS:
                x = np.array([_get(r, key) for r in runs])
                stats[mk, split, key] = (x.mean(), x.std(ddof=1))
    best = {(s, key): max(round(stats[mk, s, key][0], 3) for mk, _ in _E3_TABLE_MODELS)
            for s in splits for key, _ in _E3_TABLE_COLUMNS}

    def _cell(mk, s, key):
        mu, sd = stats[mk, s, key]
        txt = f"{mu:.3f} \\pm {sd:.3f}"
        return f"$\\mathbf{{{txt}}}$" if round(mu, 3) == best[s, key] else f"${txt}$"

    ncol = len(_E3_TABLE_COLUMNS)
    w = max(len(n) for _, n in _E3_TABLE_MODELS)
    lines = [
        r"\begin{table*}[ht]", r"\centering",
        r"\caption{Fault attribution performance ($\text{Mean} \pm \text{Std}$, 3 seeds) "
        r"on the Random and Temporal splits.}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{l" + "c" * (2 * ncol) + "}", r"\hline",
        "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$\\uparrow$ {lab}}}"
                          for _, lab in _E3_TABLE_COLUMNS) + r" \\",
        " ".join(f"\\cline{{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(ncol)),
        r"\textbf{Model} & " + " & ".join([r"\textbf{Random} & \textbf{Temporal}"] * ncol)
        + r" \\",
        r"\hline",
    ]
    for mk, name in _E3_TABLE_MODELS:
        lines.append(f"{name:<{w}} & " + " & ".join(
            _cell(mk, s, key) for key, _ in _E3_TABLE_COLUMNS for s in splits) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\label{tab:results_hw_fault}", r"\end{table*}"]
    tex = "\n".join(lines) + "\n"
    if out_path is not None:
        with open(out_path, "w") as fh:
            fh.write(tex)
    return tex


# ---- Protocol significance on the shared out-of-time slice -------------------------

def _oot_metrics(y, p, thr, thr_neg, per_class):
    """Metrics for one arm on one (resampled) set of rows, at that arm's own cuts."""
    def _f1(y_pos, pred_pos):
        tp = np.count_nonzero(y_pos & pred_pos)
        fp = np.count_nonzero(~y_pos & pred_pos)
        fn = np.count_nonzero(y_pos & ~pred_pos)
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    def _mcc(pred_pos):
        # At the positive-class cut, as the harness stores it.
        tp = float(np.count_nonzero(yb & pred_pos)); fp = float(np.count_nonzero(~yb & pred_pos))
        fn = float(np.count_nonzero(yb & ~pred_pos)); tn = float(len(yb)) - tp - fp - fn
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return (tp * tn - fp * fn) / den if den > 0 else 0.0
    yb = y.astype(bool)
    out = {"roc_auc": roc_auc_score(y, p), "mcc": _mcc(p >= thr)}
    if per_class:
        out["hardware.pr_auc"] = average_precision_score(y, p)
        out["payload.pr_auc"] = average_precision_score(1 - y, 1.0 - p)
        out["hardware.f1"] = _f1(yb, p >= thr)
        out["payload.f1"] = _f1(~yb, p < thr_neg)
    else:
        out["pr_auc"] = average_precision_score(y, p)
        out["f1"] = _f1(yb, p >= thr)
    return out


def _wboot_metrics(W, y, p, thr, thr_neg, per_class):
    """The metrics of `_oot_metrics` for a batch of bootstrap replicates at once.

    `W` is a (B, n) tensor of resample counts (row i drawn W[b, i] times in replicate
    b); `y` (n,) 0/1 labels and `p` (n,) float64 scores, on the same device. ROC-AUC
    and average precision follow scikit-learn's definitions -- equal scores form one
    threshold, ties count half in the AUC -- computed from cumulative weighted sums
    over one sort, so no replicate re-sorts. Returns {metric: (B,) float64 tensor}.
    """
    import torch
    yf = y.to(torch.float32)

    def _auc_ap(yv, sv):
        order = torch.argsort(sv, descending=True)
        s_, ys, Ws = sv[order], yv[order], W[:, order]
        last = torch.ones_like(s_, dtype=torch.bool)
        last[:-1] = s_[1:] != s_[:-1]                      # last row of each tie group
        tp = torch.cumsum(Ws * ys, 1)[:, last].double()
        fp = torch.cumsum(Ws * (1 - ys), 1)[:, last].double()
        P, N = tp[:, -1], fp[:, -1]
        z = torch.zeros_like(tp[:, :1])
        dtp = torch.diff(tp, dim=1, prepend=z)
        dfp = torch.diff(fp, dim=1, prepend=z)
        auc = (dtp * (N[:, None] - fp) + 0.5 * dtp * dfp).sum(1) / (P * N)
        ap = (dtp * tp / (tp + fp).clamp_min(1e-12)).sum(1) / P
        return auc, ap

    def _counts(pos, pred):
        pos, pred = pos.to(torch.float32), pred.to(torch.float32)
        tp = (W @ (pos * pred)).double()
        fp = (W @ ((1 - pos) * pred)).double()
        fn = (W @ (pos * (1 - pred))).double()
        tn = W.sum(1).double() - tp - fp - fn
        return tp, fp, fn, tn

    def _f1(tp, fp, fn):
        den = 2 * tp + fp + fn
        return torch.where(tp > 0, 2 * tp / den.clamp_min(1e-12), torch.zeros_like(tp))

    def _mcc(tp, fp, fn, tn):
        den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return torch.where(den > 0, (tp * tn - fp * fn) / den.clamp_min(1e-12),
                           torch.zeros_like(tp))

    auc, ap = _auc_ap(yf, p)
    tp, fp, fn, tn = _counts(yf, p >= thr)
    out = {"roc_auc": auc, "mcc": _mcc(tp, fp, fn, tn)}
    if per_class:
        out["hardware.pr_auc"] = ap
        out["payload.pr_auc"] = _auc_ap(1 - yf, -p)[1]
        out["hardware.f1"] = _f1(tp, fp, fn)
        ptp, pfp, pfn, _ = _counts(1 - yf, p < thr_neg)
        out["payload.f1"] = _f1(ptp, pfp, pfn)
    else:
        out["pr_auc"] = ap
        out["f1"] = _f1(tp, fp, fn)
    return out


def _oot_boot_one(experiment, lib, seed, n_boot, boot_seed, results, device=None):
    """Bootstrap (temporal - random) per metric for one model and seed. The resample
    indices depend only on `boot_seed` and the replicate number, so every seed and
    model sees the same resamples and the per-seed differences can be averaged."""
    exp_tag = "e3_fault" if experiment == "e3" else experiment
    per_class = experiment == "e3"
    arms = {}
    for arm in ("random", "temporal"):
        idx, p = load_predictions(exp_tag, lib, f"{arm}__oot", seed=seed, calibrated=False)
        run = results[lib][f"{arm}__seed{seed}"]
        thr = run["hardware"]["threshold"] if per_class else run["threshold"]
        thr_neg = run["payload"]["threshold"] if per_class else None
        arms[arm] = (np.asarray(idx), np.asarray(p, dtype=np.float64), thr, thr_neg)
    if not np.array_equal(arms["random"][0], arms["temporal"][0]):
        raise ValueError(f"{lib} seed {seed}: the arms' OOT predictions cover different rows")
    idx = arms["random"][0]
    if per_class:
        keep = np.asarray(np.load(os.path.join(DATA_ROOT, "failed.npy"), mmap_mode="r")[idx]) == 1
        y = np.asarray(np.load(os.path.join(DATA_ROOT, "hw.npy"), mmap_mode="r")[idx])[keep].astype(np.int8)
    else:
        keep = np.ones(len(idx), dtype=bool)
        y = np.asarray(np.load(os.path.join(DATA_ROOT, "failed.npy"), mmap_mode="r")[idx]).astype(np.int8)
    pr, pt = arms["random"][1][keep], arms["temporal"][1][keep]

    def _delta(sel):
        a = _oot_metrics(y[sel], pt[sel], *arms["temporal"][2:], per_class)
        b = _oot_metrics(y[sel], pr[sel], *arms["random"][2:], per_class)
        return {k: a[k] - b[k] for k in a}, a, b

    obs, obs_t, obs_r = _delta(slice(None))
    boots = {k: np.empty(n_boot) for k in obs}
    n = len(y)
    if device is not None:
        # Poisson bootstrap on `device`: each row's resample count is Poisson(1), drawn
        # from a generator seeded by (boot_seed, batch start), so every model and seed
        # sees the same replicates. Batch size keeps the (B, n) work under ~4 GB.
        import torch
        dev = torch.device(device)
        yt = torch.as_tensor(y, device=dev)
        st = torch.as_tensor(pt, dtype=torch.float64, device=dev)
        sr = torch.as_tensor(pr, dtype=torch.float64, device=dev)
        bsz = max(1, min(n_boot, int(4e9 // (n * 4 * 10))))
        for start in range(0, n_boot, bsz):
            b = min(bsz, n_boot - start)
            g = torch.Generator(device=dev)
            g.manual_seed(int(boot_seed) * 1_000_003 + start)
            W = torch.poisson(torch.ones((b, n), device=dev), generator=g)
            mt = _wboot_metrics(W, yt, st, *arms["temporal"][2:], per_class)
            mr = _wboot_metrics(W, yt, sr, *arms["random"][2:], per_class)
            for k in boots:
                boots[k][start:start + b] = (mt[k] - mr[k]).cpu().numpy()
            del W, mt, mr
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return lib, seed, obs, obs_t, obs_r, boots
    for b in range(n_boot):
        sel = np.random.default_rng([boot_seed, b]).integers(0, n, n)
        d, _, _ = _delta(sel)
        for k, v in d.items():
            boots[k][b] = v
    return lib, seed, obs, obs_t, obs_r, boots


def _least_busy_gpu():
    """'cuda:<i>' for the least-utilised GPU (most free memory breaks ties), or None
    without CUDA. Utilisation comes from nvidia-smi; when CUDA_VISIBLE_DEVICES remaps
    the indices, or nvidia-smi is unavailable, free memory alone decides."""
    if not torch.cuda.is_available():
        return None
    n = torch.cuda.device_count()
    free = [torch.cuda.mem_get_info(i)[0] for i in range(n)]
    util = [0] * n
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        try:
            import subprocess
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.split()
            if len(out) == n:
                util = [int(u) for u in out]
        except Exception:                     # noqa: BLE001 - fall back to memory only
            pass
    return f"cuda:{min(range(n), key=lambda i: (util[i], -free[i]))}"


def oot_protocol_significance(experiment="e3", models=None, seeds=(0, 1, 2),
                              n_boot=1000, boot_seed=0, n_jobs=-1, results_path=None,
                              device="auto"):
    """Paired bootstrap test of Temporal-OOT vs Random-OOT on the shared OOT slice.

    Both arms' saved predictions on the identical out-of-time rows are resampled
    together (same rows for both arms, every seed and every model), each metric is
    recomputed per arm at that arm's own tuned threshold, and the temporal-minus-random
    difference is averaged over seeds. Reported per model and metric: the observed
    mean difference, its 95% percentile interval, a two-sided bootstrap p-value, the
    Holm-adjusted p across models within each metric, and how many seeds agree in sign.

    On a GPU (the default when one is free) the resampling is a Poisson bootstrap --
    each row's count drawn from Poisson(1) -- computed in batches from one sort per
    model; `device="cpu"` keeps the per-replicate scikit-learn path.

    The interval reflects test-set sampling only; seed-to-seed variation enters via the
    average and the sign count, not the interval. E3 metrics are computed on failed
    jobs only, as in the harness.
    """
    from joblib import Parallel, delayed
    results = load_results(experiment, results_path)
    models = models or [m for m in PREFERRED_MODEL_ORDER if m in results]
    # Per model, only the seeds both arms have finished: a model still running (or a
    # seed that was skipped) contributes what exists rather than failing the whole run.
    pairs = [(m, s) for m in models for s in seeds
             if f"random__seed{s}" in results.get(m, {})
             and f"temporal__seed{s}" in results.get(m, {})]
    missing = sorted({m for m in models} - {m for m, _ in pairs})
    if missing:
        print(f"[significance] no seed with both arms for: {', '.join(missing)}")
    models = [m for m in models if m not in missing]
    # device="auto": the GPU with the most free memory if there is one (batched
    # Poisson bootstrap, one model/seed at a time), else the CPU path. device="cpu"
    # forces the original per-replicate scikit-learn path across `n_jobs` processes.
    if device == "auto":
        device = _least_busy_gpu()
    elif device == "cpu":
        device = None
    if device is not None:
        print(f"[significance] Poisson bootstrap on {device}", flush=True)
        jobs = [_oot_boot_one(experiment, m, s, n_boot, boot_seed, results, device=device)
                for m, s in pairs]
    else:
        jobs = Parallel(n_jobs=n_jobs)(
            delayed(_oot_boot_one)(experiment, m, s, n_boot, boot_seed, results)
            for m, s in pairs)

    rows = []
    for m in models:
        per = [j for j in jobs if j[0] == m]
        for k in per[0][2]:
            obs = np.mean([j[2][k] for j in per])
            dist = np.mean([j[5][k] for j in per], axis=0)
            p = 2 * min((dist <= 0).mean(), (dist >= 0).mean())
            rows.append({
                "model": MODEL_DISPLAY_NAMES.get(m, m), "metric": k,
                "random": np.mean([j[4][k] for j in per]),
                "temporal": np.mean([j[3][k] for j in per]),
                "delta": obs, "ci_lo": np.percentile(dist, 2.5),
                "ci_hi": np.percentile(dist, 97.5),
                "p": max(p, 1.0 / n_boot),
                "seeds_agree": f"{sum(np.sign(j[2][k]) == np.sign(obs) for j in per)}/{len(per)}",
            })
    df = pd.DataFrame(rows)
    # Holm step-down within each metric, across the models compared.
    df["p_holm"] = np.nan
    for _, g in df.groupby("metric"):
        order = g["p"].sort_values().index
        m_ = len(order)
        running = 0.0
        for rank, ix in enumerate(order):
            running = max(running, min(1.0, (m_ - rank) * df.at[ix, "p"]))
            df.at[ix, "p_holm"] = running
    return df


def oot_campaign_split_report(experiment="e3", models=None, seeds=(0, 1, 2),
                              results_path=None):
    """Score both arms' saved OOT predictions separately by campaign familiarity.

    August jobs fall into three groups: a POMS campaign that also has jobs in the
    training window ("seen"), a campaign with none ("unseen" -- nothing about it could
    have been memorised), and user jobs with no campaign ("user"; their batch id is not
    in the matrices, so familiarity cannot be decided). "Seen" is judged against the
    training window both arms share (window minus the random arm's test set). Each arm
    is scored at its own tuned threshold, mean over seeds. No retraining.
    """
    from eval.dataset import load_experiment
    results = load_results(experiment, results_path)
    models = models or [m for m in PREFERRED_MODEL_ORDER if m in results]
    exp_tag = "e3_fault" if experiment == "e3" else experiment
    per_class = experiment == "e3"

    d = load_experiment(experiment)
    cols = d.xmatch_cols
    cn, ct = cols.index("CampaignName"), cols.index("CampaignType")
    user_code = int(next(k for k, v in d.schema["CAMPAIGN_TYPE_CODES"].items() if v == "User"))
    tr = d.train("temporal")
    tr_c = np.asarray(d.Xmatch[tr][:, [cn, ct]]).astype(np.int64)
    seen_campaigns = np.unique(tr_c[tr_c[:, 1] != user_code, 0])
    failed = np.load(os.path.join(DATA_ROOT, "failed.npy"), mmap_mode="r")
    hw = np.load(os.path.join(DATA_ROOT, "hw.npy"), mmap_mode="r")

    rows, groups_cache = [], {}
    for m in models:
        for arm in ("random", "temporal"):
            acc = {}
            for seed in seeds:
                idx, p = load_predictions(exp_tag, m, f"{arm}__oot", seed=seed,
                                          calibrated=False)
                idx = np.asarray(idx)
                key = idx.tobytes()[:64] + bytes(str(len(idx)), "ascii")
                if key not in groups_cache:
                    c = np.asarray(d.Xmatch[idx][:, [cn, ct]]).astype(np.int64)
                    user = c[:, 1] == user_code
                    seen = ~user & np.isin(c[:, 0], seen_campaigns)
                    groups_cache[key] = {"all": np.ones(len(idx), bool), "seen": seen,
                                         "unseen": ~user & ~seen, "user": user}
                grp = groups_cache[key]
                run = results[m][f"{arm}__seed{seed}"]
                thr = run["hardware"]["threshold"] if per_class else run["threshold"]
                thr_neg = run["payload"]["threshold"] if per_class else None
                if per_class:
                    keep = np.asarray(failed[idx]) == 1
                    y = np.asarray(hw[idx]).astype(np.int8)
                else:
                    keep = np.ones(len(idx), bool)
                    y = np.asarray(failed[idx]).astype(np.int8)
                p = np.asarray(p, dtype=np.float64)
                for g, gm in grp.items():
                    sel = gm & keep
                    if sel.sum() == 0 or len(np.unique(y[sel])) < 2:
                        continue
                    mm = _oot_metrics(y[sel], p[sel], thr, thr_neg, per_class)
                    mm["n"] = int(sel.sum())
                    mm["positive_rate"] = float(y[sel].mean())
                    for k, v in mm.items():
                        acc.setdefault((g, k), []).append(v)
            for (g, k), vals in acc.items():
                rows.append({"model": MODEL_DISPLAY_NAMES.get(m, m), "arm": arm, "group": g,
                             "metric": k, "mean": float(np.mean(vals)),
                             "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")})
    return pd.DataFrame(rows)

