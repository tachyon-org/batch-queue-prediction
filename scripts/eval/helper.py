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
from eval.paths import DATA_ROOT, all_model_dirs as _all_model_dirs

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


def pinball_loss(y_true_log, pred_log_by_tau):
    """Mean pinball (quantile) loss in log1p space, averaged over the quantiles.

    `pred_log_by_tau` maps tau -> row-aligned predictions. The mixture fit puts a
    conditional SD of ~1.17 in log space even when the component is known, so a
    point estimate is the wrong output object for this task; pinball is the loss a
    set of quantile heads actually optimises, and it is what makes two interval
    predictors comparable. Returns the per-tau losses and their mean.
    """
    y = np.asarray(y_true_log, dtype=np.float64)
    per = {}
    for tau, pred in sorted(pred_log_by_tau.items()):
        d = y - np.asarray(pred, dtype=np.float64)
        per[float(tau)] = float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))
    return {"pinball_per_tau": per,
            "pinball_mean": float(np.mean(list(per.values()))) if per else float("nan")}


def interval_metrics(y_true_log, lo_log, hi_log):
    """Coverage and width of a predicted interval, in log1p space.

    `coverage` is the fraction of jobs whose true wait falls inside [lo, hi]; for
    heads fitted at tau = 0.1/0.9 an honest model lands near 0.80. `width_log` is
    the mean interval width in log1p units, so exp(width_log) is the multiplicative
    factor the interval spans -- the number a user actually feels.
    """
    y = np.asarray(y_true_log, dtype=np.float64)
    lo = np.asarray(lo_log, dtype=np.float64)
    hi = np.asarray(hi_log, dtype=np.float64)
    w = np.maximum(hi - lo, 0.0)
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "width_log": float(np.mean(w)),
        "width_factor": float(np.exp(np.mean(w))),
        "median_width_factor": float(np.exp(np.median(w))),
    }


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


def bootstrap_ci(y_true, scores, metric="pr_auc", n_boot=200, seed=DEFAULT_SEED, alpha=0.05):
    """Percentile bootstrap interval for a ranking metric on one test set.

    Answers "is this gap larger than test-set noise?", which is a different
    question from seed variance (see `aggregate_seeds`) -- this resamples the test
    rows and holds the fit fixed, that one refits and holds the test set fixed. A
    paper claiming one model beats another wants both.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    fn = average_precision_score if metric == "pr_auc" else roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        yb = y_true[b]
        # A resample with a single class leaves both metrics undefined.
        if yb.min() == yb.max():
            continue
        vals.append(fn(yb, scores[b]))
    vals = np.sort(vals)
    return {
        "point": float(fn(y_true, scores)),
        "lo": float(np.quantile(vals, alpha / 2)) if len(vals) else float("nan"),
        "hi": float(np.quantile(vals, 1 - alpha / 2)) if len(vals) else float("nan"),
        "n_boot": int(len(vals)),
    }


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


def collect_seed_runs(exp_name, lib, split, results_dir=None):
    """Gather every seed's metrics for one model/split out of a results JSON.

    Repeated runs are stored under "<split>" (the default seed) and
    "<split>__seed<N>" (the rest), so this pulls them back together for
    `aggregate_seeds`.
    """
    results_dir = results_dir if results_dir is not None else os.path.join(os.getcwd(), "results")
    path = os.path.join(results_dir, f"{exp_name}_results.json")
    with open(path) as f:
        data = json.load(f)
    entry = data.get(lib, {})
    runs = [v for k, v in entry.items() if k == split or k.startswith(f"{split}__seed")]
    return runs


def seed_summary_table(exp_name, libs, split, metrics, results_dir=None):
    """mean +/- std across seeds, one row per model -- the variance table a paper needs.

    `metrics` are dotted paths into the metric dict, e.g. "hardware.pr_auc".
    Models fitted under a single seed report std 0.0 with n=1, which is a placeholder
    and not evidence of stability; report the seed count alongside.
    """
    rows = {}
    for lib in libs:
        runs = collect_seed_runs(exp_name, lib, split, results_dir)
        if not runs:
            continue
        agg = aggregate_seeds(runs)
        rows[lib] = {m: agg.get(m) for m in metrics}
        rows[lib]["n_seeds"] = agg.get("n_seeds", 0)
    return rows


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


# ---- Entity memorization ----
def entity_seen_mask(entity, train_mask, test_mask):
    """Boolean over the TEST rows: True where that entity also occurs in training.

    Splitting the test set this way measures entity memorization without retraining
    and without changing the time period, which is what separates it from the
    random-vs-temporal gap.
    """
    entity = np.asarray(entity)
    seen = set(np.unique(entity[np.asarray(train_mask, dtype=bool)]).tolist())
    return np.array([e in seen for e in entity[np.asarray(test_mask, dtype=bool)]], dtype=bool)


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


def matched_test_masks(qs, cutoff, label_time=None, test_frac=0.5, seed=DEFAULT_SEED,
                       verbose=True):
    """Two training protocols scored on one identical test set.

    The random-vs-temporal gap as usually run moves four things at once: chronology,
    the test population, its class prevalence, and entity overlap. Attributing the
    whole gap to memorization is therefore not supported.

    This holds the evaluation set fixed. A random half of the post-cutoff period is
    reserved as the test set for BOTH protocols; the other half stays available for
    training. The two training pools are then subsampled to an identical size, so
    the only thing that differs between them is whether training contains jobs
    contemporaneous with the test period:

      train_honest  -- pre-cutoff jobs only (temporally honest)
      train_leaky   -- pre-cutoff jobs plus the post-cutoff jobs held out of the test

    Any remaining gap cannot be explained by a different test population, a different
    base rate, or a different test size, because all three are identical by
    construction.
    """
    qs = np.asarray(qs, dtype=np.float64)
    pre, post = qs < cutoff, qs >= cutoff

    rng = np.random.default_rng(seed)
    post_idx = np.where(post)[0]
    perm = rng.permutation(len(post_idx))
    n_test = int(round(test_frac * len(post_idx)))
    test_idx = np.sort(post_idx[perm[:n_test]])
    post_train_idx = np.sort(post_idx[perm[n_test:]])

    test = np.zeros_like(pre); test[test_idx] = True

    honest_pool = pre.copy()
    if label_time is not None:
        lt = np.asarray(label_time, dtype=np.float64)
        usable = np.isfinite(lt) & (lt > 0)
        honest_pool &= ~(usable & (lt >= cutoff))

    leaky_pool = honest_pool.copy(); leaky_pool[post_train_idx] = True

    # Match training size so the comparison is not confounded by sample size.
    n_match = int(min(honest_pool.sum(), leaky_pool.sum()))
    def _subsample(mask):
        idx = np.where(mask)[0]
        if len(idx) <= n_match:
            return mask
        keep = np.sort(rng.choice(idx, n_match, replace=False))
        out = np.zeros_like(mask); out[keep] = True
        return out

    train_honest, train_leaky = _subsample(honest_pool), _subsample(leaky_pool)
    stats = {
        "n_test": int(test.sum()),
        "n_train_honest": int(train_honest.sum()),
        "n_train_leaky": int(train_leaky.sum()),
        "n_post_in_leaky_train": int(train_leaky[post_train_idx].sum()),
    }
    assert not (train_honest & test).any() and not (train_leaky & test).any(), "test leaked into train"
    if verbose:
        print(f"[matched-test split] test {stats['n_test']:,} (identical for both protocols) | "
              f"train honest {stats['n_train_honest']:,} | train leaky "
              f"{stats['n_train_leaky']:,} of which {stats['n_post_in_leaky_train']:,} "
              f"are contemporaneous with the test period", flush=True)
    return train_honest, train_leaky, test, stats


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


def _instantiate_pytorch_model(m_name, n_feats, state_dict=None):
    """Dynamically reconstructs PyTorch architectures by inspecting state_dict weight shapes without assuming fixed key names."""
    m_name = m_name.lower()

    if m_name == "mlp":
        from train.mlp import MLP

        if isinstance(state_dict, dict) and len(state_dict) > 0:
            # 1. Identify embedding weights (keys with 'emb' in name)
            emb_keys = sorted(
                [
                    k
                    for k, v in state_dict.items()
                    if isinstance(v, torch.Tensor)
                    and v.ndim == 2
                    and "emb" in k.lower()
                ],
                key=lambda k: k,
            )
            cards_f = [state_dict[k].shape[0] for k in emb_keys]
            sum_emb_dim = sum(state_dict[k].shape[1] for k in emb_keys)

            # 2. Identify all 2D linear weight matrices excluding embeddings
            linear_keys = [
                k
                for k, v in state_dict.items()
                if isinstance(v, torch.Tensor)
                and v.ndim == 2
                and k not in emb_keys
            ]

            if linear_keys:
                # The last linear layer is the output head; preceding ones are hidden layers
                head_key = linear_keys[-1]
                body_keys = linear_keys[:-1]

                if body_keys:
                    first_linear_in = state_dict[body_keys[0]].shape[1]
                    hidden = [state_dict[k].shape[0] for k in body_keys]
                else:
                    first_linear_in = state_dict[head_key].shape[1]
                    hidden = []

                n_num = first_linear_in - sum_emb_dim
                out_dim = state_dict[head_key].shape[0]

                net = MLP(
                    cards_f=cards_f,
                    n_num=n_num,
                    out=out_dim,
                    hidden=tuple(hidden),
                )
                net.load_state_dict(state_dict, strict=False)
                return net

        net = MLP(cards_f=[], n_num=n_feats)
        if isinstance(state_dict, dict):
            net.load_state_dict(state_dict, strict=False)
        return net

    elif m_name in ("ft", "ft_transformer"):
        from train.ft import FTTransformer

        d_token = 32
        depth = 2
        heads = 4
        if isinstance(state_dict, dict):
            if "feature_embedder.weight" in state_dict:
                d_token = state_dict["feature_embedder.weight"].shape[-1]
            transformer_keys = [
                k for k in state_dict if "transformer.layers." in k
            ]
            if transformer_keys:
                depth = (
                    max(
                        [
                            int(k.split("transformer.layers.")[1].split(".")[0])
                            for k in transformer_keys
                            if k.split("transformer.layers.")[1]
                            .split(".")[0]
                            .isdigit()
                        ]
                    )
                    + 1
                )

        net = FTTransformer(
            num_features=n_feats, d_token=d_token, depth=depth, heads=heads
        )
        if isinstance(state_dict, dict):
            net.load_state_dict(state_dict, strict=False)
        return net

    elif m_name == "saint":
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

        net = SAINT(
            n_num=n_num,
            cat_dims=cat_dims,
            d_token=d_token,
            depth=depth,
            heads=heads,
        )
        if isinstance(state_dict, dict):
            net.load_state_dict(state_dict, strict=False)
        return net

    return None

def compute_permutation_importance(
    model, X_eval, y_eval, device="cuda", batch_size=32768, sample_size=None
):
    """Computes Permutation Feature Importance (% ROC-AUC drop) for PyTorch DL models.

    Uses 100% of X_eval if sample_size is None or if len(X_eval) <= sample_size.
    """
    if hasattr(model, "eval"):
        model.eval()

    if sample_size is not None and len(X_eval) > sample_size:
        idx = np.random.choice(len(X_eval), size=sample_size, replace=False)
        X_sub, y_sub = X_eval[idx], y_eval[idx]
    else:
        X_sub, y_sub = X_eval.copy(), y_eval.copy()

    def _predict(X_data):
        preds = []
        for i in range(0, len(X_data), batch_size):
            bx = torch.from_numpy(X_data[i : i + batch_size]).float().to(device)
            with torch.no_grad():
                out = model(bx)
                if hasattr(out, "logits"):
                    out = out.logits
                if out.ndim > 1:
                    out = out.squeeze(-1)
                preds.append(torch.sigmoid(out.float()).cpu().numpy())
        return np.concatenate(preds)

    base_preds = _predict(X_sub)
    base_auc = roc_auc_score(y_sub, base_preds)

    n_features = X_sub.shape[1]
    importance_scores = np.zeros(n_features)

    for f_idx in range(n_features):
        X_perm = X_sub.copy()
        np.random.shuffle(X_perm[:, f_idx])

        perm_preds = _predict(X_perm)
        perm_auc = roc_auc_score(y_sub, perm_preds)

        importance_scores[f_idx] = max(0.0, base_auc - perm_auc)

    total_imp = importance_scores.sum()
    if total_imp > 0:
        importance_scores = importance_scores / total_imp

    return importance_scores


def extract_tree_importance(model_name, file_path, n_feats):
    """Loads saved decision tree models and extracts normalized feature gain importances (summing to 1.0)."""
    m_name = model_name.lower()

    if m_name in ("lightgbm", "lgb"):
        import lightgbm as lgb

        bst = lgb.Booster(model_file=file_path)
        imp = bst.feature_importance(importance_type="gain")

    elif m_name in ("catboost", "cat"):
        from catboost import CatBoostClassifier

        cb = CatBoostClassifier()
        cb.load_model(file_path)
        imp = cb.get_feature_importance()

    elif m_name in ("xgboost", "xgb"):
        import xgboost as xgb

        m = xgb.XGBClassifier()
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
):
    """Scans directories, loads saved Tree, TabNet, and PyTorch DL models, and returns.

    an imp_dict keyed by (model, split) with normalized importances.

    `device=None` resolves to CUDA-if-present at CALL time. It must not be a default
    argument expression: Python evaluates those at import, so a `torch.cuda` probe
    there ran every time this module was imported or reloaded -- including from
    analysis notebooks that never touch a model -- and blocked outright whenever the
    driver was wedged. The other loaders here already take `device=None`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
    imp_dict = {}

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
                    imp_dict[(m, split)] = extract_tree_importance(
                        m_lower, matched_path, n_feats
                    )
                    print(f"[Loaded Tree Model] {m:10s} ({split:8s}) <- {matched_path}")
                except Exception as e:
                    print(f"[Error] Failed loading tree model {m}/{split} from {matched_path}: {e}")

            elif m_lower == "tabnet":
                try:
                    from pytorch_tabnet.tab_model import TabNetClassifier

                    clf = TabNetClassifier()
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
                        net, X_eval, y_eval, device=device
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


def plot_confusions(cm_dict, class_labels, title, ncols=None, normalize=True):
    keys = list(cm_dict)
    if not keys:
        print(f"[skip] {title}: no data (run the producing cell first)")
        return
    ncols = ncols or len(keys)
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.8 * ncols, 4.5 * nrows), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for i, k in enumerate(keys):
        ax = axes.flat[i]
        ax.axis("on")
        cm = np.asarray(cm_dict[k], float)
        M = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
        sns.heatmap(
            M,
            annot=True,
            fmt=".2f" if normalize else ".0f",
            cmap="Blues",
            vmin=0,
            vmax=1 if normalize else None,
            cbar=False,
            square=True,
            xticklabels=class_labels,
            yticklabels=class_labels,
            annot_kws={"size": 12},
            ax=ax,
        )
        ax.set_title(_klabel(k), pad=10)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.suptitle(title, y=1.03)
    plt.tight_layout()
    plt.show()


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
    title: str = "Normalized Feature Importance",
    cmap: str = "Blues",
    min_importance_threshold: float = 0.1,
    output_prefix: str = "temporal_imp_side_by_side",
):
    """Renders two side-by-side heatmaps separating Categorical vs. Non-Categorical features

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

    rows = {}
    for m in ordered_models:
        val = imp_dict[(m, split_key)]

        if isinstance(val, np.ndarray):
            series = pd.Series(val, index=feats[: len(val)])
        elif isinstance(val, (pd.Series, dict)):
            series = pd.Series(val).reindex(feats)
        else:
            continue

        series = series.fillna(0.0)

        # Row-wise max-normalization
        max_val = np.nanmax(np.abs(series.values))
        if max_val > 0:
            series = series / max_val

        disp_name = display_names.get(m.lower(), m)
        rows[disp_name] = series

    if not rows:
        print(
            f"[skip] {title}: no valid model data found for split '{split_key}'"
        )
        return

    df = pd.DataFrame(rows).T
    df.columns = feats

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

    # --- Step 2: Apply Threshold & Sorting ---
    if min_importance_threshold > 0:
        if not df_cat.empty:
            df_cat = df_cat.loc[
                :, df_cat.abs().max(axis=0) >= min_importance_threshold
            ]
        if not df_non_cat.empty:
            df_non_cat = df_non_cat.loc[
                :, df_non_cat.abs().max(axis=0) >= min_importance_threshold
            ]

    if not df_cat.empty:
        sort_cat = df_cat.abs().max(axis=0).sort_values(ascending=False).index
        df_cat = df_cat[sort_cat]

    if not df_non_cat.empty:
        sort_non_cat = (
            df_non_cat.abs().max(axis=0).sort_values(ascending=False).index
        )
        df_non_cat = df_non_cat[sort_non_cat]

    if df_cat.empty and df_non_cat.empty:
        print(f"[skip] {title}: all features fell below importance threshold")
        return

    # --- Step 3: Render Side-by-Side Figure ---
    sns.plotting_context("talk")
    plt.rcParams.update({"font.family": "serif"})

    n_cat_cols = max(len(df_cat.columns), 1)
    n_non_cat_cols = max(len(df_non_cat.columns), 1)
    total_cols = n_cat_cols + n_non_cat_cols

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(5 + 1.2 * total_cols, 2.5 + 0.8 * len(rows)),
        gridspec_kw={
            "width_ratios": [n_cat_cols, n_non_cat_cols],
            "wspace": 0.08,
        },
    )

    # Plot 1: Categoricals
    if not df_cat.empty:
        sns.heatmap(
            df_cat,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 20},
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=ax1,
        )
        ax1.set_title(
            "Categorical Features", pad=12, fontsize=24, fontweight="bold"
        )
    else:
        ax1.text(
            0.5,
            0.5,
            "No Categorical Features",
            ha="center",
            va="center",
            fontsize=18,
        )

    # Plot 2: Continuous Features
    if not df_non_cat.empty:
        sns.heatmap(
            df_non_cat,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 20},
            cbar_kws={
                "label": "Normalized Importance ($I / I_{\\max}$)",
                "shrink": 0.6,
                "pad": 0.02,
            },
            linewidths=0.5,
            linecolor="white",
            ax=ax2,
        )
        ax2.set_title(
            "Continuous Features",
            pad=12,
            fontsize=24,
            fontweight="bold",
        )
    else:
        ax2.text(
            0.5,
            0.5,
            "No Continuous Features",
            ha="center",
            va="center",
            fontsize=18,
        )

    # --- Formatting Axes ---
    for ax in (ax1, ax2):
        ax.tick_params(length=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", fontsize=22
        )

    # Y-axis styling
    ax1.set_yticklabels(
        ax1.get_yticklabels(), rotation=0, fontweight="bold", fontsize=22
    )
    ax2.tick_params(
        left=False, labelleft=False
    )  # Hide duplicate Y-labels on right plot

    # Colorbar styling
    if not df_non_cat.empty and len(ax2.collections) > 0:
        cbar = ax2.collections[0].colorbar
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(
            "Normalized Importance ($I / I_{\\max}$)",
            fontsize=22,
            fontweight="bold",
            labelpad=12,
        )

    fig.suptitle(title, y=1.02, fontsize=30)
    plt.tight_layout()

    # Save logic
    notebook_dir = os.getcwd()
    figures_dir = os.path.join(notebook_dir, "output")
    os.makedirs(figures_dir, exist_ok=True)

    pdf_path = os.path.join(figures_dir, f"{output_prefix}.pdf")
    png_path = os.path.join(figures_dir, f"{output_prefix}.png")

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.show()

    return df_cat, df_non_cat


def imp_heatmap(
    imp_dict: dict,
    feats: list[str],
    split_key: str = "temporal",
    models: list[str] = None,
    title: str = "Normalized Feature Importance",
    cmap: str = "Blues",
    min_importance_threshold: float = 0.1,
    output_prefix: str = "temporal_imp_heatmap",
):
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

    rows = {}
    for m in ordered_models:
        val = imp_dict[(m, split_key)]

        # Map numpy array, pandas Series, or dict to feature list
        if isinstance(val, np.ndarray):
            series = pd.Series(val, index=feats[: len(val)])
        elif isinstance(val, (pd.Series, dict)):
            series = pd.Series(val).reindex(feats)
        else:
            continue

        series = series.fillna(0.0)

        # Row-wise max-normalization: Maps each model's maximum importance to [0.0, 1.0]
        max_val = np.nanmax(np.abs(series.values))
        if max_val > 0:
            series = series / max_val

        disp_name = display_names.get(m.lower(), m)
        rows[disp_name] = series

    if not rows:
        print(
            f"[skip] {title}: no valid model data found for split '{split_key}'"
        )
        return

    df = pd.DataFrame(rows).T
    df.columns = feats

    # Filter out features below the importance threshold across all models
    if min_importance_threshold > 0:
        df = df.loc[:, df.abs().max(axis=0) >= min_importance_threshold]

    # Sort columns by maximum importance in descending order
    sorted_cols = df.abs().max(axis=0).sort_values(ascending=False).index
    df = df[sorted_cols]

    # Global style updates
    sns.plotting_context("talk")
    plt.rcParams.update({"font.family": "serif"})

    # Dynamic figure size matching imp_diff_heatmap proportions
    fig, ax = plt.subplots(
        figsize=(4 + 1.2 * len(df.columns), 2.5 + 1.1 * len(rows))
    )

    sns.heatmap(
        df,
        cmap=cmap,  # Sequential colormap for non-negative [0, 1] scale
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 20},
        cbar_kws={
            "label": "Normalized Importance ($I / I_{\\max}$)",
            "shrink": 0.5,
            "pad": 0.01,
        },
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    # Colorbar formatting
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(
        "Normalized Importance ($I / I_{\\max}$)",
        fontsize=22,
        fontweight="bold",
        labelpad=12,
    )

    # Axis formatting
    ax.tick_params(length=0)
    ax.set_title(title, pad=15, fontsize=30)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize=22
    )
    ax.set_yticklabels(
        ax.get_yticklabels(), rotation=0, fontweight="bold", fontsize=22
    )

    plt.tight_layout()

    # Save logic matching imp_diff_heatmap
    notebook_dir = os.getcwd()
    figures_dir = os.path.join(notebook_dir, "output")
    os.makedirs(figures_dir, exist_ok=True)

    pdf_path = os.path.join(figures_dir, f"{output_prefix}.pdf")
    png_path = os.path.join(figures_dir, f"{output_prefix}.png")

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.show()

    return df

def imp_diff_heatmap(
    imp_dict: dict,
    feats: list[str],
    n_cat: int = None,  # Pass NCAT_MATCH or NCAT_SUB from metadata
    cards: dict | list = None,  # Pass cards from metadata
    cat_cols: list[str] = None,  # Explicit list of categorical feature names
    cardinality_threshold: int = 50,
    models: list[str] = None,
    title: str = "Feature Gain Shift: Random vs. Temporal Split Protocol",
    min_importance_threshold: float = 0.1,  # Filters features with max absolute shift < 0.1
    output_prefix: str = "imp_diff_heatmap_side_by_side",
):
    """Renders heatmaps comparing (Temporal - Random) feature importance shifts,

    optionally splitting columns side-by-side into Categorical vs.
    Numerical/Dynamic features.
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

    rows = {}
    for m in ordered_models:
        imp_rand = _clean_and_normalize(imp_dict[(m, "random")])
        imp_temp = _clean_and_normalize(imp_dict[(m, "temporal")])

        # Feature gain shift (Temporal - Random)
        delta = imp_temp - imp_rand

        # Row-wise Max-Abs scaling: Maps each model's maximum shift to [-1.0, +1.0]
        max_abs = np.nanmax(np.abs(delta.values))
        if max_abs > 0:
            delta = delta / max_abs

        disp_name = display_names.get(m.lower(), m)
        rows[disp_name] = delta

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
    if min_importance_threshold > 0:
        if not df_cat.empty:
            df_cat = df_cat.loc[
                :, df_cat.abs().max(axis=0) >= min_importance_threshold
            ]
        if not df_non_cat.empty:
            df_non_cat = df_non_cat.loc[
                :, df_non_cat.abs().max(axis=0) >= min_importance_threshold
            ]

    if not df_cat.empty:
        sort_cat = df_cat.abs().max(axis=0).sort_values(ascending=False).index
        df_cat = df_cat[sort_cat]

    if not df_non_cat.empty:
        sort_non_cat = (
            df_non_cat.abs().max(axis=0).sort_values(ascending=False).index
        )
        df_non_cat = df_non_cat[sort_non_cat]

    if df_cat.empty and df_non_cat.empty:
        print(
            f"[skip] {title}: all features fell below shift threshold ({min_importance_threshold})"
        )
        return

    # --- Step 3: Render Heatmap(s) ---
    sns.plotting_context("talk")
    plt.rcParams.update({"font.family": "serif"})

    # Fallback to single subplot if all features fell into one category
    if df_cat.empty or df_non_cat.empty:
        df_single = df_cat if not df_cat.empty else df_non_cat
        fig, ax = plt.subplots(
            figsize=(4 + 1.2 * len(df_single.columns), 2.5 + 0.8 * len(rows))
        )

        sns.heatmap(
            df_single,
            cmap="coolwarm",
            center=0,
            vmin=-1.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 20},
            cbar_kws={
                "label": "Relative Shift (Δ / Max |Δ|)",
                "shrink": 0.5,
                "pad": 0.01,
            },
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )

        cbar = ax.collections[0].colorbar
        cbar.set_ticks([-1.0, 0.0, 1.0])
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(
            "Relative Shift (Δ / Max |Δ|)",
            fontsize=22,
            fontweight="bold",
            labelpad=12,
        )

        ax.tick_params(length=0)
        ax.set_title(title, pad=15, fontsize=30)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", fontsize=22
        )
        ax.set_yticklabels(
            ax.get_yticklabels(), rotation=0, fontweight="bold", fontsize=22
        )

    else:
        # Render Side-by-Side Subplots
        n_cat_cols = len(df_cat.columns)
        n_non_cat_cols = len(df_non_cat.columns)
        total_cols = n_cat_cols + n_non_cat_cols

        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(5 + 1.2 * total_cols, 2.5 + 0.8 * len(rows)),
            gridspec_kw={
                "width_ratios": [n_cat_cols, n_non_cat_cols],
                "wspace": 0.08,
            },
        )

        # Plot 1: Categorical Shift
        sns.heatmap(
            df_cat,
            cmap="coolwarm",
            center=0,
            vmin=-1.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 20},
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=ax1,
        )
        ax1.set_title(
            "Categorical Features", pad=12, fontsize=24, fontweight="bold"
        )

        # Plot 2: Continuous Shift
        sns.heatmap(
            df_non_cat,
            cmap="coolwarm",
            center=0,
            vmin=-1.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 20},
            cbar_kws={
                "label": "Relative Shift (Δ / Max |Δ|)",
                "shrink": 0.6,
                "pad": 0.02,
            },
            linewidths=0.5,
            linecolor="white",
            ax=ax2,
        )
        ax2.set_title(
            "Continuous Features",
            pad=12,
            fontsize=24,
            fontweight="bold",
        )

        # Axis Formatting
        for ax in (ax1, ax2):
            ax.tick_params(length=0)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right", fontsize=22
            )

        ax1.set_yticklabels(
            ax1.get_yticklabels(), rotation=0, fontweight="bold", fontsize=22
        )
        ax2.tick_params(
            left=False, labelleft=False
        )  # Hide redundant Y labels on right subplot

        # Colorbar Formatting
        if len(ax2.collections) > 0:
            cbar = ax2.collections[0].colorbar
            cbar.set_ticks([-1.0, 0.0, 1.0])
            cbar.ax.tick_params(labelsize=18)
            cbar.set_label(
                "Relative Shift (Δ / Max |Δ|)",
                fontsize=22,
                fontweight="bold",
                labelpad=12,
            )

        fig.suptitle(title, y=1.03, fontsize=30)

    plt.tight_layout()

    # Save logic
    notebook_dir = os.getcwd()
    figures_dir = os.path.join(notebook_dir, "output")
    os.makedirs(figures_dir, exist_ok=True)

    pdf_path = os.path.join(figures_dir, f"{output_prefix}.pdf")
    png_path = os.path.join(figures_dir, f"{output_prefix}.png")

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.show()

    return df


if __name__ == "__main__":
    DATA_DIR = DATA_ROOT
    targets = np.load(os.path.join(DATA_DIR, "targets_and_masks.npz"))

    Xmatch = np.load(os.path.join(DATA_DIR, "Xmatch.npy"), mmap_mode="r")
    Xsub = np.load(os.path.join(DATA_DIR, "Xsub.npy"), mmap_mode="r")

    failed = np.load(os.path.join(DATA_DIR, "failed.npy"), mmap_mode="r")
    hw = np.load(os.path.join(DATA_DIR, "hw.npy"), mmap_mode="r")
    wait_sv = np.load(os.path.join(DATA_DIR, "wait_sv.npy"), mmap_mode="r")
    tr_mask = np.load(os.path.join(DATA_DIR, "tr_mask.npy"), mmap_mode="r")
    te_mask = np.load(os.path.join(DATA_DIR, "te_mask.npy"), mmap_mode="r")

    yv = failed
    idx = np.arange(len(yv))

    # Temporal split indices
    tri_t = np.where(tr_mask)[0]
    tei_t = np.where(te_mask)[0]

    # Random split indices
    rng = np.random.default_rng(0)
    perm = rng.permutation(idx)
    rte = np.sort(perm[: len(tei_t)])
    rtr = np.sort(perm[len(tei_t) :])

    print(f"Data loaded: Total {len(yv):,} rows | Feature matrix {Xmatch.shape}")
    print(f"Random split: {len(rtr):,} train / {len(rte):,} test")
    print(f"Temporal split: {len(tri_t):,} train / {len(tei_t):,} test")

    evaluate_protocol_models(
        X_all=Xmatch,
        y_all=yv,
        rtr=rtr,
        rte=rte,
        tri_t=tri_t,
        tei_t=tei_t
    )