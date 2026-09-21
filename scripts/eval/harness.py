import os
import gc
import json
import time
import datetime as dt
import traceback
import numpy as np
import psutil
import torch
import argparse
import joblib
from sklearn.metrics import confusion_matrix
from eval.helper import (
    _empty_gpu, _nfeat, _get_slice, log_result, thr_sample, pick_thr,
    cls_metrics, cls_metrics_per_class, reg_metrics, THR_GRID, holdout_split,
    set_global_seed, save_predictions, load_predictions, align_predictions,
    cascade_metrics, DEFAULT_SEED, fit_calibrator, calibration_report,
    terminal_time, temporal_masks, pinball_loss, interval_metrics,
    wait_reference_metrics, skill_score, WAIT_MIX, WAIT_REGIMES,
    aggregate_seeds,
)
from eval.paths import DATA_ROOT, model_path
from eval.wandb_logger import WandbRun, load_config, config_summary
from train.tree import (_xgb_reg, _sk_reg, _xgb_cls, _xgb_prep, sk_gain, _sk_cls,
                        _sk_fit, _xgb_quantile, _lgbm_quantile, _xgb_aft, WAIT_TAUS)
from train.mlp import mlp_fit_eval
from train.tabnet import tabnet_fit_eval
from train.saint import saint_fit_eval
from train.ft import ft_fit_eval
from train.tabr import tabr_fit_eval
from train.tsmixer import tsmixer_fit_eval
from train.hierarchical import hierarchical_fit_eval

# Resolved once in __main__; one run per (experiment, model, split, seed).
WB_CFG = {"enabled": False}

CUTOFF_TAG = None


def _result_key(split, seed):
    """Result key: split, suffixed with seed/cutoff only when non-default, so
    existing result files stay readable and sweeps accumulate."""
    key = split
    if seed != DEFAULT_SEED:
        key += f"__seed{seed}"
    if CUTOFF_TAG:
        key += f"__cut{CUTOFF_TAG}"
    return key


def report_seed_variance(got, metrics, label=""):
    """Prints mean +/- std across seeds for one model, and returns the aggregate.

    `got` is the run_*_model result dict, whose keys carry the seed suffix. Runs are
    grouped by split so the spread is reported within a protocol, which is the only
    way it is interpretable: random and temporal differ systematically, so pooling
    them would report protocol effect as seed noise.
    """
    by_split = {}
    for key, mm in (got or {}).items():
        if not isinstance(mm, dict):
            continue
        split = key.split("__")[0]
        by_split.setdefault(split, []).append(mm)

    out = {}
    for split, runs in sorted(by_split.items()):
        if len(runs) < 2:
            continue
        agg = aggregate_seeds(runs)
        out[split] = agg
        print(f"\n  seed variance{label} [{split}] over {agg['n_seeds']} seeds:",
              flush=True)
        for m in metrics:
            blk = agg.get(m)
            if isinstance(blk, dict):
                print(f"    {m:24s} {blk['mean']:.4f} +/- {blk['std']:.4f} "
                      f"(min {blk['min']:.4f}, max {blk['max']:.4f})", flush=True)
    if not out:
        print(f"  seed variance{label}: single seed, no spread to report "
              f"(pass --seeds 0,1,2,3,4)", flush=True)
    return out


def _flatten_metrics(d, prefix=""):
    """Nested per-class blocks -> dotted keys (hardware.pr_auc), since W&B charts
    scalars."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_metrics(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def _wb_start(experiment, model, split, seed, **extra):
    """Opens a W&B run for one (experiment, model, split, seed) cell."""
    return WandbRun.start(WB_CFG, experiment=experiment, model=model, split=split,
                          seed=seed, extra=extra)


def mem_gb():
    return psutil.Process().memory_info().rss / (1024 ** 3)

def save_experiment_results(exp_name, lib, got_metrics, output_dir=None):
    """Loads existing experiment JSON, updates the entries for the model, and saves back to disk."""
    output_dir = output_dir if output_dir is not None else os.getcwd() + "/results"
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{exp_name}_results.json")

    data = {}
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}

    # Merge new split metrics for this model library
    if lib not in data:
        data[lib] = {}
    data[lib].update(got_metrics)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[{exp_name.upper()}] Appended results for model '{lib}' to {json_path}", flush=True)

def fit_eval_binary(
    lib, parts, tri, tei, trs, yv, spw, want_imp=False, ncat=None, split=None, exp_tag=None,
    class_names=None, eval_mask=None, seed=DEFAULT_SEED, save_preds=True,
    calibrate="isotonic",
):
    """Fits the requested binary model library and returns metrics, importance, and CM.

    `class_names` as a (positive, negative) pair switches the metric block to the
    per-class form -- used by E3, where both classes name a real task. Left as None
    (E1) the metrics stay positive-class only.

    `eval_mask` is a boolean over `tei` selecting the rows the metrics are computed
    on. Predictions are still produced, and saved, for all of `tei`. E3 uses this to
    score every test job -- which the cascade needs -- while reporting its own
    conditional metrics on the failed subset only.
    """
    set_global_seed(seed)
    yva = np.asarray(yv)
    nfeat = _nfeat(parts)
    imp = None

    if lib == "mlp":
        p_te, p_trs = mlp_fit_eval(
            parts, tri, tei, ncat, "bin", yv, spw=spw, trs=trs, split=split, exp_tag=exp_tag
        )

    elif lib == "tabnet":
        p_te, p_trs, imp = tabnet_fit_eval(
            parts, tri, tei, ncat, "bin", yv, spw=spw, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag
        )

    elif lib == "saint":
        p_te, p_trs, imp = saint_fit_eval(
            parts, tri, tei, ncat, "bin", yv, spw=spw, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag
        )

    elif lib == "ft":
        p_te, p_trs, imp = ft_fit_eval(
            parts, tri, tei, ncat, "bin", yv, spw=spw, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag
        )

    elif lib == "tabr":
        p_te, p_trs, imp = tabr_fit_eval(
            parts, tri, tei, ncat, "bin", yv, spw=spw, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag
        )

    elif lib == "tsmixer":
        p_te, p_trs, imp = tsmixer_fit_eval(
            parts, tri, tei, ncat, "bin", yv, spw=spw, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag
        )


    elif lib == "xgboost":
        Xtr = _get_slice(parts, tri)
        m = _xgb_cls(spw, seed)
        m.fit(Xtr, yva[tri])

        X_trs = _xgb_prep(_get_slice(parts, trs))
        X_tei = _xgb_prep(_get_slice(parts, tei))

        p_trs = m.predict_proba(X_trs)[:, 1]
        p_te = m.predict_proba(X_tei)[:, 1]

        if hasattr(p_trs, "cpu"):
            p_trs = p_trs.cpu().numpy()
            p_te = p_te.cpu().numpy()

        if want_imp:
            imp = sk_gain(m, nfeat)

        if split is not None:
            save_path = model_path(exp_tag, lib, "bin", split, ".json")
            m.save_model(save_path)
            print(f"[{lib}] Saved model to {save_path}", flush=True)

        del m, Xtr, X_trs, X_tei

    else:
        Xtr = _get_slice(parts, tri)
        m = _sk_fit(_sk_cls(lib, spw, seed), Xtr, yva[tri])
        p_trs = m.predict_proba(_get_slice(parts, trs))[:, 1]
        p_te = m.predict_proba(_get_slice(parts, tei))[:, 1]
        if want_imp:
            imp = sk_gain(m, nfeat)

        if split is not None:
            save_path = model_path(exp_tag, lib, "bin", split, ".txt")
            if lib in ("lightgbm", "lgb"):
                m.booster_.save_model(save_path)
            elif lib in ("catboost", "cb"):
                m.save_model(save_path)
            else:
                joblib.dump(m, save_path)
            print(f"[{lib}] Saved model to {save_path}", flush=True)

        del m, Xtr

    _empty_gpu()
    gc.collect()

    # Calibrate on the held-out slice `trs`, which the model was not fitted on and
    # which carries the natural class balance. spw-weighted training distorts the
    # output probabilities; that is harmless for a single stage but breaks the
    # cascade product, so both raw and calibrated scores are persisted.
    p_te_cal, cal_report, _cal = p_te, None, None
    if calibrate:
        try:
            _cal = fit_calibrator(p_trs, yva[trs], method=calibrate)
            p_te_cal = _cal(p_te)
        except Exception as e:
            print(f"[{lib}] calibration skipped: {e}", flush=True)
            p_te_cal, _cal = p_te, None

    # Persist the test scores: the cascade, bootstrap intervals and seed-variance
    # reporting all read these back rather than reloading a model and re-running
    # inference.
    if save_preds and exp_tag is not None and split is not None:
        save_predictions(exp_tag, lib, split, np.asarray(tei), p_te, seed=seed,
                         probs_cal=np.asarray(p_te_cal, dtype=np.float32))

    m_idx = slice(None) if eval_mask is None else np.asarray(eval_mask, dtype=bool)
    y_eval, p_eval = yva[tei][m_idx], p_te[m_idx]

    # Score calibration on the population the stage is DEFINED over, not on every row
    # it happened to score. E3 estimates P(hardware | failure) and is calibrated on a
    # failures-only slice, so judging it against the unconditional base rate over all
    # test jobs would report a correctly calibrated stage as badly miscalibrated.
    if calibrate and p_te_cal is not p_te:
        cal_report = calibration_report(y_eval, p_eval, np.asarray(p_te_cal)[m_idx])
        print(f"[{lib}] calibration ({calibrate}): Brier "
              f"{cal_report['brier_raw']:.5f} -> {cal_report['brier_cal']:.5f} | ECE "
              f"{cal_report['ece_raw']:.4f} -> {cal_report['ece_cal']:.4f} | mean pred "
              f"{cal_report['mean_pred_raw']:.4f} -> {cal_report['mean_pred_cal']:.4f} "
              f"(base {cal_report['base_rate']:.4f})", flush=True)

    thr = pick_thr(yva[trs], p_trs)
    if class_names is not None:
        # Each task gets its own operating point, tuned on the same held-out sample.
        thr_neg = pick_thr(yva[trs], p_trs, target=0)
        mm = cls_metrics_per_class(
            y_eval, p_eval, thr, thr_neg=thr_neg,
            pos_name=class_names[0], neg_name=class_names[1],
        )
    else:
        mm = cls_metrics(y_eval, p_eval, thr)
    mm["seed"] = int(seed)
    if cal_report is not None:
        mm["calibration"] = cal_report
    # `pick_thr` returns a cut on the RAW score, but downstream consumers -- the
    # cascade above all -- read the calibrated scores by default. Calibration is
    # monotone, so the equivalent calibrated cut is the raw cut pushed through the
    # same map; storing it keeps the two scales from being mixed up.
    if _cal is not None:
        mm["threshold_calibrated"] = float(np.asarray(_cal(np.array([thr])))[0])
        if class_names is not None:
            mm["threshold_neg_calibrated"] = float(np.asarray(_cal(np.array([thr_neg])))[0])
            mm[class_names[0]]["threshold_calibrated"] = mm["threshold_calibrated"]
            mm[class_names[1]]["threshold_calibrated"] = mm["threshold_neg_calibrated"]
    cm = confusion_matrix(
        y_eval.astype(np.int8), (p_eval >= thr).astype(np.int8), labels=[0, 1]
    )
    return mm, imp, cm

def fit_eval_reg(
    lib, parts, tri, tei, trs, yv, want_imp=False, ncat=None, split=None,
    exp_tag=None, seed=DEFAULT_SEED, refit_idx=None
):
    """Fits the requested regression model library and returns wait time regression metrics and importance.

    `seed` must be threaded through: the neural trainers draw init and shuffling from
    the global RNGs, and the tree regressors subsample rows and columns, so without it
    every seed refits the identical model and the spread across seeds is exactly 0.

    `refit_idx` is the FULL training window. Models that select an epoch count on the
    validation holdout use `tri`/`trs` for that, then refit on `refit_idx`; models with
    nothing to select (the boosted trees, the hierarchical estimator) skip selection
    and fit on `refit_idx` directly. Both paths end with every model's final fit seeing
    the same rows, which is what makes the cross-model comparison mean anything --
    holding the most recent 10% out of the final fit cost XGBoost 0.13 R2_log on the
    temporal split despite it having no epoch to select. See docs/validation-protocol.md.
    """
    # Models with no selection step fit here; the holdout would only take data away.
    _fit_i = tri if refit_idx is None else np.asarray(refit_idx)
    set_global_seed(seed)
    yva = np.asarray(yv)
    nfeat = _nfeat(parts)
    imp = None

    if lib == "mlp":
        p_te, p_trs = mlp_fit_eval(
            parts, _fit_i, tei, ncat, "reg", yv, trs=trs, split=split, exp_tag=exp_tag
        )

    elif lib == "tabnet":
        p_te, p_trs, imp = tabnet_fit_eval(
            parts, tri, tei, ncat, "reg", yv, trs=trs, want_imp=want_imp, split=split,
            exp_tag=exp_tag, seed=seed, refit_idx=refit_idx
        )

    elif lib == "saint":
        p_te, p_trs, imp = saint_fit_eval(
            parts, tri, tei, ncat, "reg", yv, trs=trs, want_imp=want_imp, split=split,
            exp_tag=exp_tag, is_regression=True, refit_idx=refit_idx
        )

    elif lib == "ft":
        p_te, p_trs, imp = ft_fit_eval(
            parts, tri, tei, ncat, "reg", yv, trs=trs, want_imp=want_imp, split=split,
            exp_tag=exp_tag, is_regression=True, refit_idx=refit_idx
        )

    elif lib == "tabr":
        p_te, p_trs, imp = tabr_fit_eval(
            parts, tri, tei, ncat, "reg", yv, trs=trs, want_imp=want_imp, split=split, exp_tag=exp_tag, is_regression=True
        )

    elif lib == "tsmixer":
        p_te, p_trs, imp = tsmixer_fit_eval(
            parts, tri, tei, ncat, "reg", yv, trs=trs, want_imp=want_imp, split=split,
            exp_tag=exp_tag, is_regression=True, refit_idx=refit_idx
        )

    elif lib == "hierarchical":
        p_te, p_trs, imp = hierarchical_fit_eval(
            parts, _fit_i, tei, ncat, "reg", yv, trs=trs, want_imp=want_imp, split=split,
            exp_tag=exp_tag, seed=seed
        )

    elif lib in ("xgboost", "xgb"):
        Xtr = _get_slice(parts, _fit_i)
        m = _xgb_reg(seed)
        m.fit(Xtr, yva[_fit_i])

        X_trs = _xgb_prep(_get_slice(parts, trs))
        X_tei = _xgb_prep(_get_slice(parts, tei))

        p_trs = m.predict(X_trs)
        p_te = m.predict(X_tei)

        if hasattr(p_trs, "cpu"):
            p_trs = p_trs.cpu().numpy()
            p_te = p_te.cpu().numpy()

        if want_imp:
            imp = sk_gain(m, nfeat)

        if split is not None:
            save_path = model_path(exp_tag, lib, "reg", split, ".json")
            m.save_model(save_path)
            print(f"[{lib}] Saved regression model to {save_path}", flush=True)

        del m, Xtr, X_trs, X_tei

    else:
        Xtr = _get_slice(parts, _fit_i)
        m = _sk_fit(_sk_reg(lib, seed), Xtr, yva[_fit_i])
        p_trs = m.predict(_get_slice(parts, trs))
        p_te = m.predict(_get_slice(parts, tei))
        if want_imp:
            imp = sk_gain(m, nfeat)

        if split is not None:
            save_path = model_path(exp_tag, lib, "reg", split, ".txt")
            if lib in ("lightgbm", "lgb"):
                m.booster_.save_model(save_path)
            elif lib in ("catboost", "cb"):
                m.save_model(save_path)
            else:
                joblib.dump(m, save_path)
            print(f"[{lib}] Saved regression model to {save_path}", flush=True)

        del m, Xtr

    _empty_gpu()
    gc.collect()

    mm = reg_metrics(yva[tei], p_te)
    return mm, imp, p_te


def fit_eval_reg_dist(lib, parts, tri, tei, trs, yv, taus=WAIT_TAUS, head="quantile",
                      censored=None, aft_scale=1.17, seed=DEFAULT_SEED, split=None,
                      exp_tag="e2"):
    """Fits a DISTRIBUTIONAL wait-time head and scores it as an interval predictor.

    `fit_eval_reg` trains squared error on log1p and emits one number per job. The
    FermiGrid mixture fit says that is the wrong output object: even a model that
    knew a job's component exactly would still face a conditional SD of ~1.17 in
    log space, so the honest deliverable is a range, not a point.

    Two heads:
      "quantile" -- fits `taus` directly (one multi-output XGBoost model, or one
        LightGBM model per tau). Trains and predicts in log1p space like the rest
        of E2, so the outputs drop straight into `reg_metrics`.
      "aft" -- accelerated failure time with a normal loss, i.e. a conditional
        lognormal. Predicts a scale in SECONDS; quantiles are recovered as
        pred * exp(aft_scale * z_tau), which assumes the homoscedastic sigma the
        objective was fitted with. `aft_scale` defaults to the within-component
        SD of the fitted mixture. This head is the one that can consume censored
        rows -- pass `censored`, a boolean row-aligned mask marking jobs whose
        wait is a lower bound (queued but never ran).

    Returns (metrics, preds_by_tau). `metrics` carries the point metrics of the
    median head plus pinball loss and interval coverage/width.
    """
    yva = np.asarray(yv, dtype=np.float64)
    taus = tuple(float(t) for t in taus)
    set_global_seed(seed)
    Xtr = _get_slice(parts, tri)
    preds = {}

    if head == "aft":
        # AFT labels live in original time units; the objective takes the log.
        lo = np.maximum(np.expm1(yva[tri]), 1e-3)
        hi = lo.copy()
        if censored is not None:
            c = np.asarray(censored, dtype=bool)[tri]
            hi[c] = np.inf
            print(f"[{lib}/aft] {int(c.sum()):,} of {len(c):,} training rows right-censored",
                  flush=True)
        bst = _xgb_aft(Xtr, lo, hi, seed=seed, scale=aft_scale)
        import xgboost as _xgb
        from scipy.stats import norm as _norm

        # aft_loss_distribution_scale is a fixed hyperparameter of the LOSS -- XGBoost
        # never estimates it -- so it is the wrong number to build quantiles from.
        # Using it directly gave intervals a quarter of the width the quantile heads
        # produced, and 0.58 coverage against a nominal 0.80. The spread is instead
        # estimated from the training residuals in log space, on the UNCENSORED rows
        # only: a censored row's residual is a lower bound, so including it would drag
        # the estimate down and re-narrow the intervals.
        fit_pred = bst.predict(_xgb.DMatrix(_xgb_prep(Xtr)))
        obs = np.isfinite(hi)
        resid = np.log(np.maximum(lo[obs], 1e-3)) - np.log(np.maximum(fit_pred[obs], 1e-3))
        sigma = float(np.std(resid))
        print(f"[{lib}/aft] residual sigma_log = {sigma:.3f} "
              f"(loss scale was {aft_scale:.2f}; quantiles use the residual)", flush=True)

        scale_pred = bst.predict(_xgb.DMatrix(_xgb_prep(_get_slice(parts, tei))))
        for t in taus:
            preds[t] = np.log1p(np.maximum(scale_pred * np.exp(sigma * _norm.ppf(t)), 0.0))
        del bst

    elif lib in ("xgboost", "xgb"):
        m = _xgb_quantile(taus, seed=seed)
        m.fit(Xtr, yva[tri])
        q = np.asarray(m.predict(_xgb_prep(_get_slice(parts, tei))), dtype=np.float64)
        q = q.reshape(len(q), -1)
        for i, t in enumerate(taus):
            preds[t] = q[:, i]
        del m

    else:
        Xte = _get_slice(parts, tei)
        for t in taus:
            m = _sk_fit(_lgbm_quantile(t, seed=seed), Xtr, yva[tri])
            preds[t] = np.asarray(m.predict(Xte), dtype=np.float64)
            del m
        del Xte

    del Xtr
    _empty_gpu()
    gc.collect()

    # Quantile heads can cross; sorting each row restores monotonicity without
    # changing any individual head's marginal calibration much.
    order_t = sorted(preds)
    stack = np.sort(np.column_stack([preds[t] for t in order_t]), axis=1)
    preds = {t: stack[:, i] for i, t in enumerate(order_t)}

    med = preds[min(taus, key=lambda t: abs(t - 0.5))]
    mm = reg_metrics(yva[tei], med)
    mm.update(pinball_loss(yva[tei], preds))
    lo_t, hi_t = min(taus), max(taus)
    iv = interval_metrics(yva[tei], preds[lo_t], preds[hi_t])
    mm.update({f"iv_{k}": v for k, v in iv.items()})
    mm["iv_taus"] = [lo_t, hi_t]
    mm["head"] = head
    return mm, preds


def run_e2_reference(wait_log, splits, results=None):
    """Scores the two no-feature reference predictors on every split.

    These are the floor E2 has to clear, and without them an R2_log of 0.27 has no
    scale: a single constant already reaches within-2x ~ 0.25 on this distribution
    because log wait is wide but unimodal. Fitted on the training slice only, so
    they are admissible baselines and not oracles.
    """
    yva = np.asarray(wait_log, dtype=np.float64)
    out = {}
    for split, a, b in splits:
        tri = a[~np.isnan(yva[a])]
        tei = b[~np.isnan(yva[b])]
        ref = wait_reference_metrics(yva[tri], yva[tei])
        out[split] = ref
        for name, d in ref.items():
            log_result("E2", model=f"ref:{name}", split=split,
                       **{k: v for k, v in d.items() if k != "pred_s"})
            print(f"{'ref:' + name:14s} {split:9s} pred={d['pred_s']:8,.0f}s  "
                  f"R2(log) {d['r2_log']:+.3f}  within-2x {d['within2x']:.3f}  "
                  f"MAE(log1p) {d['mae_log1p']:.3f}", flush=True)
    if results is not None:
        results.setdefault("reference", {}).update(out)
    return out


def run_e2_dist(lib, Xsub, wait_log, splits, ncat, head="quantile", censored=None,
                order=None, seed=DEFAULT_SEED, results=None):
    """Executes the distributional wait head across splits, alongside `run_e2_model`."""
    got = {}
    for split, a, b in splits:
        wb = _wb_start("e2dist", f"{lib}:{head}", split, seed,
                       head=head, n_train=int(len(a)), n_test=int(len(b)))
        try:
            yva = np.asarray(wait_log, dtype=np.float64)
            tri_all = a[~np.isnan(yva[a])]
            tei = b[~np.isnan(yva[b])]
            fit_i, val_i = holdout_split(
                tri_all, order=order if split == "temporal" else None)
            tri = fit_i
            trs = thr_sample(val_i)

            t0 = time.perf_counter()
            mm, _ = fit_eval_reg_dist(lib, [Xsub], tri, tei, trs, yva, head=head,
                                      censored=censored, seed=seed, split=split,
                                      exp_tag="e2dist")
            got[split] = mm
            wb.log_final(mm)
            log_result("E2dist", model=f"{lib}:{head}", split=split,
                       **{k: v for k, v in mm.items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool)})
            print(
                f"{'model':14s} {'split':9s}  R2(log)  Within-2x  Pinball  Cover  Width(x)\n"
                f"{lib + ':' + head:14s} {split:9s}  {mm['r2_log']:.3f}    "
                f"{mm['within2x']:.3f}      {mm['pinball_mean']:.3f}    "
                f"{mm['iv_coverage']:.3f}  {mm['iv_width_factor']:.0f}",
                flush=True,
            )
            print(f"Total time: {time.perf_counter() - t0:.2f} seconds")
        except Exception as e:
            traceback.print_exc()
            print(f"  [skip] {lib}/{head}/{split}: {e}", flush=True)
            _empty_gpu()
            gc.collect()
        finally:
            wb.finish()
    if results is not None:
        results.setdefault(f"{lib}:{head}", {}).update(got)
    return got


def run_e1_model(lib, Xm, yv, splits, imp_store, cm_store, ncat, order=None, seed=DEFAULT_SEED):
    """Executes evaluation across all splits for a given model architecture.

    `order` is the per-row time key used to carve the threshold-selection slice off
    the end of the training window on temporal splits; see `holdout_split`.
    """
    yva = np.asarray(yv)
    got = {}
    for split, a, b in splits:
        wb = _wb_start("e1", lib, split, seed, n_train=int(len(a)), n_test=int(len(b)))
        try:
            fit_i, val_i = holdout_split(a, order=order if split == "temporal" else None)
            trs = thr_sample(val_i)
            spw = float((yva[fit_i] == 0).sum() / max((yva[fit_i] == 1).sum(), 1))
            print(
                f"[{lib}/{split}] fit {len(fit_i):,} | threshold-selection holdout "
                f"{len(val_i):,} ({len(val_i) / len(a):.0%}"
                f"{', most recent' if order is not None and split == 'temporal' else ', random'})",
                flush=True,
            )
            t_start = time.perf_counter()
            mm, imp, cm = fit_eval_binary(
                lib,
                [Xm],
                fit_i,
                b,
                trs,
                yv,
                spw,
                want_imp=True,
                ncat=ncat,
                split=split,
                exp_tag="e1",
                seed=seed,
            )
            if imp is not None:
                imp_store[(lib, split)] = imp

            cm_store[(lib, split)] = cm
            got[_result_key(split, seed)] = mm
            wb.log_final(mm)
            if imp is not None:
                wb.log_importance([f"f{i}" for i in range(len(imp))], imp)
            # log_result("E1", model=lib, split=split, **mm)
            print(
                f"{'model':9s} {'split':9s}  ROC    PR     F1     Prec   Rec\n"
                f"{lib:9s} {split:9s}  {mm['roc_auc']:.3f}  {mm['pr_auc']:.3f}  {mm['f1']:.3f}  "
                f"{mm['precision']:.3f}  {mm['recall']:.3f}",
                flush=True,
            )
            t_end = time.perf_counter()
            print(f"Total time: {t_end - t_start:.2f} seconds")
        except Exception as e:
            traceback.print_exc()
            print(f"  [skip] {lib}/{split}: {e}", flush=True)
            _empty_gpu()
            gc.collect()
        finally:
            wb.finish()

    if "random" in got and "temporal" in got:
        d, t = got["random"], got["temporal"]
        print(
            f"{lib:9s} {'Delta':9s}  {t['roc_auc'] - d['roc_auc']:+.3f}  {t['pr_auc'] - d['pr_auc']:+.3f}  "
            f"{t['f1'] - d['f1']:+.3f}  {t['precision'] - d['precision']:+.3f}  {t['recall'] - d['recall']:+.3f}"
        )

    if got:
        save_experiment_results("protocol_eval", lib, got)

    return got


def run_e2_model(lib, Xsub, wait_log, splits, imp_store, ncat, order=None,
                 seed=DEFAULT_SEED):
    """Executes submit-time wait regression across requested splits for a given model architecture.

    `order` is the per-row time key used to carve the validation slice off the end of
    the training window on temporal splits; see `holdout_split`.
    """
    got = {}
    for split, a, b in splits:
        wb = _wb_start("e2", lib, split, seed, n_train=int(len(a)), n_test=int(len(b)))
        try:
            # Mask out unobserved or NaN wait times
            vw_a = ~np.isnan(wait_log[a])
            vw_b = ~np.isnan(wait_log[b])
            tri_all = a[vw_a]
            tei = b[vw_b]
            # Genuine holdout, as E1/E3 already do. `trs` was previously a subsample
            # of the training rows themselves, so a neural trainer validating on it
            # was validating on data it had fit -- early stopping could not see
            # overfitting at all. Fit on fit_i, validate on a slice held out of it.
            fit_i, val_i = holdout_split(
                tri_all, order=order if split == "temporal" else None)
            tri = fit_i
            trs = thr_sample(val_i)
            print(f"[{lib}/{split}] fit {len(fit_i):,} | validation holdout "
                  f"{len(val_i):,} ({len(val_i) / len(tri_all):.0%}"
                  f"{', most recent' if order is not None and split == 'temporal' else ', random'})",
                  flush=True)

            t_start = time.perf_counter()
            mm, imp, _ = fit_eval_reg(
                lib,
                [Xsub],
                tri,
                tei,
                trs,
                wait_log,
                want_imp=True,
                ncat=ncat,
                split=split,
                exp_tag="e2",
                seed=seed,
                refit_idx=tri_all,
            )
            if imp is not None:
                imp_store[(lib, split)] = imp

            got[_result_key(split, seed)] = mm
            wb.log_final(mm)
            if imp is not None:
                wb.log_importance([f"f{i}" for i in range(len(imp))], imp)
            log_result("E2", model=lib, split=split, **mm)
            print(
                f"{'model':9s} {'split':9s}  R2(log)  Med-AE(s)  Within-2x  "
                f"MAE(<2m)  MAE(2m-45m) MAE(45m-1d) MAE(>1d)\n"
                f"{lib:9s} {split:9s}  {mm['r2_log']:.3f}    {mm['median_ae_s']:7.0f}s  {mm['within2x']:.3f}     "
                f"{mm['mae_inst']:8.0f}s  {mm['mae_turn']:8.0f}s   {mm['mae_prov']:8.0f}s  "
                f"{mm['mae_park']:8.0f}s",
                flush=True,
            )
            t_end = time.perf_counter()
            print(f"Total time: {t_end - t_start:.2f} seconds")
        except Exception as e:
            traceback.print_exc()
            print(f"  [skip] {lib}/{split}: {e}", flush=True)
            _empty_gpu()
            gc.collect()
        finally:
            wb.finish()

    if got:
        save_experiment_results("wait_time", lib, got)

    return got

E3_TASKS = ("hardware", "payload")
_E3_HDR = f"{'model':10s} {'split':10s} {'task':9s}  Thr   PR     F1     Prec   Rec     Support"


def _fmt_e3_block(lib, split, mm):
    """One row per attribution task at its own tuned threshold, then shared metrics."""
    lines = [_E3_HDR]
    for task in E3_TASKS:
        t = mm[task]
        # A cut pinned to the end of the search grid means F1 for that task was still
        # climbing -- the operating point is a grid artifact, not an optimum.
        edge = " *" if t["threshold"] <= THR_GRID[0] or t["threshold"] >= THR_GRID[-1] else ""
        lines.append(
            f"{lib:10s} {split:10s} {task:9s}  {t['threshold']:.2f}  {t['pr_auc']:.3f}  "
            f"{t['f1']:.3f}  {t['precision']:.3f}  {t['recall']:.3f}  {t['support']:>9,d}{edge}"
        )
    lines.append(
        f"{'':10s} {'':10s} {'shared':9s}  ROC {mm['roc_auc']:.3f}  MCC {mm['mcc']:.3f}  "
        f"Acc {mm['accuracy']:.3f}  BalAcc {mm['balanced_accuracy']:.3f}  "
        f"MacroF1 {mm['macro_f1']:.3f} (tuned {mm['macro_f1_tuned']:.3f})"
    )
    if any(
        mm[t]["threshold"] <= THR_GRID[0] or mm[t]["threshold"] >= THR_GRID[-1]
        for t in E3_TASKS
    ):
        lines.append(
            f"{'':10s} {'':10s} * threshold at the edge of the "
            f"[{THR_GRID[0]:.2f}, {THR_GRID[-1]:.2f}] search grid"
        )
    return "\n".join(lines)


def _fmt_e3_delta(lib, rnd, tmp):
    """Random -> temporal shift, per task. Negative means the temporal split is harder."""
    lines = [f"{'model':10s} {'split':10s} {'task':9s}  dThr   dPR    dF1    dPrec  dRec"]
    for task in E3_TASKS:
        d, t = rnd[task], tmp[task]
        lines.append(
            f"{lib:10s} {'Delta':10s} {task:9s}  {t['threshold'] - d['threshold']:+.2f}  "
            f"{t['pr_auc'] - d['pr_auc']:+.3f}  "
            f"{t['f1'] - d['f1']:+.3f}  {t['precision'] - d['precision']:+.3f}  "
            f"{t['recall'] - d['recall']:+.3f}"
        )
    lines.append(
        f"{'':10s} {'':10s} {'shared':9s}  dROC {tmp['roc_auc'] - rnd['roc_auc']:+.3f}  "
        f"dMCC {tmp['mcc'] - rnd['mcc']:+.3f}  "
        f"dBalAcc {tmp['balanced_accuracy'] - rnd['balanced_accuracy']:+.3f}  "
        f"dMacroF1 {tmp['macro_f1'] - rnd['macro_f1']:+.3f}"
    )
    return "\n".join(lines)


def run_e3_model(lib, Xm, failed, hw, splits, imp_store, cm_store, ncat, order=None, seed=DEFAULT_SEED):
    """Executes fault attribution evaluation (hardware vs. payload failure) conditioned on job failure.

    `order` is the per-row time key used to carve the threshold-selection slice off
    the end of the training window on temporal splits; see `holdout_split`.
    """
    hw_a = np.asarray(hw)
    failed_a = np.asarray(failed)
    got = {}

    total_failed = (failed_a == 1).sum()
    total_hw = ((failed_a == 1) & (hw_a == 1)).sum()
    total_payload = ((failed_a == 1) & (hw_a == 0)).sum()
    
    print("\n" + "=" * 60, flush=True)
    print(f"  Total Failed Jobs: {total_failed:,}", flush=True)
    print(f"  Hardware Faults  : {total_hw:,} ({total_hw / total_failed * 100:.2f}%)", flush=True)
    print(f"  Payload Faults   : {total_payload:,} ({total_payload / total_failed * 100:.2f}%)", flush=True)
    print("=" * 60 + "\n", flush=True)

    for split, a, b in splits:
        wb = _wb_start("e3", lib, split, seed, n_train=int(len(a)), n_test=int(len(b)),
                       n_failed=int(total_failed), n_hw=int(total_hw))
        try:
            # Fit on failed jobs only: E3 is the conditional task.
            tri = a[failed_a[a] == 1]
            # Train on failures only -- the conditional task -- but SCORE every test
            # job. At match time nothing knows which jobs will fail, so the cascade
            # needs P(hardware | failure) on the whole test population; `eval_mask`
            # keeps E3's own reported metrics on the failed subset it is defined for.
            tei = b
            eval_mask = failed_a[b] == 1

            # Fit and threshold-selection rows must be disjoint, so the operating
            # point is chosen on predictions this model has not already seen.
            fit_i, val_i = holdout_split(tri, order=order if split == "temporal" else None)
            trs = thr_sample(val_i)
            spw = float((hw_a[fit_i] == 0).sum() / max((hw_a[fit_i] == 1).sum(), 1))
            print(
                f"[{lib}/{split}] fit {len(fit_i):,} | threshold-selection holdout "
                f"{len(val_i):,} ({len(val_i) / len(tri):.0%}"
                f"{', most recent' if order is not None and split == 'temporal' else ', random'}"
                f", {(hw_a[val_i] == 1).mean() * 100:.2f}% hardware) | "
                f"scoring {len(tei):,} test jobs, reporting on {int(eval_mask.sum()):,} failed",
                flush=True,
            )

            t_start = time.perf_counter()
            mm, imp, cm = fit_eval_binary(
                lib,
                [Xm],
                fit_i,
                tei,
                trs,
                hw,
                spw,
                want_imp=True,
                ncat=ncat,
                split=split,
                exp_tag="e3_fault",
                class_names=("hardware", "payload"),
                eval_mask=eval_mask,
                seed=seed,
            )
            if imp is not None:
                imp_store[(lib, split)] = imp

            cm_store[(lib, split)] = cm
            got[_result_key(split, seed)] = mm
            # E3 reports per-class blocks (hardware/payload); flatten them so each
            # class's metrics get their own W&B keys instead of one nested blob.
            wb.log_final(_flatten_metrics(mm))
            if imp is not None:
                wb.log_importance([f"f{i}" for i in range(len(imp))], imp)
            log_result("E3", model=lib, split=split, **mm)
            print(_fmt_e3_block(lib, split, mm), flush=True)
            t_end = time.perf_counter()
            print(f"Total time: {t_end - t_start:.2f} seconds")
        except Exception as e:
            traceback.print_exc()
            print(f"  [skip] {lib}/{split}: {e}", flush=True)
            _empty_gpu()
            gc.collect()
        finally:
            wb.finish()

    if "random" in got and "temporal" in got:
        print(_fmt_e3_delta(lib, got["random"], got["temporal"]), flush=True)

    if got:
        save_experiment_results("fault_attr", lib, got)

    return got

def run_cascade(e1_lib, e3_lib, split, failed, hw, seed=DEFAULT_SEED,
                thr_fail=None, thr_attr=None, results=None):
    """Composes the saved E1 and E3 test scores into a match-time hardware detector.

    E3 on its own answers "given that this job failed, was it hardware?", which is
    not a question anything can ask at match time -- the failure is not known yet.
    Chaining the two stages, P(hardware) = P(failure) * P(hardware | failure),
    restores a prediction over the whole test population, which is the form a
    deployed alert would actually take.

    Reads both stages from their saved prediction files rather than refitting, so
    this is cheap to re-run and any (E1, E3) pairing can be scored.
    """
    idx1, p_fail = load_predictions("e1", e1_lib, split, seed=seed)
    idx3, p_attr = load_predictions("e3_fault", e3_lib, split, seed=seed)
    idx, p_fail, p_attr = align_predictions(idx1, p_fail, idx3, p_attr)

    n1, n3 = len(idx1), len(idx3)
    if len(idx) != n1:
        print(
            f"  [warn] E1 scored {n1:,} rows, E3 scored {n3:,}, overlap {len(idx):,}. "
            f"The cascade is evaluated on the overlap; a shortfall means E3 was run "
            f"before it scored the full test set.",
            flush=True,
        )

    y_hw = (np.asarray(hw)[idx] == 1).astype(np.int8)

    # Default operating points: tuned per stage on their own held-out slices and
    # recorded in the results JSON. Falling back to 0.5 would be arbitrary.
    if (thr_fail is None or thr_attr is None) and results is not None:
        thr_fail = thr_fail if thr_fail is not None else results.get("e1_threshold")
        thr_attr = thr_attr if thr_attr is not None else results.get("e3_threshold")

    mm = cascade_metrics(y_hw, p_fail, p_attr, thr_fail=thr_fail, thr_attr=thr_attr)
    mm["e1_model"], mm["e3_model"], mm["seed"] = e1_lib, e3_lib, int(seed)

    print("\n" + "=" * 68, flush=True)
    print(f"  Cascade  E1={e1_lib}  E3={e3_lib}  split={split}  seed={seed}", flush=True)
    print(f"  Test jobs {mm['n_test']:,} | hardware {mm['n_hardware']:,} "
          f"({mm['prevalence'] * 100:.2f}%) <- AP baseline", flush=True)
    print("-" * 68, flush=True)
    print(f"  {'score':22s} {'AP':>8s} {'AUROC':>8s}   {'lift over base':>14s}", flush=True)
    for key, label in (("soft", "P(fail) x P(hw|fail)"),
                       ("p_fail_only", "P(fail) alone"),
                       ("p_attr_only", "P(hw|fail) alone")):
        b = mm[key]
        print(f"  {label:22s} {b['pr_auc']:8.4f} {b['roc_auc']:8.4f}   "
              f"{b['pr_auc'] / mm['prevalence']:13.1f}x", flush=True)
    if "stage1" in mm:
        s1 = mm["stage1"]
        print("-" * 68, flush=True)
        print(f"  stage 1 gate @ {s1['threshold']:.2f}: flags {s1['flagged']:,} jobs "
              f"({s1['flagged_frac'] * 100:.1f}%)", flush=True)
        print(f"  hardware faults surviving the gate: "
              f"{s1['hardware_recall_ceiling'] * 100:.1f}%  <- ceiling on cascade recall",
              flush=True)
    if "hard" in mm:
        h = mm["hard"]
        print(f"  deployed rule @ ({h['threshold_fail']:.2f}, {h['threshold_attr']:.2f}): "
              f"P {h['precision']:.3f}  R {h['recall']:.3f}  F1 {h['f1']:.3f}  "
              f"MCC {h['mcc']:.3f}", flush=True)
        print(f"  alerts raised: {h['alerts']:,} ({h['alert_frac'] * 100:.2f}% of test jobs)",
              flush=True)
    print("=" * 68 + "\n", flush=True)

    save_experiment_results("cascade", f"{e1_lib}__{e3_lib}", {split: mm})
    return mm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Tabular Model Evaluation Harness (E1/E2)"
    )
    parser.add_argument(
        "experiment",
        type=str,
        choices=["e1", "e2", "e2dist", "e3", "cascade"],
        help="Experiment: 'e1' (failure classification), 'e2' (wait-time regression), "
             "'e2dist' (wait-time interval prediction: quantile or AFT head), "
             "'e3' (fault attribution given failure), or 'cascade' (match-time "
             "hardware detection composing saved e1 and e3 scores)",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model identifier (e.g., ft, saint, tsmixer, xgb, lgb, cat, mlp, tabnet). "
             "For 'cascade', give the two stages as 'e1model:e3model' (e.g. "
             "'lightgbm:saint'), or a single name to use it for both stages.",
    )
    parser.add_argument(
        "split",
        type=str,
        nargs="?",
        default="both",
        choices=["random", "temporal", "both"],
        help="Split protocol to run: 'random', 'temporal', or 'both' (default: both)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for the fit, and the suffix on saved models, predictions and "
             f"result keys (default: {DEFAULT_SEED}). Repeat a run under several "
             f"seeds to report the spread across them.",
    )
    parser.add_argument(
        "--cutoff", type=str, default="2025-07-01",
        help="Deployment cutoff for the temporal split: ISO date (UTC) or epoch "
             "seconds. Results are keyed by cutoff, so sweeps accumulate.",
    )

    # ---- Weights & Biases ----
    parser.add_argument("--wandb", action="store_true",
                        help="Log to Weights & Biases (see config/wandb.yaml.example).")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Force logging off even if enabled in the config.")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project (overrides config).")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity/team (overrides config).")
    parser.add_argument("--wandb-mode", type=str, default=None,
                        choices=["online", "offline", "disabled"],
                        help="W&B mode; 'offline' writes to ./wandb for `wandb sync`.")
    parser.add_argument("--wandb-tag", type=str, action="append", default=None,
                        help="Extra tag; repeatable.")
    parser.add_argument("--wandb-notes", type=str, default=None,
                        help="Free-text note attached to the run.")
    parser.add_argument(
        "--wandb-tree-rounds", action="store_true",
        help="Per-boosting-round tree curves. Needs an eval_set (an extra pass over "
             "the full matrix) and is scored on a training subsample, not validation.",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="quantile",
        choices=["quantile", "aft"],
        help="e2dist only: 'quantile' fits the tau grid directly in log1p space; "
             "'aft' fits a conditional lognormal (survival:aft, normal loss) that "
             "can also consume right-censored never-ran jobs.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to run in sequence, e.g. '0,1,2,3,4'. Overrides "
             "--seed and reports mean/std across the runs.",
    )
    args = parser.parse_args()
    SEEDS = ([int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed])

    # Resolve W&B config once: CLI > env > config/wandb.yaml.
    WB_CFG = load_config(args)
    globals()["WB_CFG"] = WB_CFG
    print(config_summary(WB_CFG), flush=True)

    SAVE_DIR = DATA_ROOT

    def _parse_cutoff(v):
        """ISO date or epoch seconds -> epoch seconds. Pinned to UTC: the naive
        form uses the machine's local zone and has shifted this cutoff before."""
        v = str(v).strip()
        if v.isdigit():
            return int(v)
        d = dt.datetime.strptime(v, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        return int(d.timestamp())

    CUTOFF_EPOCH = _parse_cutoff(args.cutoff)
    CUTOFF_LABEL = dt.datetime.fromtimestamp(
        CUTOFF_EPOCH, dt.timezone.utc).strftime("%Y-%m-%d")
    print(f"temporal cutoff: {CUTOFF_LABEL} ({CUTOFF_EPOCH})", flush=True)
    # 2025-01-01: timestamps below this are unset sentinels, not real dates.
    WINDOW_FLOOR = 1735689600
    targets = np.load(os.path.join(SAVE_DIR, "targets_and_masks.npz"))

    # Queue-start time: the key the temporal split is cut on, and so the ordering
    # used to hold out the most recent slice of training for threshold selection.
    QS = targets["qs"]

    Xmatch = np.load(os.path.join(SAVE_DIR, "Xmatch.npy"), mmap_mode="r")
    Xsub = np.load(os.path.join(SAVE_DIR, "Xsub.npy"), mmap_mode="r")

    failed = np.load(os.path.join(SAVE_DIR, "failed.npy"), mmap_mode="r")
    hw = np.load(os.path.join(SAVE_DIR, "hw.npy"), mmap_mode="r")

    # Jobs that never started are kept in the feature matrices because they occupied
    # the queue and so shape the contention features, but they are not valid targets
    # for failure prediction or fault attribution -- nothing ran, so there is no
    # run-time outcome to predict. E1 and E3 restrict to RAN below. E2 needs no mask:
    # a job that never started has no wait time and is dropped by its own target.
    _ran_path = os.path.join(SAVE_DIR, "ran.npy")
    RAN = (np.load(_ran_path, mmap_mode="r").astype(bool)
           if os.path.exists(_ran_path) else None)
    if RAN is None:
        print("[warn] ran.npy not found -- every row is treated as a valid target. "
              "Re-run the feature pipeline to regenerate it.", flush=True)
    wait_sv = np.load(os.path.join(SAVE_DIR, "wait_sv.npy"), mmap_mode="r")

    # Compute log-transformed target for wait time regression
    wait_log = np.log1p(np.maximum(wait_sv, 0))

    # Categorical-column counts come from the feature pipeline's schema_meta.json.
    # This previously read globals(), which is empty in a fresh process -- so ncat was
    # always None and the neural trainers treated every column as numeric, losing the
    # entity embeddings. The globals() lookup is kept as a fallback for notebook use.
    _meta_path = os.path.join(SAVE_DIR, "schema_meta.json")
    _SCHEMA = {}
    if os.path.exists(_meta_path):
        with open(_meta_path) as _f:
            _SCHEMA = json.load(_f)
    else:
        print(f"[warn] {_meta_path} missing -- ncat falls back to None, which disables "
              f"categorical embeddings in the neural models.", flush=True)
    NCAT_MATCH = _SCHEMA.get("NCAT_MATCH", globals().get("NCAT_MATCH", None))
    NCAT_SUB = _SCHEMA.get("NCAT_SUB", globals().get("NCAT_SUB", None))
    print(f"ncat: match={NCAT_MATCH} sub={NCAT_SUB}", flush=True)

    globals()["CUTOFF_TAG"] = None if args.cutoff == "2025-07-01" else CUTOFF_LABEL
    WB_CFG.setdefault("tags", [])
    WB_CFG["tags"] = list(WB_CFG["tags"]) + [f"cut{CUTOFF_LABEL}"]

    print(f"Loaded Xmatch {Xmatch.shape} and Xsub {Xsub.shape}")

    # Index setup
    idx = np.arange(len(QS))

    # 1. Temporal split indices.
    #
    # Always cut on when each target became OBSERVABLE, never on submission time. A
    # job submitted in June that terminates in July carries a label a model deployed
    # at the cutoff could not have known; cutting on QDate would leave it in training.
    #
    # Wait time is known at execution start, failure and fault attribution only at
    # termination, so E2 gets an earlier observation time than E1/E3.
    _label_src = targets["jst"] if args.experiment in ("e2", "e2dist") else targets["comp"]
    tau, tau_src = terminal_time(
        _label_src,
        job_start=targets["jst"],
        # Present once the feature pipeline has been re-run; without it, jobs that
        # ran but were removed fall back to queue time.
        wall_clock=targets["wall"] if "wall" in targets.files else None,
        qdate=QS,
        floor=WINDOW_FLOOR,
    )
    n_fb = int((tau_src == 2).sum())
    print(f"Label-observation basis: {args.experiment} uses "
          f"{'JobStartDate' if args.experiment in ('e2', 'e2dist') else 'CompletionDate'}; "
          f"{n_fb:,} rows ({n_fb / len(tau) * 100:.1f}%) fall back to QDate as a "
          f"lower bound (removed before ever starting).")
    tr_t, te_t, _tstats = temporal_masks(QS, CUTOFF_EPOCH, label_time=tau)
    tri_t, tei_t = np.where(tr_t)[0], np.where(te_t)[0]
    print(f"Temporal split at {CUTOFF_LABEL}: train {len(tri_t):,} | test {len(tei_t):,}")

    # 2. Random split indices
    rng = np.random.default_rng(0)
    perm = rng.permutation(idx)
    rte = np.sort(perm[: len(tei_t)])
    rtr = np.sort(perm[len(tei_t) :])

    SPLITS = []
    if args.split in ("random", "both"):
        SPLITS.append(("random", rtr, rte))
    if args.split in ("temporal", "both"):
        SPLITS.append(("temporal", tri_t, tei_t))

    if RAN is not None and args.experiment in ("e1", "e3"):
        _before = sum(len(a) + len(b) for _, a, b in SPLITS)
        SPLITS = [(nm, a[RAN[a]], b[RAN[b]]) for nm, a, b in SPLITS]
        _after = sum(len(a) + len(b) for _, a, b in SPLITS)
        print(f"Target mask (Ran): {_before - _after:,} never-ran rows removed from the "
              f"{args.experiment.upper()} population; they remain in the feature matrices "
              f"as queue context.", flush=True)
        for nm, a, b in SPLITS:
            print(f"  {nm:9s} train {len(a):>12,} | test {len(b):>12,}", flush=True)

    IMP = {}
    print(
        f"Splits prepared: Random ({len(rtr):,} train / {len(rte):,} test) | "
        f"Temporal ({len(tri_t):,} train / {len(tei_t):,} test)"
    )

    n_tot = len(failed)
    n_fail = int(np.sum(failed == 1))
    n_hw = int(np.sum(hw == 1))
    n_payload = n_fail - n_hw
    ftype = targets["fault_type"] if "fault_type" in targets else None

    if ftype is not None:
        n_succ = int(np.sum(ftype == -1))
        n_cancel = int(np.sum(ftype == -2))
        n_app = int(np.sum(ftype == 0))
        n_sub = int(np.sum(ftype == 2))
        n_completed = n_tot - n_cancel

        print(
            f"  ├─ Clean Successes       : {n_succ:,} ({n_succ/n_tot*100:.2f}%)\n",
            f"  ├─ Voluntary User Cancels: {n_cancel:,} ({n_cancel/n_tot*100:.2f}%)\n",
            f"  └─ Genuine Failures      : {n_fail:,} ({n_fail/n_tot*100:.2f}% of total)\n",
            f"-"*50,
            f"\nFailure rate (excl. user removal): {n_fail/(n_tot-n_cancel)*100:.2f}%\n",
            f"-"*50,
        )

    if args.experiment == "cascade":
        e1_lib, _, e3_lib = args.model.partition(":")
        e3_lib = e3_lib or e1_lib
        for split_name, _, _ in SPLITS:
            for sd in SEEDS:
                run_cascade(e1_lib, e3_lib, split_name, failed, hw, seed=sd)

    elif args.experiment == "e1":
        print(f"Running Experiment E1 (Job Failure Classification) [{args.model}]")
        CM = {}
        _got = {}
        for sd in SEEDS:
            _got.update(run_e1_model(
                args.model, Xmatch, failed, SPLITS, IMP, CM, NCAT_MATCH, order=QS, seed=sd
            ) or {})
        report_seed_variance(_got, ["roc_auc", "pr_auc", "f1", "mcc"],
                             label=f" [e1/{args.model}]")
    elif args.experiment == "e2":
        print(f"Running Experiment E2 (Wait Time Regression) [{args.model}]")
        # Floor first: a single constant already reaches within-2x ~ 0.25 on this
        # distribution, so the model numbers below are only interpretable next to it.
        print("Reference predictors (no features):")
        run_e2_reference(wait_log, SPLITS)
        _got = {}
        for sd in SEEDS:
            _got.update(run_e2_model(
                args.model, Xsub, wait_log, SPLITS, IMP, NCAT_SUB, order=QS, seed=sd
            ) or {})
        report_seed_variance(_got, ["r2_log", "within2x", "mae_log1p", "median_ae_s"],
                             label=f" [e2/{args.model}]")

    elif args.experiment == "e2dist":
        print(f"Running Experiment E2dist (Wait Time Intervals, {args.head} head) "
              f"[{args.model}]")
        print("Reference predictors (no features):")
        run_e2_reference(wait_log, SPLITS)
        # Never-ran jobs are right-censored, not missing: their wait is known only
        # to exceed the point at which they left the queue. Only the AFT head can
        # use them, and only if the RAN mask is available here.
        CENS = (~RAN) if (args.head == "aft" and "RAN" in dir() and RAN is not None) else None
        for sd in SEEDS:
            run_e2_dist(args.model, Xsub, wait_log, SPLITS, NCAT_SUB,
                        head=args.head, censored=CENS, order=QS, seed=sd)

    elif args.experiment == "e3":
        if ftype is not None:
            print("FAULT ATTRIBUTION (Genuine Failures Only):")
            print(f"  ├─ Payload Faults (App + Sub) : {n_payload:,} ({n_payload/n_fail*100:.1f}%)")
            print(f"  │    ├─ Application Faults    : {n_app:,} ({n_app/n_fail*100:.1f}%)")
            print(f"  │    └─ Submission Faults     : {n_sub:,} ({n_sub/n_fail*100:.1f}%)")
            print(f"  └─ Hardware Faults            : {n_hw:,} ({n_hw/n_fail*100:.1f}%)")
        else:
            print(f"Total Submitted Jobs  : {n_tot:,}")
            print(f"Genuine Failures      : {n_fail:,} ({n_fail/n_tot*100:.2f}%)")
            print(f"  ├─ Payload Faults   : {n_payload:,} ({n_payload/n_fail*100:.1f}%)")
            print(f"  └─ Hardware Faults  : {n_hw:,} ({n_hw/n_fail*100:.1f}%)")

        print(f"Running Experiment E3 (Fault Attribution: Hardware vs. Payload) [{args.model}]")
        CM = {}
        _got = {}
        for sd in SEEDS:
            _got.update(run_e3_model(
                args.model, Xmatch, failed, hw, SPLITS, IMP, CM, NCAT_MATCH, order=QS,
                seed=sd,
            ) or {})
        report_seed_variance(
            _got, ["hardware.pr_auc", "hardware.f1", "payload.pr_auc", "mcc"],
            label=f" [e3/{args.model}]")
