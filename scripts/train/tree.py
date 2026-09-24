import os

import numpy as np
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import torch
import gc
from eval.helper import _empty_gpu

# config
_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if _GPU else "cpu")
XGB_DEV = "cuda" if _GPU else "cpu"
CB_TASK = "GPU" if _GPU else "CPU"

# CatBoost's GPU allocator aborts the PROCESS on OOM (SIGABRT, exit 134) rather than
# raising, so the CPU fallback in _sk_fit can never catch it -- the interpreter is
# already gone. On a shared GPU the only workable guard is to look before leaping.
CB_MIN_FREE_GB = float(os.environ.get("FIFE_CB_MIN_FREE_GB", "14"))

# CatBoost builds categorical target statistics ("tree-ctrs") for combinations of up
# to `max_ctr_complexity` categorical columns, default 4. With 15 categoricals in
# Xmatch that is a combinatorial blow-up: it asked for 4.2 GB of ctr memory and died
# on a shared GPU. Capping at 2 keeps pairs and drops the 3- and 4-way combinations,
# which on this data are near row-identifiers (Owner x CampaignName x MatchEntry) and
# are exactly what the temporal split is meant to punish. Lower to 1 if it still OOMs.
CB_CTR_COMPLEXITY = int(os.environ.get("FIFE_CB_CTR_COMPLEXITY", "2"))

# Columns with at most this many levels are one-hot encoded instead of getting a
# target-statistic counter. Measured on the E1 matrix (38M rows, 15 categoricals):
# without it CatBoost GPU died on tree-ctr memory at every complexity setting; with
# it the same fit completes in 154 s. This, not the complexity cap, is what makes
# CatBoost run at this scale.
CB_ONE_HOT = int(os.environ.get("FIFE_CB_ONE_HOT", "255"))

# Upper bound on the fraction of the card CatBoost may claim. The old 0.5 ceiling was
# the binding constraint even when 22 GB were free; the actual share is still derived
# from free memory below, so a busy card still backs it off.
CB_MAX_PART = float(os.environ.get("FIFE_CB_MAX_PART", "0.9"))

# When set (the default), a fit that cannot get a GPU raises instead of quietly
# dropping to CPU. Set FIFE_REQUIRE_GPU=0 to allow the fallback.
REQUIRE_GPU = os.environ.get("FIFE_REQUIRE_GPU", "1") not in ("0", "false", "False")


def _cb_device():
    """(task_type, devices, gpu_ram_part) for CatBoost, chosen from free memory now.

    `gpu_ram_part` is sized against what is actually free rather than a fixed fraction
    of the card, so a partly occupied GPU is still usable instead of aborting mid-fit.
    Raises when no device has headroom and REQUIRE_GPU is set, rather than returning
    CPU: a run that silently takes the slow path is worse than one that stops.
    """
    if not _GPU:
        if REQUIRE_GPU:
            raise RuntimeError(
                "CatBoost requires a GPU but torch reports no CUDA device. "
                "Set FIFE_REQUIRE_GPU=0 to allow CPU training.")
        return "CPU", None, None
    best, best_free, total = None, 0, 0
    for d in range(torch.cuda.device_count()):
        try:
            free, tot = torch.cuda.mem_get_info(d)
        except Exception:
            continue
        if free > best_free:
            best, best_free, total = d, free, tot
    if best is None or best_free < CB_MIN_FREE_GB * 1e9:
        msg = (f"CatBoost needs {CB_MIN_FREE_GB:.0f} GB of free GPU memory but the "
               f"emptiest device has {best_free / 1e9:.1f} GB. Wait for the GPU to "
               f"free up, lower FIFE_CB_MIN_FREE_GB, or set FIFE_REQUIRE_GPU=0.")
        if REQUIRE_GPU:
            raise RuntimeError(msg)
        print(f"  [catboost] {msg} Falling back to CPU.", flush=True)
        return "CPU", None, None
    part = round(min(CB_MAX_PART, (best_free * 0.8) / total), 3)
    print(f"  [catboost] GPU {best}: {best_free / 1e9:.1f} GB free, "
          f"gpu_ram_part={part}", flush=True)
    return "GPU", str(best), part
# LightGBM's wheel is built without CUDA (-DUSE_CUDA=1); only the OpenCL "gpu" device
# is available, and it is SLOWER than CPU here -- measured 1.05 s against 0.41 s on
# 200k x 42 with 15 categorical columns, because OpenCL only accelerates histogram
# construction and loses to the threaded CPU build at this feature count. CPU is
# therefore the default on merit, not by oversight. Set FIFE_LGBM_DEV=gpu to override.
LGBM_DEV = os.environ.get("FIFE_LGBM_DEV", "cpu")

# --- categorical handling -------------------------------------------------
# The leading `ncat` columns of every matrix are integer category codes (Owner,
# CampaignName, MatchSite ...). Treated as numbers they impose an ordering that does
# not exist; each library has a native mechanism for them instead.

# A column gets native categorical treatment only if it has few enough distinct
# levels in TRAINING. The high-cardinality code columns are identities -- users,
# campaigns, campaign stages -- and partition-based splits let a model fit a
# per-identity effect that does not survive the cutoff: 17.7% of July test rows carry
# a CampaignStageId never seen in training. Measured on the temporal arm, treating
# every code column as categorical drops r2_log from 0.270 to 0.125 at 7.1M training
# rows, and the gap widens with data. Restricting to the low-cardinality columns
# keeps the in-distribution gain (random arm 0.8383 -> 0.8406) without it.
CAT_MAX_LEVELS = int(os.environ.get("FIFE_CAT_MAX_LEVELS", "50"))

# Rows sampled when counting levels. Cardinality does not need all 48M rows.
_CAT_SAMPLE_STEP = 40


def _cat_idx(ncat, X=None, max_levels=None, step=None):
    """Indices of the leading `ncat` code columns to treat as categorical.

    Without `X` every code column qualifies, which is the caller saying it has
    already made the selection. With `X` the levels are counted on the training rows
    only -- never the test set -- so the choice carries no information about the
    period being predicted.

    `step` subsamples those rows. Pass 1 when `X` has already been thinned by the
    caller: undercounting levels admits a high-cardinality column, which is the
    failure mode this filter exists to prevent, so the two thinnings must not compose
    silently.
    """
    n = int(ncat) if isinstance(ncat, (int, np.integer)) else 0
    if n <= 0:
        return []
    if X is None:
        return list(range(n))
    lim = CAT_MAX_LEVELS if max_levels is None else int(max_levels)
    st = _CAT_SAMPLE_STEP if step is None else int(step)
    Xs = np.asarray(X[::st, :n])
    return [i for i in range(n) if len(np.unique(Xs[:, i])) <= lim]


def _cb_frame(X, ncat, cat_idx=None):
    """CatBoost input with integer code columns.

    Passing `cat_features` alongside a float array raises outright ("no categorical
    features, but 'cat_features' parameter specifies ..."), so the codes have to be a
    genuine integer dtype. Returns None if the conversion is not worth attempting, in
    which case the caller falls back to numeric treatment.
    """
    idx = cat_idx if cat_idx is not None else _cat_idx(ncat)
    if not idx:
        return None
    try:
        import pandas as pd
        X = np.asarray(X)
        cols = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols, copy=False)
        for i in idx:
            df[cols[i]] = df[cols[i]].astype(np.int32)
        return df
    except (MemoryError, ImportError, ValueError) as e:
        print(f"  [catboost] categorical conversion skipped ({type(e).__name__}: {e}); "
              f"codes will be treated as numeric", flush=True)
        return None


def sk_gain(m, nfeat):
    if "lightgbm" in type(m).__module__:
        v = np.asarray(m.booster_.feature_importance("gain"), float)
    elif "xgboost" in type(m).__module__:
        v = np.asarray(m.feature_importances_, float)
    else:                                              # CatBoost
        v = np.asarray(m.get_feature_importance(), float)
    if len(v) < nfeat:
        v = np.concatenate([v, np.zeros(nfeat - len(v))])
    s = v.sum(); return v / s if s > 0 else v

def _xgb_cls(spw, seed=42, ncat=0, nfeat=None, X=None, cat_idx=None):
    """Classifier counterpart of `_xgb_reg`, including its categorical treatment.

    Without `cat_idx` the code columns are read as ordered integers, which is a
    different model class from the partition splits E2 gets -- so E1/E3 would be
    comparing a weaker tree against the same neural models.
    """
    kw = dict(
        tree_method="hist", device=XGB_DEV, n_estimators=200, max_depth=8,
        learning_rate=0.1, scale_pos_weight=spw, eval_metric="logloss",
        random_state=seed,
    )
    idx = cat_idx if cat_idx is not None else _cat_idx(ncat, X)
    ft = (['c' if i in set(idx) else 'q' for i in range(int(nfeat))]
          if (nfeat is not None and idx) else None)
    if ft is not None:
        kw.update(enable_categorical=True, feature_types=ft, max_cat_to_onehot=1)
    return xgb.XGBClassifier(**kw)

# Row/column subsampling for the regressors, matching what _sk_cls already does.
# Two reasons: at 48M rows a tree that sees every row costs far more for little gain
# (LightGBM took 1,061 s per split against XGBoost's 36 s on GPU), and with no
# sampling at all `hist` boosting is deterministic -- `random_state` has nothing to
# act on, so a multi-seed run returns identical fits and a spread of exactly 0.
_SUBSAMPLE = 0.1        # 10% of rows per tree (~4.8M at this scale)
_COLSAMPLE = 0.8


def _xgb_reg(seed=42, ncat=0, nfeat=None, X=None, cat_idx=None):
    kw = dict(
        tree_method="hist", device=XGB_DEV, n_estimators=200, max_depth=8,
        learning_rate=0.1, subsample=_SUBSAMPLE, colsample_bytree=_COLSAMPLE,
        random_state=seed,
    )
    idx = cat_idx if cat_idx is not None else _cat_idx(ncat, X)
    ft = (['c' if i in set(idx) else 'q' for i in range(int(nfeat))]
          if (nfeat is not None and idx) else None)
    if ft is not None:
        # max_cat_to_onehot=1 forces partition-based splits for every code column
        # rather than one-hot for the low-cardinality ones, so all of them are
        # handled the same way.
        kw.update(enable_categorical=True, feature_types=ft, max_cat_to_onehot=1)
    return xgb.XGBRegressor(**kw)

def _xgb_prep(arr):
    """Contiguous float32 NumPy on the CPU, which is what the XGBoost GPU engine wants."""
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    return np.ascontiguousarray(arr, dtype=np.float32)


def _sk_cls(lib, spw, seed=42, ncat=0, X=None, cat_idx=None):
    if lib == "lightgbm":
        import os
        # Leave headroom. LightGBM is CPU-bound and will take every core it is given;
        # at n_jobs=all it starved a concurrent GPU trainer of the host cores it needs
        # for batching (an MLP seed ran 1.84x slower alongside it). FIFE_LGBM_JOBS
        # overrides; the default keeps ~25% of the machine free for whatever else runs.
        _all = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 4
        avail_cores = int(os.environ.get("FIFE_LGBM_JOBS", max(1, int(_all * 0.75))))
        
        return LGBMClassifier(
            n_estimators=100,
            num_leaves=31,          # Standard leaf limit (fast & effective)
            max_depth=-1,           # Eliminates depth truncation & "best gain: -inf" warnings
            learning_rate=0.1,
            n_jobs=avail_cores,
            subsample=0.1,          # Bagging: samples 10% (4.8M rows) per tree for high speed
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=1000, # Stops micro-splits on 48M dataset
            scale_pos_weight=spw,
            # subsample=0.1 above makes this fit genuinely stochastic, so the seed
            # is load-bearing here rather than cosmetic.
            random_state=seed,
            verbose=-1,
            verbosity=-1            # Explicitly quets C++ core warnings
        )
    elif lib == "catboost":
        _task, _dev, _part = _cb_device()
        kw = dict(
            task_type=_task,
            iterations=200,
            depth=8,
            learning_rate=0.1,
            verbose=False,
            allow_writing_files=False,
            scale_pos_weight=spw,
            random_seed=seed
        )
        if _task == "GPU":
            kw["devices"] = _dev
            kw["gpu_ram_part"] = _part
        _ci = cat_idx if cat_idx is not None else _cat_idx(ncat, X)
        if _ci:
            kw["cat_features"] = _ci
            kw["max_ctr_complexity"] = CB_CTR_COMPLEXITY
            kw["one_hot_max_size"] = CB_ONE_HOT
        return CatBoostClassifier(**kw)

def _sk_reg(lib, seed=42, ncat=0, X=None, cat_idx=None):
    if lib == "lightgbm":
        # subsample_freq must be >= 1 or LightGBM ignores subsample entirely.
        return LGBMRegressor(device=LGBM_DEV, n_estimators=200, num_leaves=255, max_depth=8,
                             learning_rate=0.1, subsample=_SUBSAMPLE, subsample_freq=1,
                             colsample_bytree=_COLSAMPLE, n_jobs=-1, verbose=-1,
                             random_state=seed)
    elif lib == "catboost":
        # CatBoost's default Bayesian bootstrap ignores `subsample`; Bernoulli takes it.
        _task, _dev, _part = _cb_device()
        kw = dict(task_type=_task, iterations=200, depth=8, learning_rate=0.1,
              bootstrap_type="Bernoulli", subsample=_SUBSAMPLE, rsm=_COLSAMPLE,
              verbose=False, allow_writing_files=False, random_seed=seed)
        if _task == "GPU":
            kw["devices"] = _dev
            kw["gpu_ram_part"] = _part
            kw.pop("rsm", None)          # rsm is unsupported on CatBoost GPU
        _ci = cat_idx if cat_idx is not None else _cat_idx(ncat, X)
        if _ci:
            kw["cat_features"] = _ci
            kw["max_ctr_complexity"] = CB_CTR_COMPLEXITY
            kw["one_hot_max_size"] = CB_ONE_HOT
        return CatBoostRegressor(**kw)

def _sk_predict(m, X, ncat=0, cat_idx=None):
    """Predict, rebuilding the integer-column frame CatBoost needs when it was fitted
    with cat_features. XGBoost and LightGBM take the raw array."""
    if "catboost" in type(m).__module__ and m.get_params().get("cat_features"):
        frame = _cb_frame(X, ncat, cat_idx)
        if frame is not None:
            return m.predict(frame)
    return m.predict(X)


def _sk_fit(m, Xtr, ytr, sample_weight=None, ncat=0, cat_idx=None):
    """Fit, passing whatever categorical argument the library wants at fit time.

    LightGBM takes `categorical_feature` on fit(); CatBoost takes `cat_features` on
    the estimator but needs integer columns in the data, so the frame is rebuilt.
    XGBoost is configured entirely at construction.
    """
    idx = cat_idx if cat_idx is not None else _cat_idx(ncat, Xtr)
    is_cb = "catboost" in type(m).__module__
    is_lgb = "lightgbm" in type(m).__module__

    if is_cb and idx:
        frame = _cb_frame(Xtr, ncat, idx)
        if frame is None:                      # conversion declined; drop cat handling
            m.set_params(cat_features=None)
        else:
            Xtr = frame

    def _do(mm):
        kw = {}
        if is_lgb and idx:
            kw["categorical_feature"] = idx
        if sample_weight is not None:
            kw["sample_weight"] = sample_weight
        return mm.fit(Xtr, ytr, **kw)
    try:
        _do(m); return m
    except Exception as e:
        if (getattr(m, "get_params", lambda: {})().get("task_type") == "GPU"
                and not REQUIRE_GPU):
            print(f"  [fit failed: {e}]", flush=True)
            _empty_gpu(); gc.collect()
            p = dict(m.get_params()); p["task_type"] = "CPU"; p.pop("devices", None); p.pop("gpu_ram_part", None)
            m2 = type(m)(**p); _do(m2); return m2
        raise