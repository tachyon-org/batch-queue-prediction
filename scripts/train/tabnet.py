import os
import gc
import numpy as np
import torch
from eval.helper import _empty_gpu, _get_slice
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from eval.wandb_logger import log_epoch, log_summary
from pytorch_tabnet.callbacks import Callback
from eval.paths import model_path



def tabnet_fit_eval(
    parts, tri, tei, ncat, kind, y, spw=None, trs=None, want_imp=False, split=None,
    exp_tag=None, seed=42, refit_idx=None
):
    split = split if split is not None else "default"
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    pin_mem = DEV != "cpu"
    ya = np.asarray(y)

    print(f"device={DEV} | split={split} | kind={kind}", flush=True)

    Xtr = _get_slice(parts, tri)
    X_te = _get_slice(parts, tei)
    X_trs = _get_slice(parts, trs) if trs is not None else None

    Xtr_np = np.asarray(Xtr, dtype=np.float32)
    X_te_np = np.asarray(X_te, dtype=np.float32)
    X_trs_np = np.asarray(X_trs, dtype=np.float32) if X_trs is not None else None

    cat_idxs, cat_dims = [], []
    if isinstance(ncat, int) and ncat > 0:
        cat_idxs = list(range(ncat))
        cat_dims = [
            int(max(Xtr_np[:, i].max(), X_te_np[:, i].max()) + 1)
            for i in cat_idxs
        ]

    eval_size = min(50000, len(X_te_np))
    # eval_set drives TabNet's own early stopping (patience=3), so it must not be the
    # test slice -- that would select the model on the data it is reported against.
    _val_X, _val_i = ((X_trs_np, np.asarray(trs)) if X_trs_np is not None
                      else (X_te_np, np.asarray(tei)))
    eval_size = min(eval_size, len(_val_X))
    eval_idx = np.random.choice(len(_val_X), size=eval_size, replace=False)
    X_eval_sub = _val_X[eval_idx]
    y_eval_sub = ya[_val_i][eval_idx]

    y_tr = ya[tri]

    # TabNetRegressor requires 2D targets shape (N, 1)
    if kind == "reg":
        y_tr = y_tr.reshape(-1, 1)
        y_eval_sub = y_eval_sub.reshape(-1, 1)

    tabnet_params = dict(
        device_name=DEV,
        n_d=8,
        n_a=8,
        n_steps=3,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 5, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="sparsemax",
        # pytorch_tabnet's TabModel declares `seed: int = 0` and calls
        # torch.manual_seed(self.seed) in its own constructor, AFTER the harness has
        # called set_global_seed(). Leaving this out meant every seed trained from 0
        # and the three runs came back bit-identical, so the spread was exactly 0.
        seed=seed,
        verbose=1,
    )

    if kind == "reg":
        clf = TabNetRegressor(**tabnet_params)
        eval_metric = ["mse"]
    else:
        clf = TabNetClassifier(**tabnet_params)
        eval_metric = ["auc"]

    # TabNet owns its training loop, so the per-epoch W&B curve comes from a
    # callback rather than an inline hook. pytorch_tabnet hands each callback the
    # epoch's logs dict, which already carries the train loss and every eval_metric.
    class _WandbEpoch(Callback):
        def on_epoch_end(self, epoch, logs=None):
            log_epoch(epoch + 1,
                      {f"train/{k}" if k == "loss" else f"val/{k}": v
                       for k, v in (logs or {}).items()
                       if isinstance(v, (int, float))},
                      phase=f"tabnet[{kind}]")

    clf.fit(
        X_train=Xtr_np,
        y_train=y_tr,
        eval_set=[(X_eval_sub, y_eval_sub)],
        eval_name=["test"],
        eval_metric=eval_metric,
        callbacks=[_WandbEpoch()],
        max_epochs=10,
        patience=3,
        batch_size=16384,
        virtual_batch_size=2048,
        num_workers=0,
        pin_memory=pin_mem,
        drop_last=False,
        compute_importance=False,
    )

    if refit_idx is not None:
        # pytorch_tabnet drives its own early stopping from `eval_set`, so the epoch
        # count is selected there. Refit from scratch on the FULL training window for
        # exactly that many epochs with NO eval_set, so nothing can stop it early and
        # the final fit sees every row the trees see. See docs/validation-protocol.md.
        n_ep = int(getattr(clf, "best_epoch", 0) or 0) + 1
        refit_idx = np.asarray(refit_idx)
        print(f"    [TabNet] refit: {len(refit_idx):,} rows x {n_ep} epoch(s) "
              f"(best_epoch={getattr(clf, 'best_epoch', None)} on the holdout)",
              flush=True)
        log_summary(selected_epoch=n_ep,
                    final_fit_rows=int(len(refit_idx)))
        X_rf = np.asarray(_get_slice(parts, refit_idx), dtype=np.float32)
        y_rf = ya[refit_idx]
        if kind == "reg":
            y_rf = y_rf.reshape(-1, 1)
        clf = (TabNetRegressor(**tabnet_params) if kind == "reg"
               else TabNetClassifier(**tabnet_params))
        clf.fit(
            X_train=X_rf,
            y_train=y_rf,
            max_epochs=n_ep,
            patience=0,                 # 0 disables early stopping in pytorch_tabnet
            batch_size=16384,
            virtual_batch_size=2048,
            num_workers=0,
            pin_memory=pin_mem,
            drop_last=False,
            compute_importance=False,
        )
        del X_rf, y_rf
        gc.collect()

    save_path = model_path(exp_tag, "tabnet", kind, split)
    clf.save_model(save_path)
    print(f"--> Generating predictions for test splits... | split={split}", flush=True)

    if kind == "reg":
        p_te = clf.predict(X_te_np).reshape(-1)
        p_trs = (
            clf.predict(X_trs_np).reshape(-1) if X_trs_np is not None else None
        )
    else:
        p_te = clf.predict_proba(X_te_np)[:, 1]
        p_trs = (
            clf.predict_proba(X_trs_np)[:, 1] if X_trs_np is not None else None
        )

    imp = getattr(clf, "feature_importances_", None) if want_imp else None

    del clf, Xtr, X_te, X_trs, Xtr_np, X_te_np, X_trs_np, X_eval_sub
    _empty_gpu()
    gc.collect()

    return p_te, p_trs, imp