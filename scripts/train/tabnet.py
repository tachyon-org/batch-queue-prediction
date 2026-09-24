import os
import gc
import numpy as np
import torch
from eval.helper import _empty_gpu, _get_slice
from eval.helper import NEURAL_EPOCHS, NEURAL_CAT
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from eval.wandb_logger import log_epoch, log_summary
from pytorch_tabnet.callbacks import Callback
from eval.paths import model_path



def tabnet_fit_eval(
    parts, tri, tei, ncat, kind, y, spw=None, trs=None, want_imp=False, split=None,
    exp_tag=None, seed=42, full_train_idx=None
):
    split = split if split is not None else "default"
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    pin_mem = DEV != "cpu"
    ya = np.asarray(y)

    print(f"device={DEV} | split={split} | kind={kind}", flush=True)

    # Fixed epoch budget: train once on the full window. This MUST happen before the
    # slice below: slicing first pairs features from the 90% fit set with labels from
    # the full window, and pytorch_tabnet accepts mismatched lengths without raising,
    # so the model silently trains on misaligned rows and can only learn the mean.
    if full_train_idx is not None:
        tri = np.asarray(full_train_idx)

    Xtr = _get_slice(parts, tri)
    X_te = _get_slice(parts, tei)
    X_trs = _get_slice(parts, trs) if trs is not None else None

    Xtr_np = np.asarray(Xtr, dtype=np.float32)
    X_te_np = np.asarray(X_te, dtype=np.float32)
    X_trs_np = np.asarray(X_trs, dtype=np.float32) if X_trs is not None else None

    # Categorical embeddings are OFF by default, and that is a measured decision, not
    # an oversight. On 10.7M rows, embedding the code columns costs R2_log on BOTH
    # arms while reaching LOWER training loss -- the signature of memorising entity
    # identity that does not recur after the cutoff:
    #
    #     arm       no cat    low-cardinality only    all columns (1.8M)
    #     random    +0.764    +0.288                  +0.442
    #     temporal  +0.177    -11.647                 -0.029
    #
    # Restricting to low-cardinality columns (the rule the trees use) does not help;
    # it is worse. Every saved model in Pre-Trained-Models/neural-models/old_082026
    # also has cat_idxs=[], which is why those runs scored positive.
    # Set FIFE_NEURAL_CAT=1 to re-enable across all neural trainers.
    cat_idxs, cat_dims = [], []
    if NEURAL_CAT and isinstance(ncat, int) and ncat > 0:
        cat_idxs = list(range(ncat))
        # Size the embedding tables from TRAINING rows only. Taking the max over the
        # test slice too would let the test period set the table width -- a small leak,
        # but the same class as the one this protocol exists to remove. Codes run
        # 0..max_train, and index max_train+1 is a dedicated UNK slot that every level
        # first seen after the cutoff maps to.
        _tr_max = [int(Xtr_np[:, i].max()) for i in cat_idxs]
        cat_dims = [m + 2 for m in _tr_max]

        def _clip_unseen(A):
            """Map post-cutoff levels onto the trained UNK index."""
            if A is None:
                return None
            A = A.copy()
            for i, m in zip(cat_idxs, _tr_max):
                np.clip(A[:, i], 0, m + 1, out=A[:, i])
                A[A[:, i] > m, i] = m + 1
            return A

        _n_unseen = sum(int((X_te_np[:, i] > m).sum()) for i, m in zip(cat_idxs, _tr_max))
        if _n_unseen:
            print(f"    [TabNet] {_n_unseen:,} test cells hold a level unseen in "
                  f"training; mapped to UNK", flush=True)
        X_te_np = _clip_unseen(X_te_np)
        X_trs_np = _clip_unseen(X_trs_np)

    eval_size = min(50000, len(X_te_np))
    # eval_set drives TabNet's own early stopping (patience=3), so it must not be the
    # test slice -- that would select the model on the data it is reported against.
    if X_trs_np is None:
        # Falling back to the test slice here would select the epoch on the data the
        # model is then reported against -- the exact leak this protocol exists to
        # prevent. Fail loudly instead of doing it silently.
        raise ValueError(
            "no validation holdout: pass trs=. Validating on the test slice would "
            "select the model on the data it is scored against.")
    _val_X, _val_i = X_trs_np, np.asarray(trs)
    eval_size = min(eval_size, len(_val_X))
    eval_idx = np.random.choice(len(_val_X), size=eval_size, replace=False)
    X_eval_sub = _val_X[eval_idx]
    y_eval_sub = ya[_val_i][eval_idx]

    y_tr = ya[tri]
    assert len(Xtr_np) == len(y_tr), (
        f"feature/label mismatch: {len(Xtr_np):,} rows of X against {len(y_tr):,} "
        f"labels. pytorch_tabnet does not check this and will train on misaligned rows.")

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
        # lr 0.02 with the scheduler stepping every 5 epochs
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 5, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="sparsemax",
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
                      phase=None)

    clf.fit(
        X_train=Xtr_np,
        y_train=y_tr,
        eval_set=[(X_eval_sub, y_eval_sub)],
        eval_name=["val"],
        eval_metric=eval_metric,
        callbacks=[_WandbEpoch()],
        max_epochs=NEURAL_EPOCHS,
        # patience=0 skips pytorch_tabnet's EarlyStopping callback, which is what
        # restores best weights. That is deliberate: the eval slice sits inside the
        # training window (full_train_idx = tri_all), so restoring the best epoch
        # would select on rows the model fitted. Fixed budget => final epoch.
        patience=0,
        batch_size=16384,
        virtual_batch_size=2048,
        num_workers=0,
        pin_memory=pin_mem,
        drop_last=False,
        compute_importance=False,
    )


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