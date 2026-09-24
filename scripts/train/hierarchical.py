"""TROUT: hierarchical deep-learning wait-time estimator (Lovell et al., SC-W 2024).

Reimplemented from the paper, "A Hierarchical Deep Learning Approach for Predicting
Job Queue Times in HPC Systems" (DOI 10.1109/SCW63240.2024.00086). The authors' source
was not available, so every choice below is either quoted from the paper or listed in
UNSPECIFIED at the bottom of this docstring.

From the paper (Sec. III, Fig. 1, Algorithm 1):

  * "Two densely connected feed-forward neural networks were used to estimate job
    start time using these features." There is no attention mechanism; "hierarchical"
    names the two-stage structure, not the architecture.
  * Stage 1 is "a fully connected binary classification model with two hidden layers"
    predicting "whether jobs will start in ten minutes or less".
  * Stage 2 "predicts the queue time in minutes for jobs predicted to take more than
    ten minutes by the classification model", and is trained on the "Original dataset
    with only long queue times" (Fig. 1).
  * Algorithm 1: if the classifier says short, report the short class and stop;
    otherwise run the regressor.
  * Threshold is 10 minutes. 5 and 30 minutes were both tried; 5 min gave "over twice
    the mean absolute percentage error", 30 min was only marginally better and was
    rejected over user experience and classifier training data.
  * Class balance: SMOTE, "under-sampling the majority class ... and oversampling the
    minority class through artificial data creation".
  * "The regression model's architecture contains 33 input features and three hidden
    layers. The exponential linear unit (ELU) activation function was used for all
    layers except the output layer."
  * "The model utilized the smooth L1 loss function, a combination of mean absolute
    error and mean squared error."
  * "Both models made use of the Adam optimizer."
  * "a natural log transformation was applied to all features."
  * Batch normalization was tested and rejected. Min-max and box-cox scaling were
    tested and gave no benefit.

DEVIATIONS, all forced by this dataset rather than chosen:

  * SMOTE synthesises minority rows by interpolation, which is not tractable at 48.5M
    rows. Stage 1 is balanced by random under-sampling of the majority class instead.
    Note the majority class is inverted here: on Anvil 87% of jobs waited under 10
    minutes, on FIFE most jobs wait longer, so the class that gets cut is the long one.
  * The natural log transform is NOT applied: this pipeline delivers the numeric
    columns already standardized, which serves the same purpose. Applying it as well
    would transform z-scores rather than raw counts. See `_log_features`.
  * Algorithm 1 returns the string "Predicted to take less than 10 minutes" for the
    short class, i.e. no number. E2 scores a number for every job, so a short-class
    job is assigned the training-set median wait of that class -- the constant that
    minimises absolute error over it, and so the most favourable reading of a model
    that declines to predict there.

UNSPECIFIED in the paper, because Optuna chose them and the chosen values are not
reported ("The hyperparameters investigated include the learning rate, the number of
epochs ... the number of hidden layers for the model, the size of each layer, the size
of the dropout layers ... and which activation function to use"). Values here follow
the other neural trainers in this repo so the comparison stays like-for-like:

    classifier hidden   (256, 128)          regressor hidden  (256, 128, 64)
    dropout             0.1                 optimiser         Adam, lr 1e-3
    batch               16,384              epochs            5 (fixed budget)
"""

import gc
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from eval.helper import _get_slice, NEURAL_EPOCHS
from eval.paths import model_path

# 10 minutes, in the log1p(seconds) space the E2 target uses.
QUICK_THRESHOLD_S = 600.0
QUICK_THRESHOLD = float(np.log1p(QUICK_THRESHOLD_S))

CLF_HIDDEN = (256, 128)
REG_HIDDEN = (256, 128, 64)
DROPOUT = 0.1
LR = 1e-3
BATCH = 16384
# Follows the shared neural budget. Noted for the record: TROUT converges slower than
# the single-network models -- two networks from scratch, and its regressor sees only
# the long-wait subset. A synthetic sweep at the live schema shape gave test MAE 2.48
# at 5 epochs against 1.34 at 10, and the real 10-epoch run's regressor loss was still
# falling at the cap. Raise FIFE_TROUT_EPOCHS to trade wall clock for convergence.
MAX_EPOCHS = int(os.environ.get("FIFE_TROUT_EPOCHS", NEURAL_EPOCHS))
PATIENCE = 5


def _log_features(X, n_cat):
    """Paper Sec. III log-transforms every feature, to "manage the highly skewed
    nature of the data and reduce the input scale".

    That step is a NO-OP here, deliberately. Every numeric column of Xmatch/Xsub
    arrives z-scored already (SCALERS in schema_meta.json covers all 16 that were
    previously raw), which does the job the paper's transform is there to do.
    Applying log1p on top would squash standard scores rather than the skewed counts
    the paper targets -- a second scaling of data that has already been scaled.

    Kept as a named function, and still called at each site, so the deviation stays
    visible rather than silently disappearing. `n_cat` is unused for the same reason.
    """
    return np.nan_to_num(np.asarray(X, dtype=np.float32),
                         nan=0.0, posinf=0.0, neginf=0.0)


class _FeedForward(nn.Module):
    """Densely connected feed-forward stack. ELU on every layer but the output."""

    def __init__(self, n_in, hidden, n_out=1, dropout=DROPOUT):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train(model, X, y, loss_fn, dev, epochs, val=None, patience=PATIENCE, tag=""):
    """Adam training loop. With `val` (X, y, scorer) it early-stops on that score and
    returns the epoch count that produced the best one; otherwise it runs `epochs`."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=BATCH, shuffle=True, drop_last=False,
    )
    best_score, best_epoch, best_state, stale = np.inf, 0, None, 0
    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(xb)
        msg = f"    [TROUT{tag}] epoch {ep + 1}/{epochs} loss {total / len(X):.5f}"
        if val is not None:
            score = val[2](model, val[0], val[1], dev)
            msg += f" | holdout {score:.5f}"
            if score < best_score - 1e-6:
                best_score, best_epoch, stale = score, ep + 1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        print(msg, flush=True)
        if val is not None and stale >= patience:
            break
    # No best-weight restoration -- the validation slice is inside the training
    # window, so selecting an epoch on it would select on fitted rows.
    _ = best_state
    return best_epoch or epochs


@torch.no_grad()
def _infer(model, X, dev, sigmoid=False):
    model.eval()
    out = []
    for i in range(0, len(X), BATCH):
        xb = torch.from_numpy(X[i:i + BATCH]).to(dev)
        p = model(xb)
        out.append((torch.sigmoid(p) if sigmoid else p).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class TROUT:
    """Two-stage hierarchical estimator. Stage 1 gates, Stage 2 regresses."""

    def __init__(self, n_features, n_cat=0, seed=42, device=None):
        self.n_features = int(n_features)
        self.n_cat = int(n_cat)
        self.seed = int(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clf = None
        self.reg = None
        self.short_value = 0.0
        self.reg_epochs = 0

    # -- stage 1 -------------------------------------------------------------
    def _fit_classifier(self, Xl, y, dev, rng, max_epochs):
        is_long = (y > QUICK_THRESHOLD).astype(np.float32)
        idx_long = np.flatnonzero(is_long == 1.0)
        idx_short = np.flatnonzero(is_long == 0.0)
        if len(idx_long) == 0 or len(idx_short) == 0:
            raise ValueError("one wait class is empty; cannot train the gate")

        # Under-sample the majority to the minority's size (SMOTE stand-in).
        k = min(len(idx_long), len(idx_short))
        sel = np.concatenate([rng.choice(idx_long, k, replace=False),
                              rng.choice(idx_short, k, replace=False)])
        rng.shuffle(sel)
        print(f"    [TROUT] gate: {len(idx_short):,} short / {len(idx_long):,} long "
              f"-> balanced to {k:,} each", flush=True)

        self.clf = _FeedForward(self.n_features, CLF_HIDDEN).to(dev)
        _train(self.clf, Xl[sel], is_long[sel], nn.BCEWithLogitsLoss(), dev,
               max_epochs, tag="/gate")

    # -- stage 2 -------------------------------------------------------------
    def _fit_regressor(self, Xl, y, dev, max_epochs, val_Xl=None, val_y=None):
        long_i = np.flatnonzero(y > QUICK_THRESHOLD)
        print(f"    [TROUT] regressor: fitting on {len(long_i):,} long-wait jobs",
              flush=True)
        self.reg = _FeedForward(self.n_features, REG_HIDDEN).to(dev)

        val = None
        if val_Xl is not None and len(val_Xl):
            def _score(model, Xv, yv, d):
                # Selection is on the COMBINED pipeline, not the regressor alone:
                # the gate's mistakes are part of what the epoch count has to suit.
                return float(np.mean(np.abs(self._combine(Xv, d, reg=model) - yv)))
            val = (val_Xl, val_y, _score)

        self.reg_epochs = _train(self.reg, Xl[long_i], y[long_i], nn.SmoothL1Loss(),
                                 dev, max_epochs, val=val, tag="/reg")

    # -- algorithm 1 ---------------------------------------------------------
    def _combine(self, Xl, dev, reg=None):
        gate = _infer(self.clf, Xl, dev, sigmoid=True) >= 0.5
        out = np.full(len(Xl), self.short_value, dtype=np.float32)
        if gate.any():
            out[gate] = _infer(reg if reg is not None else self.reg, Xl[gate], dev)
        return out

    def fit(self, X, y, X_val=None, y_val=None, max_epochs=MAX_EPOCHS):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = torch.device(self.device)
        y = np.asarray(y, dtype=np.float32)
        Xl = _log_features(X, self.n_cat)

        short = y[y <= QUICK_THRESHOLD]
        self.short_value = float(np.median(short)) if len(short) else 0.0

        self._fit_classifier(Xl, y, dev, rng, max_epochs)
        val_Xl = _log_features(X_val, self.n_cat) if X_val is not None else None
        self._fit_regressor(Xl, y, dev, max_epochs, val_Xl,
                            np.asarray(y_val, dtype=np.float32) if y_val is not None else None)
        return self

    def predict(self, X):
        return self._combine(_log_features(X, self.n_cat), torch.device(self.device))

    def save(self, path):
        torch.save({"clf": self.clf.state_dict(), "reg": self.reg.state_dict(),
                    "n_features": self.n_features, "n_cat": self.n_cat,
                    "short_value": self.short_value, "reg_epochs": self.reg_epochs,
                    "clf_hidden": CLF_HIDDEN, "reg_hidden": REG_HIDDEN}, path)


def hierarchical_fit_eval(
    parts, tri, tei, ncat, kind, y, spw=None, trs=None, want_imp=False,
    split=None, exp_tag=None, seed=42, full_train_idx=None,
):
    """Harness entry point for TROUT wait-time regression."""
    ya = np.asarray(y, dtype=np.float32)
    tri = np.asarray(tri)
    # Fixed epoch budget: train once on the full window.
    if full_train_idx is not None:
        tri = np.asarray(full_train_idx)
    n_cat = ncat if isinstance(ncat, int) and ncat > 0 else 0

    Xtr = np.ascontiguousarray(np.asarray(_get_slice(parts, tri), dtype=np.float32))
    X_te = np.ascontiguousarray(np.asarray(_get_slice(parts, tei), dtype=np.float32))
    X_trs = (np.ascontiguousarray(np.asarray(_get_slice(parts, trs), dtype=np.float32))
             if trs is not None else None)

    print(f"    [TROUT - Wait Time Regression] {Xtr.shape[0]:,} rows, "
          f"{Xtr.shape[1]} features, n_cat={n_cat}, "
          f"threshold {QUICK_THRESHOLD_S:.0f}s", flush=True)

    model = TROUT(n_features=Xtr.shape[1], n_cat=n_cat, seed=seed)
    model.fit(Xtr, ya[tri], X_trs, ya[np.asarray(trs)] if trs is not None else None)


    save_path = model_path(exp_tag, "hierarchical", kind, split, ".pt")
    model.save(save_path)
    print(f"--> Saved TROUT model to {save_path}", flush=True)

    p_te = model.predict(X_te)
    p_trs = model.predict(
        np.ascontiguousarray(np.asarray(_get_slice(parts, trs), dtype=np.float32))
    ) if trs is not None else None

    # Attribution for TROUT comes from the post-hoc permutation path, like the other
    # neural models; there is no intrinsic importance to return here.
    return p_te, p_trs, None
