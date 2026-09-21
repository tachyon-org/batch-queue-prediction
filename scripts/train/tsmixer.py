import copy
import gc
import os
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from eval.helper import _empty_gpu, _get_slice
from eval.wandb_logger import log_epoch, log_summary
from eval.paths import model_path


class Block(nn.Module):
    """Core TSMixer Block with Time-Mixing and Feature-Mixing MLPs."""

    def __init__(
        self, sequence_length, channels, expansion_factor=2, dropout=0.1
    ):
        super().__init__()
        # Time-Mixing: Mixes information across the feature/time sequence axis
        self.norm1 = nn.LayerNorm([sequence_length, channels])
        self.time_mlp = nn.Sequential(
            nn.Linear(sequence_length, sequence_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Feature-Mixing: Mixes information across channels within each token
        self.norm2 = nn.LayerNorm([sequence_length, channels])
        self.feature_mlp = nn.Sequential(
            nn.Linear(channels, channels * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * expansion_factor, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x shape: (B, sequence_length, channels)
        # 1. Time-Mixing
        res = x
        x_norm = self.norm1(x)
        x_time = self.time_mlp(x_norm.transpose(1, 2)).transpose(1, 2)
        x = res + x_time

        # 2. Feature-Mixing
        res = x
        x_norm = self.norm2(x)
        x_feat = self.feature_mlp(x_norm)
        return x + x_feat


class TSMixer(nn.Module):
    """TSMixer Architecture for Tabular and Telemetry Logs."""

    def __init__(
        self,
        num_features,
        d_model=32,
        depth=3,
        expansion_factor=2,
        dropout=0.1,
    ):
        super().__init__()
        self.num_features = num_features
        # Feature tokenization: projects scalar feature values into d_model channels
        self.feature_proj = nn.Linear(1, d_model)

        self.blocks = nn.ModuleList(
            [
                Block(
                    sequence_length=num_features,
                    channels=d_model,
                    expansion_factor=expansion_factor,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.head_norm = nn.LayerNorm([num_features, d_model])
        self.head = nn.Linear(num_features * d_model, 1)

    def forward(self, x):
        B, N = x.shape
        x_emb = self.feature_proj(x.unsqueeze(-1))  # Shape: (B, N, d_model)

        for block in self.blocks:
            x_emb = block(x_emb)

        x_emb = self.head_norm(x_emb)
        out = self.head(x_emb.view(B, -1))
        return out.squeeze(-1)


def tsmixer_fit_eval(
    parts,
    tri,
    tei,
    ncat,
    kind,
    y,
    spw=None,
    trs=None,
    want_imp=False,
    split=None,
    is_regression=False,
    exp_tag=None,
    refit_idx=None,
):
    # Auto-detect regression task
    if kind in ["reg", "regression"]:
        is_regression = True

    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ya = np.asarray(y, dtype=np.float32)

    Xtr = _get_slice(parts, tri)
    X_te = _get_slice(parts, tei)
    X_trs = _get_slice(parts, trs) if trs is not None else None

    Xtr_np = np.ascontiguousarray(np.asarray(Xtr, dtype=np.float32))
    X_te_np = np.ascontiguousarray(np.asarray(X_te, dtype=np.float32))
    X_trs_np = (
        np.ascontiguousarray(np.asarray(X_trs, dtype=np.float32))
        if X_trs is not None
        else None
    )

    num_features = Xtr_np.shape[1]
    use_amp = DEV.type == "cuda"
    task_name = "Regression" if is_regression else "Classification"
    print(
        f"    [TSMixer - {task_name}] Training on {DEV} | amp={use_amp} | batch_size=16384 | split={split}",
        flush=True,
    )

    def _build():
        """Fresh model + optimizer. The refit trains from scratch rather than
        continuing the selection model: only the epoch count carries over."""
        m = TSMixer(
            num_features=num_features, d_model=32, depth=3, dropout=0.1
        ).to(DEV)
        return m, torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)

    model, optimizer = _build()

    # --- Loss function & task setup ---
    if is_regression:
        criterion = nn.MSELoss()
    else:
        pw = torch.tensor(float(spw), device=DEV) if spw is not None else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Parallel DataLoader streaming
    ds_tr = TensorDataset(torch.from_numpy(Xtr_np), torch.from_numpy(ya[tri]))
    loader_tr = DataLoader(
        ds_tr,
        batch_size=16384,
        shuffle=True,
        drop_last=False,
        pin_memory=use_amp,
        num_workers=4 if use_amp else 0,
        persistent_workers=True if use_amp else False,
    )

    eval_size = min(30000, len(X_te_np))
    # Validation for epoch selection must NOT be the test slice: the loop keeps the
    # best-scoring epoch, so validating on `tei` selects the model on the data it
    # is then reported against. `trs` is the harness's held-out slice.
    _val_X, _val_i = ((X_trs_np, np.asarray(trs)) if X_trs_np is not None
                      else (X_te_np, np.asarray(tei)))
    eval_size = min(eval_size, len(_val_X))
    eval_idx = np.random.choice(len(_val_X), size=eval_size, replace=False)
    X_eval_tensor = torch.from_numpy(_val_X[eval_idx]).to(
        DEV, non_blocking=True
    )
    y_eval = ya[_val_i][eval_idx]

    best_score = -float("inf")
    best_epoch = 0
    patience, patience_counter, best_weights = 5, 0, None

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for bx, by in loader_tr:
            bx = bx.to(DEV, non_blocking=True)
            by = by.to(DEV, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(bx)
                loss = criterion(out, by)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader_tr)

        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                val_raw = model(X_eval_tensor)

            if is_regression:
                val_preds = val_raw.float().cpu().numpy()
                val_score = r2_score(y_eval, val_preds)
                metric_label = "Val R2"
            else:
                val_preds = torch.sigmoid(val_raw.float()).cpu().numpy()
                val_score = roc_auc_score(y_eval, val_preds)
                metric_label = "Val AUC"

        print(
            f"    [TSMixer] Epoch {epoch+1:02d}/10 | Loss: {avg_loss:.4f} | {metric_label}: {val_score:.5f} (Best: {max(best_score, val_score):.5f})",
            flush=True,
        )
        log_epoch(epoch + 1,
                  {"train/loss": avg_loss, "val/score": val_score,
                   "val/best_score": max(best_score, val_score)},
                  phase=f"tsmixer[{kind}]")

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    if refit_idx is not None and best_epoch > 0:
        # Selection is done and its model is discarded. Retrain from scratch on the
        # FULL training window for exactly the epoch count the holdout chose, with no
        # validation and no early stopping, so the final fit sees every row the trees
        # see. See docs/validation-protocol.md.
        refit_idx = np.asarray(refit_idx)
        print(f"    [TSMixer] refit: {{len(refit_idx):,}} rows x {{best_epoch}} epoch(s) "
              f"(selected on {{len(_val_i):,}} holdout rows, best {{best_score:.5f}})",
              flush=True)
        # Rebound rather than deleted: the cleanup at the end of this function still
        # names them, and `del` here would make that a NameError on the refit path.
        ds_tr = loader_tr = Xtr_np = None
        gc.collect()

        X_rf = np.ascontiguousarray(
            np.asarray(_get_slice(parts, refit_idx), dtype=np.float32))
        loader_rf = DataLoader(
            TensorDataset(torch.from_numpy(X_rf), torch.from_numpy(ya[refit_idx])),
            batch_size=16384, shuffle=True, drop_last=False,
            pin_memory=use_amp, num_workers=0,
        )
        log_summary(selected_epoch=best_epoch,
                    selection_score=float(best_score),
                    holdout_rows=int(len(_val_i)),
                    final_fit_rows=int(len(refit_idx)))
        model, optimizer = _build()
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        for epoch in range(best_epoch):
            running_loss = 0.0
            model.train()
            for bx, by in loader_rf:
                bx = bx.to(DEV, non_blocking=True)
                by = by.to(DEV, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = criterion(model(bx), by)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
            avg = running_loss / len(loader_rf)
            print(f"    [TSMixer] refit epoch {{epoch+1:02d}}/{{best_epoch}} | "
                  f"Loss: {{avg:.4f}}", flush=True)
            log_epoch(epoch + 1, {{"refit/loss": avg}}, phase="tsmixer-e2-refit")
        del loader_rf, X_rf
        gc.collect()


    # Save trained model to disk
    save_path = model_path(exp_tag, "tsmixer", kind, split, ".pt")
    torch.save(model.state_dict(), save_path)
    print(f"    [TSMixer] Saved checkpoint to {save_path}", flush=True)

    # Evaluation predictions
    model.eval()
    with torch.no_grad():

        def _predict_in_chunks(X_data):
            if X_data is None:
                return None
            preds = []
            ds = TensorDataset(torch.from_numpy(X_data))
            loader = DataLoader(
                ds,
                batch_size=32768,
                pin_memory=use_amp,
                num_workers=2 if use_amp else 0,
            )
            for (bx,) in loader:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(bx.to(DEV, non_blocking=True))
                if is_regression:
                    preds.append(
                        out.float().cpu().numpy()
                    )  # Unconstrained linear predictions for regression
                else:
                    preds.append(
                        torch.sigmoid(out.float()).cpu().numpy()
                    )  # Probabilities for classification
            return np.concatenate(preds)

        p_te = _predict_in_chunks(X_te_np)
        p_trs = _predict_in_chunks(X_trs_np)

    imp = None
    del model, Xtr, X_te, X_trs, Xtr_np, X_te_np, X_trs_np, ds_tr, loader_tr
    _empty_gpu()
    gc.collect()

    return p_te, p_trs, imp