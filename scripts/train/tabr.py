import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from eval.helper import _empty_gpu, _get_slice
from eval.wandb_logger import log_epoch


class TabR(nn.Module):
    """TabR: Retrieval-Augmented Tabular Deep Learning Model."""

    def __init__(self, in_dim, d_main=128, d_retrieval=64, split=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, d_main),
            nn.BatchNorm1d(d_main),
            nn.ReLU(),
        )

        self.key_proj = nn.Linear(d_main, d_retrieval)
        self.query_proj = nn.Linear(d_main, d_retrieval)

        self.head = nn.Sequential(
            nn.Linear(d_main * 2, d_main),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_main, 1),
        )

    def forward(self, x, context_x=None, context_y=None):
        B = x.size(0)
        h = self.encoder(x)

        # Self-retrieval / batch context fallback if context pool isn't passed separately
        if context_x is None:
            context_h = h
        else:
            context_h = self.encoder(context_x)

        queries = self.query_proj(h)
        keys = self.key_proj(context_h)

        # Dot-product retrieval attention
        scores = torch.matmul(queries, keys.T) / (keys.shape[-1] ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        retrieved_context = torch.matmul(attn, context_h)

        combined = torch.cat([h, retrieved_context], dim=-1)
        return self.head(combined).squeeze(-1)


CONTEXT_SIZE = 4096   # rows sampled from train each epoch as the retrieval bank


def tabr_fit_eval(
    parts, tri, tei, ncat, kind, y, spw=None, trs=None, want_imp=False, split=None,
    is_regression=False, exp_tag=None,
):
    # `is_regression` and `exp_tag` are passed by the harness to every trainer; their
    # absence here raised TypeError and made TabR fail on every split and seed.
    if kind in ("reg", "regression"):
        is_regression = True
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = DEV.type == "cuda"
    ya = np.asarray(y, dtype=np.float32)

    Xtr = _get_slice(parts, tri)
    X_te = _get_slice(parts, tei)
    X_trs = _get_slice(parts, trs) if trs is not None else None

    Xtr_np = np.ascontiguousarray(np.asarray(Xtr, dtype=np.float32))
    X_te_np = np.ascontiguousarray(np.asarray(X_te, dtype=np.float32))
    X_trs_np = np.ascontiguousarray(np.asarray(X_trs, dtype=np.float32)) if X_trs is not None else None

    in_dim = Xtr_np.shape[1]
    ctx_size = min(CONTEXT_SIZE, len(Xtr_np))
    print(f"    [TabR] Training on {DEV} | amp={use_amp} | batch=16384 | context={ctx_size} | split={split}", flush=True)

    pw = torch.tensor(float(spw), device=DEV) if spw is not None else None
    model = TabR(in_dim=in_dim, d_main=128, d_retrieval=64, split=split).to(DEV)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    rng = np.random.default_rng(42)
    ds_tr = TensorDataset(torch.tensor(Xtr_np), torch.tensor(ya[tri]))
    loader_tr = DataLoader(ds_tr, batch_size=16384, shuffle=True, drop_last=False,
                           pin_memory=use_amp, num_workers=0)

    eval_size = min(30000, len(X_te_np))
    # Validation for epoch selection must NOT be the test slice: the loop keeps the
    # best-scoring epoch, so validating on `tei` selects the model on the data it
    # is then reported against. `trs` is the harness's held-out slice.
    _val_X, _val_i = ((X_trs_np, np.asarray(trs)) if X_trs_np is not None
                      else (X_te_np, np.asarray(tei)))
    eval_size = min(eval_size, len(_val_X))
    eval_idx = rng.choice(len(_val_X), size=eval_size, replace=False)
    X_eval_tensor = torch.tensor(_val_X[eval_idx]).to(DEV)
    y_eval = ya[_val_i][eval_idx]

    best_auc, patience, patience_counter, best_weights = 0.0, 5, 0, None

    for epoch in range(15):
        running_loss = 0.0
        # Resample the context bank each epoch so retrieval sees varied neighbours.
        ctx_idx = rng.choice(len(Xtr_np), size=ctx_size, replace=False)
        ctx_bank = torch.tensor(Xtr_np[ctx_idx]).to(DEV)  # [CONTEXT_SIZE, in_dim]

        model.train()
        for bx, by in loader_tr:
            bx = bx.to(DEV, non_blocking=True)
            by = by.to(DEV, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Pass fixed context bank → attention is [B, CONTEXT_SIZE], not [B, B]
                out = model(bx, context_x=ctx_bank)
                loss = criterion(out, by)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader_tr)

        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                val_logits = model(X_eval_tensor, context_x=ctx_bank)
            val_preds = torch.sigmoid(val_logits.float()).cpu().numpy()
            val_auc = roc_auc_score(y_eval, val_preds)

            val_preds = torch.sigmoid(val_logits.float()).cpu().numpy()
            val_auc = roc_auc_score(y_eval, val_preds)

        print(
            f"    [TabR] Epoch {epoch+1:02d}/15 | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.5f} (Best: {max(best_auc, val_auc):.5f})",
            flush=True,
        )
        log_epoch(epoch + 1,
                  {"train/loss": avg_loss, "val/auc": val_auc,
                   "val/best_auc": max(best_auc, val_auc)},
                  phase=f"tabr[{kind}]")
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Use the final context bank for inference (fixed seed for reproducibility)
    infer_ctx_idx = rng.choice(len(Xtr_np), size=ctx_size, replace=False)
    infer_ctx = torch.tensor(Xtr_np[infer_ctx_idx]).to(DEV)

    model.eval()
    with torch.no_grad():
        def _predict_in_chunks(X_data):
            if X_data is None:
                return None
            preds = []
            loader = DataLoader(TensorDataset(torch.tensor(X_data)), batch_size=32768,
                                pin_memory=use_amp, num_workers=0)
            for (bx,) in loader:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(bx.to(DEV, non_blocking=True), context_x=infer_ctx)
                preds.append(torch.sigmoid(logits.float()).cpu().numpy())
            return np.concatenate(preds)

        p_te = _predict_in_chunks(X_te_np)
        p_trs = _predict_in_chunks(X_trs_np)

    imp = None
    del model, Xtr, X_te, X_trs, Xtr_np, X_te_np, X_trs_np, ds_tr, loader_tr, ctx_bank, infer_ctx
    _empty_gpu()
    gc.collect()

    return p_te, p_trs, imp