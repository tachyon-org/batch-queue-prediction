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



class VectorizedTokenizer(nn.Module):
    """Tokenizes all numerical features in parallel using a single matrix operation."""

    def __init__(self, num_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_token) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.01)

    def forward(self, x):
        # x shape: (B, F) -> Output shape: (B, F + 1, d_token)
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat([cls_tokens, tokens], dim=1)


class FastTransformerBlock(nn.Module):
    """Transformer Encoder using FlashAttention / Memory-Efficient SDPA kernels."""

    def __init__(self, d_token, heads, ffn_factor=2, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = d_token // heads
        self.qkv = nn.Linear(d_token, d_token * 3)
        self.out = nn.Linear(d_token, d_token)
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * ffn_factor),
            nn.GELU(),
            nn.Linear(d_token * ffn_factor, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, N, D = x.shape
        x_norm = self.norm1(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)

        # Multi-head reshape
        q = q.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, D)

        x = x + self.out(attn)
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    """Streamlined FT-Transformer tuned for large tabular data."""

    def __init__(self, num_features=0, d_token=32, depth=2, heads=4):
        super().__init__()
        self.tokenizer = VectorizedTokenizer(num_features, d_token)
        self.blocks = nn.ModuleList(
            [FastTransformerBlock(d_token, heads) for _ in range(depth)]
        )
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x):
        x_emb = self.tokenizer(x)
        for block in self.blocks:
            x_emb = block(x_emb)

        # Extract [CLS] token representation
        cls_out = self.head_norm(x_emb[:, 0])
        return self.head(cls_out).squeeze(-1)


def ft_fit_eval(
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
    exp_tag=None,
    is_regression=False,
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

    n_features = Xtr_np.shape[1]
    n_cat = ncat if isinstance(ncat, int) and ncat > 0 else 0
    n_num = n_features - n_cat

    cat_dims = []
    if n_cat > 0:
        cat_dims = [
            int(max(Xtr_np[:, i].max(), X_te_np[:, i].max()) + 1)
            for i in range(n_cat)
        ]

    use_amp = DEV.type == "cuda"
    task_name = "Regression" if is_regression else "Classification"
    print(
        f"    [FT-Transformer - {task_name}] Training on {DEV} | amp={use_amp} | batch_size=16384 | split={split}",
        flush=True,
    )

    def _build():
        """Fresh model + optimizer. The refit phase trains from scratch rather than
        continuing the selection model, so the final weights are never influenced by
        the 90% fit -- only the epoch count carries over."""
        m = FTTransformer(
            num_features=n_num + len(cat_dims), d_token=32, depth=2, heads=4
        ).to(DEV)
        return m, torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)

    model, optimizer = _build()

    # --- Task-specific loss function ---
    if is_regression:
        criterion = nn.MSELoss()
    else:
        pw = torch.tensor(float(spw), device=DEV) if spw is not None else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ds_tr = TensorDataset(torch.tensor(Xtr_np), torch.tensor(ya[tri]))
    loader_tr = DataLoader(
        ds_tr,
        batch_size=16384,
        shuffle=True,
        drop_last=False,
        pin_memory=use_amp,
        num_workers=0,
    )

    eval_size = min(30000, len(X_te_np))
    # Validation for epoch selection must NOT be the test slice: the loop keeps the
    # best-scoring epoch, so validating on `tei` selects the model on the data it
    # is then reported against. `trs` is the harness's held-out slice.
    _val_X, _val_i = ((X_trs_np, np.asarray(trs)) if X_trs_np is not None
                      else (X_te_np, np.asarray(tei)))
    eval_size = min(eval_size, len(_val_X))
    eval_idx = np.random.choice(len(_val_X), size=eval_size, replace=False)
    X_eval_tensor = torch.tensor(_val_X[eval_idx]).to(DEV)
    y_eval = ya[_val_i][eval_idx]

    best_score = -float("inf")
    patience, patience_counter, best_weights = 5, 0, None
    best_epoch = 0

    for epoch in range(10):
        running_loss = 0.0
        model.train()
        for bx, by in loader_tr:
            bx = bx.to(DEV, non_blocking=True)
            by = by.to(DEV, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(bx)
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
            f"    [FT-Transformer] Epoch {epoch+1:02d}/10 | Loss: {avg_loss:.4f} | {metric_label}: {val_score:.5f} (Best: {max(best_score, val_score):.5f})",
            flush=True,
        )
        log_epoch(epoch + 1,
                  {"train/loss": avg_loss, "val/score": val_score,
                   "val/best_score": max(best_score, val_score)},
                  phase=f"ft[{kind}]")

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            patience_counter = 0
            best_weights = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    if refit_idx is not None and best_epoch > 0:
        # Selection is done; its model is discarded. Retrain from scratch on the FULL
        # training window for exactly the epoch count the holdout chose, with no
        # validation and no early stopping, so the final fit sees every row the trees
        # see. See docs/validation-protocol.md.
        refit_idx = np.asarray(refit_idx)
        print(f"    [FT-Transformer] refit: {len(refit_idx):,} rows x {best_epoch} "
              f"epoch(s) (selected on {len(_val_i):,} holdout rows, "
              f"{metric_label} {best_score:.5f})", flush=True)
        # Rebound rather than deleted: the cleanup at the end of this function still
        # names them, and `del` here would make that a NameError on the refit path.
        ds_tr = loader_tr = Xtr_np = None
        gc.collect()

        X_rf = np.ascontiguousarray(
            np.asarray(_get_slice(parts, refit_idx), dtype=np.float32))
        loader_rf = DataLoader(
            TensorDataset(torch.tensor(X_rf), torch.tensor(ya[refit_idx])),
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
            avg_loss = running_loss / len(loader_rf)
            print(f"    [FT-Transformer] refit epoch {epoch+1:02d}/{best_epoch} | "
                  f"Loss: {avg_loss:.4f}", flush=True)
            log_epoch(epoch + 1, {"refit/loss": avg_loss}, phase=f"ft[{kind}]-refit")
        del loader_rf, X_rf
        gc.collect()

    save_path = model_path(exp_tag, "ft", kind, split, ".pt")
    torch.save(model.state_dict(), save_path)

    model.eval()
    with torch.no_grad():

        def _predict_in_chunks(X_data):
            if X_data is None:
                return None
            preds = []
            loader = DataLoader(
                TensorDataset(torch.tensor(X_data)),
                batch_size=32768,
                pin_memory=use_amp,
                num_workers=0,
            )
            for (bx,) in loader:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(bx.to(DEV, non_blocking=True))
                if is_regression:
                    preds.append(
                        out.float().cpu().numpy()
                    )  # Linear predictions for regression
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