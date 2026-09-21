import os
import gc
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, r2_score, roc_auc_score
from eval.helper import _empty_gpu, _get_slice
from eval.wandb_logger import log_epoch, log_summary
from eval.paths import model_path


class Attention(nn.Module):
    """Fast Feature Attention using PyTorch 2.0+ Scaled Dot-Product Attention (SDPA)."""
    def __init__(self, d_token, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = d_token // heads
        self.qkv = nn.Linear(d_token, d_token * 3)
        self.out = nn.Linear(d_token, d_token)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        # Reshape for multi-head attention: (B, heads, N, head_dim)
        q = q.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        
        # Uses FlashAttention / Memory-Efficient Attention C++ kernels automatically
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out(out)


class SAINT(nn.Module):
    """Vectorized & Optimized SAINT Architecture."""
    def __init__(self, n_num, cat_dims, d_token=32, depth=2, heads=4, split=None):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_dims)

        if n_num > 0:
            self.num_embed = nn.Parameter(torch.randn(n_num, d_token))
            self.num_bias = nn.Parameter(torch.randn(n_num, d_token))

        # Fused Categorical Embedding Table using offsets (eliminates ModuleList loops)
        if self.n_cat > 0:
            offsets = torch.tensor([0] + list(np.cumsum(cat_dims)[:-1]), dtype=torch.long)
            self.register_buffer("cat_offsets", offsets)
            self.cat_embed = nn.Embedding(sum(cat_dims), d_token)

        # Layers using fast SDPA attention
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(d_token=d_token, heads=heads),
                nn.LayerNorm(d_token),
                nn.Sequential(
                    nn.Linear(d_token, d_token * 2),
                    nn.ReLU(),
                    nn.Linear(d_token * 2, d_token)
                ),
                nn.LayerNorm(d_token)
            ]))

        total_cols = n_num + self.n_cat
        self.head = nn.Linear(total_cols * d_token, 1)

    def forward(self, x):
        B = x.size(0)
        tokens = []

        # Single vectorized lookup for all categorical features
        if self.n_cat > 0:
            x_cat = x[:, :self.n_cat].long() + self.cat_offsets
            cat_tokens = self.cat_embed(x_cat)
            tokens.append(cat_tokens)

        if self.n_num > 0:
            x_num = x[:, self.n_cat:]
            num_tokens = x_num.unsqueeze(-1) * self.num_embed + self.num_bias
            tokens.append(num_tokens)

        x_emb = torch.cat(tokens, dim=1)

        for attn, norm1, ffn, norm2 in self.layers:
            x_emb = norm1(x_emb + attn(x_emb))
            x_emb = norm2(x_emb + ffn(x_emb))

        out = x_emb.view(B, -1)
        return self.head(out).squeeze(-1)


def saint_fit_eval(
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
    task_type = "Regression" if is_regression else "Classification"
    print(
        f"    [SAINT - {task_type}] Training on {DEV} | amp={use_amp} | batch_size=16384",
        flush=True,
    )

    def _build():
        """Fresh model + optimizer. The refit trains from scratch rather than
        continuing the selection model: only the epoch count carries over."""
        m = SAINT(
            n_num=n_num,
            cat_dims=cat_dims,
            d_token=32,
            depth=2,
            heads=4,
            split=split,
        ).to(DEV)
        return m, torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)

    model, optimizer = _build()

    # --- REGRESSION VS CLASSIFICATION SETUP ---
    if is_regression:
        criterion = nn.MSELoss()
    else:
        pw = torch.tensor(float(spw), device=DEV) if spw is not None else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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

    # Validation set for epoch selection. This MUST NOT be the test slice: the
    # loop below keeps the best-scoring epoch, so validating on `tei` would be
    # selecting the model on the data it is then reported against. `trs` is the
    # harness's held-out slice (a genuine holdout for E1/E3; a training subsample
    # for E2, where it makes early stopping inert rather than leaky).
    _val_X, _val_i = ((X_trs_np, np.asarray(trs)) if X_trs_np is not None
                      else (X_te_np, np.asarray(tei)))
    eval_size = min(30000, len(_val_X))
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
                val_score = r2_score(
                    y_eval, val_preds
                )  # Score metric: R^2 for regression
                metric_name = "Val R2"
            else:
                val_preds = torch.sigmoid(val_raw.float()).cpu().numpy()
                val_score = roc_auc_score(
                    y_eval, val_preds
                )  # Score metric: ROC-AUC for classification
                metric_name = "Val AUC"

        print(
            f"    [SAINT] Epoch {epoch+1:02d}/10 | Loss: {avg_loss:.4f} | {metric_name}: {val_score:.5f} (Best: {max(best_score, val_score):.5f})",
            flush=True,
        )
        log_epoch(epoch + 1,
                  {"train/loss": avg_loss, "val/score": val_score,
                   "val/best_score": max(best_score, val_score)},
                  phase=f"saint[{kind}]")

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    save_path = model_path(exp_tag, "saint", kind, split, ".pt")
    torch.save(model.state_dict(), save_path)

    if best_weights is not None:
        model.load_state_dict(best_weights)

    if refit_idx is not None and best_epoch > 0:
        # Selection is done and its model is discarded. Retrain from scratch on the
        # FULL training window for exactly the epoch count the holdout chose, with no
        # validation and no early stopping, so the final fit sees every row the trees
        # see. See docs/validation-protocol.md.
        refit_idx = np.asarray(refit_idx)
        print(f"    [SAINT] refit: {{len(refit_idx):,}} rows x {{best_epoch}} epoch(s) "
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
            print(f"    [SAINT] refit epoch {{epoch+1:02d}}/{{best_epoch}} | "
                  f"Loss: {{avg:.4f}}", flush=True)
            log_epoch(epoch + 1, {{"refit/loss": avg}}, phase="saint-e2-refit")
        del loader_rf, X_rf
        gc.collect()


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
                    )  # Raw linear output for regression
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