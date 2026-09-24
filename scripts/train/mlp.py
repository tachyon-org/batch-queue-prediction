import os
import gc
import numpy as np
import torch
import torch.nn as nn
from eval.helper import _empty_gpu, _get_slice, _nfeat
from eval.wandb_logger import log_epoch
from eval.paths import model_path

from eval.helper import NEURAL_EPOCHS, NEURAL_CAT
MLP_EPOCHS, MLP_BS, MLP_LR, MLP_EMB_CAP = NEURAL_EPOCHS, 16384, 1e-3, 32

class MLP(nn.Module):
    """Standard Feedforward Neural Network (MLP) with Categorical Embeddings."""

    def __init__(self, cards_f, n_num, out=1, hidden=(256, 128)):
        super().__init__()
        self.embs = nn.ModuleList(
            [
                nn.Embedding(c + 1, min(MLP_EMB_CAP, (c + 1) // 2 + 1))
                for c in cards_f
            ]
        )
        d = sum(e.embedding_dim for e in self.embs) + n_num
        layers = []
        for a, b in zip((d,) + tuple(hidden), hidden):
            layers += [
                nn.Linear(a, b),
                nn.BatchNorm1d(b),
                nn.ReLU(),
                nn.Dropout(0.1),
            ]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden[-1], out)
        self.nout = out

    def forward(self, xc, xn):
        if len(self.embs) > 0:
            e_outs = [e(xc[:, i]) for i, e in enumerate(self.embs)]
            h = torch.cat(e_outs + [xn], dim=1)
        else:
            h = xn
        z = self.head(self.body(h))
        return z.squeeze(1) if self.nout == 1 else z


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def mlp_fit_eval(
    parts,
    tri,
    tei,
    ncat,
    kind,
    y,
    spw=None,
    num_class=None,
    trs=None,
    class_w=None,
    split=None,
    exp_tag=None,
):
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    ya = np.asarray(y)

    Xtr = _get_slice(parts, tri)

    # Determine categorical vs numerical dimensions
    # NEURAL_CAT is off by default: embedding the code columns memorises entity
    # identity that does not survive the cutoff. See eval/helper.py.
    n_cat_cols = (ncat if isinstance(ncat, int) else 0) if NEURAL_CAT else 0
    cards_f = (
        [int(Xtr[:, i].max() + 1) for i in range(n_cat_cols)]
        if n_cat_cols > 0
        else []
    )
    n_num_cols = Xtr.shape[1] - n_cat_cols

    # Standardize numerical features for neural network stability
    xn_raw = Xtr[:, n_cat_cols:].astype(np.float32)
    mean = np.nanmean(xn_raw, axis=0, keepdims=True)
    std = np.nanstd(xn_raw, axis=0, keepdims=True)
    std[std == 0] = 1.0
    mean = np.nan_to_num(mean)

    xn_tr_scaled = np.nan_to_num((xn_raw - mean) / std)

    net = MLP(
        cards_f, n_num_cols, out=(num_class if kind == "multi" else 1)
    ).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=MLP_LR)

    # Disable FP16 mixed precision for regression to prevent overflow/underflow
    use_amp = (DEV == "cuda") and (kind != "reg")

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    xc_tr = torch.from_numpy(Xtr[:, :n_cat_cols].astype(np.int64))
    xn_tr = torch.from_numpy(xn_tr_scaled.astype(np.float32))
    y_tr = torch.from_numpy(ya[tri])

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xc_tr, xn_tr, y_tr),
        batch_size=MLP_BS,
        shuffle=True,
    )

    pw = (
        torch.tensor(float(spw), device=DEV)
        if (kind == "bin" and spw is not None)
        else None
    )

    print(
        f"    mlp[{kind}] in-memory training: {len(tri):,} rows on ({DEV})",
        flush=True,
    )
    for ep in range(MLP_EPOCHS):
        net.train()
        tot, nb_ = 0.0, 0
        for xc_b, xn_b, yb_b in loader:
            xc_b, xn_b, yb_b = (
                xc_b.to(DEV, non_blocking=True),
                xn_b.to(DEV, non_blocking=True),
                yb_b.to(DEV, non_blocking=True),
            )

            with torch.autocast(
                device_type="cuda" if DEV == "cuda" else "cpu",
                enabled=use_amp,
            ):
                o = net(xc_b, xn_b)
                if kind == "bin":
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        o, yb_b.float(), pos_weight=pw
                    )
                elif kind == "reg":
                    loss = nn.functional.mse_loss(o, yb_b.float())
                elif kind == "multi":
                    loss = nn.functional.cross_entropy(o, yb_b.long())

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            nb_ += 1

        print(
            f"    mlp[{kind}] epoch {ep + 1}/{MLP_EPOCHS} mean loss {tot / max(nb_, 1):.4f}",
            flush=True,
        )
        # Per-epoch curve for W&B. No-op unless a run is active, so this stays
        # runnable with wandb absent.
        log_epoch(ep + 1, {"train/loss": tot / max(nb_, 1)}, phase=None)

    if split is not None:
        save_path = model_path(exp_tag, "mlp", kind, split, ".pt")
        torch.save(net.state_dict(), save_path)

    net.eval()

    def _pred(idxs):
        if len(idxs) == 0:
            return np.array([])
        Xeval = _get_slice(parts, idxs)
        xc_ev = torch.from_numpy(Xeval[:, :n_cat_cols].astype(np.int64))
        xn_ev_raw = Xeval[:, n_cat_cols:].astype(np.float32)
        xn_ev_scaled = np.nan_to_num((xn_ev_raw - mean) / std)
        xn_ev = torch.from_numpy(xn_ev_scaled.astype(np.float32))

        eval_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xc_ev, xn_ev),
            batch_size=131072,
            shuffle=False,
        )

        out = []
        with torch.no_grad():
            for xc_b, xn_b in eval_loader:
                pred = net(xc_b.to(DEV), xn_b.to(DEV)).float().cpu().numpy()
                out.append(pred)
        res_arr = np.concatenate(out)
        if kind == "bin":
            return _sigmoid(res_arr)
        return res_arr

    p_te = _pred(tei)
    p_trs = _pred(trs) if trs is not None else None

    del net, opt, Xtr, xc_tr, xn_tr, y_tr
    _empty_gpu()
    gc.collect()
    return p_te, p_trs