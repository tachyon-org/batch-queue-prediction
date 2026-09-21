import gc
import os
import numpy as np
import lightgbm as lgb
from eval.helper import _get_slice
import joblib
from eval.paths import model_path

class Hierarchical:
    """
    Two-Stage Hierarchical Wait-Time Estimator (Lovell et al. SC-W 2024).
    Stage 1: Global Gradient Boosted Trees for macro queue estimation.
    Stage 2: Hierarchical empirical residual correction across categorical submission tiers.
    """

    def __init__(
        self,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        smoothing_weight=20,
        seed=42,
        subsample=0.8,
    ):
        self.seed = seed
        self.subsample = subsample
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.smoothing_weight = (
            smoothing_weight  # Shrinkage weight for small groups
        )

        self.global_model = None
        self.group_residuals = {}
        self.global_residual_mean = 0.0

    def fit(self, X_tr, y_tr, n_cat=0):
        # --- Stage 1: Fit Global Macro Model ---
        self.global_model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            # Was a hardcoded 42, so the seed never reached this learner. Row
            # subsampling (subsample_freq >= 1, or LightGBM ignores it) is what makes
            # the fit stochastic at all -- without it the seed has nothing to act on
            # and a multi-seed run reports a spread of exactly 0.
            random_state=self.seed,
            subsample=self.subsample,
            subsample_freq=1,
            n_jobs=-1,
            verbosity=-1,
        )

        # Categorical feature indices for LightGBM
        cat_features = list(range(n_cat)) if n_cat > 0 else None
        self.global_model.fit(X_tr, y_tr, categorical_feature=cat_features)

        # Predict on training set to compute Stage 1 residuals
        pred_tr_macro = self.global_model.predict(X_tr)
        residuals = y_tr - pred_tr_macro
        self.global_residual_mean = float(np.mean(residuals))

        # --- Stage 2: Calculate Hierarchical Residual Corrections ---
        if n_cat > 0:
            # Use primary categorical column (e.g., JobSubGroup or Owner)
            # Column 0 is treated as the primary tier index
            primary_cats = X_tr[:, 0].astype(int)
            unique_groups = np.unique(primary_cats)

            for g in unique_groups:
                mask = primary_cats == g
                group_res = residuals[mask]
                n_samples = len(group_res)

                # Empirical Bayes / Laplace smoothing to prevent overfitting small queues:
                # residual_corr = (sum_res) / (count + smoothing_weight)
                smoothed_offset = np.sum(group_res) / (
                    n_samples + self.smoothing_weight
                )
                self.group_residuals[g] = smoothed_offset

        return self

    def predict(self, X, n_cat=0):
        # Macro predictions
        macro_preds = self.global_model.predict(X)

        # Apply Micro Hierarchical Corrections
        if n_cat > 0:
            primary_cats = X[:, 0].astype(int)
            micro_offsets = np.array(
                [
                    self.group_residuals.get(g, self.global_residual_mean)
                    for g in primary_cats
                ]
            )
            final_preds = macro_preds + micro_offsets
        else:
            final_preds = macro_preds + self.global_residual_mean

        return final_preds

    @property
    def feature_importances_(self):
        if self.global_model is not None:
            return self.global_model.feature_importances_
        return None


def hierarchical_fit_eval(
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
    seed=42,
):
    """
    Harness-compliant entry point for Hierarchical wait-time regression.
    """
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

    n_cat = ncat if isinstance(ncat, int) and ncat > 0 else 0

    print(
        f"    [Hierarchical - Wait Time Regression] Training on {Xtr_np.shape[0]} samples (Hierarchical Grouping n_cat={n_cat})...",
        flush=True,
    )

    model = Hierarchical(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        smoothing_weight=20,
        seed=seed,
    )

    # Fit Stage 1 & Stage 2
    model.fit(Xtr_np, ya[tri], n_cat=n_cat)

    save_path = os.path.join(
        model_path(exp_tag, "hierarchical", kind, split, ".joblib")
    )
    joblib.dump(model, save_path)
    print(f"--> Saved Hierarchical model to {save_path}", flush=True)

    # Generate Test & Validation predictions
    p_te = model.predict(X_te_np, n_cat=n_cat)
    p_trs = (
        model.predict(X_trs_np, n_cat=n_cat) if X_trs_np is not None else None
    )

    imp = model.feature_importances_ if want_imp else None

    # Cleanup
    del Xtr, X_te, X_trs, Xtr_np, X_te_np, X_trs_np
    gc.collect()

    return p_te, p_trs, imp