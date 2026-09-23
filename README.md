## Batch-Scheduled HEP Job Prediction

A benchmarking and evaluation suite for evaluating tree-based models and deep tabular architectures on High-Performance Computing (HPC) batch job execution logs. This repository provides end-to-end pipelines for model training, threshold selection, feature importance extraction, and protocol evaluation under both random and temporal data split schemes. We also include analyses of submit-time vs. match-time feature matrices, hold reason codes, and exit code/signal evaluation.

### Prediction Tasks

1. Job Failure (classification) at match-time (**E1**)
2. Queue wait time (regression) at submit-time (**E2**)
3. Fault attribution (classification) at match-time (**E3**)

### Supported Models

* XGBoost, LightGBM, CatBoost, MLP
* TabNet
* TabR
* FT-Transformer
* SAINT
* Hierarchical NN
* TSMixer

### Repo Structure

```
fife-batch-jobs/
├── scripts/
│   ├── config/
│   │    └── wandb.yaml.example				# Copy to wandb.yaml and fill in
│   ├── eval/
│   │   ├── harness.py              		# Main CLI evaluation harness
│   │   └── helper.py               		# Model loaders, metrics, and prediction routines
│   ├── logs/                       		# Run logs from long-running sweeps (gitignored)
│   ├── output/                     		# Figures (PDF) and per-run evaluation JSONs
│   ├── results/                      		# Job prediction results (JSON format)
│   ├── train/                      		# Model implementations
│   ├── vis/								# Data explorer visualization
|   ├── data-explorer.ipynb    				# Job data visualization app example (work in progress)
|   ├── data-analysis.ipynb    				# Initial job data analysis, distribution fits
|   ├── feat-engineering.ipynb 				# Training and testing setup for job prediction tasks
|   ├── pred-analysis.ipynb    				# Job outcome prediction results
|   ├── wait_time_regression.ipynb			# Queue wait time prediction results
|   ├── load.py								# Module to load/process the job logs
|   ├── run_sweep.sh						# Full sweep: all models x both splits x all tasks
├── .gitignore                      		# Dataset and runtime configuration
├── CLAUDE.md                       		# Global instructions for Claude  
├── FIFE-Docs.md                    		# FIFE Batch Queue data documentation  
├── MODELING.md                     		# Detailed modeling design notes
├── README.md                       		# Repo info
└── config.json								# Dataset and runtime configuration
```

The notebooks live directly in `scripts/`, beside `eval/` and `train/`, rather than in
a `notebooks/` subdirectory. Jupyter sets the working directory to the notebook's own
folder, so this is what lets `from eval.dataset import load_experiment` resolve with no
`sys.path` manipulation, and it makes the relative paths in the notebooks (`results/`,
`output/`) the same ones the harness uses. Run them with `scripts/` as the working
directory.

### Requirements

#### Environment setup

1. Create a virtual environment

```
   python3 -m venv ~/envs/batch-eval
```

2. Activate

```
  source ~/envs/batch-eval/bin/activate
```

3. Install dependencies

```
  pip install --no-cache-dir -r requirements.txt
```

#### Dataset(s)

TBD

##### Structure

Paths resolve through `scripts/eval/paths.py` and each root takes an environment
override: `FIFE_DATA_ROOT` (feature matrices and targets, default fast local NVMe),
`FIFE_MODEL_ROOT` (saved models, default bulk storage), `FIFE_PRED_ROOT` (saved test
predictions). Data stays on scratch because training mmaps tens of GB out of it
repeatedly; `/media/storage0` is NFS and would be far slower.

Saved models are laid out one directory per experiment and model, with the seed in
the filename so seeds no longer overwrite each other:

```
models/
├── e1/xgboost/xgboost_bin_temporal_s42.json
├── e2/lightgbm/lightgbm_reg_random_s0.txt
└── e3/saint/saint_bin_temporal_s1.pt
```

- `Xmatch.npy` & `Xsub.npy`: feature matrices (match-time and submit-time features)
- `failed.npy`: finary failure target labels
- `wait_sv.npy`: raw queue wait times in seconds
- `tr_mask.npy` & `te_mask.npy`: train and test split marks (`tr` for random and `te` for temporal)
- `targets_and_masks.npz`: aggregated dataset dictionary

### How to run

All training and evaluation tasks are managed through the CLI in `scripts/eval/harness.py`.
Run it as a module with `scripts/` as the working directory:

```
cd scripts
python3 -m eval.harness <experiment> <model> [split]
```

#### Arguments

- `<experiment>`:
  - `e1` -- run job failure classification
  - `e2` -- run queue wait time regression
  - `e3` -- run fault attribution classification
- `<model>`: `xgb`, `lgb`, `cat`, `mlp`, `tabnet`, `saint`, `ft`, `tsmixer`, `tabr`, `hierarchical`
- `[split]`: specify training split protocol (optional) `random`, `temporal`, or `both` (default)

Additional experiments:

- `e2dist` -- queue wait as an *interval* rather than a point estimate (`--head quantile|aft`)
- `cascade` -- match-time hardware detection composing saved E1 and E3 scores

#### Flags

| Flag                    | Default        | What it does                                                                                               |
| ----------------------- | -------------- | ---------------------------------------------------------------------------------------------------------- |
| `--seed N`            | `42`         | Seeds Python, NumPy and Torch, and suffixes saved models, predictions and result keys.                     |
| `--seeds 0,1,2,3,4`   | --             | Runs each seed in sequence and prints mean ± std per split at the end. Overrides`--seed`.               |
| `--cutoff YYYY-MM-DD` | `2025-07-01` | Deployment cutoff the temporal protocol simulates. Also accepts raw epoch seconds. Parsed as**UTC**. |

**Seeding.** `--seed` is the only source of run-to-run randomness: it seeds Python's`random`passed to every model constructor. The train/test *partition* is deliberately not affected -- it is built with `np.random.default_rng`, which is immune to`np.random.seed`, so every seed sees exactly the same data and the spread measures model variance alone. A single seed reports `std 0.0` with `n=1`, which is a placeholder and not evidence of stability; use `--seeds` with at least three values for anything reported as an error bar.

**Cutoffs.** Results are keyed by cutoff when it is not the default, e.g.`temporal__seed7__cut2025-06-01`. Only the temporal protocol depends on the cutoff; the random split has none.

**Split basis.** The temporal split always cuts on when each job's *label became observable*, never on submission time. A job submitted June 28 that finishes July 5 has an outcome that was unknowable at a July 1 cutoff, so cutting on `QDate` would leave a future label in training. It is separate from feature admissibility (`Xsub` vs `Xmatch`), which governs which *columns* exist at prediction time. On this dataset the two bases differ by 22,370 jobs, 0.052% of training.

#### Usage

Run from `scripts/` with the package on the path:

```
cd scripts
```

1. Job failure classification using XGBoost, both splits:

```
python3 -m eval.harness e1 xgboost both
```

2. Five seeds, logged to W&B:

```
python3 -m eval.harness e1 xgboost both --seeds 0,1,2 --wandb
```

3. Cutoff sensitivity -- same protocol, different deployment date:

```
python3 -m eval.harness e1 xgboost temporal --cutoff 2025-06-01
python3 -m eval.harness e1 xgboost temporal --cutoff 2025-08-01
```

### Running the full sweep

`scripts/run_sweep.sh` runs every model across both splits for E1/E2/E3, then the interval heads, the cutoff-sensitivity runs and the cascade. Logs land in
`scripts/logs/sweep_seed<seed>_<timestamp>/`, one file per cell, and a one-line summary of each is echoed as it finishes.

```
cd scripts
./run_sweep.sh                                   # seed 42, default cutoff
SEEDS=0,1,2,3,4 ./run_sweep.sh                   # multi-seed error bars
CUTOFFS=2025-06-01,2025-08-01 ./run_sweep.sh     # add cutoff sensitivity
WANDB=0 ./run_sweep.sh                           # no logging
WANDB_MODE=offline ./run_sweep.sh                # log locally, sync later
```

| Variable          | Default                              | Meaning                                                                                                                                              |
| ----------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SEEDS`         | `42`                               | Comma-separated seeds. The first is the "primary" seed used for single-seed stages (cascade, cutoff sweep).                                          |
| `CUTOFFS`       | `2025-06-01,2025-07-01,2025-08-01` | Cutoffs for the sensitivity stage.`2025-07-01` is skipped there because the main sweep already covers it.                                          |
| `CUTOFF_MODELS` | `xgboost lightgbm`                 | Models used for cutoff sensitivity. Kept small on purpose -- this answers "is the gap an artifact of one cutoff", which does not need the full grid. |
| `WANDB`         | `1`                                | `0` disables logging entirely.                                                                                                                     |
| `WANDB_MODE`    | unset                                | `offline` writes to `wandb/` for a later `wandb sync`.                                                                                         |

The sweep must be run **after** the feature pipeline has been regenerated; see the notebook order in `scripts/` (`data-analysis.ipynb` -> `feat-engineering.ipynb`)

### Experiment tracking (Weights & Biases)

Logging is **off by default** and the code runs unchanged without `wandb` installed.

**Setup** -- one file, copied once. `config/wandb.yaml` is gitignored:

```
cd scripts
cp config/wandb.yaml.example config/wandb.yaml     # fill in entity + api_key
python3 -m eval.harness e1 xgboost both --wandb
```

If you already ran `wandb login` or set `WANDB_API_KEY`, leave `api_key` blank. Blank
keys fall back to the defaults in `eval/wandb_logger.py`.

### Metrics

##### Job failure classification/fault attribution

Precision, recall, F1, ROC-AUC, PR-AUC

##### Queue wait time

- log1p MAE: mean absolute error evaluated on $log(1 + wait\_time)$
- raw MAE (s): mean absolute error converted back to raw seconds
- sMAPE (%): symmetric mean absolute percentage error (bounded between $0\%$ and $200\%$)
- within-2x ratio: proportion of predictions falling within a factor of 2 of actual wait times

Error is also broken out by **queueing regime**. These bins are crossover points of a three-component lognormal mixture fitted to the
FermiGrid wait distribution (`scripts/data-analysis.ipynb`).

| Key      | Range           | Mechanism                                         |
| -------- | --------------- | ------------------------------------------------- |
| `inst` | < 2 min         | matched into an already-idle pilot slot           |
| `turn` | 2 min -- 45 min | waiting for an occupied slot to turn over         |
| `prov` | 45 min -- 1 day | waiting for new pilot provisioning                |
| `park` | > 1 day         | beyond the fitted range (not a regime; see below) |

### References

Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. [https://doi.org/10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NeurIPS 30), 3146–3154.

Prokhorenkova, L., et al. (2018). *CatBoost: unbiased boosting with categorical features*. Advances in Neural Information Processing Systems (NeurIPS 31), 6638–6648.

Arik, S. O., & Pfister, T. (2021). *TabNet: Attentive Interpretable Tabular Learning*. Proceedings of the AAAI Conference on Artificial Intelligence, 35(8), 6679–6687. [https://arxiv.org/abs/1908.07442](https://arxiv.org/abs/1908.07442)

Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*. Advances in Neural Information Processing Systems (NeurIPS 34), 18932–18943. [https://arxiv.org/abs/2106.11959](https://arxiv.org/abs/2106.11959)

Somepalli, G., Goldblum, M., Suri, A., Geiping, J., & Goldstein, T. (2021). *SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-training*. arXiv preprint arXiv:2106.01342. [https://arxiv.org/abs/2106.01342](https://arxiv.org/abs/2106.01342)

Chen, S. A., Li, C. L., Yoder, N., Sercu, T., & Pfister, T. (2023). *TSMixer: Lightweight MLP-Mixer for Multivariate Time Series Forecasting*. arXiv preprint arXiv:2303.06053. [https://arxiv.org/abs/2303.06053](https://arxiv.org/abs/2303.06053)

Lovell, A., et al. (2024). *A Hierarchical Deep Learning Approach for Predicting Job Queue Times in HPC Systems*. IEEE/ACM International Conference for High Performance Computing, Networking, Storage and Analysis Workshops (SC24 Workshops).
