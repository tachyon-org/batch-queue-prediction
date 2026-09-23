"""Filesystem layout for datasets, saved models and saved predictions.

Every path in the project resolves through here, in the project's usual order:

    1. environment    FIFE_DATA_ROOT, FIFE_MODEL_ROOT, FIFE_PRED_ROOT
    2. config file    scripts/config/paths.yaml, `hosts:` section for this host
    3. config file    scripts/config/paths.yaml, top level
    4. default        ./data, ./models, ./predictions, relative to scripts/

The `hosts:` section exists because a checkout on a shared filesystem is visible
from several machines while their fast local storage is not, so one top-level value
cannot serve them all. Host names appear only in that file, never here.

Nothing here names a particular machine. Training mmaps tens of GB out of these
roots repeatedly, so on a cluster they should point at fast local storage rather
than a network mount -- which is what the config file is for. The defaults are
repo-relative so a fresh checkout runs without configuration.

Model layout is one directory per experiment and model:

    models/e1/xgboost/xgboost_bin_temporal.json
    models/e2/lightgbm/lightgbm_reg_random.txt
    models/e3/saint/saint_bin_temporal.pt

A multi-seed run writes the same path each time, so the file left on disk is the
LAST seed. That is deliberate: per-seed metrics are what the variance report needs
and they are kept in the results JSON under `<split>__seed<n>` keys, while keeping
one model per seed would multiply the on-disk footprint for artifacts nothing reads.
"""

import os
import socket

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
CONFIG_PATH = os.environ.get("FIFE_PATHS_CONFIG",
                             os.path.join(_SCRIPTS, "config", "paths.yaml"))

_DEFAULTS = {
    "data_root": os.path.join(_SCRIPTS, "data"),
    "model_root": os.path.join(_SCRIPTS, "models"),
    "pred_root": os.path.join(_SCRIPTS, "predictions"),
}


HOSTNAME = socket.gethostname().split(".")[0]


def _load_config():
    """Read scripts/config/paths.yaml if present, returning (top_level, this_host).

    Absent or unreadable is not an error -- the defaults stand and the roots can
    still be set from the environment.
    """
    if not os.path.isfile(CONFIG_PATH):
        return {}, {}
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        def _strs(d):
            return {k: v for k, v in (d or {}).items()
                    if isinstance(v, str) and v.strip()}
        global _CFG_RAW, _HOST_RAW
        _CFG_RAW = cfg
        _HOST_RAW = (cfg.get("hosts") or {}).get(HOSTNAME) or {}
        return _strs(cfg), _strs(_HOST_RAW)
    except Exception as e:                      # noqa: BLE001 - config must not break a run
        print(f"[paths] ignoring {CONFIG_PATH}: {e}")
        return {}, {}


_CFG_RAW, _HOST_RAW = {}, {}
_CFG, _HOST_CFG = _load_config()


def _resolve(env_var, key):
    return os.path.expanduser(
        os.environ.get(env_var) or _HOST_CFG.get(key) or _CFG.get(key)
        or _DEFAULTS[key])


# Results are tracked in git, unlike the data and model roots, so they live in the
# repository at a fixed location. Deliberately not configurable and not relative to
# the working directory: a run started from anywhere must update the same file.
RESULTS_DIR = os.path.join(_SCRIPTS, "results")

DATA_ROOT = _resolve("FIFE_DATA_ROOT", "data_root")

MODEL_ROOT = _resolve("FIFE_MODEL_ROOT", "model_root")
PRED_ROOT = _resolve("FIFE_PRED_ROOT", "pred_root")

# exp_tag values used by the harness -> the directory they belong in.
_EXP_ALIASES = {"e3_fault": "e3"}


def experiment_dir_name(exp_tag):
    if not exp_tag:
        return "misc"
    return _EXP_ALIASES.get(str(exp_tag), str(exp_tag))


def model_dir(exp_tag, lib, create=True):
    """<MODEL_ROOT>/<experiment>/<model>/"""
    d = os.path.join(MODEL_ROOT, experiment_dir_name(exp_tag), str(lib))
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def model_path(exp_tag, lib, kind, split, ext="", create=True):
    """Full path for one saved model. `ext` includes the dot, or is empty for
    libraries that append their own (TabNet). Not seed-qualified: a multi-seed run
    overwrites, leaving the last seed's model."""
    return os.path.join(model_dir(exp_tag, lib, create=create),
                        f"{lib}_{kind}_{split}{ext}")


def all_model_dirs():
    """Every per-model directory under MODEL_ROOT, for post-hoc loaders that scan
    for saved models. Includes MODEL_ROOT itself so flat legacy layouts still load."""
    out = [MODEL_ROOT]
    if not os.path.isdir(MODEL_ROOT):
        return out
    for exp in sorted(os.listdir(MODEL_ROOT)):
        p = os.path.join(MODEL_ROOT, exp)
        if not os.path.isdir(p):
            continue
        out.append(p)
        out.extend(os.path.join(p, m) for m in sorted(os.listdir(p))
                   if os.path.isdir(os.path.join(p, m)))
    return out


def data_path(name):
    return os.path.join(DATA_ROOT, name)


def describe():
    """Where each root came from, for a run log or a bug report."""
    src = {}
    for env_var, key, val in (("FIFE_DATA_ROOT", "data_root", DATA_ROOT),
                              ("FIFE_MODEL_ROOT", "model_root", MODEL_ROOT),
                              ("FIFE_PRED_ROOT", "pred_root", PRED_ROOT)):
        if os.environ.get(env_var):
            where = f"env {env_var}"
        elif _HOST_CFG.get(key):
            where = f"config hosts.{HOSTNAME}"
        elif _CFG.get(key):
            where = "config (top level)"
        else:
            where = "default"
        src[key] = (val, where)
    return src


if __name__ == "__main__":
    print(f"host {HOSTNAME}   config {CONFIG_PATH}"
          f"{'' if os.path.isfile(CONFIG_PATH) else '  (absent)'}")
    for k, (v, where) in describe().items():
        exists = "" if os.path.isdir(v) else "   [missing]"
        print(f"  {k:12s} {v}{exists}    [{where}]")
