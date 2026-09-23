"""Weights & Biases logging for the evaluation harness.

Config: copy config/wandb.yaml.example to config/wandb.yaml and fill it in.
Override order is CLI > environment > config/wandb.yaml > DEFAULTS below.

Every entry point is a no-op when logging is off or wandb is missing, so nothing
here can break a fit that has already burned GPU-hours.
"""

import os
import sys

import yaml

CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "wandb.yaml")

DEFAULTS = {
    "enabled": False,
    "project": "fife-batch-jobs",
    "entity": None,
    "mode": "online",
    "api_key": None,
    "tags": [],
    "notes": None,
    "log_epochs": True,
    "log_importance": True,
    "log_tree_rounds": False,
}

_ENV = {"project": "WANDB_PROJECT", "entity": "WANDB_ENTITY",
        "mode": "WANDB_MODE", "api_key": "WANDB_API_KEY"}

_ACTIVE = None
_WARNED = set()


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[wandb] {msg}", file=sys.stderr, flush=True)


def load_config(args=None):
    """Resolves CLI > env > config/wandb.yaml > DEFAULTS into one flat dict."""
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG):
        try:
            with open(CONFIG) as f:
                # None means "not filled in", so it must not blank out a default.
                cfg.update({k: v for k, v in (yaml.safe_load(f) or {}).items()
                            if v is not None})
        except Exception as e:
            _warn_once("yaml", f"could not read {CONFIG}: {e}")

    for key, env in _ENV.items():
        if os.environ.get(env):
            cfg[key] = os.environ[env]

    if args is not None:
        if getattr(args, "no_wandb", False):
            cfg["enabled"] = False
        elif getattr(args, "wandb", False):
            cfg["enabled"] = True
        for attr, key in (("wandb_project", "project"), ("wandb_entity", "entity"),
                          ("wandb_mode", "mode"), ("wandb_notes", "notes")):
            if getattr(args, attr, None):
                cfg[key] = getattr(args, attr)
        if getattr(args, "wandb_tag", None):
            cfg["tags"] = list(cfg["tags"]) + list(args.wandb_tag)
        if getattr(args, "wandb_tree_rounds", False):
            cfg["log_tree_rounds"] = True
    return cfg


def config_summary(cfg):
    if not cfg.get("enabled"):
        return "wandb: disabled"
    who = f"{cfg['entity']}/" if cfg.get("entity") else ""
    return f"wandb: {cfg['mode']} -> {who}{cfg['project']}"


class WandbRun:
    """One run. Every method is safe to call when logging is off."""

    def __init__(self, run=None, cfg=None):
        self.run = run
        self.cfg = cfg or DEFAULTS

    @classmethod
    def start(cls, cfg, experiment, model, split, seed, extra=None):
        global _ACTIVE
        if not cfg.get("enabled"):
            _ACTIVE = cls(None, cfg)
            return _ACTIVE
        try:
            import wandb
        except ImportError:
            _warn_once("import", "wandb not installed; logging disabled "
                                 "(pip install wandb, or drop --wandb)")
            _ACTIVE = cls(None, cfg)
            return _ACTIVE

        if cfg.get("api_key"):
            os.environ.setdefault("WANDB_API_KEY", str(cfg["api_key"]))
        run_cfg = {"experiment": experiment, "model": model, "split": split,
                   "seed": seed, **(extra or {})}
        try:
            run = wandb.init(
                project=cfg["project"], entity=cfg.get("entity"),
                mode=cfg.get("mode", "online"),
                # Grouped by task+model with the split as job_type, so the
                # random-vs-temporal contrast reads as one comparison.
                name=f"{experiment}-{model}-{split}-s{seed}",
                group=f"{experiment}-{model}",
                job_type=str(split),
                tags=[str(t) for t in cfg["tags"]]
                     + [str(experiment), str(model), str(split), f"seed{seed}"],
                notes=cfg.get("notes"), config=run_cfg, reinit=True,
            )
        except Exception as e:
            _warn_once("init", f"wandb.init failed ({e}); continuing without logging")
            _ACTIVE = cls(None, cfg)
            return _ACTIVE
        _ACTIVE = cls(run, cfg)
        return _ACTIVE

    def finish(self):
        global _ACTIVE
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as e:
                _warn_once("finish", f"wandb.finish failed: {e}")
        if _ACTIVE is self:
            _ACTIVE = None

    def log(self, metrics, step=None):
        if self.run is None or not metrics:
            return
        try:
            self.run.log(_scalars(metrics), step=step)
        except Exception as e:
            _warn_once("log", f"wandb.log failed: {e}")

    def log_final(self, metrics, prefix="final"):
        """Logs metrics to history and to the run summary, which is what the runs
        table sorts and filters on."""
        if self.run is None:
            return
        flat = _scalars({f"{prefix}/{k}": v for k, v in (metrics or {}).items()})
        self.log(flat)
        try:
            for k, v in flat.items():
                self.run.summary[k] = v
        except Exception as e:
            _warn_once("summary", f"wandb summary write failed: {e}")

    def log_importance(self, names, values, name="feature_importance"):
        if self.run is None or not self.cfg.get("log_importance", True):
            return
        try:
            import wandb
            rows = sorted(zip(names, [float(v) for v in values]), key=lambda r: -r[1])
            self.run.log({name: wandb.Table(columns=["feature", "importance"],
                                            data=[[str(n), v] for n, v in rows])})
        except Exception as e:
            _warn_once("imp", f"wandb importance table failed: {e}")


def _scalars(metrics):
    """Keeps only chartable values. NaN appears for empty strata and would render as
    a gap and sort unpredictably, so those keys are dropped for that step."""
    out = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, (int, float)):
            f = float(v)
            if f == f and abs(f) != float("inf"):
                out[k] = f
    return out


def active():
    return _ACTIVE if (_ACTIVE is not None and _ACTIVE.run is not None) else None


def log_epoch(step, metrics, phase=None):
    """Called from training loops; returns immediately when no run is active.
    `phase` namespaces keys so a model training several heads does not collide."""
    run = active()
    if run is None or not run.cfg.get("log_epochs", True):
        return
    pre = f"{phase}/" if phase else ""
    run.log({**{f"{pre}{k}": v for k, v in (metrics or {}).items()},
             f"{pre}epoch": step})


def log_summary(**kv):
    """Record one-off values on the active run's summary (no step). Used for things
    that are properties of the fit rather than of an epoch -- the selected epoch
    count, the number of rows the final fit saw."""
    run = active()
    if run is None:
        return
    try:
        for k, v in kv.items():
            run.run.summary[k] = v
    except Exception as e:
        _warn_once("summary", f"wandb summary write failed ({e})")


def tree_rounds_on():
    run = active()
    return run is not None and run.cfg.get("log_tree_rounds", False)
