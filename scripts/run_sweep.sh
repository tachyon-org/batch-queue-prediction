#!/usr/bin/env bash
# Seed-42 sweep for E1 (failure prediction) and E3 (fault attribution), both split
# protocols, under the revised labels and the label-observation temporal split.
#
# Prerequisite: the feature pipeline must be regenerated first, e.g. `python3 -m eval.featurize --overwrite`.
# Run from scripts/.  Usage:  ./run_sweep.sh [seed]
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# A single seed produces no spread; error bars need >= 3.
SEEDS="${SEEDS:-${1:-42}}"
PRIMARY_SEED="${SEEDS%%,*}"

# Checks the leakage gap is not an artifact of where the cutoff falls.
CUTOFFS="${CUTOFFS:-2025-06-01,2025-07-01,2025-08-01}"
CUTOFF_MODELS="${CUTOFF_MODELS:-xgboost lightgbm}"

LOGDIR="logs/sweep_seed${PRIMARY_SEED}_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOGDIR"

# WANDB=0 disables; WANDB_MODE=offline defers upload. Credentials: config/wandb.yaml.
WANDB="${WANDB:-1}"
if [ "$WANDB" = "1" ]; then
  WB_ARGS=(--wandb --wandb-tag "sweep$(date +%Y%m%d)")
  [ -n "${WANDB_MODE:-}" ] && WB_ARGS+=(--wandb-mode "$WANDB_MODE")
else
  WB_ARGS=(--no-wandb)
fi
echo "wandb: $([ "$WANDB" = 1 ] && echo enabled || echo disabled)"

# Trees first: fast, and they surface data problems before the expensive fits start.
# Override to run a subset, e.g.  MODELS="xgboost lightgbm catboost" ./run_sweep.sh
read -r -a MODELS <<< "${MODELS:-xgboost lightgbm catboost mlp tabnet saint ft tsmixer}"

echo "sweep -> $LOGDIR  (seeds $SEEDS)"
for exp in e1 e3; do
  for m in "${MODELS[@]}"; do
    log="$LOGDIR/${exp}_${m}.log"
    echo "=== $exp / $m ==="
    if python3 -m eval.harness "$exp" "$m" both --seeds "$SEEDS" "${WB_ARGS[@]}" >"$log" 2>&1; then
      grep -E "ROC|PR |hardware |payload |calibration" "$log" | tail -4
    else
      echo "  FAILED (exit $?) -- see $log"
      tail -5 "$log" | sed 's/^/    /'
    fi
  done
done

# The harness prints the no-feature reference floor before each model's numbers.
echo
echo "=== E2 wait time ==="
for m in "${MODELS[@]}"; do
  log="$LOGDIR/e2_${m}.log"
  echo "=== e2 / $m ==="
  python3 -m eval.harness e2 "$m" both --seeds "$SEEDS" "${WB_ARGS[@]}" >"$log" 2>&1 \
    && grep -E "ref:|R2\(log\)" "$log" | tail -6 \
    || { echo "  FAILED -- see $log"; tail -5 "$log" | sed 's/^/    /'; }
done

# Temporal only: the random split has no cutoff.
echo
echo "=== cutoff sensitivity (temporal only) ==="
for cut in ${CUTOFFS//,/ }; do
  if [ "$cut" = "2025-07-01" ]; then
    echo "  $cut -- already covered by the main sweep, skipping"
    continue
  fi
  for m in $CUTOFF_MODELS; do
    for exp in e1 e3; do
      log="$LOGDIR/cut${cut}_${exp}_${m}.log"
      echo "=== cutoff $cut / $exp / $m ==="
      python3 -m eval.harness "$exp" "$m" temporal --cutoff "$cut" \
          --seed "$PRIMARY_SEED" "${WB_ARGS[@]}" >"$log" 2>&1 \
        && grep -E "temporal cutoff|ROC|hardware " "$log" | tail -3 \
        || { echo "  FAILED -- see $log"; tail -5 "$log" | sed 's/^/    /'; }
    done
  done
done

echo
echo "=== cascade over the saved predictions ==="
for m in "${MODELS[@]}"; do
  python3 -m eval.harness cascade "${m}:${m}" temporal \
    --seed "$PRIMARY_SEED" "${WB_ARGS[@]}" \
    >"$LOGDIR/cascade_${m}.log" 2>&1 \
    && grep -E "P\(fail\)|hardware faults surviving" "$LOGDIR/cascade_${m}.log" \
    || echo "  cascade $m failed -- see $LOGDIR/cascade_${m}.log"
done
echo
echo "logs: $LOGDIR"
