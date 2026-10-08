#!/usr/bin/env bash
# Run eval.harness jobs on this machine's GPUs (lab servers: gpu3, redbull): one job
# per experiment x model x seed (x arm with -a split), each pinned to one GPU and
# started when a GPU has room. The lab-server counterpart of the Perlmutter script.
#
# Detaches by default, so it survives logging out; the scheduler's own output and
# one log per job go to scripts/logs/local/<host>_<time>/.  ./run_gpus.sh -h  for
# usage; see man/run_gpus.1.
#
# Options override environment variables, which override the defaults below.
# Parallel seeds are safe: predictions are named per seed and the per-model results
# file (results/<experiment>/<model>.json) is written under a lock.
set -uo pipefail
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
ORIG_ARGS=("$@")

# Python: the repo venv if there is one, else the miniconda install shared over home
# (non-interactive shells on gpu3 otherwise get /usr/bin/python3, which has no torch).
VENV="${FIFE_VENV:-$(dirname "$SCRIPTS")/venv}"
if [ -n "${FIFE_PYTHON:-}" ]; then PY="$FIFE_PYTHON"
elif [ -x "$VENV/bin/python3" ]; then PY="$VENV/bin/python3"
elif [ -x "$HOME/miniconda3/bin/python3" ]; then PY="$HOME/miniconda3/bin/python3"
else PY=python3; fi

EXPS="${EXPS:-}"
MODELS="${MODELS:-}"
SEEDS="${SEEDS:-0,1,2}"
ARM="${ARM:-both}"                           # both | random | temporal | split
GPUS="${FIFE_GPUS:-}"                        # empty: every GPU on the machine
MIN_FREE_GB="${FIFE_MIN_FREE_GB:-16}"        # free GPU memory needed to start a job
MAX_UTIL="${FIFE_MAX_UTIL:-100}"             # skip GPUs busier than this (%); 100 = ignore
MIN_RAM_GB="${FIFE_MIN_RAM_GB:-40}"          # host memory available needed to start a job
GAP="${FIFE_LAUNCH_GAP:-180}"                # s after a launch, so its data load shows in RAM
WANDB="${WANDB:-1}"
TAG="${WANDB_TAG:-$(hostname -s)}"
DRY=0
FG=0

usage() {
  cat <<EOF
Usage: $(basename "$0") -e EXPS -m MODELS [options] [-- extra harness args]

  -e EXPS     experiments, comma-separated: e1,e2,e3                 [\$EXPS]
  -m MODELS   models, comma-separated, e.g. mlp,saint,ft             [\$MODELS]
  -s SEEDS    seeds, comma-separated; one job each                   [$SEEDS]
  -a ARM      both | random | temporal | split (one job per arm)     [$ARM]
  -g GPUS     GPU indices to use, e.g. 0,1 (default: all)            [\$FIFE_GPUS]
  -F GB       free GPU memory needed to start a job                  [$MIN_FREE_GB]
  -U PCT      skip GPUs above this utilisation (100 = ignore)        [$MAX_UTIL]
  -R GB       host memory available needed to start a job            [$MIN_RAM_GB]
  -T TAG      W&B tag                                                [$TAG]
  -W          no W&B logging
  -f          stay in the foreground (default: detach)
  -n          dry run: list the jobs, start nothing
  -h          this help

Python: $PY
Example:  $(basename "$0") -e e2 -m mlp,tabnet,saint,ft,tsmixer
EOF
}

while getopts "e:m:s:a:g:F:U:R:T:Wfnh" opt; do
  case "$opt" in
    e) EXPS="$OPTARG" ;;   m) MODELS="$OPTARG" ;;     s) SEEDS="$OPTARG" ;;
    a) ARM="$OPTARG" ;;    g) GPUS="$OPTARG" ;;       F) MIN_FREE_GB="$OPTARG" ;;
    U) MAX_UTIL="$OPTARG" ;; R) MIN_RAM_GB="$OPTARG" ;; T) TAG="$OPTARG" ;;
    W) WANDB=0 ;;          f) FG=1 ;;                 n) DRY=1 ;;
    h) usage; exit 0 ;;    *) usage >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ "${1:-}" = "--" ] && shift
EXTRA="$*"

die() { echo "error: $*" >&2; exit 2; }
[ -n "$EXPS" ] && [ -n "$MODELS" ] || { usage >&2; die "-e and -m are required"; }
for e in ${EXPS//,/ }; do
  case "$e" in e1|e2|e3) ;; *) die "unknown experiment '$e'" ;; esac
done
case "$ARM" in both|random|temporal) ARMS=("$ARM") ;; split) ARMS=(random temporal) ;;
  *) die "-a must be both, random, temporal or split" ;; esac
command -v nvidia-smi >/dev/null || die "nvidia-smi not found -- no GPUs here?"
[ -n "$GPUS" ] || GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)"

if [ "$WANDB" = 1 ]; then WB="--wandb --wandb-tag $TAG"; else WB="--no-wandb"; fi

NAMES=(); CMDS=()
for e in ${EXPS//,/ }; do
  for m in ${MODELS//,/ }; do
    for s in ${SEEDS//,/ }; do
      for a in "${ARMS[@]}"; do
        NAMES+=("$e-$m-$a-s$s")
        CMDS+=("-m eval.harness $e $m $a --seeds $s $WB $EXTRA")
      done
    done
  done
done

if [ "$DRY" = 1 ]; then
  echo "host $(hostname -s) | GPUs $GPUS | python $PY"
  for i in "${!NAMES[@]}"; do printf '  %-28s %s %s\n' "${NAMES[$i]}" "$PY" "${CMDS[$i]}"; done
  echo "${#NAMES[@]} job(s) would run, at most one per GPU at a time"
  exit 0
fi

LOGDIR="${FIFE_LOCAL_LOGDIR:-$SCRIPTS/logs/local/$(hostname -s)_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOGDIR"

# Detach: re-run this script under setsid/nohup with the same arguments and return.
if [ "$FG" = 0 ] && [ -z "${FIFE_RUN_GPUS_CHILD:-}" ]; then
  FIFE_RUN_GPUS_CHILD=1 FIFE_LOCAL_LOGDIR="$LOGDIR" \
    setsid nohup "$0" "${ORIG_ARGS[@]}" >"$LOGDIR/scheduler.log" 2>&1 </dev/null &
  echo "${#NAMES[@]} job(s) queued on $(hostname -s), GPUs $GPUS"
  echo "progress:  tail -f $LOGDIR/scheduler.log"
  echo "stop:      kill \$(cat $LOGDIR/scheduler.pid)   (also stops its running jobs)"
  exit 0
fi

echo $$ >"$LOGDIR/scheduler.pid"
cd "$SCRIPTS"
export PYTHONPATH="$SCRIPTS:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID              # CUDA indices == nvidia-smi indices
ts() { date "+%m-%d %H:%M:%S"; }

declare -A RUNNING=()   # gpu -> pid
declare -A RUNNAME=()   # gpu -> job name
trap 'echo "$(ts) stopping; killing ${RUNNING[*]:-nothing}"; kill ${RUNNING[@]} 2>/dev/null; rm -f "$LOGDIR/scheduler.pid"; exit 143' INT TERM

gpu_ok() {   # free memory and utilisation as nvidia-smi sees them now
  local used total util
  IFS=', ' read -r used total util < <(nvidia-smi -i "$1" \
    --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
  [ $(( (total - used) / 1024 )) -ge "$MIN_FREE_GB" ] && [ "$util" -le "$MAX_UTIL" ]
}
ram_ok() {
  [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge "$MIN_RAM_GB" ]
}
reap() {
  local g pid rc
  for g in "${!RUNNING[@]}"; do
    pid="${RUNNING[$g]}"
    kill -0 "$pid" 2>/dev/null && continue
    wait "$pid"; rc=$?
    if [ "$rc" = 0 ]; then echo "$(ts) done    ${RUNNAME[$g]} (GPU $g)"
    else echo "$(ts) FAILED  ${RUNNAME[$g]} (GPU $g, exit $rc) -- see $LOGDIR/${RUNNAME[$g]}.log"; FAILS=$((FAILS + 1)); fi
    unset "RUNNING[$g]" "RUNNAME[$g]"
  done
}

echo "$(ts) $(hostname -s): ${#NAMES[@]} job(s) on GPUs $GPUS | python $PY"
echo "  start a job when a GPU has >= ${MIN_FREE_GB} GB free and <= ${MAX_UTIL}% util, host >= ${MIN_RAM_GB} GB"
next=0; FAILS=0; waited=0
while [ "$next" -lt "${#NAMES[@]}" ] || [ "${#RUNNING[@]}" -gt 0 ]; do
  reap
  launched=0
  if [ "$next" -lt "${#NAMES[@]}" ]; then
    for g in ${GPUS//,/ }; do
      [ -n "${RUNNING[$g]:-}" ] && continue
      gpu_ok "$g" && ram_ok || continue
      name="${NAMES[$next]}"
      # shellcheck disable=SC2086 -- CMDS entries are word lists
      CUDA_VISIBLE_DEVICES="$g" "$PY" ${CMDS[$next]} >"$LOGDIR/$name.log" 2>&1 &
      RUNNING[$g]=$!; RUNNAME[$g]="$name"
      echo "$(ts) start   $name on GPU $g (pid $!)"
      next=$((next + 1)); launched=1; waited=0
      break
    done
    if [ "$launched" = 0 ] && [ "${#RUNNING[@]}" = 0 ] && [ $((waited % 20)) = 0 ]; then
      echo "$(ts) waiting: no GPU in $GPUS has room (or host RAM < ${MIN_RAM_GB} GB)"
    fi
  fi
  # Sleep in the background and wait on it, so a kill runs the trap at once.
  if [ "$launched" = 1 ]; then sleep "$GAP" & else sleep 30 & waited=$((waited + 1)); fi
  wait $!
done
echo "$(ts) all done: ${#NAMES[@]} job(s), $FAILS failed. logs in $LOGDIR"
rm -f "$LOGDIR/scheduler.pid"
