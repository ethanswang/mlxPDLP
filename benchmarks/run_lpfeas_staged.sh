#!/bin/bash
# Run the LPfeas Metal/FP32 benchmark in three serial stages (easy, medium,
# hard) with the current geometric-mean protocol. The comparison tool pairs
# these results with the recorded pre-geometric baselines.
#
# Usage:
#   run_lpfeas_staged.sh <easy|medium|hard|all> [results_dir]
#
# Environment:
#   BUILD_DIR           build tree containing mlxpdlp_lpfeas_benchmark
#                       (default: ../build relative to this script)
#   LPFEAS_DATA_DIR     directory with the .mps.gz corpus (default: data/lpfeas)
#   COOLDOWN_SECONDS    pause between instances (default: 120)
#
# Notes:
#   - Every instance runs SERIALLY (--jobs 1).
#   - A results_dir/<name>.json that already exists is skipped, so an
#     interrupted run resumes where it left off.
#   - COOLDOWN_SECONDS matters: back-to-back serial sweeps throttle the GPU
#     (degme measured 1073s hot vs 422s cool on identical code), which makes
#     wall-time comparisons against benchmarks/results meaningless.
#   - Stage assignment follows the last recorded wall time:
#       easy   <= 300s,  medium  300-3600s,  hard > 3600s.
#     Hard cases need ~21 hours of serial wall time in total.

set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR=${BUILD_DIR:-$(cd "$SCRIPT_DIR/../build" && pwd)}
DATA_DIR=${LPFEAS_DATA_DIR:-$(cd "$SCRIPT_DIR/data/lpfeas" && pwd)}
COOLDOWN=${COOLDOWN_SECONDS:-120}

STAGE=${1:-}
RESULTS_DIR=${2:-$SCRIPT_DIR/results/lpfeas-staged-geomean12}
if [ -z "$STAGE" ]; then
    echo "usage: $0 <easy|medium|hard|all> [results_dir]" >&2
    exit 2
fi

EASY="a2864 neos5052403 neos-3025225 stormG2_1000 scpm1 Primal2_1000 tpl-tub-ws16 L2CTA3D set-cover"
MEDIUM="degme thk_48 Dual2_5000 square41 rail4284 s82 thk_63"
HARD="fhnw-bin1 L1_sixm1000obs ns1688926 ns1687037 dlr1 bdry2 dlr2"

case "$STAGE" in
    easy)   INSTANCES=$EASY ;;
    medium) INSTANCES=$MEDIUM ;;
    hard)   INSTANCES=$HARD ;;
    all)    INSTANCES="$EASY $MEDIUM $HARD" ;;
    *)
        echo "unknown stage '$STAGE' (expected easy|medium|hard|all)" >&2
        exit 2
        ;;
esac

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/progress.log"

run_one() {
    local name=$1
    if [ -f "$RESULTS_DIR/$name.json" ]; then
        echo "[$(date +%H:%M:%S)] skip $name (result exists)" | tee -a "$LOG"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] starting $name" | tee -a "$LOG"
    "$BUILD_DIR/mlxpdlp_lpfeas_benchmark" \
        --data "$DATA_DIR" \
        --instance "$name" --jobs 1 --device metal \
        --tolerance 1e-4 --solver-tolerance 5e-5 \
        --time-limit 1200 --iteration-limit 1500000 \
        --correction-time-limit 600 --correction-iteration-limit 300000 \
        --host-double-polishing --host-double-time-limit 180 --host-double-iteration-limit 300000 \
        --geometric-mean-iterations 12 --curtis-reid-iterations 20 --sv-max-iterations 5000 \
        --fail-on-validation \
        --output-prefix "$RESULTS_DIR/$name" >> "$LOG" 2>&1
    echo "[$(date +%H:%M:%S)] finished $name" | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] cooldown ${COOLDOWN}s" | tee -a "$LOG"
    sleep "$COOLDOWN"
}

for name in $INSTANCES; do
    run_one "$name"
done
echo "[$(date +%H:%M:%S)] stage '$STAGE' complete: $INSTANCES" | tee -a "$LOG"
