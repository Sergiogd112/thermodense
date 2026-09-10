#!/bin/bash
# Submit with qsub -S /bin/bash -pe smp 2; no changes to existing job environments.
set -euo pipefail

: "${RUN_ROOT:?Set RUN_ROOT to the isolated global-activity run directory}"
PYTHON="${BENCHMARK_PYTHON:-$HOME/.local/share/thermodense/source/.venv/bin/python}"
export PYTHONPATH="$RUN_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2

case "${1:-}" in
    parcorr)
        case "${2:-}" in all|low|moderate|high) ;; *) exit 2 ;; esac
        exec "$PYTHON" -m thermodense.benchmarks.global_activity_pcmci \
            --manifest "$RUN_ROOT/input/manifest.json" \
            --output "$RUN_ROOT/outputs/parcorr-$2" --group "$2"
        ;;
    polynomial)
        exec "$PYTHON" -m thermodense.benchmarks.global_activity_polynomial \
            --manifest "$RUN_ROOT/input/manifest.json" \
            --output "$RUN_ROOT/outputs/polynomial" --lag 180 --epochs 15 \
            --threads 2 --batch-size 64 --lambdas 0.0001,0.001,0.01 --seed 0
        ;;
    *)
        >&2 printf 'Usage: %s parcorr {all|low|moderate|high} | polynomial\n' "$0"
        exit 2
        ;;
esac
