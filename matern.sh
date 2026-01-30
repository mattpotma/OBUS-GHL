#!/bin/bash
set -euo pipefail

source ~/.asdf/asdf.sh

# Ensure Python can find ghlobus
export PYTHONPATH=.

# Disable GPU
export CUDA_VISIBLE_DEVICES=""

# Safe LD_LIBRARY_PATH update (won't error if unset)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(python3-config --prefix)/lib"

# ---- Paths ----
C3_ROOT="/home/ella/src/OBUS-GHL/dicoms/c3"
LCP_ROOT="/home/ella/src/OBUS-GHL/dicoms/lcp"

RESULTS_DIR="/home/ella/src/OBUS-GHL/results_ga_benchmark"
mkdir -p "$RESULTS_DIR"

MASTER_CSV="$RESULTS_DIR/ga_benchmark_tflite_all.csv"
echo "patient,sweep,probe,pytorch_full,tflite_uniform_50,tflite_matern_50" > "$MASTER_CSV"

# ---- Models ----
TFLITE_DIR="./tflite_models"
TFLITE_MODEL="ghlobus_ga_model_50.tflite"

PYTORCH_MODEL="checkpoints/ga_experiment_mnv2_e39.ckpt"
CNN_NAME="MobileNet_V2"

run_probe() {
    local PROBE_NAME="$1"
    local EXAM_ROOT="$2"

    echo "==== Processing probe: $PROBE_NAME ===="

    run_tflite() {
        local cmd_output
        local status
        set +e
        cmd_output=$("$@" 2>&1)
        status=$?
        set -e
        if [[ $status -ne 0 ]]; then
            echo "  [ERROR] Command failed (exit $status): $*" >&2
            echo "$cmd_output" >&2
            echo ""
            return 0
        fi
        echo "$cmd_output"
    }

    for examdir in "$EXAM_ROOT"/patient*; do
        [[ -d "$examdir" ]] || continue

        local patient
        patient=$(basename "$examdir")
        echo "  Patient: $patient"

        for sweep in "$examdir"/*.dcm; do
            [[ -f "$sweep" ]] || continue

            local sweep_name
            sweep_name=$(basename "$sweep")
            echo "    Sweep: $sweep_name"

            # ---------- TFLite Uniform 50 ----------
            tfl_uniform_out=$(run_tflite uv run ghlobus/inference/tflite_inference.py \
                --task=GA \
                --tflite_dir="$TFLITE_DIR" \
                --dicom="$sweep" \
                --outdir=./test_tflite \
                --tflite_model_name="$TFLITE_MODEL" \
                --frame_mode=uniform_50)

            tfl_uniform_ga=$(echo "$tfl_uniform_out" | grep "Predicted GA (Days):" | awk '{print $4}')

            # ---------- TFLite Matern 50 ----------
            tfl_matern_out=$(run_tflite uv run ghlobus/inference/tflite_inference.py \
                --task=GA \
                --tflite_dir="$TFLITE_DIR" \
                --dicom="$sweep" \
                --outdir=./test_tflite \
                --tflite_model_name="$TFLITE_MODEL" \
                --frame_mode=matern_50)

            tfl_matern_ga=$(echo "$tfl_matern_out" | grep "Predicted GA (Days):" | awk '{print $4}')

            # ---------- PyTorch Full Sequence ----------
            pyt_out=$(run_tflite uv run ghlobus/inference/inference.py \
                --task=GA \
                --modelpath="$PYTORCH_MODEL" \
                --cnn_name="$CNN_NAME" \
                --file="$sweep")

            pyt_ga=$(echo "$pyt_out" | grep "Predicted GA (Days):" | awk '{print $4}')

            # Append to master CSV
            echo "$patient,$sweep_name,$PROBE_NAME,$pyt_ga,$tfl_uniform_ga,$tfl_matern_ga" >> "$MASTER_CSV"
        done
    done
}

# Run for each probe independently
run_probe "C3"  "$C3_ROOT"
run_probe "LCP" "$LCP_ROOT"

echo "Done! Master CSV saved at: $MASTER_CSV"
