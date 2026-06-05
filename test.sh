#!/bin/bash

# Test script for SPEGC with TTT functionality and model configuration switching
# This script automatically switches between different model configurations (A/B/C/D/E)
# Each configuration uses corresponding weight file and dataset combination

# =============================================================================
# CONFIGURATION - Change this to switch between models and datasets
# =============================================================================
MODEL_CONFIG="B"  # Options: A, B, C, D, E

# Configure environment variables
CUDA_VISIBLE_DEVICES=0
export DETECTRON2_VERBOSITY=0
export FVCORE_CACHE_VERBOSITY=0

# Configuration mapping
case $MODEL_CONFIG in
    "A") DATASET_LINE=5; MODEL_FILE="model_A.pth" ;;
    "B") DATASET_LINE=7; MODEL_FILE="model_B.pth" ;;
    "C") DATASET_LINE=9; MODEL_FILE="model_C.pth" ;;
    "D") DATASET_LINE=11; MODEL_FILE="model_D.pth" ;;
    "E") DATASET_LINE=13; MODEL_FILE="model_E.pth" ;;

    # "A") DATASET_LINE=16; MODEL_FILE="model_A.pth" ;; 
    # "B") DATASET_LINE=17; MODEL_FILE="model_B.pth" ;;
    # "C") DATASET_LINE=18; MODEL_FILE="model_C.pth" ;;
    # "D") DATASET_LINE=19; MODEL_FILE="model_D.pth" ;;
    # "E") DATASET_LINE=16; MODEL_FILE="model_E.pth" ;;
    *) echo "[ERROR] Invalid MODEL_CONFIG: $MODEL_CONFIG. Use A, B, C, D, or E."; exit 1 ;;
esac

# Build file path
FILE_PATH="weights/fundus_source/$MODEL_FILE"

echo "[TEST] Starting SPEGC test with configuration: $MODEL_CONFIG"
echo "[TEST] Using weight file: $MODEL_FILE"
echo "[TEST] Using config: configs/test_segment.yaml"
echo "[TEST] Expected to see TTT debug messages during execution"

# Update YAML configuration - comment out all dataset lines first
sed -i 's/^  TEST:/  # TEST:/g' configs/test_segment.yaml
sed -i 's/^  # # TEST:/  # TEST:/g' configs/test_segment.yaml

# Activate the selected configuration line
sed -i "${DATASET_LINE}s/^  # TEST:/  TEST:/" configs/test_segment.yaml

# Define base command
BASE_COMMAND="python train_net.py --eval-only --config configs/test_segment.yaml MODEL.WEIGHTS"

# Build complete command
FULL_COMMAND="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${BASE_COMMAND} ${FILE_PATH}"

echo "[TEST] Running: ${FULL_COMMAND}"

# Execute command
eval "${FULL_COMMAND}"

echo "[TEST] Test completed for configuration $MODEL_CONFIG"