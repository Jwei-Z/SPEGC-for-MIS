#!/usr/bin/env bash

# Source Model Training Script for SPEGC
# This script trains the source model on the source domain datasets.

# =============================================================================
# CONFIGURATION - Set your GPU ID and output directory name below
# =============================================================================
CUDA_VISIBLE_DEVICES=0
OUTPUT_NAME="fundus_source"  # Checks and outputs will be saved under output/<OUTPUT_NAME>

# Configure environment variables to suppress excessive framework warnings
export DETECTRON2_VERBOSITY=0
export FVCORE_CACHE_VERBOSITY=0

echo "[TRAIN] Starting Source Model Training..."
echo "[TRAIN] Configuration file: configs/seg_res50fpn_source.yaml"
echo "[TRAIN] Output directory: output/${OUTPUT_NAME}"

# Execute training command
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train_net.py \
      --num-gpus 1 \
      --config configs/seg_res50fpn_source.yaml \
      OUTPUT_DIR "output/${OUTPUT_NAME}"

echo "[TRAIN] Training process completed. Checkpoints are saved under output/${OUTPUT_NAME}/"
