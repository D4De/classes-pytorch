#!/usr/bin/bash
# SET EXPERIMENT VARIABLES HERE

# --- DIRECTORIES ---
export CLASSES_DIR="/home/miele/WORKSPACE/classes-simulator"
export EXP_DIR="${CLASSES_DIR}/experiments"

# --- NETWORKS ---
# network list is ("alexnet_cifar10" "mobilenetv2_gtsrb" "res50_cifar10" "deeplabv3_oxfordpet")
export NETWORKS=("alexnet_cifar10" "mobilenetv2_gtsrb" "res50_cifar10" "deeplabv3_oxfordpet")
export CONFIGS=("nv_16x32_b1_dat-524288_wt-131072_int8")
export IN=("100")
export ERR=("160")

# --- EXPERIMENT CONFIGURATIONS ---
export ERROR_MODELS_DIR="${CLASSES_DIR}/error_models/conv_models"
export HW_CONFIG_ID="16x32_int8"
export MODELS_DF_NAME="unique_complete_df.xlsx"
export USE_SINGLE_BATCH="True"
export UNIFORM_SPATIAL_CLASSES="True"
export SDC_FREQUENCIES_FILE_NAME="SDC_frequencies.yaml"
export TOLERANCE="0.001"
export COMPUTE_SINGLE_METRICS="False"
export NUM_THREADS="15"