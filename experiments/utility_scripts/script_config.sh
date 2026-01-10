#!/usr/bin/bash
# SET EXPERIMENT VARIABLES HERE

# --- DIRECTORIES ---
export CLASSES_DIR="/home/alberto/MasterThesis/classes"
export EXP_DIR="${CLASSES_DIR}/experiments"

# --- NETWORKS ---
# network list is ("alexnet_cifar10" "mobilenetv2_gtsrb" "res50_cifar10" "deeplabv3_oxfordpet")
export NETWORKS=("mobilenetv2_gtsrb")
export CONFIGS=("nv_8x8_b1_dat-524288_wt-32768_int8")
export IN=("100")
export ERR=("1")

# --- EXPERIMENT CONFIGURATIONS ---
export ERROR_MODELS_DIR="${CLASSES_DIR}/error_models/conv_models"
export HW_CONFIG_ID="8x8_int8"
export MODELS_DF_NAME="unique_complete_df.xlsx"
export USE_SINGLE_BATCH="True"
export UNIFORM_SPATIAL_CLASSES="True"
export SDC_FREQUENCIES_FILE_NAME="SDC_frequencies.xlsx"
export TOLERANCE="0.001"
export COMPUTE_SINGLE_METRICS="False"
export NUM_THREADS="15"