#!/usr/bin/bash
# SET EXPERIMENT VARIABLES HERE

# --- DIRECTORIES ---
export CLASSES_DIR="/home/miele/WORKSPACE/classes-simulator"
export EXP_DIR="${CLASSES_DIR}/experiments"

# --- NETWORKS ---
# network list is ("alexnet_cifar10" "mobilenetv2_gtsrb" "mobilenetv2-large_gtsrb" "res50_cifar10" "res9_cifar10" "deeplabv3_oxfordpet" "yolov11_coco")
# configs list is (\
#    nv_8x8_b1_dat-524288_wt-32768_int8 \
#    nv_16x16_b1_dat-524288_wt-65536_int8 \
#    nv_16x32_b1_dat-524288_wt-131072_int8 \
#    nv_8x8_b1_dat-1048576_wt-65536_int16 \
#    nv_32x16_b1_dat-1048576_wt-131072_int16 \
#    nv_32x32_b1_dat-1048576_wt-262144_int16 \
#    nv_32x8_b1_dat-2097152_wt-131072_int32 \
#    nv_8x16_b1_dat-2097152_wt-262144_int32 \
#    nv_32x32_b1_dat-2097152_wt-524288_int32 \
#)

# export NETWORKS=("deeplabv3_oxfordpet")
# export CONFIGS=(\
#    nv_8x8_b1_dat-524288_wt-32768_int8 \
# #   nv_16x16_b1_dat-524288_wt-65536_int8 \
# #   nv_16x32_b1_dat-524288_wt-131072_int8 \
# )

export NETWORKS=("alexnet_cifar10" "mobilenetv2_gtsrb" "res50_cifar10" "deeplabv3_oxfordpet" "yolov11_coco")
export CONFIGS=(\
   nv_8x8_b1_dat-524288_wt-32768_int8 \
   nv_16x16_b1_dat-524288_wt-65536_int8 \
   nv_16x32_b1_dat-524288_wt-131072_int8 \
   nv_8x8_b1_dat-1048576_wt-65536_int16 \
   nv_32x16_b1_dat-1048576_wt-131072_int16 \
   nv_32x32_b1_dat-1048576_wt-262144_int16 \
   nv_32x8_b1_dat-2097152_wt-131072_int32 \
   nv_8x16_b1_dat-2097152_wt-262144_int32 \
   nv_32x32_b1_dat-2097152_wt-524288_int32 \
)

export IN=("100")
export ERR=("160")

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