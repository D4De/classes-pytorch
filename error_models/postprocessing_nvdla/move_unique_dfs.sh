#!/usr/bin/bash

BASE_DIR="/home/miele/WORKSPACE/classes-simulator/error_models"
POSTPROCESS_DIR="${BASE_DIR}/injection_campaign_postprocessing"
MODELS_DIR="${BASE_DIR}/conv_models"

models=(\
    "8x8_int8" \
    "8x8_int16" \
    "8x16_int32" \
    "16x16_int8" \
    "16x32_int8" \
    "32x8_int32" \
    "32x16_int16" \
    "32x32_int16" \
    "32x32_int32" \
)

configs=(\
   nv_8x8_b1_dat-524288_wt-32768_int8 \
   nv_8x8_b1_dat-1048576_wt-65536_int16 \
   nv_8x16_b1_dat-2097152_wt-262144_int32 \
   nv_16x16_b1_dat-524288_wt-65536_int8 \
   nv_16x32_b1_dat-524288_wt-131072_int8 \
   nv_32x8_b1_dat-2097152_wt-131072_int32 \
   nv_32x16_b1_dat-1048576_wt-131072_int16 \
   nv_32x32_b1_dat-1048576_wt-262144_int16 \
   nv_32x32_b1_dat-2097152_wt-524288_int32 \
)

for i in "${!models[@]}"; do 
  echo "${models[$i]} - ${configs[$i]}"
  cp "${POSTPROCESS_DIR}/models_${models[$i]}/unique_complete_df.xlsx" "${MODELS_DIR}/${configs[$i]}"
done