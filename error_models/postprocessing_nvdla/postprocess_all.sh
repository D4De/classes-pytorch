#!/usr/bin/bash

## RUN FROM CLASSES ROOT DIRECTORY

# models=(\
#     "8x8_int8" \
#     "8x8_int16" \
#     "8x16_int32" \
#     "16x16_int8" \
#     "16x32_int8" \
#     "32x8_int32" \
#     "32x16_int16" \
#     "32x32_int16" \
#     "32x32_int32" \
# )

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

for model_id in "${models[@]}"; do
    echo "${model_id}"
    python -m error_models.postprocessing_nvdla.postprocess_step1 "error_models/injection_campaign_postprocessing/models_${model_id}/postprocessing_config.yaml" &&
    python -m error_models.postprocessing_nvdla.postprocess_step2 "error_models/injection_campaign_postprocessing/models_${model_id}/postprocessing_config.yaml"
done
