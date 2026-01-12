#!/usr/bin/bash

workdir=`pwd`
source ${workdir}/script_config.sh
cd ${CLASSES_DIR}

SCRIPT=tools/rescale_experiment_results.py

METRICS_OUTPUT_FILENAME="experiment_metrics.xlsx"
VULNERABILITY_OUTPUT_FILENAME="vulnerability.xlsx"

SHORT_CONFIGS=(\
   8x8_int8 \
   16x16_int8 \
   16x32_int8 \
   8x8_int16 \
   32x16_int16 \
   32x32_int16 \
   32x8_int32 \
   8x16_int32 \
   32x32_int32 \
)
NUM_CONFIGS=${#CONFIGS[@]}

for network in "${NETWORKS[@]}"; do
    for i in "${!CONFIGS[@]}"; do
        config=${CONFIGS[i]}
        short=${SHORT_CONFIGS[i]}
        python $SCRIPT experiments/exp_${network}/${config} ${network} ${short} experiments/aggregated_results/${VULNERABILITY_OUTPUT_FILENAME} experiments/aggregated_results/${METRICS_OUTPUT_FILENAME} &&
        echo "${network}: configuration $i done"
    done
done