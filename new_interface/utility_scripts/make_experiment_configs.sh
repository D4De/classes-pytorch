#!/usr/bin/bash

# RUN FROM utility_scripts DIRECTORY

workdir=`pwd`
source ${workdir}/script_config.sh
cd ${CLASSES_DIR}

make_config_file() {
# ARGS
# 1: configuration name
# 2: short configuration name
# 3: network and dataset name
# 4: number of errors per layer
# 5: num of inputs
# 6: experiment directory
# 7: output filepath
echo "\
experiment_name: $3 $5 inputs $4 errors
network_dataset_id: $3
hw_config_id: $2
error_models_path: ${ERROR_MODELS_DIR}/$1
error_models_df_path: ${ERROR_MODELS_DIR}/$1/${MODELS_DF_NAME}
use_single_batch: ${USE_SINGLE_BATCH}
batch_size: $5
uniform_spatial_classes: ${UNIFORM_SPATIAL_CLASSES}
num_faults_per_module: $4
fault_list_path: $6/fault_list_${4}errors.tar
SDC_frequencies_path: $6/${SDC_FREQUENCIES_FILE_NAME}
tolerance: ${TOLERANCE}
compute_single_metrics: ${COMPUTE_SINGLE_METRICS}
num_threads: ${NUM_THREADS}
force_single_channel: ${FORCE_SINGLE_CHANNEL}" > $7
}

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    short_config="${SHORT_IDS[$i]}"
    for network in "${NETWORKS[@]}"; do
        for num_err in "${ERR[@]}"; do
            for num_input in "${IN[@]}"; do
                single_exp_dir=${EXP_DIR}/exp_${network}/${config}
                conf_file_path=${single_exp_dir}/conf_${num_input}in_${num_err}err.yaml

                # make experiment directory if necessary
                if [ ! -d "${single_exp_dir}" ]; then
                    mkdir -p "${single_exp_dir}"
                fi
                make_config_file $config $short_config $network $num_err $num_input $single_exp_dir $conf_file_path
            done
        done
    done
done