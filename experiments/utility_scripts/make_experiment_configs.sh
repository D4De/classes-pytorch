#!/usr/bin/bash

workdir=`pwd`
source ${workdir}/script_config.sh
cd ${CLASSES_DIR}

make_config_file() {
# ARGS
# 1: configuration name
# 2: network and dataset name
# 3: number of errors per layer
# 4: num of inputs
# 5: experiment directory
# 6: output filepath
echo "\
experiment_name: $2 $4 inputs $3 errors
network_dataset_id: $2
hw_config_id: ${HW_CONFIG_ID}
error_models_path: ${ERROR_MODELS_DIR}/$1
error_models_df_path: ${ERROR_MODELS_DIR}/$1/${MODELS_DF_NAME}
use_single_batch: ${USE_SINGLE_BATCH}
batch_size: $4
uniform_spatial_classes: ${UNIFORM_SPATIAL_CLASSES}
num_faults_per_module: $3
fault_list_path: $5/fault_list_${3}errors.tar
SDC_frequency_dict_path: $5/${SDC_FREQUENCIES_FILE_NAME}
tolerance: ${TOLERANCE}
compute_single_metrics: ${COMPUTE_SINGLE_METRICS}
num_threads: ${NUM_THREADS}" > $6
}

for config in "${CONFIGS[@]}"; do
    for network in "${NETWORKS[@]}"; do
        for num_err in "${ERR[@]}"; do
            for num_input in "${IN[@]}"; do
                single_exp_dir=${EXP_DIR}/exp_${network}/${config}
                conf_file_path=${single_exp_dir}/conf_${num_input}in_${num_err}err.yaml

                # make experiment directory if necessary
                if [ ! -d "${single_exp_dir}" ]; then
                    mkdir -p "${single_exp_dir}"
                fi
                make_config_file $config $network $num_err $num_input $single_exp_dir $conf_file_path
            done
        done
    done
done