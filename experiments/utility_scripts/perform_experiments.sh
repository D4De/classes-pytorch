#!/usr/bin/bash

# RUN FROM utility_scripts DIRECTORY

workdir=`pwd`
source ${workdir}/script_config.sh

cd ${CLASSES_DIR}

for config in "${CONFIGS[@]}"; do
    for network in "${NETWORKS[@]}"; do
        for num_err in "${ERR[@]}"; do
            for num_input in "${IN[@]}"; do
                single_exp_dir=${EXP_DIR}/exp_${network}/${config}
                config_name=conf_${num_input}in_${num_err}err.yaml
                saved_outputs_dir=saved_rankings_${num_input}in_${num_err}err
                # run experiment - add flag -rf to regenerate fault lists and sdc frequency files if necessary
                python -m experiments.run_experiment ${single_exp_dir} -cf ${config_name} &&
                # change saved outputs dir name
                mv ${single_exp_dir}/outputs/saved_rankings ${single_exp_dir}/outputs/${saved_outputs_dir}
            done
        done
    done
done