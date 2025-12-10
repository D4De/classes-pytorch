#!/usr/bin/bash

compress_saved_outputs() {
    tar -c -I 'xz -9 -T0' -f $1.tar.xz $1 &&
    rm -r $1
}

# NETWORKS=("res50_cifar10" "mobilenetv2_gtsrb" "alexnet_cifar10" "deeplabv3_oxfordpet")
NETWORKS=("res50_cifar10")
CONFIGS=("nv_8x8_b1_dat-524288_wt-32768_int8")
ERR=("160")
IN=("100")

for network in "${NETWORKS[@]}"; do
    for config in "${CONFIGS[@]}"; do
        for num_err in "${ERR[@]}"; do
            for num_input in "${IN[@]}"; do
                exp_dir=experiments/exp_${network}/${config}
                config_name=conf_${num_input}in_${num_err}err.yaml
                saved_outputs_dir=saved_rankings_${num_input}in_${num_err}err
                # run experiment
                python -m experiments.run_experiment ${exp_dir} -cf ${config_name} &&
                # change saved outputs dir name
                mv ${exp_dir}/outputs/saved_rankings ${exp_dir}/outputs/${saved_outputs_dir} &&
                # start compressing saved outputs in the background
                compress_saved_outputs ${exp_dir}/outputs/${saved_outputs_dir} &
            done
        done
    done
done