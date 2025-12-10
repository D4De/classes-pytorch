#!/usr/bin/bash

# NETWORKS=("resnet50_cifar10" "mobilenetv2_gtsrb" "alexnet_cifar10" "deeplabv3_oxfordpet")
NETWORKS=("mobilenetv2_gtsrb" "resnet50_cifar10" "deeplabv3_oxfordpet")
CONFIGS=("nv_8x8_b1_dat-524288_wt-32768_int8")
ERR=("1000")
IN=("100")

for network in "${NETWORKS[@]}"; do
    for config in "${CONFIGS[@]}"; do
        for num_err in "${ERR[@]}"; do
            for num_input in "${IN[@]}"; do
                python -m experiments.run_experiment experiments/exp_${network}/${config} -cf conf_${num_input}in_${num_err}err.yaml &&
                mv experiments/exp_${network}/${config}/outputs/saved_rankings experiments/exp_${network}/${config}/outputs/saved_rankings_${num_input}in_${num_err}err
            done
        done
    done
done