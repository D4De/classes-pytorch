#!/usr/bin/bash

compress_saved_outputs() {
    tar -c -I 'xz -9 -T0' -f $1.tar.xz $1 &&
    rm -r $1
}

source ./experiments/script_config.sh

for config in "${CONFIGS[@]}"; do
    for network in "${NETWORKS[@]}"; do
        for num_err in "${ERR[@]}"; do
            for num_input in "${IN[@]}"; do
                exp_dir=experiments/exp_${network}/${config}
                saved_outputs_dir=saved_rankings_${num_input}in_${num_err}err
                # start compressing saved outputs in the background
                echo "Starting tar compression of $network for configuration $config"
                compress_saved_outputs ${exp_dir}/outputs/${saved_outputs_dir} &
            done
        done
    done
done