#!/usr/bin/bash

SCRIPT='make_experiment_blobs.py'

OUTPUT_DIR='.'
CLASSES_BASE_DIR='/home/miele/WORKSPACE/results-storage/error_simulation/experiments'
HYPERS_BASE_DIR='/home/miele/WORKSPACE/results-storage/network_hypers'

NETWORKS=(
    'alexnet_cifar10' \
    'deeplabv3_oxfordpet' \
    'mobilenetv2_gtsrb' \
    'res9_cifar10' \
    'res50_cifar10' \
    'yolov11_coco' \
)
EXP_DIRS=(
    'exp_alexnet_cifar10' \
    'exp_deeplabv3_oxfordpet_v1' \
    'exp_mobilenetv2-large_gtsrb' \
    'exp_res9_cifar10' \
    'exp_res50_cifar10' \
    'exp_yolov11_coco' \
)
CONFIGS=(
  'nv_8x8_b1_dat-524288_wt-32768_int8' \
  'nv_16x16_b1_dat-524288_wt-65536_int8' \
  'nv_16x32_b1_dat-524288_wt-131072_int8' \
  'nv_8x8_b1_dat-1048576_wt-65536_int16' \
  'nv_32x16_b1_dat-1048576_wt-131072_int16' \
  'nv_32x32_b1_dat-1048576_wt-262144_int16' \
  'nv_32x8_b1_dat-2097152_wt-131072_int32' \
  'nv_8x16_b1_dat-2097152_wt-262144_int32' \
  'nv_32x32_b1_dat-2097152_wt-524288_int32' \
)
SPATIAL_CLASSES=(   # in the applev files, only consider these spatial classes
    'Single' \
    'FullChannels' \
    'MultiChannelBlock' \
    'BulletWake' \
    'Rectangles' \
    'ShatteredChannel' \
    'QuasiShatteredChannel' \
    'SameRow' \
    'SingleBlock' \
    'Skip4' \
    'SingleChannelRandom' \
)

python ${SCRIPT} --output_dir ${OUTPUT_DIR} --classes_base_dir ${CLASSES_BASE_DIR} --hypers_base_dir ${HYPERS_BASE_DIR} \
    --networks ${NETWORKS[@]} --exp_dirs ${EXP_DIRS[@]} --configs ${CONFIGS[@]} --spatial_classes ${SPATIAL_CLASSES[@]}