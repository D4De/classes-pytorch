#!/usr/bin/bash

WORKSPACE_DIR="/home/miele/WORKSPACE"

SCRIPT="${WORKSPACE_DIR}/classes-simulator/tools/model_extraction/get_module_hyperparameters.py"
SAVEDIR="${WORKSPACE_DIR}/results-storage/network_reports"
mkdir "${SAVEDIR}"

TCAD_DIR="${WORKSPACE_DIR}/tcad2025"

NETWORKS=(\
    "alexnet" \
    "deeplabv3" \
    "mobilenetv2-large" \
    "res9" \
    "res50" \
    "yolov11" \
)
DATASETS=(\
    "cifar10" \
    "oxfordpet" \
    "gtsrb" \
    "cifar10" \
    "cifar10" \
    "coco"
)

DATASET_PATHS=(\
    "${WORKSPACE_DIR}/pytorch-models-data/datasets/CIFAR10" \
    "${TCAD_DIR}/models/other_nets/segmentation/oxfordpet/oxfordpet_data" \
    "${TCAD_DIR}/models/other_nets/classification/gtsrb/gtsrb_data" \
    "${WORKSPACE_DIR}/pytorch-models-data/datasets/CIFAR10" \
    "${WORKSPACE_DIR}/pytorch-models-data/datasets/CIFAR10" \
    "${TCAD_DIR}/models/other_nets/detection/coco/coco_data" \
)
TRAIN_PATHS=(\
    "${WORKSPACE_DIR}/pytorch-models-data/nets/alexnet" \
    "${TCAD_DIR}/models/other_nets/segmentation/oxfordpet/models/" \
    "${TCAD_DIR}/models/other_nets/classification/gtsrb/models/" \
    "${WORKSPACE_DIR}/pytorch-models-data/nets/resnet9" \
    "${WORKSPACE_DIR}/pytorch-models-data/nets/resnet50" \
    "${TCAD_DIR}/models/other_nets/detection/coco/models" \
)

for i in {0..5}; do
    echo ${NETWORKS[i]}
    export TORCH_DATASETPATH=${DATASET_PATHS[i]}
    export TORCH_TRAINPATH=${TRAIN_PATHS[i]}

    python ${SCRIPT} ${TCAD_DIR} ${NETWORKS[i]} ${DATASETS[i]} ${SAVEDIR}
done