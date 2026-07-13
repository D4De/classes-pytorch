#!/usr/bin/bash

TOPDIR=`pwd` # LAUNCH FROM additional_pictures DIRECTORY

OUTDIR="/home/miele/WORKSPACE/results-storage/pictures/extra_heatmaps_and_plots"
ORGANIZED_PLOTS_DIR="/home/miele/WORKSPACE/results-storage/pictures/results_dfs/cross_layer_aggregates" # contains one directory per network; each directory contains the applev_class_groups csv files
RAW_LAYER_CSV="$/home/miele/WORKSPACE/results-storage/network_reports/raw_layer_results.csv"
RAW_UNITS_CSV="$/home/miele/WORKSPACE/results-storage/network_reports/raw_unit_results.csv"

python ./extra_heatmaps_and_plots.py --outdir ${OUTDIR} --layer_csv_path ${RAW_LAYER_CSV} --unit_csv_path ${RAW_UNITS_CSV} --other_results_dir ${OUTDIR} --organized_plots_dir ${ORGANIZED_PLOTS_DIR} --show_annotations