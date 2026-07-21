# Postprocessing scripts
The scripts in this directory are used to collect experimental results and to produce several plots from them. If you have the `results-storage` directory available, you can see many output examples.

`make_experiment_blobs.py` collects results from the applev files produced by the error simulation experiments and combines them with layer hyperparameters to produce a single blob .csv file for each network. For convenience, you can use `make_experiment_blobs.sh` to invoke the script, but be sure to adjust all paths appropriately. The script expects the following arguments:
- `output_dir`: path to the directory to save the resulting blob files in
- `classes_base_dir`: path to the directory containing the outputs of error simulation experiments (this directory is usually called `experiments` and contains one `exp_*` subdirectory for each tested network)
- `hypers_base_dir`: path to the directory containing the .csv files listing the layer hyperparameters for each network. These files are produced via the scripts in `tools/model_extraction`
- `networks`: list of network names
- `exp_dirs`: names of the experiment directories (`exp_*`) reachable from `classes_base_dir`; network order should be the same one defined by `networks` (see `make_experiment_blobs.sh` for an example)
- `configs`: complete ids of the NVDLA configurations used in the experiments. Should match the directory names in each experiment directory (`exp_*`)
- `spatial_classes`: names (in Pascal case) of the spatial classes to consider in the applev files; all other spatial classes will be ignored

`build_results_dfs.py` produces a series of pandas dataframes from the experimental results and plots these dataframes as heatmaps using a green-to-red colormap to highlight especially high frequencies. This script uses the parameters specified in `args.yaml`, so ensure that all of them, especially the paths, are set correctly. In general, the script needs to access the experimental results (same as `make_experiment_blobs.py`), the CLASSES error models, and some fault injection results (check `tcad2025/outdir/reports` if you have the directory available).
Look at the files in `/home/miele/WORKSPACE/results-storage/pictures/results_dfs` to see what types of pictures are produced. A few words about each type:
- `sdc_and_classes` contains heatmaps representing the occurrence frequencies of spatial classes across error models (`class_distribution` files) and also the SDC frequencies associated with each model. Labels on the y-axis identify the error models with their set of hyperparameter values; they are, in order, channels_out (K), channels_in (C), input_size (W), kernel_size (R), and padding
- `cross_layer_aggregates` contains heatmaps showing the different FIT values for each network, the distribution of layer frequencies of critical failure multiplied by spatial class occurrence frequencies (`applev_class_groups`), the distribution of total criticality values for each layer (`total_crit`; this is the sum of all critical failure * class occurrence products), and the delta criticality plots (these are the average layer deviations of total criticality from the network average in the various configurations; they can be used to spot particularly vulnerable configurations)

`count_experiment_errors.py` is a simple script that scans applev files and counts the total amount of errors generated for each network, grouping the counts by bitwidth. Edit the script directly to set its main parameters at the top if you need to run it.

`gather_layer_aggregation_pieces.py` produces a csv file containing the area and liveness factors for each unit, layer, and configuration. Edit the paths at the top of the script if you need to run it. It takes as input the results of the fault injection campaigns and the global `results.csv` aggregation file, which can be found in `results-storage/network_reports`.

`raw_layer_results.py` is used to produce `raw_unit_results.csv` and `raw_layer_results.csv`, storing results for all layers and units, respectively. Edit the paths at the top of the script if you need to run it. It takes as input the `results.csv` blob file containing all fault injection results, the fault injection results themselves, the results of the application-level error simulation experiments (and the additional single channel experiments, though this can be removed if necessary), and the csv files containing the network hyperparameters. Look at the various directories in `results-storage` to find examples of these files.

Directory `additional_pictures` contains more scripts to produce plots:
- `applev_plots.py` creates plots that show the layer distribution of hyperparameters and critical failure rates throughout a CNN.  Edit the paths at the top of the script if you need to run it. It takes as input the application blobs and the network reports called `*_final.csv`, which can be found in `results-storage/network_reports`.
- `extra_heatmaps_and_plots.py` generates observability and susceptibility heatmaps and scatter plots, both for single units and for entire CNN layers. It can be more conveniently be invoked via `make_heatmaps.sh`, whose paths should be adjusted if necessary. It uses the `applev_class_groups` csv files stored in `results-storage/pictures/results_dfs/cross_layer_aggregates`, as well as the raw layer and units csv reports stored in `results-storage/network_reports`.
- `individual_plots.py` makes additional plots related to single layers or specific groups of layers, rather than the entire layer set. Take a look at the outputs in `results-storage/pictures` to get a better idea. Edit the paths at the top of the script if you need to run it. It takes as input the global `results.csv` file found in `results-storage/network_reports` and storing the results of all fault injection campaigns, and `raw_layer_results.csv`.
- `layer_rankings.py` produces configuration rankings of both average unit observability and layer susceptibility for all injected layers. The rankings are saved in a yaml file and subsequently plotted for easy visual inspection. Edit the paths at the top of the script if you need to run it. It takes as input `raw_layer_results.csv`.
- `layer_time_area_charts.py` produces aggregation coefficients plots (so-called "rectangle charts") for each layer, hardware unit, and configuration. These charts show one rectangle per unit; the width is the liveness (active time) of the unit, while the height is the area (fraction of registers belonging to the unit). Note that only rectangles whose area is sufficient for the plot are shown. Edit the paths at the top of the script if you need to run it. It takes as input `aggregation_pieces.csv`, stored in `results-storage/network_reports`.
- `network_fit_graphs.py` produces simple bar charts for every network, showing the distribution of its FIT values across the NVDLA configurations. Edit the paths at the top of the script if you need to run it. It takes as input `network_fit_values.csv`, stored in `results-storage/pictures/results_dfs/cross_layer_aggregates`.
- `spatial_class_distribution.py` plots stacked bar charts showing the average spatial class frequencies for each hardware unit, as well as the total average frequencies. One such chart is produced for every NVDLA configuration. Edit the paths at the top of the script if you need to run it. It takes as input `raw_layer_results.csv` and the individual unit `results_*` csv files stored in `results-storage/network_reports`.

# An example
This example shows how the most important postprocessing scripts work by executing them on the results of fault injection and error simulation for AlexNet (in one NVDLA configuration).

Download the necessary files [here](https://miele.faculty.polimi.it/postprocessing_example.tar.xz) and extract wherever you want with
```
tar -x -I xz -f postprocessing_example.tar.xz
```
The `experiments` directory contains the essential output files from an AlexNet experiment carried out using the error models for the 8x8_int8 NVDLA configuration.
The `experiments_single_channel` directory contains similar results obtained by forcing the error simulation experiments to single channel errors only.
The `network_hypers` directory contains a csv file listing the hyperparameters of AlexNet's convolutional layers, which are generally produced using the scripts in `tools/model_extraction`.
The `network_reports` directory contains several additional files, either obtained from these postprocessing scripts or by using those of the NVDLA RTL-level simulator.
The `outdir` directory uses the same structure as the output directory of the NVDLA simulator and contains some basic results for AlexNet.

You can now use the downloaded files as follows:
1) `make_experiment_blobs`: edit `make_experiment_blobs.sh` and set
- `OUTPUT_DIR` to a directory of your choice
- `CLASSES_BASE_DIR` to the `experiments` directory among the files you just downloaded
- `HYPERS_BASE_DIR` to the `network_hypers` directory among the files you just downloaded
- `NETWORKS` to the single entry `alexnet_cifar10`
- `EXP_DIRS` to the single entry `exp_alexnet_cifar10` (name of the network directory within `experiments`)
- `CONFIGS` to the single entry matching the configuration ('nv_8x8_b1_dat-524288_wt-32768_int8')
- leave `SPATIAL_CLASSES` as-is
Run `./make_experiment_blobs.sh` to create the `alexnet_cifar10_application.csv` aggregated file.

2) `build_results_dfs`: edit `args.yaml` and set
- `error_models_base_path` to the CLASSES directory containing the error models (`error_models/nvdla_models`)
- `experiments_base_path` to the `experiments` directory you downloaded
- `final_reports_base_path` to the `network_reports` directory you downloaded
- `output_dir` to an output directory of your choice
- `network_dataset_ids` to the single entry `alexnet_cifar10`
- `configuration_ids` to the single entry corresponding to the configuration ('nv_8x8_b1_dat-524288_wt-32768_int8')
- `short_configuration_ids` to the single entry corresponding to the short configuration id ('8x8_int8')
Run `python build_results_dfs.py <path/to/args.yaml>` to produce a set of aggregation dataframes and corresponding heatmaps.

3) `gather_layer_aggregation_pieces`: edit the script directly and set
- `benchmarks_source_dir` to the `outdir` directory you downloaded
- `results_path` so that it leads to the `results.csv` file in `network_reports`
- `output_path` to the location you want for the output csv file (generally, the name is `aggregation_pieces.csv` or something similar)
- `allowed_configs` to the only configuration we're considering ('nv_8x8_b1_dat-524288_wt-32768_int8')
Then simply run the script to create the csv file.

4) `raw_layer_results`: edit the script directly and set
- `results_path` so that it leads to the `results.csv` file
- `benchmarks_path` so that it leads to the `outdir/benchmarks` directory
- `classes_exp_path` so that it leads to the `experiments` directory
- `classes_extra_results_path` so that it leads to the `experiments_single_channel` directory
- `networks` to the single entry 'alexnet_cifar10'
- `network_hypers_path` to the `network_hypers` directory
- `out_layer_path` to the location you want for the `raw_layer_results.csv` file
- `out_units_path` to the location you want for the `raw_units_results.csv` file
- `configs` to the single entry corresponding to the configuration ('nv_8x8_b1_dat-524288_wt-32768_int8')
Run the script to create the two csv files.

You should now have all the files needed to run the other scripts in `additional_pictures`. Edit the parameters at the top of each script by following the input details explained in the previous section.
