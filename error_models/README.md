# Error models (initial set)

This folder contains the error models defined for NVIDIA GPU. Models have been derived by means of fault injection experiments on cudaDNN test applications running individual DNN operators and then mining frequently-observed patterns. Fault injection has been performed by means of  [NVBitFI](https://github.com/fernandoFernandeSantos/nvbitfi)), rmodels definition by means of the CNN error classifier tool (available in this [repository](https://github.com/D4De/cnn-error-classifier) ). We defined two sets of models, each one stored in a different folder:
- `faultythread_models`, derived by injecting a random value in a single CUDA thread
- `faultywarp_models`, derived by injecting a random value in all the CUDA threads of the same warp

# Error models (NVDLA set)

In 2025, a new set of error models was obtained via fault injection campaigns on the NVDLA accelerator. These campaigns targeted convolutional layers accelerated in different configurations of the accelerator. The new set is in `conv_models`, where each folder contains the models associated with a configuration and a `unique_complete_df.xlsx` Excel file, which is crucial for the proper operation of the simulator.

In addition, `injection_campaign_postprocessing` contains scripts used to derive the final error models from the raw output of the Error Classifier. The next section explains how these scripts work.

## Running injection campaign postprocessing to obtain Error Models

The scripts in `injection_campaign_postprocessing` transform the Error Classifier outputs into the final Error Models and, importantly, produce a `.xlsx` file, usually called `unique_complete_df.xlsx`; this file is used by the experiment scripts to interpolate the available error models when selecting one for each injectable layer of a network. There is no implementation-specific reason for this file to be in Excel format, it is just a matter of convenience in case it needs to be manually inspected and edited.

Once the injection campaign results for a configuration are ready, create a postprocessing output directory to store the results (usually, the directory name starts with `models_`, e.g. `models_8x8_int8`) and copy `postprocessing_config_example.yaml` into it. Rename this file if you want and edit it to fit the configuration and networks you have available; refer to the comments in the file to know what each fields does. Then, run (from the CLASSES root directory):
```
python -m error_models.injection_campaign_postprocessing_postprocess_step1 <path/to/YAML/configuration/file>
```
This first step produces an intermediate `step1_complete_df.xlsx` file in the postprocessing directory. The script will print a set of instructions to follow before running step 2: inspect the produced Excel file and identify possible outlier layers to be discarded (note that several Z-score values have already been calculated to make this process quicker). If you find any outlier, delete the corresponding row(s) from the first sheet of the file to discard it from the final set of models, then save the file and run:
```
python -m error_models.injection_campaign_postprocessing_postprocess_step2 <path/to/YAML/configuration/file>
```
This step produces, aside from `step2_unique_complete_df.xlsx`, a `reconstruction_test_df.xlsx` file. Before you proceed, examine this file and look at the rightmost column: a cell which displays "True" indicates that the model in the corresponding row was unable to be accurately reconstructed via interpolation of the other models. If you do not need to keep that model, consider deleting the its row from `step2_unique_complete_df.xlsx` before moving on.

Once you're done, move all error models from the postprocessing subdirectory `merged_error_models` to the final model storage directory you will use for the experiments, along with `unique_complete_df.xlsx`. By convention, this final directory is located in `error_models/conv_models` and is named after the full id of the hardware configuration from which the models were obtained (example: `nv_16x32_b1_dat-524288_wt-131072_int8`).

If you need to postprocess the results from many configurations, consider editing and using the utility script `postprocess_all.sh` after you have created all YAML configuration files.