# Experiment setup and execution (NVDLA)

This directory contains all scripts needed to set up and run error simulation experiments using sets of error models specifically derived from NVDLA fault injection campaigns. However, only a few scripts and files actually depend on the structure of NVDLA campaign outputs and the following methods can be easily extended to results from other sources. Similarly, these methods currently only work for convolutional layers, but extensions should be simple to implement.

## Setting up and running an experiment

Begin by choosing a network/dataset pair to run an experiment on and decide how to structure the experiment. Our running example is as follows:
- network: AlexNet
- dataset: CIFAR10
- configuration: nv_8x8_b1_dat-524288_wt-32768_int8
- batch size (number of input images): 100
- number of errors (per layer and per spatial classes): 160
Each network/dataset pair is associated with an id that's used by the experiment scripts and is checked by different scripts, so ensure that you type it correctly. In our case, the id is `alexnet_cifar10`. The full list of ids is defined in `network_getter.py`: it's the set of keys in the `available` dictionary.

If you don't already have them, make a `dataset_data` and a `weights` directory inside `experiments`; store the network weight files (typically `.pth` or `.pt` for PyTorch) for your network in `weights` and, if the dataset is not automatically downloaded from TorchVision or similar services, download it yourself and store it into `dataset_data` (an example of the latter case is the COCO dataset). The current versions of the experiment scripts download CIFAR10 automatically, so there is no need to download it manually; a weights file must still be provided, though.

Move into `utility_scripts` and edit `script_config.sh`. In particular:
- set `CLASSES_DIR` to the path of your CLASSES root directory
- add the network/dataset id to the `NETWORKS` list (in our case, "alexnet_cifar10")
- add the full hardware configuration id of the error models you want to use to the `CONFIGS` list. This should be the name of the error models directory in `error_models/conv_models` (in our case, `nv_8x8_b1_dat-524288_wt-32768_int8`). Since the configuration id is quite long, also add a shorter one to use for output filenames (in our case, `8x8_int8`) to the `SHORT_IDS` list. Make sure that the two lists have the same number of elements and that the shorter ids are in the same position as their corresponding full id
- add to `IN` the number of input images you want to use (in our case, "100") and to `ERR` the number of errors to generate for each layer (in our case, "160")
- variables in the `EXPERIMENT CONFIGURATIONS` section are used to generate the experiment configuration files. If you want to know what each field does, take a look at `template_exp_config.yaml`. In most cases, you won't need to modify them

`NETWORKS`, `CONFIGS`, `SHORT_IDS`, `IN` and `ERR` are all lists and can be used to set up a sequence of experiments to run. Be aware, though, that the scripts will run all possible combinations of values in the lists, which may result in many experiments being run (for example, if you list 3 networks, 4 configs and 1 value each for `IN` and `ERR`, you will run 12 experiments). The scripts do **NOT** currently run experiments in parallel.

You can now run `./make_experiment_configs.sh`, which will create an experiment directory in `experiments`. In our example, it will be called `exp_alexnet_cifar10` and will contain one directory with the hardware configuration name, the same one used for the error models' directory. Inside this second directory will be an experiment configuration YAML file. If you specified more than one configuration in `script_config.sh`, you will see one subdirectory per configuration, each with its own configuration file inside.

Now you can simply run `./perform_experiments.sh`, which will automatically handle the rest of the process. Since experiments are usually quite long, it is recommended to run the script as a background process by appending `&` to the command. Also consider using `nohup` if you are connected to a remote shell, like so: `nohup ./perform_experiments.sh &`. Once the experiment is over, the experiment directory will contain:
- a complete log in the `logs` subdirectory
- a fault list tar file encoding all produced errors, which can be used to exactly replicate the experiment
- an `SDC_frequencies.xlsx` file listing the SDC and spatial class frequencies for each targeted layer in the network. Most frequencies are the ones obtained by interpolating the error models
- an `outputs` subdirectory, containing an overall report that lists the outcome of each error and other metadata, such as the experiment runtime, and either a tar archive with all generated errors or a CSV file, depending on the value of `COMPUTE_SINGLE_METRICS` that was used in `script_config.sh` to generate the configuration file
- an "applev" file, which is the most important output: for each layer and each spatial class, it lists the obtained Masked, Safe and Critical frequencies. These values can be directly used in aggregation formulas, such as that of the FIT metric.

NOTE: if you are running an experiment with `compute_single_metrics` set to False in the configuration file, the corrupted outputs of each network run will be saved to the experiment directory. This will typically make the experiment runtime shorter, but will also require a lot of additional storage space. If you need to compress the results, run `tar_pack_outputs.sh` without modifying `script_config.sh` to archive them as `.tar.xz` files once the experiments are over.

## A word on the submodules

The two submodules `nets_repo` and `other_nets` already contain scripts to instantiate several common networks along with their datasets. However, they do not contain weight files, which must still be provided separately. If you want to extend the experiment process to different networks, consider checking the submodules first, as the network you are interested in may already be provided by them.

## Extending the set of networks and datasets

Follow this section if you want to run an experiment on a network/dataset pair which is not already provided in the repo.

As we've seen, each network/dataset pair is assigned a unique id, which is used throughout the experiment setup. The list of available ids is stored in the `available` dictionary defined in `network_getter.py` and you will need to add your own to it, along with the network information needed for the experiment. For the sake of consistency, use `<network>_<dataset>` as your id.

Then, in the same file, you need to extend the `match` statement in the `get_network_and_exp_functions()` function to cover the case of your new id. This function must return 5 objects: the model instance, the dataset (in the form of a PyTorch DataLoader), the network info in the `available` dictionary, a run function and a metrics function.

The model instance and the dataloader are usually provided by external functions. The scripts for the networks provided by the repo, for instance, are stored in the two submodules mentioned above.

The run function is called for each generated error and is responsible for executing both a golden run and a "corrupted" run on the network, and returning the results. The run functions are defined in `network_runner.py` and they are usually very similar, so have a look at them to see how they are structured. If your network performs standard classification or segmentation, you can probably just use one of the available functions in `network_runner.py`; instead, if your network uses some kind of custom API or packs its output in a customized way, as is the case with Ultralytics YOLO, you should write your own run function.

The metrics function takes the results produced by the run function and computes whatever metrics are suitable for the experiment you're running. These functions are defined in `experiments/metrics.py`. Once again, have a look at them to see their structure. It's likely you will have to write your own metrics function, as different experiments usually require different metrics to be computed.
Note the following: an experiment can either save all erroneous tensors it generates for subsequent analysis or directly compute a set of metrics for each error, which are called "single metrics" (each set is appended to a CSV file while the experiment runs. The CSV header structure for each network/dataset is defined by the `available` dictionary in `network_getter.py`). This behavior is determined by the `COMPUTE_SINGLE_METRICS` field in `script_config.sh`; for each network/dataset, `metrics.py` defines a function for metrics calculation, which receives this parameter as a boolean value to determine the type of result to produce. If you wish to support both cases, make sure to include both branches in your custom metrics function.

A **note** on the run and metrics functions: since these are called in the same way regardless of the underlying network, they should all use the same signatures. Look at how they are called in `run_experiment.py` to see which parameters are mandatory. All other parameters you may need to pass to your custom functions should be directly passed in `network_getter` via partial application, as is already the case with all networks currently supported.

So, **to recap**:
- prepare scripts to create a model instance and a dataloader
- add an entry to the `available` dictionary in `network_getter.py` using as a key the new id of your network/dataset pair
- if necessary, write your custom run function in `network_runner.py` and your custom metrics function in `metrics.py`
- extend `get_network_and_exp_functions()` in `network_getter.py` so that it returns the model, the dataloader, and the two functions (partially applied where needed) when your new id is passed to it