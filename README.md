# Classes for PyTorch

CLASSES (Cross-Layer AnalysiS framework for Soft-Errors effectS in CNNs), a novel cross-layer framework for an early, accurate and fast reliability analysis of CNNs accelerated onto GPUs when affected by SEUs.
A theoretical description of the implemented framework can be found in: <br>
C. Bolchini, L. Cassano, A. Miele and A. Toschi, "[Fast and Accurate Error Simulation for CNNs Against Soft Errors](https://ieeexplore.ieee.org/document/9799516)," in IEEE Transactions on Computers, 2022, doi: 10.1109/TC.2022.3184274. <br>

If you use CLASSES in your research, we would appreciate a citation to:

> @ARTICLE{bc+2022ea,<br>
> author={Bolchini, Cristiana and Cassano, Luca and Miele, Antonio and Toschi, Alessandro},<br>
> journal={IEEE Transactions on Computers}, <br>
> title={{Fast and Accurate Error Simulation for CNNs Against Soft Errors}}, <br>
> year={2022},<br>
> volume={},<br>
> number={},<br>
> pages={1-14},<br>
> doi={10.1109/TC.2022.3184274}<br>
> }

## Copyright & License

Copyright (C) 2024 Politecnico di Milano.

This framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](https://www.gnu.org/licenses/) for more details.

Neither the name of Politecnico di Milano nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

## Installation

NOTE: This newer version of CLASSES supports only PyTorch (for now). The older version of CLASSES supports only Tensorflow2 and is available [here](https://github.com/D4De/classes)

1. Create a virtual environment and activate it:

```
python -m venv .venv
source .venv/bin/activate
```

2. Install the correct PyTorch version for your setup using the [PyTorch official installation guide](https://pytorch.org/get-started/locally/).

3. Install the remaining requirements:

```
pip install torchinfo numpy tqdm
```

## Installation

To install the framework you only need to clone the repository:

```
git clone https://github.com/D4De/classes-pytorch.git
```

and import the `classes` module.

## How it works

To fully understand how CLASSES works, we strongly suggest to read our paper [Fast and Accurate Error Simulation for CNNs Against Soft Errors](https://ieeexplore.ieee.org/document/9799516), that explains the philosophy of the CLASSES framework. <br>
Nonetheless, we want to provide a small description of its operation.
The following image provides a high level representation of the whole framework: we executed an error injection campaign
at GPU level using [NVBitFI](https://www.mbsullivan.info/attachments/papers/tsai2021nvbitfi.pdf) for each of the [supported layers](#supported-operators) in order to create a database of error models.
<br>
![](framework.png)
<br>
These models are then used by CLASSES in conjunction with either TensorFlow or PyTorch to simulate the presence of an error
during the execution of a given model.

Here we can see more in depth how CLASSES works.
<br>
![](classes.png)
<br>
It uses the concept of a *saboteur* implemented in PyTorch as a hook. The hook is executed at the end of the execution of the layer
and corrupts the result of the layer.

### Supported operators

The operators supported are the following:

- Convolution: 2d GEMM convolution
- Pooling: Max and Average 2d
- Activations: Relu, Clipped Relu, Sigmoid, Tanh, Elu
- Bias Add

Other operators can be added after generating their error models by performing platform-level fault injection campaigns.

### Error models

We executed an error injection campaign on GPU to create error models that could resemble real model behavior as much as possible. The older error models used in CLASSES were classified according to three parameters:

- Cardinality: the number of corrupted values observed in a faulty tensor with respect to the expected version.
- Pattern: the spatial distribution of the corrupted values.
- Values: the numerical value of the corrupted data.

These models were revised and rationalized in the thesis [A novel approach for error modeling in a cross-layer reliability analysis of convolutional neural networks](https://www.politesi.polimi.it/bitstream/10589/210019/4/tesi_passarello_finale_2.pdf) and used in the following publications where CLASSES was employed.<br>
Now the models are built hierarchically, first modeling the errors by considering their spatial patterns, i.e. classes of corrupted tensors that have similar spatial distribution patterns. Examples of spatial patterns are extracted from experiments performed on an Ampere GPU using [NVBITFI](https://www.mbsullivan.info/attachments/papers/tsai2021nvbitfi.pdf). We found the following recurring patterns:

1. **Single Point**: a single erroneous value is found in the output tensor.

2. **Same Row**: all the erroneous values are located on the same row of a single channel. It is also possible to have non-corrupted values between two erroneous ones.

3. **Bullet Wake**: similar to _(2) Same Row_ model, but the erroneous values are located along the channel axis. It is also possible to have channels without erroneous values on the axis.

4. **Skip X**: erroneous values are located with a stride equal to _X_ positions in the linearized tensor.

5. **Single Block**: there is a single corrupted channel, where all the erroneous values are adjacent and the cardinality is comprised between 10 and 64.

6. **Single Channel Alternated Blocks**: multiple nonadjacent blocks of at least 16 consecutive erroneous values located in the same channel.

7. **Full Channel**: more than 70% of the values of the same channel are erroneous.

8. **Multichannel Block**: an extension of the _(5) Single Block_ model where multiple channels are corrupted, each with a single block pattern.

9. **Shattered Channel**: a combination of the _(3) Bullet Wake_ and _Single Map Random_ models where erroneous values are located on a line along the channel axis and at least one channel is randomly corrupted.

10. **Random**: either a single channel or multiple ones are corrupted with an irregular pattern.

For each occurring pattern the error models contain other two sub-characterizations:

- **Spatial parameters**, which allow to better generalize and characterize the spatial distributions. Examples of spatial parameters are:
  - Percentage of channels of the tensors that are corrupted.
  - Maximum number of faults that are corrupted.
  - Interval (or skip) between two corrupted values in the linearized tensor.
    Each spatial class has its own set of parameters.
- **Domain models**, which model the distribution of the corrupted values.

To derive the domains models, each corrupted value is first sorted into one of four **Value Classes**:

- In Range: the corrupted value is still within the operating range of the golden tensor (i.e. `[min(golden_tensor), max(golden_tensor)]`), but it is not exactly zero.
- Out of Range: the corrupted value is outside the operating range of the golden tensor.
- NaN: the value is NaN, or infinity.
- Zero: the value is exactly zero.

Then the tensor is classified in categories of Value Class distributions, i.e. is one of the following:

- Single Value Class: all errors belong to the same value class.
- Double Value Class: all errors belong to only two different value classes; in this case, the proportion of values is stored in the class.
- Random. the tensor is not in one of the previous classes.

The error models have been obtained by first performing a fault injection at the architectural level (for example using [NVBitFI](https://github.com/fernandoFernandeSantos/nvbitfi)) and then using the CNN error classifier tool (available in this [repository](https://github.com/D4De/cnn-error-classifier) ), which analyzes the fault injection results and creates the models. The models can be found in this repo in the `error_models` folder, containing a different model json file per operator. CLASSES will read from these models and generate the corresponding error patterns.
For a given operator, the json file contains the relative frequency of each spatial pattern; for each spatial pattern, the relative frequency for each configuration of spatial parameters and domain parameters is provided.

The injection site generator will randomly pick a spatial pattern (with a probability equal to its relative frequency) and a configuration of spatial and domain parameters. The generator then picks the corrupted locations in the tensor by calling the pattern generator function corresponding to the picked spatial pattern (the code pattern generator functions are in `classes/pattern_generators`). For each corrupted location, a value is picked according to the domain parameters distribution, and the corrupted tensor is inserted in the network by the error simulator module described below.

## Architecture

The framework is composed of two decoupled parts: the backend and the frontend.

## Backend

The backend code is in charge of generating the faults (or errors) to be injected into the execution. The generation is done by the `FaultGenerator (classes.fault_generator.FaultGenerator)`, in turn guided by the specific error models corresponding to the operator of the current target layer (loaded from the model's json and stored in an `ErrorModel` object). The `FaultGenerator`, given the error model, the output shape of the target layer to corrupt and the operating range (minimum and maximum values of the output tensor during normal operation), generates a `Fault` (or multiple faults inside a `FaultBatch`).

The `Fault` object represents a framework-independent fault and is characterized by:

- A **mask** (`corrupted_value_mask`), a multi-dimensional integer numpy array with the same shape of the output tensor. The locations
  are the ones where a fault will be injected. The values different from zero represent the `ValueClass` of the corrupted value.
- A **1d corrupted value array** (`corrupted_values`), which contains the actual values that will be injected. The i-th value of the array is
  the value that replaces the i-th corrupted location in the unrolled mask, so the length of the array is equal to the count of non-zero values inside the mask.
- Other **metadata** of the faults (`spatial_pattern_name`, `sp_parameters`) that is not directly used in the injection process, but it is
  useful to store them for data analysis purpose.

Summarizing, the backend generates the faults using the error models, thus creating framework-independent `Fault` objects that the frontend
will inject in the target layers of the network.

## Frontend

The frontends of CLASSES inject the faults generated in the backend in the execution of the inference. Frontends are framework-specific, as they need to use the functions provided by the deep learning framework used by the network. This repository only contains a frontend for PyTorch. A frontend for TensorFlow can be implemented in the future by reusing the same backend.

The frontend has the following responsibilities:

- Call the `FaultGenerator`.
- Optionally, persist the fault list on disk, so that it may be reused, enabling reproducible experiments.
- Load and convert the framework-independent output of the `FaultGenerator` (containing numpy arrays) to a format compatible with the used framework.
- Inject the fault in the target layer.

### PyTorch Frontend

The frontend for PyTorch contains utilities to inject the faults generated by the backend into PyTorch networks. The frontend
treats faults as subclasses of `torch.utils.data.Dataset`, enabling the use of PyTorch DataLoaders, in a very similar way
as iterating over an input dataset. There are two ways to generate and load fault list datasets in PyTorch:

- Lazy mode: generates the faults lazily and then converts and injects them. This mode is implemented in the class `PyTorchLazyFaultList`,
  which subclassess `IterableDataset`. This mode does not require to generate faults in advance, but the faults are not automatically
  persisted on disk.
- Persisted mode: first, a `PyTorchFaultList` object is instantiated to pre-generate the fault list and persist it on disk inside a `.tar` archive. Then, the user can load the faults using a `FaultListFromTarFile` dataset object, a map-style dataset (indexable
  using the `dataset[i]` notation).

Both of these `Dataset` objects can be wrapped into a `Dataloader` and iterated over in a for loop that returns one `PyTorchFault` (or fault batch) at a time. The faults are already converted to `torch.Tensor` and can be moved to the device (GPU) using the `.to(device)` method, very similarly to how it's done for input data.
The faults can then be injected by using the forward hooks provided by CLASSES in the `classes.simulators.simulator_hook` module. A forward hook is a function that can be attached to a PyTorch module (a layer) and will be executed after the inference of that layer terminates. The value returned by the function replaces the original output tensor of the targeted module.

### Directory Structure

The `classes` directory contains the main code for the library. It contains the following submodules:

- `error_models` (backend): contains logic for loading from json files and for representing the errors models.
- `fault_generator` (backend): contains the code for randomly generating the faults from the error models.
  - `fault_generator.py`: implements the `FaultGenerator` class.
  - `fault.py`: implements the `Fault` and `FaultBatch` classes, representing frontend-independent faults.
- `pattern_generators` (backend): contains the code to generate the mask marking the locations of corrupted values for each spatial class.
  - Each file contains a function to generate a specific spatial class. All functions have the same common interface (except for their name):
    - `output_shape`: the shape of the output tensor to corrupt.
    - `params`: a dict containing the spatial parameters picked by the `FaultGenerator`. The mask will be generated according to these parameters.
    - `layout`: the layout to use (CHW = channel first, HWC = channel last).
  - `generator_utils.py`: contains various common functions used by all pattern generators.
- `value_generators` (backend): contains the code to generate the values that will replace golden values in the output of the targetd layer.
  - `value_class.py`: enum that contains the various value classes, and the code to generate values belonging to those classes.
  - `value_class_distribution.py`: models the concept of value class distribution (as explained in the [error model section](#error-models)) and contains the code for realizing these distributions.
  - `value_generators.py` and `float_utils.py`: utility functions for generating the float values.
- `tests` (backend): contains CLI tools to test and debug the `pattern_generators`. These tools are standalone and can be called from the root directory using the command `python -m classes.test.\[name_of_the_module\]`.
  - `fault_list_visualizer.py`: generates fault masks starting from the error models and draws them to multiple 2d heatmaps so that the developer can check that the generated faults are similar to the ones coming from the platform-level injection campaign.
  - `pattern_tester.py`: generates fault masks from the error models by using various input shape configurations, also checking that the generator does not throw errors during generation.
- `simulators` (frontend): contains the various frontends. Only the `pytorch` frontend is implemented:
  - `keras`: (NOT YET IMPLEMENTED)
  - `pytorch`:
    - `fault_list.py`: contains the iterator that lazily generates faults (by calling the backend) and the functions to persist a fault list.
    - `pytorch_fault.py`: implements the `Fault` object with PyTorch's Tensor object. `PyTorchFault` can be easily moved to GPU using the `.to()` method (as can PyTorch tensors).
    - `simulator_hook.py`: contains the code for creating and applying the CLASSES error simulator hook.
    - `fault_list_datasets.py`: implements the two modes for loading CLASSES faults (lazy and persisted).
    - `error_model_mapper.py`: contains a function that maps the PyTorch module operator types to their corresponding error models.
    - `module_profiler.py`: contains utility profiler functions that profile:
      - `output shape`: useful for getting the output shapes for each module of the netrork. The output is used by CLASSES to determine the shape of the output value.
      - `operating range`: for getting the normal operating range of the output of each module. The operating range is used by the value generator to generate values `InRange` and `OutOfRange`.
      - `execution time` of each module.
- `examples`: Contains tutorials. Start from `getting_started.py` and then follow `advanced.py` to see how to run reproducible experiments.

## Examples and Tutorial

You can find commented examples in `examples`,
which showcase the main features of CLASSES.

You can run them with

```
python -m examples.getting_started
python -m examples.advanced

```

from the root folder of the repo.

## Obtaining error models, configuring experiments and running them (for NVDLA)

The `experiments` directory contains all scripts needed to set up and run error simulation experiments using sets of error models specifically derived from NVDLA fault injection campaigns. However, only a few scripts and files actually depend on the structure of NVDLA campaign outputs and the following methods can be easily extended to results from other sources. Similarly, these methods currently only work for convolutional layers, but possible extensions should be simple.

### Running injection campaign postprocessing.

In terms of error models, `error_models/injection_campaign_postprocessing` contains a few scripts that transform the NVDLA outputs into error models and, importantly, produce a `.xlsx` file, usually called `unique_complete_df.xlsx`; this file is used by the experiment scripts to interpolate the available error models when selecting one for each injectable layer of a network. There is no implementation-specific reason for this file to be in Excel format, it is just a matter of convenience in case it needs to be manually inspected.

Once the injection campaign results for a configuration are ready, create a postprocessing directory to store the results (e.g. `models_8x8_int8`) and copy `error_models/injection_campaign_postprocessing/postprocessing_config_example.yaml` into it. Rename this file if you want and edit this file to fit the configuration and networks you have available, then call (from the CLASSES root directory):
```
python -m error_models.injection_campaign_postprocessing_postprocess_step1 <path/to/YAML/configuration/file>
```
This first step produces an intermediate "step1" `.xlsx` file in the postprocessing directory. Following the step 1 script's output directions, inspect the file and identify possible outlier layers to be discarded (note that several Z-score figures have already been calculated to make this process quicker). If you find any outlier, delete the corresponding row(s) from the first sheet of the file, then save it and run:
```
python -m error_models.injection_campaign_postprocessing_postprocess_step2 <path/to/YAML/configuration/file>
```
This step produces, aside from `step2_unique_complete_df.xlsx`, a "reconstruction test" Excel file. Before you proceed, examine this file and look at the rightmost column: a cell which displays "True" indicates that the model in its row was unable to be accurately reconstructed via interpolation of the other models. If there is no background knowledge requiring you to keep that model (for example, you may know that layer to be particularly vulnerable when compared with the other layers), consider deleting the model's row from `step2_unique_complete_df.xlsx` before moving on.

Once you're done, move all error models from the postprocessing subdirectory `merged_models` to the final model storage directory you will use for the experiments, along with `step2_unique_complete_df.xlsx`, which, as mentioned previously, is typically renamed to just `unique_complete_df.xlsx`. By convention, this final directory is located in `error_models/conv_models` and is named after the full id of the hardware configuration from which the models were obtained (example: `nv_16x32_b1_dat-524288_wt-131072_int8`).

### Setting up and running an experiment

Let's return to the `experiments` directory and select a network/dataset to run an experiment on. As a running example, let's choose AlexNet with the CIFAR10 dataset. Each network/dataset pair is associated with an id that's used by the experiment scripts and is found in different files, so be careful when typing it manually. In our case, the id is `alexnet_cifar10`.

First of all, if you don't already have them, make a `dataset_data` and a `weights` directory inside `experiments`; store the network weight files (typically `.pth` or `.pt` for PyTorch) for your network in `weights` and, if the dataset does not automatically download from TorchVision or similar services, download it yourself and store it into `dataset_data` (an example of the latter case is the COCO dataset). The current versions of the experiment scripts download CIFAR10 automatically, so there is no need to download it manually; a weights file must still be provided, though.

Now head to `experiments/utility_scripts` and edit `script_config.sh`. In particular:
- set `CLASSES_DIR` to the path of your CLASSES root directory
- add the network/dataset id to the `NETWORKS` list (in our case, "alexnet_cifar10")
- add the full hardware configuration id of the error models (the one used for the name of the error models final directory) to the `CONFIGS` list
- add to `IN` the number of input images you want to use and to `ERR` the number of errors to generate for each layer
- variables in the `EXPERIMENT CONFIGURATIONS` section are used to generate the experiment configuration files. If you want to know what each field does, take a look at `experiments/template_exp_config.yaml`

Notice that several of the variables in `script_config.sh` are lists and can be used to set up a sequence of experiments to run. Be aware, though, that the scripts will run all possible combinations of said variables, which may result in many experiments being run. A possible use case is to run experiments for a set of networks over several configurations, possibly with different numbers of inputs and errors, but you will probably run experiments for several networks over a single configuration, single number of inputs and single number of errors. The scripts currently do not run experiments in parallel.

You can now run `./make_experiment_configs.sh`, which will create an experiment directory in `experiments`. In our example, it will be called `exp_alexnet_cifar10` and will contain a directory with the hardware configuration name, the same one used for the error models' directory. Inside this second directory will be an experiment configuration YAML file.

Now you can simply run `./perform_experiments.sh`, which will automatically handle the rest of the process. Since experiments are usually quite long, it is recommended to run the script as a background process by appending `&` to the command (also consider using `nohup` if you are connected to a remote shell). Once the experiment is over, the experiment directory will contain several output files, including the tar file of the generated fault list, which can be used to run the same experiment again if needed.

NOTE: if you are running an experiment with `compute_single_metrics` set to True in the configuration file, the corrupted outputs of each network run will be saved to the experiment directory. This will typically make the experiment runtime shorter, but will also require a lot of additional storage space. If you need to compress the results, `experiments/utility_scripts/tar_pack_outputs.sh` can be used to archive them as `.tar.xz` files once the experiments are over.

### A word on the submodules

The two submodules `nets_repo` and `other_nets` already contain scripts to instantiate several common networks along with their datasets. However, they do not contain weight files, which must still be provided separately. If you want to extend the experiment process to different networks, consider checking the submodules first, as the network you are interested in may already be provided by them.

### Extending the set of networks and datasets

Follow this section if you want to run an experiment on a network/dataset pair which is not already provided in the repo.

As we've seen, each network/dataset pair is assigned a unique id, which is used throughout the experiment setup. The list of available ids is stored in the `available` dictionary defined in `experiments/network_getter.py` and you will need to add your own to it, along with the network information needed for the experiment.

Then, in the same file, you need to extend the `match` statement in the `get_network_and_exp_functions()` function to cover the case of your new id. This function must return 5 objects: the model instance, the dataset (in the form of a PyTorch DataLoader), the network info in the `available` dictionary, a run function and a metrics function.

The model instance and the dataloader are usually provided by external functions. The scripts for the networks provided by the repo, for instance, are stored in the two submodules mentioned above.

The run function is called for each generated error and is responsible for executing both a golden run and a "corrupted" run on the network, and returning the results. These functions are collected in `experiments/network_runner.py`, so have a look at them to see how they are structured. If your network is "classical", e.g. a standard classification or segmentation network, you can probably just use one of the available functions in `network_runner.py`; instead, if your network uses some kind of custom interface, as is the case with Ultralytics YOLO, you should write your own run function.

The metrics function takes the results produced by the run function and computes whatever metrics are suitable for the experiment you're running. These functions are collected in `experiments/metrics.py`. Once again, have a look at them to see their structure. It's likely you will have to write your own metrics function, as different experiments may need different metrics to be computed.

A **note** on the run and metrics functions: since these are called in the same way regardless of the underlying network, they should all use the same signatures. Look at how they are called in `experiments/run_experiment.py` to see which parameters are mandatory. All other parameters you may need to pass to your custom functions should be directly passed in `network_getter` via partial application, as is already the case with all networks currently supported.

So, **to recap**:
- prepare scripts to create a model instance and a dataloader
- add an entry to the `available` dictionary in `network_getter.py` using as a key the new id of your network/dataset pair
- if necessary, write your custom run function in `network_runner.py` and your custom metrics function in `metrics.py`
- extend `get_network_and_exp_functions()` in `network_getter.py` so that it returns the model, the dataloader, and the two functions (partially applied where needed) when your new id is passed to it