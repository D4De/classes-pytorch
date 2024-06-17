# Classes
CLASSES (Cross-Layer AnalysiS framework for Soft-Errors effectS in CNNs), a novel cross-layer framework for an early, accurate and fast reliability analysis of CNNs accelerated onto GPUs when affected by SEUs.
A theoretical description of the implemented framework can be found in:
C. Bolchini, L. Cassano, A. Miele and A. Toschi, "Fast and Accurate Error Simulation for CNNs Against Soft Errors," in IEEE Transactions on Computers, 2022, doi: 10.1109/TC.2022.3184274. <br>

If you use Classes in your research, we would appreciate a citation to:

>@ARTICLE{bc+2022ea,<br>
>  author={Bolchini, Cristiana and Cassano, Luca and Miele, Antonio and Toschi, Alessandro},<br>
>  journal={IEEE Transactions on Computers}, <br>
>  title={{Fast and Accurate Error Simulation for CNNs Against Soft Errors}}, <br>
>  year={2022},<br>
>  volume={},<br>
>  number={},<br>
>  pages={1-14},<br>
>  doi={10.1109/TC.2022.3184274}<br>
>}

## Table of Contents

1. [Copyright & License](#copyright--license)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Usage](#usage)
    1. [TensorFlow2 - As a K function](#as-a-k-function)
    2. [TensorFlow2 - As a layer](#as-a-layer)

## Copyright & License

Copyright (C) 2024 Politecnico di Milano.

This framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](https://www.gnu.org/licenses/) for more details.

Neither the name of Politecnico di Milano nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

## Installation 
1. Create a virtual environment and activate it
```
python -m venv .venv
source .venv/bin/activate
```

2. Install the correct PyTorch version for your setup using the ![PyTorch official installation guide](https://pytorch.org/get-started/locally/).

3. Install the remaining requirements:
```
pip install torchinfo numpy tqdm
```

## Installation 

To install the framework you only need to clone the repository 
```
git clone https://github.com/D4De/CLASSES.git
```
and import the `classes` module.

## How it works
To fully understand how Classes works we strongly suggest to read our paper _Fast and Accurate Error Simulation for CNNs Against Soft Errors_, that explains the philosophy of the CLASSES framework.
Nonetheless, we want to provide a small description of its operation. 
The following image provides a high level representation of the whole framework, we executed an error injection campaign
at GPU level using NVBitFI for each of the [supported layers](#operators-supported) in order to create a database of error models.
<br>
![](framework.png)
<br>
These models are then used by Classes in conjunction with either TensorFlow or PyTorch to simulate the presence of an error
during the execution of a given model.

Here we can see more in depth how Classes works.
<br>
![](classes.png)
<br>
It uses the concept of a saboteur implemented in PyTorch as a hook. The hook is executed at the end of the execution of the layer
and corrupts the result of the layer. 


### Supported operators
The operators supported aarethe following:
- Convolution: 2d GEMM convolution 
- Pooling: Max and Average 2d
- Activations: Relu, Clipped Relu, Sigmoid, Tanh, Elu
- Bias Add

Other operators can be added after generating their error models performing platform-level fault injection campaigns.

### Error models
We executed an error injection campaign on GPU to create error models that could be as loyal as possible to the real world. The older error models used in classes were defined based on three parameters
- Cardinality: the number of corrupted values observed in a faulty tensor with respect to the expected version.
- Pattern: the spatial distribution of the corrupted values.
- Values: the numerical value of the corrupted data.

These models were revised and rationalized in the thesis ![A novel approach for error modeling in a cross-layer reliability analysis of convolutional neural networks](https://www.politesi.polimi.it/bitstream/10589/210019/4/tesi_passarello_finale_2.pdf) and used in the following publications where CLASSES was employed. Now the models are build hierarchically, modeling the errors first by their spatial patterns, that are classes of corrupted tensors that have similar spatial distribution patterns. Example of spatial patterns are extracted from experiments performed on an Ampere GPU using NVBITFI we found these recurring patterns:
1. **Single Point**: a single erroneous value is found in the output tensor.

2. **Same Row**: all the erroneous values are located on the same row of a single channel. It is also possible to have non-corrupted values between two erroneous ones.

3. **Bullet Wake**: similar to *Same Row* model, but the erroneous values are located along the channel axis. It is also possible to have channels without erroneous values on the axis.

4. **Skip X**: erroneous values are located with a stride equal to *X* positions in the linearized tensor.

5. **Single Block**: there is a single corrupted channel, where all the erroneous values are adjacent and the cardinality is comprised between 10 and 64.

6. **Single Channel Alternated Blocks**: multiple nonadjacent blocks of at least 16 consecutive erroneous values located in the same channel.

7. **Full Channel**: more than 70% of the values of the same channel are erroneous.

8. **Multichannel Block**: an extension of the *Single Block* model where multiple channels are corrupted, each with a single block pattern.

9. **Shattered Channel**: a combination of the *Bullet Wake* and *Single Map Random* models where erroneous values are located on a line along the channel axis and at least one channel is randomly corrupted.

10. **Random**: either a single channel or multiple ones are corrupted with an irregular pattern.
    
For each occourring pattern the error models contains other two sub characterizations:
* Spatial parameters, that allow to better generalize and characterized the spatial distributions. Examples of spatial parameters are:
    - Percentage of channels of the tensors that are corrupted
    - Maxium number of faults that are corrupted
    - Interval (or skip) between two corrupted values in the linearized tensor.
    Each spatial class has its own set of parameters.
* Domain models, that models the distribution of the corrupted values.

To derive the domains models each corrupted value is first put in one of four Value Classes:
- In Range: The corrupted value remains inside the operating range of the golden tensor (i.e. ``[min(golden_tensor), max(golden_tensor)]``), but it is not exactly zero.
- Out of Range: The corrupted value falls of the operating range of the golden tensor.
- NaN: The value is NaN, or infinity
- Zero: The value is exactly zero.
 
Then the tensor is classified in categories of Value Class distributions that is one of these:
- Single Value Class: All errors belong to the same value class
- Double Value Class: All errors belong to only two different value classes. In this case the proportion of values is stored in the class.
- Random. The tensor is not on one of these two classes.

The error models are obtained by first performing a fault injection at the architectural level (for example using NVBitFI) and then using the CNN error classifier tool (available in this ![https://github.com/D4De/cnn-error-classifier](repository) ) that analyzes the fault injection results and creates the models. The models can be found in this repo in the ```models``` folder, that contains one different model json file per operator. CLASSES will read from these models and will generate error patters using the models specified in the json. 
For a given operator the json file contains the relative frequency of each spatial pattern, and for each spatial pattern, there are the relative frequency for each configuration of spatial parameters and domain parameters. 

The injection site generator will pick at random a spatial pattern (with a probability equal to its relative frequency) and a configuration of spatial and domain parameters. The generator then picks the corrupted locations in the tensor by calling the pattern generator function corresponding to the picked spatial pattern (the code pattern generator functions are in ```src/pattern_generators```). For each corrupted location a value is picked based on the domain parameters distribution, and then the corrupted tensor is inserted in the network by the error simulator module described below.

## Architecture

The framework is composed by two decoupled parts: the backend and the frontend.

## Backend

The backend code is in charge of generating the faults (or errors) to be injected into the execution. The generation is done by the ``FaultGenerator (classes.fault_generator.FaultGenerator)`` guided the specific error models corresponding to the operator of the current target layer (loaded from the models json and stored in a ``ErrorModel`` object). The ``FaultGenerator`` given the error model, the shape of the output of the target lauyer to corrupt and the operating range (minimum and maximum values of the output tensor during normal operation), generates a ``Fault`` (or multiple faults inside a ``FaultBatch``).

The ``Fault`` object represents a framework independent fault and is characterized by:

* A mask (``corrupted_value_mask``), a multi dimensional integer numpy array of the same shape as the output tensor. The locations
are the ones where a fault will be injected. The values different from zero represent the ``ValueClass`` of the corrupted value.
* A 1d corrupted value array (``corrupted_values``) that contains the actual values that will be injected. The i-th value of the array is
the value that replaces the i-th corrupted location in the unrolled mask, so the length of the array is equal to the count of non-zero values inside the mask.
* Other metadata of the faults (``spatial_pattern_name``, ``sp_parameters``) that is not directly used in the injection process, but it is
useful to store them for data analysis purpose.

In synthesis the backend generates the faults using the error models creating framework independent ``Fault`` objects that the frontend
will inject in the target layers of the network

## Frontend

The frontends of CLASSES inject the faults generated in the backend in the execution of the inference. Frontend are framework specific because they need to use the functions present in the deep learning framework used by the network. In this repository there is a frontend only for PyTorch. A frontend for Tensorflow can be implemented in the future reusing the same backend.


The frontend has the following resposibilities:
* Call the `FaultGenerator`
* Optionally, persist the fault list on disk, so that can can be reused, enabling reproducible experiments.
* Loading and converting the framework independent output of the `FaultGenerator` (containing numpy arrays) to a format compatible with the framework used.
* Inject the fault in the target layer.

### PyTorch Frontend
The frontend for PyTorch contains utilites to inject the faults generated in the backend in pytorch networks. The frontend 
treats faults as subclasses of ``torch.utils.data.Dataset``, enabling the use of PyTorch dataloaders, in a very similar way
as iterating over a input dataset. There are two ways to generate and load fault list datasets in pytorch: 
* Lazy mode: Generates the fault lazily then converts and injects them. This mode is implemented in the class ``PyTorchLazyFaultList`` 
that subclassess ``IterableDataset``. This mode does not require to generate faults in advance, however the faults are not automatically
persisted on disk.
* Persisted mode: First a ``PyTorchFaultList`` object is instatiated to pre-generate the fault list and persist it in the disk inside a ``.tar`` archive. Then the user can load the faults using a ``FaultListFromTarFile`` dataset object, that is a map-style dataset (indexable
using the ``dataset[i]`` notation).

From both these ``Dataset`` objects can be wrapped into a ``Dataloader`` and can be iterated in a for loop that returns one ``PyTorchFault`` (or fault batch) at the time. The faults are already converted to ``torch.Tensor`` and can be moved to the devige (GPU) using the ``.to(device)`` method, similarly to how it done for input data. 
The faults can be injected then using the forward hooks provided by classes in the ``classes.simulators.simulator_hook`` module. A forward hook is a function that can be attached to a PyTorch module (a layer) and will be executed after the inference of that layer terminated. The value returned by the function replaces the original output tensor of the targeted module.

### Directory Structure

The directory ``classes`` contains the main code for the library. It contains the following submodules:
* ``error_models`` (backend):  Contains logic for loading from json files and for representing the errors models.
* ``fault_generator`` (backend): Contains the code for randomly generating the faults from the error models. 
    * ``fault_generator.py``: Implementation of the ``FaultGenerator`` class.
    * ``fault.py``: Implementation of the ``Fault`` and ``FaultBatch`` classes, that represent frontend-indepentent faults.
* ``pattern_generators`` (backend): Contains the code to generate the mask containg the locations of corrupted values for each spatial class.
    * Each file contains a function to generate a specific spatial class. All functions have the same interface in common (except for their name):
        * ``output_shape``: The shape of the output tensor to corrupt. 
        * ``params``: A dict containing the spatial parameters picked by the ``FaultGenerator``. The mask will be generated according to the paramters.
        * ``layout``. The layout to use (CHW = channel first, HWC = channel last).
    * ``generator_utils.py``. Contains various common function used by all pattern generators.
* ``value_generators`` (backend): Contains the code to generate the values that will replace golden values in the output of the targetd layer.
    * ``value_class.py``: Enum that contains the various value classes, and the code to generate values belonging to those classes.
    * ``value_class_distribution.py``: Models the concept of value class distribution (as explained in the error model section) and contains the code for realizing these distributions.
    * ``value_generators.py`` and ``float_utils.py``: Utility functions for generating the float values
* ``tests`` (backend): Contains CLI tools to test and debug the ``pattern_generators``. These tools are standalone and can be called from the root directory using the command ``python -m classes.test.\[name_of_the_module\]``
    * ``fault_list_visualizer.py``: Generates fault masks starting from the error models and draws them to multiple 2d heatmaps so that the developer can check that faults generated are similar to the ones coming from the platform level injection campaign
    * ``pattern_tester.py``: Generates fault masks from the error models using various input shape configuration checking that the generator does not throw errors during the generation.
* ``simulators`` (frontend): Contains the various frontends. Only ``pytorch`` frontend is implemented:
    * ``keras`` (NOT YET IMPLEMENTED)
    * ``pytorch``:
        * ``fault_list.py``: Contains the iterator that lazily generates faults (calling the backend) and the functions to persist a fault list.
        * ``pytorch_fault.py``: Implementation of the ``Fault`` object with PyTorch's Tensor object. ``PyTorchFault`` can be easily moved to GPU using the ``.to()`` method (as PyTorch tensor).
        * ``simulator_hook.py``: Contains the code for creating and applying the CLASSES error simulator hook
        * ``fault_list_datasets.py``: Implementation of the two modes of loading CLASSES faults (lazy and persisted)
        * ``error_model_mapper.py``: Contains a function that maps the pytorch module operators type to their correct error models
        * ``module_profiler.py``: Contains utility profiler functions that profile:
            * ``output shape`` Useful for getting the output shapes for each module of the netrork. The output is used by CLASSES to determine the shape of the output valkue
            * ``operating range``. For getting the normal operating range of the output of each module. The operating range is used by the value generator to generate values ``InRange`` and ``OutOfRange``.
            * ``execution time`` of each module




## Code Example

You can find an example with comments in ``examples/lenet5``,
where the main features of CLASSES are showcased.

You can run it with
```
python -m examples.lenet5
```
from the root folder of the repo.

