# CLASSES

## Table of Contents

1. [Dependencies](#dependencies)
2. [Installation](#installation)
3. [Usage](#usage)
    1. [TensorFlow2 - As a K function](#as-a-k-function)
    2. [TensorFlow2 - As a layer](#as-a-layer)
## Dependencies 
The following libraries are required in order to correctly use the framework. <br>
N.B. This framework has been developed for TensorFlow1, TensorFlow2 and PyTorch. If you want to use it with one of the frameworks you don't need to install also the others.

1. Python3 
1. Numpy package
2. TensorFlow 2.10 or lower
3. TensorFlow 1
4. PyTorch

## Installation 

To install the framework you only need to clone the repository 
```
git clone https://github.com/D4De/CLASSES.git
```

We suggest to create a conda environment with all the required packages. 

### Ubuntu
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create --name classes python=3.9
conda activate classes
pip install numpy
```
### MacOS

```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
conda create --name classes python=3.9
conda activate classes
pip install numpy
```

## Usage

The framework is composed by two distinct modules, the injection sites generator and the error simulator. The first one, as the name suggests, is responsible for the creaton of the injection sites and it is common among all the different implementations of the error simulator. The error simulator itself has been implemented in four different ways in order to work with both TensorFlow (1 and 2) and PyTorch. We provide here an high level description of both modules, a more precise description with examples can be found in the `example` folder. 

### Injection sites generator 

This module is used to create the injection sites needed for the error simulation campaign. The whole module is accessed through the function 
`generate_injection_sites(sites_count, layer_type, layer_name, size, models_mode='')`
its inputs are:
- sites_count: the number of injection sites to create
- layer_type: what kind of layer we are injecting, the available layers are the ones defined in the enum `OperatorType`
- layer_name: a string representing the layer name, can be empty
- size: a string that defines the output shape of the layer we want to target. It must be defined with the following format '(None, Channel, Height, Width)' 

This function returns three lists: 
- injection_sites: list of injection sites, each injection site has an index and a value, the value itself has two parameters, value_type that describes what kind of value has been selected based on our error models and raw_value which is the numerical value. 
- cardinalities: list of cardinalities selected.
- patterns: list of patterns selected.

### Fault injector

The module used to perform the fault injection campaign has been developed in four different shapes. 

### Tensorflow2
### As a K function

Using Keras backend functions allows us to split a model into two separate submodels, one that performs the computation from the input layer to the one we selected for the injection and one that produces the final result from the output of the selected layer. With this approach we can obtain the selected layer's output and modify it before feeding it to the second submodel.
If the model is not linear (e.g. a unet) this approach requires us to make sure that each skip connection is correctly set for the final output to be correctly produced. In the `example/tensorflow2/k_function` folder we provide two examples showing how to use this version of the error simulator on a linear model and on a model with skip connections.

### As a layer

We also developed the injector as a layer that can be placed into the module like any other keras layer. This approach simplifies the process in case of models with skip connections since we do not need to manually restore them but it has some drawbacks. The most important one is that due to how TensorFlow2 works we are not able to generate injection sites each time we perform a prediction, instead we need to create them at setup time and pass them as inputs to the layer. The injector will randomly select one at each prediction.
