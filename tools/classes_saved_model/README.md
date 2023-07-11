# Introduction 

The tool described here has been developed in order to use CLASSES with models for which the user is only in possession of the saved weights and no information about the code used to build the model is provided.
We assume that such file is in the h5 format, if not there are multiple tools available online to convert between the most common formats (e.g., ONNX, SavedModel) to h5.

# Copyright & License

Copyright (C) 2023 Politecnico di Milano.

This tool is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This tool is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](https://www.gnu.org/licenses/) for more details.

Neither the name of Politecnico di Milano nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# Requirements

The `h5py` library is required for this tool to work correctly, it can be installed with the following command
```
pip install h5py
```

# Usage

To run the tool we first need to setup the [`config.json`](config.json) file, it contains the following keys

1. model: path to the weights.h5 file that describes the model we are targeting.
2. new_model: path were the script will save the weights.h5 file of the new model we are creating.
3. injection_sites: an int representing the number of injections we want to perform.
4. layer_type: the name of the type of the layer we are targeting. It is required for selecting the correct error model. It should be the same as one of the json provided in the models folder without the extension (e.g, avgpool.json -> avgpool).
5. layer_output_shape_cf: a string with the output shape of the targeted layer in the format NCHW. (e.g, '(None, 16, 27, 27)').
6. layer_output_shape_cl: a string with the output shape of the targeted layer in the format NHWC. (e.g, '(None, 27, 27, 16)').
7. models_folder: path to the folder `model, by default "../../models"
8. range_min: an int that, given the possible values that the targeted layer produces, defines the minimum possible value.
9. range_max: an int that, given the possible values that the targeted layer produces, defines the maximum possible value.

Then we simply run the following command

```python
python add_classes_models.py
```

We will be asked to select which layer we want to target and then the script will automatically update the weights file 
by adding the ErrorSimulator layer.