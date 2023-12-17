# Installation Guide

### Basic Installation
If you simply desire to install the CLASSES tool, compile the package with pip install
```
pip install .
```

If you are extending the tool and desire to add some new functionalities or upload some new models .json files, follow those steps

### Customize Installation by adding a new folder for ErrorModels

1. Modify the code as you desire
2. Add the .json file in a folder inside the classes_core
3. Open the setup.py
4. Add the new folder to the package by modifing the package parameter
``` python
packages=pack + ["classes_core/new_folder"],
```
5. Add the new folder to the package package_data
``` python
package_data= {         
        "classes_core": [
            "logging.conf"  #Per il logger
        ],

        ...,

        "classes_core/new_folder":[
            "*.json"
        ]
        
        },
```
6. If desired Add the new path to ```./classes_core/__init__.py ```
``` python
new_error_model_path = str(pathlib.Path(__file__).parent.absolute()) + 'new_folder'
```

7. Install the updates as in Basic installation
```
pip install .
```
8. You are Done, now you can include classes as a normal python package
``` python
from classes_core.error_simulator_keras import ErrorSimulator, create_injection_sites_layer_simulator
import classes_core  

path_to_my_new_error_models = classes_core.new_error_model_path
```