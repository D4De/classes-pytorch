"""
An example of network graphing, operator extraction and initial inference runs on a Resnet50.
The overall process outputs the selected operators' weights, parameters and intermediate input tensors.
"""

import torch
import os

import model_inspection as inspection
import modules_extraction as extraction

from network_models.resnet50.resnet50 import get_model

if __name__ == '__main__':
    # ---PART 1: NETWORK INSPECTION---

    # prepare relevant directories
    this_dir = os.path.dirname(os.path.realpath(__file__))
    weights_dir = os.path.join(this_dir, 'network_models', 'resnet50')
    tensorboard_dir = os.path.join(this_dir, 'tensorboard', 'resnet50')
    saved_data_dir = os.path.join(this_dir, 'saved_data', 'resnet50')
  
    # get model
    gtsrb_num_classes = 43
    model = get_model(gtsrb_num_classes, weights_dir, pretrained_weights=False, return_transforms=False)
    
    # prepare input size
    input_size = torch.Size([1, 3, 10,10])
    
    # generate graph
    inspection.generate_model_graph(model, input_size, tensorboard_dir=tensorboard_dir)
    
    # list the model layers
    modules = inspection.get_submodules(model)
    inspection.list_submodules_with_filters(modules)
    
    # select the derired submodules
    selected_modules = inspection.select_submodules(modules, saved_data_dir)
    
    # ---PART 2: RUN INFERENCE AND EXTRACT DATA---
    
    # prepare relevant directories
    modules_data_dir = os.path.join(saved_data_dir, 'selected_modules_data')
    extraction.create_modules_dirs(selected_modules, modules_data_dir)
    
    # save selected modules' parameters
    extraction.save_selected_modules_parameters(selected_modules, modules_data_dir)
    
    # get device and transfer model
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu'
    model = model.to(device)
    
    # prepare input (if a dataset is available, load that instead)
    example_input = extraction.generate_random_input(input_size).to(device)
    
    # install hooks on selected modules
    handles = extraction.install_extraction_hooks(selected_modules, modules_data_dir)
    
    # run inference
    with torch.no_grad():
        model(example_input)
    
    # remove hooks
    extraction.remove_extraction_hooks(handles)