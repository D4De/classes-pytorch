"""
A more extensive run of CLASSES on a trained ResNet50 with the GTSRB dataset.
One fault is generated for each injectable layer in the network.
Remember to run this example from CLASSES' root folder.
"""
import torch
import torchvision

import os
from tqdm import tqdm

from classes.error_models.error_model import ErrorModel
from classes.fault_generator.fault_generator import FaultGenerator
from classes.simulators.pytorch.module_profiler import module_shape_profiler
from classes.simulators.pytorch.pytorch_fault import PyTorchFault
from classes.simulators.pytorch.simulator_hook import applied_hook, create_simulator_hook
from classes.simulators.pytorch.error_model_mapper import create_module_to_generator_mapper
from classes.simulators.pytorch.fault_list import PyTorchFaultList, PyTorchFaultListMetadata
from classes.simulators.pytorch.fault_list_datasets import FaultListFromTarFile

# TODO: add this path as a commandline argument
import sys
sys.path.insert(0, 'E:/University/MasterThesis/classes/examples/resnet50')

# TODO: in general, the network model should be passed as an argument
from model import get_resnet50_model
from metrics import accuracy
from gtsrb_transforms import data_transform


if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.realpath(__file__))
    
    # TODO: add as commandline argument
    data_path = os.path.join(this_dir, 'data')
    assert os.path.exists(data_path), "Dataset directory not found, check path."
    
    # TODO: add as commandline argument
    trained_model_path = os.path.join(this_dir, 'checkpoints/resnet50_gtsrb_best.pth')
    assert os.path.exists(trained_model_path), "Trained model file not found, check path."
    
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu'
    print(f'Using {device}')
    
    # TODO: add as optional commandline argument
    batch_size = 512
    
    print('Getting GTSRB test dataset')
    testset = torchvision.datasets.GTSRB(
        data_path,
        split='test',
        transform=data_transform(),
        download=True)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        pin_memory=True)
    
    gtsrb_num_classes = 43
    gtsrb_test_num_samples = len(testloader.dataset)
    criterion = torch.nn.CrossEntropyLoss()
    
    model = get_resnet50_model(gtsrb_num_classes, pretrained_weights=False, return_transforms=False)
    print('Loading pre-trained weights')
    weights = torch.load(trained_model_path)['model']
    model.load_state_dict(weights)
    model.to(device)
    model.eval()


    # perform a golden run to collect results
    golden_outputs = []
    
    with torch.no_grad():
        scores = torch.zeros((gtsrb_test_num_samples, gtsrb_num_classes), device=device)
        labels = torch.zeros((gtsrb_test_num_samples), device=device, dtype=torch.long)
        
        for batch_id, (image, label) in enumerate(tqdm(testloader, desc='Performing golden run', colour='yellow')):
            image = image.to(device)
            label = label.to(device)
            
            output = model(image)
            golden_outputs.append(output)

            start_idx = batch_id * testloader.batch_size
            end_idx = start_idx + output.size(0)
            scores[start_idx:end_idx, :] = output
            labels[start_idx:end_idx] = label
        
        top_1 = accuracy(scores, labels, k=1)
        top_5 = accuracy(scores, labels, k=5)
        loss = criterion(scores, labels)

    print('Golden results:')
    print(f'Top-1 accuracy: {top_1:.2f}%')
    print(f'Top-5 accuracy: {top_5:.2f}%')
    print(f'Cross Entropy loss: {loss:.3f}')
    
    
    # now use CLASSES to perform error simulation; this part only targets one layer
    error_model_folder = './error_models/models'
    
    conv_error_models = ErrorModel.from_model_folder(error_model_folder, 'conv_gemm')

    sample_images, _ = next(iter(testloader))
    shape_profile = module_shape_profiler(model, input_data=sample_images) 

    for layer_name, shape in shape_profile.items():
        print(f'* Module name: {layer_name} --> Shape: {shape}')

    injected_layer_name = 'conv1'   
    target_layer = model.get_submodule(injected_layer_name)

    generator = FaultGenerator(conv_error_models)

    faults = []
    for i in range(10):
        fault = generator.generate_mask(
            output_shape=shape_profile[injected_layer_name]
        )
        
        torch_fault = PyTorchFault.from_fault(fault)
        torch_fault.to(device=device)
        faults.append(torch_fault)

    
    tolerance = 10**-3
    
    for fault in faults:
        print(f'Output Shape: {fault.corrupted_value_mask.shape}')
        print(f'Spatial Pattern: {fault.spatial_pattern_name}')

        num_masked = 0
        num_sdc_safe = 0
        num_sdc_critical = 0
        
        error_simulator_pytorch_hook = create_simulator_hook(fault)

        with applied_hook(target_layer, error_simulator_pytorch_hook), torch.no_grad():
            scores = torch.zeros((gtsrb_test_num_samples, gtsrb_num_classes), device=device)
            labels = torch.zeros((gtsrb_test_num_samples), device=device, dtype=torch.long)
            
            for batch_id, (image, label) in enumerate(tqdm(testloader, colour='red')):
                image = image.to(device)
                label = label.to(device)
                
                output = model(image)
                
                # compare faulty output and golden output
                # check if each row in the output (the single output vector for one image) is equal
                # to the corresponding golden row 
                corrupted_rows_mask = find_corrupted_output_rows(output, golden_outputs[batch_id], tolerance)
                num_corrupted_rows = corrupted_rows_mask.count_nonzero()
                
                num_masked += (corrupted_rows_mask.numel() - num_corrupted_rows)
                
                # for each output row that does not match the golden counterpart, get the top-5
                # ranking (indices only) and compare with the golden one
                corrupted_rows = output[corrupted_rows_mask]
                golden_rows = golden_outputs[batch_id][corrupted_rows_mask]
                
                corrupted_rankings_mask = find_ranking_shifts(corrupted_rows, golden_rows)
                num_equal_rankings = corrupted_rankings_mask.count_nonzero()
                
                num_sdc_safe += num_equal_rankings
                num_sdc_critical += (corrupted_rankings_mask.numel() - num_equal_rankings)

                # store scores and labels to compute accuracy
                start_idx = batch_id * testloader.batch_size
                end_idx = start_idx + output.size(0)
                scores[start_idx:end_idx, :] = output
                labels[start_idx:end_idx] = label
            
            top_1 = accuracy(scores, labels, k=1)
            top_5 = accuracy(scores, labels, k=5)
            loss = criterion(scores, labels)

            print(f'Top-1 accuracy: {top_1:.2f}%')
            print(f'Top-5 accuracy: {top_5:.2f}%')
            print(f'Cross Entropy loss: {loss:.3f}')
    
    
    
    
    # let's now try a more extensive error simulation by generating a few faults
    # for each targetable layer
    fault_list_path = example_folder_path + '/resnet_gtsrb_fault_list.tar'
    num_faults_per_module = 1 # just one for now, for testing purposes

    # if the fault list does not exist, create it
    if not os.path.exists(fault_list_path):
        print(f'Fault List {fault_list_path} does not exist in the current folder.' \
              f' Generating a new one with {num_faults_per_module} faults per network layer.')

        module_to_generator_mapping = create_module_to_generator_mapper(
            model_folder_path=error_model_folder,
            conv_strategy='conv_gemm'
        )
        
        # generate the list and save it to the specified path
        fault_list = PyTorchFaultList(model, input_data=sample_images, module_to_fault_generator_fn=module_to_generator_mapping)
        fault_list.generate_and_persist_fault_list(fault_list_path, num_faults_per_module)
    
    # load the fault list metadata
    fault_list_info = PyTorchFaultListMetadata.load_fault_list_info(fault_list_path)

    for injectable_module_name in fault_list_info.injectable_layers:

        module = model.get_submodule(injectable_module_name)

        # for each network module for which a fault generator exists, build a fault dataset
        fault_list_dataset = FaultListFromTarFile(
            fault_list_path, injectable_module_name
        )
        
        fault_list_loader = torch.utils.data.DataLoader(
            fault_list_dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=True,
        )
        
        for fault in fault_list_loader:
            print(f'Module: {fault.module_name} | Type: {type(module).__name__}')
            print(f'Output Shape: {fault.corrupted_value_mask.shape}')
            print(f'Spatial Pattern: {fault.spatial_pattern_name}')
           
            fault.to(device=device)

            # create and apply the hook
            error_simulator_pytorch_hook = create_simulator_hook(fault)

            with applied_hook(module, error_simulator_pytorch_hook):
                top_1, top_5, loss = run_inference(model, testloader, gtsrb_num_classes, device)
                print(f'Top-1 accuracy: {top_1:.2f}%')
                print(f'Top-5 accuracy: {top_5:.2f}%')
                print(f'Cross Entropy loss: {loss:.3f}')