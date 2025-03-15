import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from classes.simulators.pytorch.error_model_mapper import create_module_to_generator_mapper
from classes.simulators.pytorch.fault_list import PyTorchFaultList, PyTorchFaultListMetadata
from classes.simulators.pytorch.fault_list_datasets import FaultListFromTarFile
from classes.simulators.pytorch.pytorch_fault import PyTorchFault
from classes.simulators.pytorch.simulator_hook import applied_hook, create_simulator_hook

BATCH_SIZE = 16


# Read this tutorial after completing getting_started.
# In this tutorial we will use LeNet, to create a fault list for convolutional and pooling layers
# and store it to reproduce the experiments other times



class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# NOTE: Each layer in the module MUST not be reused multiple time. 
# EACH operator defined in init MUST BE APPLIED ONLY ONCE IN THE WHOLE NETWORK
# Pay attention that this constraint is respected when using models included in models libraries or with code written by others.

# Also do not use functional operators if you want to inject them. Define always operators 
# in the __init__

# If these constraints are not respected change the code of the model.

# WHY THESE RESTRICTIONS?
# Before generating faults, CLASSES profiles modules output shapes, and uses their fully qualified
# name to index them in the result of classes.simulators.pytorch.module_profiler.module_shape_profiler() function
# If there are two layers with the same name module_shape_profiler() returns a bad output
# If there are two layers with the same name module_shape_profiler() returns a bad output and an error will
# be raised during error simulation.

"""
Example DON'T DO THIS: 
```
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2) # <-- Pool defined once
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) # <-- Used multiple times
        x = self.pool(torch.relu(self.conv2(x))) # <-- Used multiple times
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ALSO AVOID USING torch functional operators if you want to profile them and inject them with classes
```
"""


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_or_load_model(net, model_path, trainloader):
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        net.load_state_dict(torch.load(model_path))
    else:
        print("No pre-trained model found. Training from scratch...")
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # Train the model
        print("Training started...")
        for epoch in range(5):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        # Save the model
        torch.save(net.state_dict(), model_path)
    
    return net


def inference(net, testloader):
    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total}%')



def main():
    # Load and preprocess MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                    download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=2)

    # Initialize the model
    net = LeNet().to(device)

    net = train_or_load_model(net, 'examples/lenet_mnist.pth', trainloader)

    # Perform inference to evaluate the model
    inference(net, testloader)

    # FROM HERE WE START INTRODUCING CLASSES

    sample_images, labels = next(iter(testloader)) # Get a sample batch

    fault_list_path = 'examples/lenet_fault_list.tar'
    n_faults_per_module = BATCH_SIZE 

    # Here we create a fault_list if not exists
    # If it exists we skip directly to load the fault list from file, to reproduce existing experiments
    # If you run this script twice you should get the same results as before
    if not os.path.exists(fault_list_path):
        print('Fault List does not exists. Generating a new one')
        # 1. Instatiate a PyTorchFaultList object
        
        # This function loads the error models from the folder and maps
        # pytorch module type to an error models contained in model_folder_path
        # depending on the operator (pooling layers use pool.json)
        module_to_generator_mapping = create_module_to_generator_mapper(
            model_folder_path='error_models/models', # <- Model folder path
            conv_strategy='conv_gemm' # <- Select which convolution methods to use depending on the startegy
        )
        # We pass a sample image to profile the layer (will be done inside PyTorchFaultList) and the mapper
        fault_list  = PyTorchFaultList(net, input_data=sample_images, module_to_fault_generator_fn=module_to_generator_mapping)
        # Generate and save the fault list in a tarball file. 
        # The tarball contains a file for each profiled module that has error models (according to module_to_generator_mapping)
        # Each file contains n_faults_per_module errors
        # All files are grouped in a .tar file, togheter with another file containing info and metadata of the fault list
        fault_list.generate_and_persist_fault_list(fault_list_path, n_faults_per_module)
    
    # Now the fault_list exists and is persisted to file

    # Load the metadata to have information about the faultlist (for example the number of faults, and the module profile)
    # This will just load the metadata file of the fault list
    fault_list_info = PyTorchFaultListMetadata.load_fault_list_info(fault_list_path)

    # Iterate between all injectable modules (obtained from the metadata)
    for injectable_module_name in fault_list_info.injectable_layers:

        module = net.get_submodule(injectable_module_name)

        # Create a torch.utils.Dataset from the tar file containing the fault list
        fault_list_dataset = FaultListFromTarFile(
            fault_list_path, injectable_module_name
        )
        # Create a dataloader, so that the faultlist can be loaded and iterated efficiently
        # NOTE: Always specify the batch_size=None. Other batch sizes may not be supported as for now
        fault_list_loader = torch.utils.data.DataLoader(
            fault_list_dataset,
            batch_size=None,
            num_workers=8,
            pin_memory=True,
        )
        
        # Iterate trough the fault list of the module
        for fault in fault_list_loader:
            print(f'Module: {fault.module_name} Type: {type(module).__name__}')
            print(f'Output Shape: {fault.corrupted_value_mask.shape}')
            print(f'Spatial Pattern: {fault.spatial_pattern_name}')
            # Move the fault to GPU (if available) for faster simulation
            fault.to(device=device)

            # As before we create the fault hook and we apply it using applied_hook
            error_simulator_pytorch_hook = create_simulator_hook(fault)

            with applied_hook(module, error_simulator_pytorch_hook):
                # Run inference with error simulation
                inference(net, testloader)
                # The results of the inference may be used for calculating the 
                # layer vulnerability factor (LVF) comparing the faults to golden


if __name__ == '__main__':
    main()