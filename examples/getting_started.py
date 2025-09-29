import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import classes
from classes.error_models.error_model import ErrorModel
from classes.fault_generator.fault_generator import FaultGenerator
from classes.simulators.pytorch.module_profiler import module_shape_profiler
from classes.simulators.pytorch.pytorch_fault import PyTorchFault
from classes.simulators.pytorch.simulator_hook import applied_hook, create_simulator_hook


# In this example we show a simple use of classes, to apply faults to the output of
# a convolutional layer of lenet.
# Steps are
# 1. Load error model from file
# 2. Profile the model, to get list of layer names and output shapes.
# 3. Choose a layer and generate a list of Fault object (containing fault mask, and values)
# 4. Convert the Fault objects to PyTorchFault (to apply them to pytorch models)
# 5. Create a pytorch hook for applying the error to the output feature map of the target layer
# 6. Run inference with error applied
# 7. Remove the hook from the model and apply another error hook

# RUN THIS FILE FROM THE ROOT OF THIS REPO USING
# python -m examples.getting_started

# In the advanced.py tutorial we will see how to automate some of these steps
# and how to save a fault list to make reproducible experiments

BATCH_SIZE = 64

# Define LeNet architecture
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

    # Perform inference to evaluate the model [No faults]
    inference(net, testloader)

    # FROM HERE WE START INTRODUCING CLASSES

    # 1. Load error model from file
    conv_error_models = ErrorModel.from_model_folder('error_models/models', 'conv_gemm')
    # There are other ways to load the error model from a json error model file lie
    # ErrorModel.from_json_dict(dict)

    # 2. Profile the model, to get list of layer names and output shapes.

    # We need to get a dummy image (or a batch) to profile the shapes of each module
    images, labels = next(iter(testloader)) # Get a sample batch from dataloader
    shape_profile = module_shape_profiler(net, input_data=images[0]) 
    # (Alternatively we can supply input_shape if we know the shape of the image beforehand)
    # Other arguments allow to:
    # - Select only certain layers to be profiled (in case of errors)
    # - Profile input shapes instead of outputs


    # The output of the profiler is a dictionary with:
    # - As keys: the fully qualified path of the pytorch modules inside the network
    # - As values: a pytorch shape object, representing the output shape of the model
    for layer_name, shape in shape_profile.items():
        print(f'* Module name: {layer_name} --> Shape: {shape}')


    # 3. Choose a layer and generate a list of Fault object (containing fault mask, and values)

    # Select a layer name, to target for error simulation
    injected_layer_name = 'conv1'   
    # Get the pytorch module object of the layer
    target_layer = net.get_submodule(injected_layer_name)

    generator = FaultGenerator(
        conv_error_models,
        # Optional parameters for customizing fault generation, no need to specify them
        fixed_spatial_class=None, # <- Here you can force a spatial class. If None it will generate randomly the spatial classes following distributions contained in the model
        fixed_spatial_parameters=None, # <- Same for the parameters that characterize the spatial distribution of the pattern
        fixed_domain_class=None,  # <- Same for domain classes (The values that replace the correct ones)
    )

    faults = []
    # Generate 10 fault batches
    for i in range(10):
        fault = generator.generate_mask(
            output_shape=shape_profile[injected_layer_name],
        )
        
        # 4. Convert the Fault objects to PyTorchFault (to apply them to pytorch models)
        torch_fault = PyTorchFault.from_fault(fault)
        torch_fault.to(device=device)
        faults.append(torch_fault)


    for fault in faults:
        print(f'Output Shape: {fault.corrupted_value_mask.shape}')
        print(f'Spatial Pattern: {fault.spatial_pattern_name}')
        # 5. Create a pytorch hook for applying the error to the output feature map of the target layer

        # Returns a PyTorch Hook (read pytorch docs to see how they work), that modifies the output of the
        # layer corrupting it according to the fault. The hook should be applied to target layer, and then
        # the inference must be run
        error_simulator_pytorch_hook = create_simulator_hook(fault)

        # 6. Run inference with error applied

        # Applied hook is a context manager that applies a PyTorch hook to a pytorch module (in this case the target layer)
        # and then when exited (after completing inference, and leaving the with block) automatically removes
        # the hook fron the network, restoring it to the golden state
        # The hook is removed even in case of an exception
        with applied_hook(target_layer, error_simulator_pytorch_hook):
            # Here net has the error_simulator_pytorch_hook applied, and will corrupt the output of target_layer
            inference(net, testloader)

        # 7. Remove the hook from the model and apply another error hook


        # Here, outside of the with it will run regularly without any error simulation
        # This avoids to forget having multiple error simulation applied
        # You can combine hooks like this, chaining multiple with blocks or using python ExitStack 




if __name__ == '__main__':
    main()