import os
import sys
import torch
import pandas as pd

def get_conv_hyperparameters(model: torch.nn.Module, input_size: torch.Size, dummy_input, device = torch.accelerator.current_accelerator()):
    """
    Extracts the hyperparameters of all convolutional layers in a network.
    Produces a Pandas DataFrame with these columns:
    Layer,Channels_out,Channels_in,Input_size,Kernel_size,Padding,Stride
    """
    df_rows: list[list] = []
    input_sizes: list[int] = []

    def _make_input_hook():
        def _hook(module, input, output):
            input_sizes.append(input[0].shape[-1])
            return output
        return _hook
    
    handles: list[torch.utils.hooks.RemovableHandle] = []

    # extract parameters and install hooks
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            df_rows.append([
                name,
                mod.out_channels,
                mod.in_channels,
                mod.kernel_size[0],
                mod.padding[0],
                mod.stride[0],
            ])

            handle = mod.register_forward_hook(_make_input_hook())
            handles.append(handle)

    # forward pass to get input sizes
    if dummy_input is None:
        dummy_input = torch.rand((1,3,input_size, input_size), device=device)

    _ = model(dummy_input)
    for handle in handles:
        handle.remove()

    # add input_sizes to rows
    for row, input_size in zip(df_rows, input_sizes):
        row.insert(3, input_size)

    # build dataframe
    hyper_df = pd.DataFrame(df_rows, columns=['layer','K','C','W','R','padding','stride'])
    return hyper_df


def count_network_layers(model: torch.nn.Module):
    num_conv = 0
    num_linear = 0

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            num_conv += 1
        elif isinstance(mod, torch.nn.Linear):
            num_linear += 1
    
    return num_conv, num_linear


def main():
    if len(sys.argv) != 5:
        raise ValueError(f'Usage: python get_module_hyperparameters.py <tcad_dir> <network> <dataset> <savedir>')

    tcad_dir = os.path.realpath(sys.argv[1])
    network = sys.argv[2]
    dataset = sys.argv[3]
    savedir = os.path.realpath(sys.argv[4])

    sys.path.insert(1, tcad_dir)
    import models.zoo as zoo

    model, loader = zoo.models_factory(dataset, network, 1)
    for img, _ in loader: break
    hyper_df = get_conv_hyperparameters(model, None, img, device='cpu')

    save_path = os.path.join(savedir, f'{network}_{dataset}_hyper.csv')
    hyper_df.to_csv(save_path, index=False)

    # count network layers and update csv
    layer_csv_path = os.path.join(savedir, 'layer_counts.csv')
    if os.path.exists(layer_csv_path):
        layer_count_df = pd.read_csv(layer_csv_path, index_col=0)
    else:
        layer_count_df = pd.DataFrame(columns=['num_convolution', 'num_linear', 'total'])
    
    num_conv, num_linear = count_network_layers(model)
    layer_count_df.loc[network] = [num_conv, num_linear, num_conv+num_linear]
    layer_count_df.to_csv(layer_csv_path)


if __name__ == '__main__':
    main()