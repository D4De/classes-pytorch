import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

blobs_dir = '/home/miele/WORKSPACE/results-storage/error_simulation/application_blobs'
final_reports_dir = '/home/miele/WORKSPACE/results-storage/network_reports'
output_dir = '/home/miele/WORKSPACE/results-storage/pictures/applev_plots'

params = ['C', 'K', 'W', 'R', 'padding']    # layer hyperparameters to plot; in order: input channels, output channels, input size, kernel size, padding

networks = [
    'alexnet_cifar10',
    'deeplabv3_oxfordpet',
    'mobilenetv2-large_gtsrb',
    'res9_cifar10',
    'res50_cifar10',
    'yolov11_coco',
]
spatial_classes = [
    'Single',
    'FullChannels',
    'MultiChannelBlock',
    'BulletWake',
    'Rectangles',
    'ShatteredChannel',
    'QuasiShatteredChannel',
    'SameRow',
    'SingleBlock',
    'Skip4',
    'SingleChannelRandom',
]


def build_hypers_charts(blob_df: pd.DataFrame, network: str, outdir: str):
    """Since layer hyperparameters in a network never change, prepare a single set of charts with their distribution."""
    colors = cm.tab10(np.linspace(0, 1, len(params)))

    # parameters don't change with configuration: take the first set
    for _, layer_rows in blob_df.groupby('config'):
        break

    layer_rows = layer_rows.sort_values('position')
    x = layer_rows['layer'] # layer names

    fig, ax = plt.subplots(len(params), 1)
    fig.suptitle(f'Hyperparameters for {network}')

    for i, param in enumerate(params):
        y = layer_rows[param]
        ax[i].bar(x,y, color=colors[i])
        ax[i].set_title(param)

        if i < len(params) - 1:
            ax[i].tick_params(labelbottom=False) # disable x ticks
    
    ax[-1].set_xticks(layer_rows['position'], labels=layer_rows['position'])
    ax[-1].tick_params('x', rotation=90, labelsize=7)

    fig.tight_layout()

    save_path = os.path.join(outdir, f'{network}_hypers.png')
    fig.savefig(save_path)
    plt.close()


def build_class_criticality_charts(blob_df: pd.DataFrame, report_dict: dict, network: str, network_outdir: str):
    for config, config_rows in blob_df.groupby('config'):
        # get configuration parameters and build config id
        first_row = config_rows.iloc[0]
        atomicc = first_row['atomic-c'].item()
        atomick = first_row['atomic-k'].item()
        bitwidth = first_row['bitwidth'].item()
        config_id = f'{atomicc}x{atomick}_int{bitwidth}'

        # sort layers by network position
        config_rows = config_rows.sort_values('position')

        x = config_rows['layer']

        # iterate over spatial classes
        for spatial_class in spatial_classes:
            crit_values = config_rows[spatial_class]

            # take list of time exposures from report. Note that some layers may be missing
            time_values = [report_dict[config]['layers'].get(layer, {}).get('Latency', 0) for layer in x]

            # make plot with bar charts for class criticality values and time exposures
            fig, ax = plt.subplots(2,1)
            fig.suptitle(f'{network}, {config_id}, {spatial_class}')

            ax[0].bar(x, crit_values, color='b')
            ax[0].set_title('Criticality')
            ax[0].set_ylim(0,1)
            ax[0].tick_params(labelbottom=False) # disable x ticks

            ax[1].bar(x, time_values, color='g')
            ax[1].set_title('Time exposure')
            ax[1].set_xticks(config_rows['position'], labels=config_rows['position'])
            ax[1].tick_params('x', rotation=90, labelsize=7)

            fig.tight_layout()

            save_path = os.path.join(network_outdir, spatial_class, f'{config_id}.png')
            fig.savefig(save_path)
            plt.close()


def main():
    os.makedirs(output_dir, exist_ok=True)

    for network in networks:
        print(f'Building charts for {network}')
        network_outdir = os.path.join(output_dir, network)
        os.makedirs(network_outdir, exist_ok=True)

        # load application blob
        blob_path = os.path.join(blobs_dir, f'{network}_application.csv')
        blob_df = pd.read_csv(blob_path)

        build_hypers_charts(blob_df, network, network_outdir)

        # load network report
        if network.startswith('mobilenetv2'):   # special naming case for mobilenet
            report_path = os.path.join(final_reports_dir, 'mobilenetv2_gtsrb_final.yaml')
        else:
            report_path = os.path.join(final_reports_dir, f'{network}_final.yaml')
        with open(report_path) as f:
            report_dict = yaml.load(f, yaml.SafeLoader)

        # make output directories for each spatial class
        for spatial_class in spatial_classes:
            os.makedirs(os.path.join(network_outdir, spatial_class), exist_ok=True)

        build_class_criticality_charts(blob_df, report_dict, network, network_outdir)


if __name__ == '__main__':
    main()