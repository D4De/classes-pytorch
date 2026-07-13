"""
For each configuration, plots a stacked bar chart with the average spatial class frequencies per hardware unit; the final column
is the average spatial class frequencies layer-wise.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

output_dir = '/home/miele/WORKSPACE/results-storage/pictures/class_distributions'

layer_results_path = '/home/miele/WORKSPACE/results-storage/network_reports/raw_layer_results.csv'
unit_results_dir = '/home/miele/WORKSPACE/results-storage/network_reports'

units = {   # names used in the reports along with names to display
    'top.conv.csb'  : 'CSB',
    'top.conv.csc'  : 'CSC',
    'top.conv.dl'   : 'DL',
    'top.conv.wl'   : 'WL',
    'top.conv.cdma' : 'CDMA',
    'top.conv.cbuf' : 'CBUF',
    'top.conv.cmac' : 'CMAC',
    'top.conv.cacc' : 'CACC',
    'top.sdp'       : 'SDP',
}

spatial_classes = { # same order they will be plotted in the bar charts; it's recommended to order by severity
    'class-Single'              : 'Single',
    'class-SingleChannelRandom' : 'SingleChannelRandom',
    'class-SingleChannelBlock'  : 'SingleChannelBlock',
    'class-MultiChannelBlock'   : 'MultiChannelBlock',
    'class-BulletWake'          : 'BulletWake',
    'class-MultiChannelRandom'  : 'MultiChannelRandom',
    'class-SingleFullChannel'   : 'SingleFullChannel',
    'class-MultiFullChannels'   : 'MultiFullChannels',
}

cmap = matplotlib.colormaps['RdYlGn_r']

def get_config_frequencies(config, layer_results_df):
    """Builds a dictionary of NumPy arrays that associates each spatial class to its average frequencies for all units/layers."""
    result = { sp_class_name : np.ndarray(len(units) + 1, dtype=np.float32) for sp_class_name in spatial_classes.values() }

    # get layer rows for the provided configuration
    layer_rows = layer_results_df[layer_results_df['config'] == config][spatial_classes.keys()]
    # average the layer frequencies
    avg_layer_freqs = layer_rows.mean()

    for sp_class, name in spatial_classes.items():
        result[name][-1] = avg_layer_freqs[sp_class]

    # for each unit, load the report
    for i, unit in enumerate(units.keys()):
        unit_report_path = os.path.join(unit_results_dir, f'results_{unit}.csv')
        unit_results_df = pd.read_csv(unit_report_path)
        # get unit rows for the provided configuration and average
        unit_rows = unit_results_df[unit_results_df['config'] == config][spatial_classes.keys()]
        avg_unit_freqs = unit_rows.mean()

        for sp_class, name in spatial_classes.items():
            result[name][i] = avg_unit_freqs[sp_class]
    
    return result


def make_fig(
    config: str,
    weights: dict,
    save_path: str,
    figsize=(10, 5),
): 
    bar_labels = list(units.values()) + ['Layers']
    cmap_step = 1.0/len(bar_labels)
    bar_colors = [cmap(value) for value in np.arange(cmap_step, 1.0 + cmap_step, cmap_step)]

    fig, ax = plt.subplots()
    bottom = np.zeros(len(bar_labels))

    for i, (sp_class_name, frequencies) in enumerate(weights.items()):
        bars = ax.bar(bar_labels, frequencies, label=sp_class_name, color=bar_colors[i], bottom=bottom)
        bottom += frequencies

    ax.set_title(f'Avg. spatial class frequencies for {config}')
    ax.set_xlabel("Unit")
    ax.set_ylabel("Spatial Class Distribution")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right")

    # add legend
    figsize = (round(figsize[0] * 1.25), figsize[1])
    fig.set_size_inches(figsize)
    ax.legend(title="Spatial Class", loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close()


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)

    # load layer results
    layer_results_df = pd.read_csv(layer_results_path)
    # get configurations
    configs = list(layer_results_df['config'].unique())
    
    for config in configs:
        config_weights = get_config_frequencies(config, layer_results_df)
        save_path = os.path.join(output_dir, f'{config}.png')
        make_fig(config, config_weights, save_path)
