import os
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

fit_values_path = '/home/miele/WORKSPACE/results-storage/pictures/results_dfs/cross_layer_aggregates/network_fit_values.csv'
output_dir = '/home/miele/WORKSPACE/results-storage/pictures/FIT_charts'

if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)

    # load FIT df
    network_fit_df = pd.read_csv(fit_values_path, index_col=0)
    configs = list(network_fit_df.columns)

    cmap = cm.tab10(np.linspace(0, 1, len(configs)))

    for network, row in network_fit_df.iterrows():
        fig, ax = plt.subplots()
        ax.bar(configs, row, color=cmap)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('FIT')
        ax.set_title(f'FIT values for {network}')
        ax.set_xticklabels(configs, rotation=45, ha="right")
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'FIT_{network}.png')
        fig.savefig(save_path, dpi=100)
        plt.close()