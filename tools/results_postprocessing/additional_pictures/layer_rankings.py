"""
Takes the raw layer results and builds configuration rankings for both observability and susceptibility. In particular, the top-3
and bottom-3 are determined and saved, so that the rankings for the two values can be compared to pinpoint changes/inversions.
"""
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

K = 3    # how long the top and bottom pieces of the ranking should be
REGIME_THRESHOLD = 0.8  # if fetch/compute is higher than this, the layer is memory-bound, otherwise it is compute-bound

raw_layer_results_path = '/home/miele/WORKSPACE/results-storage/network_reports/raw_layer_results.csv'

output_path = '/home/miele/WORKSPACE/results-storage/network_reports/layer_rankings.yaml'
pictures_dir = '/home/miele/WORKSPACE/results-storage/pictures'


def mean_value_difference(values: list):
    return np.abs(np.diff(np.array(values)).mean()).item()

def build_layer_rankings(layer_rows: pd.DataFrame):
    def _build_value_ranking(value: str):
        # sort entries by value and compute ranking
        sorted_rows = layer_rows.sort_values(value, ascending=False)
        sorted_value_rows = sorted_rows[value]

        # determine regimes
        regimes = (sorted_rows['fetch_over_compute'] > REGIME_THRESHOLD).map({False: 'compute', True: 'memory'})
        ranking = [str(param_tuple) for param_tuple in zip(regimes, sorted_rows['c_over_atomicc'], sorted_rows['AtomicK'], sorted_rows['bitwidth'])]

        topk = ranking[:K]
        bottomk = ranking[-K:]

        topk_mean_difference = mean_value_difference(sorted_value_rows.iloc[:K])
        bottomk_mean_difference = mean_value_difference(sorted_value_rows.iloc[-K:])

        top_bottom_distance = (sorted_value_rows.iloc[K-1] - sorted_value_rows.iloc[-K]).item()

        layer_result = {
            'ranking':              ranking,
            'values':               sorted_value_rows.tolist(),
            'topk':                 topk,
            'bottomk':              bottomk,
            'topk_diff' :           topk_mean_difference,
            'bottomk_diff':         bottomk_mean_difference,
            'top_bottom_distance':  top_bottom_distance,
        }

        return layer_result
    

    obs_ranking = _build_value_ranking('observability')
    suscept_ranking = _build_value_ranking('susceptibility')

    # determine ranking differences
    # top/bottom are exactly the same
    topk_exact_match = obs_ranking['topk'] == suscept_ranking['topk']
    bottomk_exact_match = obs_ranking['bottomk'] == suscept_ranking['bottomk']

    # top/bottom elements are the same
    topk_elements_match = set(obs_ranking['topk']) == set(suscept_ranking['topk'])
    bottomk_elements_match = set(obs_ranking['bottomk']) == set(suscept_ranking['bottomk'])

    # number of top/bottom elements that differ
    topk_different_elements = len(set(obs_ranking['topk']).intersection(set(suscept_ranking['topk'])))
    bottomk_different_elements = len(set(obs_ranking['bottomk']).intersection(set(suscept_ranking['bottomk'])))

    total_result = {
        'observability':                obs_ranking,
        'susceptibility':               suscept_ranking,
        'topk_exact_match':             topk_exact_match,
        'bottomk_exact_match':          bottomk_exact_match,
        'topk_elements_match':          topk_elements_match,
        'bottomk_elements_match':       bottomk_elements_match,
        'topk_different_elements':      topk_different_elements,
        'bottomk_different_elements':   bottomk_different_elements,
    }

    return total_result


def make_rankings_file():
    print('Building rankings file')
    total_result = {}

    # load raw results csv
    raw_results_df = pd.read_csv(raw_layer_results_path)

    for layer_id, layer_rows in raw_results_df.groupby(['network', 'layer']):
        layer_name = f'{layer_id[0]}_{layer_id[1]}'
        print(f'Computing rankings for {layer_name}')
        total_result[layer_name] = build_layer_rankings(layer_rows)

    with open(output_path, 'w') as f:
        yaml.dump(total_result, f, sort_keys=False)


def plot_all_rankings(output_dir: str):
    """Plots all observability and susceptibility rankings in a single figure."""
    print(f'Plotting rankings')
    with open(output_path) as f:
        rankings_dict = yaml.load(f, yaml.SafeLoader)
    
    num_layers = len(rankings_dict)
    # arrange the layers in a square grid
    nsize = np.ceil(np.sqrt(num_layers)).astype(np.int64).item()

    fig = plt.figure()
    fig.set_size_inches(19.20 * 2, 10.8 * 2)

    for i, (layer_name, layer_dict) in enumerate(rankings_dict.items()):
        ax = fig.add_subplot(nsize, nsize, i+1)
        ax.set_axis_off()
        ax.set_title(layer_name)
        ax.title.set_size(7)

        # rankings are lists of tuples
        obs_ranking = layer_dict['observability']['ranking']
        suscept_ranking = layer_dict['susceptibility']['ranking']

        # prepare dataframe with the two rankings as columns
        table_data = list(zip(obs_ranking, suscept_ranking))

        obs_colors = cm.tab10(np.linspace(0, 1, len(obs_ranking)))
        suscept_colors = [None] * len(obs_ranking)

        # find each observation ranking entry in the susceptibility ranking and assign the same color to it
        for obs_rank_pos, rank_tuple in enumerate(obs_ranking):
            suscept_rank_pos = suscept_ranking.index(rank_tuple)
            suscept_colors[suscept_rank_pos] = obs_colors[obs_rank_pos]

        cell_colors = [[obs_col, suscept_col] for obs_col, suscept_col in zip(obs_colors, suscept_colors)]
        the_table = ax.table(cellText=table_data, cellColours=cell_colors, rowLabels=None, colLabels=['OBS', 'SDC'], loc='center', fontsize=7)
        # the_table.scale(1,0.6)
    
    plt.tight_layout()

    plot_output_path = os.path.join(output_dir, 'layer_rankings_plot.png')
    fig.savefig(plot_output_path, dpi=100)
    plt.close()


def plot_suscept_rankings(output_dir: str):
    """Plots all rankings in a single figure, but only the susceptibility ones."""
    print(f'Plotting susceptibility rankings')
    with open(output_path) as f:
        rankings_dict = yaml.load(f, yaml.SafeLoader)
    
    num_layers = len(rankings_dict)
    # arrange the layers in a square grid
    nsize = np.ceil(np.sqrt(num_layers)).astype(np.int64).item()

    fig = plt.figure()
    fig.set_size_inches(19.20 * 2, 10.8 * 2)

    for i, (layer_name, layer_dict) in enumerate(rankings_dict.items()):
        ax = fig.add_subplot(nsize, nsize, i+1)
        ax.set_axis_off()
        ax.set_title(layer_name)
        ax.title.set_size(7)

        # rankings are lists of tuples
        # table_data = [[entry] for entry in layer_dict['susceptibility']['ranking']]
        ranking = layer_dict['susceptibility']['ranking']
        regimes = []
        parameters = []
        for entry in ranking:
            strip_entry = entry[1:-1] # strip parentheses
            pieces = strip_entry.split(',')
            regimes.append(pieces[0][1:-1])
            parameters.append((f'CAC={pieces[1]}, AK={pieces[2]}, BW={pieces[3]}'))

        table_data = list(zip(regimes, parameters))

        colors = [[entry, entry] for entry in cm.tab10(np.linspace(0, 1, len(table_data)))]

        the_table = ax.table(cellText=table_data, rowLabels=None, colLabels=['Regime', 'Parameters'], cellColours=colors, loc='center', fontsize=7)
        # the_table.scale(1,0.6)
    
    plt.tight_layout()

    plot_output_path = os.path.join(output_dir, 'layer_suscept_rankings_plot.png')
    fig.savefig(plot_output_path, dpi=100)
    plt.close()


def main():
    output_dir = os.path.join(pictures_dir, 'layer_rankings')
    os.makedirs(output_dir, exist_ok=True)

    make_rankings_file()
    plot_all_rankings(output_dir)
    plot_suscept_rankings(output_dir)


if __name__ == '__main__':
    main()