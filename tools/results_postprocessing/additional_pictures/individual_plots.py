import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

results_path = '/home/miele/WORKSPACE/results-storage/network_reports/results.csv'
raw_results_path = '/home/miele/WORKSPACE/results-storage/network_reports/raw_layer_results.csv'
outdir = '/home/miele/WORKSPACE/results-storage/network_reports/pictures'

CMAP_RED_TO_GREEN = 'RdYlGn_r'


def extract_configuration_info(config_id: str):
    pieces = config_id.split('_')
    C, K = pieces[1].split('x') # skip "nv_", then split on x in "CxK"
    precision = pieces[-1][3:] # strip "int" from last piece
    return int(C), int(K), int(precision)


def calculate_trendline(x_data, y_data):
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    return p(x_data)


def make_categorical(values: np.ndarray):
    # converts all values in a numerical array into string representations. The values must be representable as integers
    return np.char.mod('%d', values)


def plot_layer_sorted_by(layer_id: tuple, layer_df: pd.DataFrame, col_name: str, by: str, xlabel: str, outdir: str, yrange):
    """
    Plots each layer across the configurations. Produces a plot for the values in the provided column.
    Values are sorted according to the provided 'by' parameter (column).
    """
    network, layer = layer_id

    # sort layers
    layer_rows = layer_df.sort_values(by)

    x_values = layer_rows[by]
    if len(x_values.unique()) < 2:  # avoid plotting if there are not at least two different parameter values
        return

    y_values = layer_rows[col_name]
    y_trendline = calculate_trendline(x_values, y_values)

    x_values = make_categorical(x_values)

    plt.clf()
    plt.gca().set_ylim(yrange)
    plt.plot(x_values, y_values, 'bo', x_values, y_trendline, "r-")
    plt.suptitle(f'{network}_{layer}')
    plt.xlabel(xlabel)
    plt.ylabel(col_name.title())

    savepath = os.path.join(outdir, f'{network}_{layer}.png')
    plt.savefig(savepath)


def plot_every_layer(df: pd.DataFrame, outdir: str):
    """For each layer in the dataframe, produces a plot of its SDC frequencies across the configurations."""
    os.makedirs(outdir, exist_ok=True)

    sorting_parameters = ['c_over_atomicc', 'k_over_atomick', 'bitwidth', 'TileSize']
    parameter_labels   = ['C/AtomicC', 'K/AtomicK', 'Bitwidth', 'TileSize']

    # find observability and susceptibility ranges to make the plots uniform
    obs_range = (df['observability'].min(), df['observability'].max())
    suscept_range = (df['susceptibility'].min(), df['susceptibility'].max())

    for param, param_label in zip(sorting_parameters, parameter_labels):
        # make main dir and subdirs
        param_dir = os.path.join(outdir, 'by_' + param)
        observ_dir = os.path.join(param_dir, 'observ')
        suscept_dir = os.path.join(param_dir, 'suscept')
        os.makedirs(param_dir, exist_ok=True)
        os.makedirs(observ_dir, exist_ok=True)
        os.makedirs(suscept_dir, exist_ok=True)

        # a layer instance is uniquely identified by layer_id = ('network', 'layer', 'config')
        # use only the first two elements to find a single layer rows across the configurations
        for layer_id, layer_rows in df.groupby(['network', 'layer']):
            plot_layer_sorted_by(layer_id, layer_rows, col_name='observability', by=param, xlabel=param_label, outdir=observ_dir, yrange=obs_range)
            plot_layer_sorted_by(layer_id, layer_rows, col_name='susceptibility', by=param, xlabel=param_label, outdir=suscept_dir, yrange=suscept_range)


def every_layer_density_plot(df: pd.DataFrame, outdir: str):
    """
    For both observability and susceptibility, plots each layer (across configurations) in a density plot, whose
    axes are C/AtomicC and K/AtomicK. The size of each drawn point is proportional to its represented value. Note that
    points overlap when their parameter pairs (C, K) coincide, thus they are drawn unfilled.
    """
    os.makedirs(outdir, exist_ok=True)

    # make main dir and subdirs
    main_dir = os.path.join(outdir, 'density_plots')
    observ_dir = os.path.join(main_dir, 'observ')
    suscept_dir = os.path.join(main_dir, 'suscept')
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(observ_dir, exist_ok=True)
    os.makedirs(suscept_dir, exist_ok=True)

    size_factor = 200

    def _plot_density(layer_id, x_values, y_values, values, title: str, savedir: str, size_factor: int):
        network, layer = layer_id

        # adjust sizes
        sizes = np.array(size_factor * values, dtype=np.int64)

        edgecolors = cm.tab10(np.linspace(0, 1, len(x_values)))

        fig, ax = plt.subplots()
        scatter = ax.scatter(x_values, y_values, c='none', s=sizes, edgecolors=edgecolors)
        ax.set_xlabel('C/AtomicC')
        ax.set_ylabel('K/AtomicK')
        ax.set_title(f'{title} for {network}_{layer}')

        # shrink box to fit legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # add legend with point values
        legend_values = [f'{value * 100:.2f}%' for value in values]
        legend_elements = [Patch(facecolor=edgecolors[i], label=legend_values[i]) for i in range(len(values))]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title='Values')

        savepath = os.path.join(savedir, f'{network}_{layer}.png')
        fig.savefig(savepath)
        plt.close()

    # a layer instance is uniquely identified by layer_id = ('network', 'layer', 'config')
    # use only the first two elements to find a single layer rows across the configurations
    for layer_id, layer_rows in df.groupby(['network', 'layer']):
        x_values = make_categorical(layer_rows['c_over_atomicc'])
        y_values = make_categorical(layer_rows['k_over_atomick'])

        observ_rows = layer_rows['observability'].squeeze()
        suscept_rows = layer_rows['susceptibility'].squeeze()

        # find min value and use it to compute the size factor
        #min_value = min(observ_rows.min(), suscept_rows.min())
        #size_factor = np.ceil(min_marker_size / min_value)

        _plot_density(layer_id, x_values, y_values, observ_rows, 'Observability', observ_dir, size_factor)
        _plot_density(layer_id, x_values, y_values, suscept_rows, 'SDC', suscept_dir, size_factor)


def plot_heatmap_cols(df: pd.DataFrame, outdir: str):
    """
    Takes the observability and susceptibility values used to build the heatmaps (where each column is one configuration)
    and builds plots with individual columns, sorted by cell values. The plots include a table with all layer hyperparameters.
    """
    os.makedirs(outdir, exist_ok=True)

    # make subdirs
    observ_dir = os.path.join(outdir, 'observ')
    suscept_dir = os.path.join(outdir, 'suscept')
    os.makedirs(observ_dir, exist_ok=True)
    os.makedirs(suscept_dir, exist_ok=True)

    # find y axis ranges
    obs_range = (df['observability'].min(), df['observability'].max())
    suscept_range = (df['susceptibility'].min(), df['susceptibility'].max())

    def _sort_and_plot(config_rows: pd.DataFrame, config_name: str, values_col: str, yrange: tuple):
        # sort layers
        sorted_rows = config_rows.sort_values(values_col)
        xlabels = [f'{network}_{layer}' for network, layer in zip(sorted_rows['network'], sorted_rows['layer'])]
        heatmap_values = sorted_rows[values_col]

        table_rows = ['C','K','W','R','padding','TileSize','NumTiles']
        table_cols = sorted_rows[table_rows].reset_index().drop(columns=['index']).T
        # add value as row
        formatted_values = [f'{value:.2f}' for value in heatmap_values.tolist()]
        table_cols.loc['Value'] = formatted_values

        fig, ax = plt.subplots()
        ax.set_ylim(yrange)
        fig.suptitle(f'{values_col.title()} values for {config_name}')

        ax.scatter(xlabels, heatmap_values)
        ax.set_xticks([])
        ax.set_ylabel(values_col.title())

        # add index text next to markers
        for x, y, text in zip(xlabels, heatmap_values, table_cols.columns):
            ax.annotate(text, xy=(x, y), xycoords='data', xytext=(-0.5, 3.0), textcoords='offset points')

        # add hyperparameter table
        the_table = plt.table(
            cellText=table_cols,
            colLabels=None,
            loc='bottom',
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(7.5)

        # add layer index legend
        legend_handles = [Patch(color='none', label=f'{layer_num}: {layer_name}') for layer_num, layer_name in enumerate(xlabels)]
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.41), title='Layers')

        plt.subplots_adjust(right=0.75, bottom=0.3)

        return fig, ax


    for config, config_rows in df.groupby('config'):
        # extract configuration parameters from the first entry
        first_entry = config_rows.iloc[0]
        atomicc = first_entry['AtomicC'].item()
        atomick = first_entry['AtomicK'].item()
        bitwidth = first_entry['bitwidth'].item()
        short_name = f'{atomicc}x{atomick}_int{bitwidth}'

        # observability
        fig, ax = _sort_and_plot(config_rows, short_name, 'observability', obs_range)
        savepath = os.path.join(observ_dir, f'{short_name}.png')
        fig.set_size_inches(19.20, 10.8)
        fig.savefig(savepath, dpi=100)
        plt.close()

        # susceptibility
        fig, ax = _sort_and_plot(config_rows, short_name, 'susceptibility', suscept_range)
        savepath = os.path.join(suscept_dir, f'{short_name}.png')
        fig.set_size_inches(19.20, 10.8)
        fig.savefig(savepath, dpi=100)
        plt.close()


def plot_colored_configurations(df: pd.DataFrame, outdir: str):
    """
    Plots all SDC frequencies, divided by configuration and sorted by one of the parameters.
    Then, puts all frequencies into the same plot, giving each configuration a different color, still sorting.
    """
    os.makedirs(outdir, exist_ok=True)
    parameters = ['padding','C','K','W','R','TileSize','NumTiles']

    for config, config_rows in df.groupby('config'):
        c, k, bitwidth = extract_configuration_info(config)
        short_name = f'{c}x{k}_int{bitwidth}'

        # make configuration dir
        config_dir = os.path.join(outdir, short_name)
        os.makedirs(config_dir, exist_ok=True)

        for param in parameters:
            sorted_config_rows = config_rows.sort_values(param)

            x_values = sorted_config_rows[param]
            y_values = sorted_config_rows['susceptibility']

            fig, ax = plt.subplots()

            ax.plot(x_values, y_values, 'bo', x_values, calculate_trendline(x_values, y_values), "r-")
            ax.set_title(f'SDC values for {short_name} by {param}')
            ax.set_xlabel(param)
            ax.set_ylabel('SDC')

            savepath = os.path.join(config_dir, f'SDC_by_{param}.png')
            save_plt_figure(fig, savepath)
            plt.close()
    
    # now put all configurations together
    colors = cm.rainbow(np.linspace(0, 1, len(df['config'].unique())))
    for param in parameters:
        savepath = os.path.join(outdir, f'global_SDC_by_{param}.png')
        sorted_df = df.sort_values(param)

        legend_names = []

        fig, ax = plt.subplots()
        ax.set_title(f'SDC values by {param}')
        ax.set_xlabel(param)
        ax.set_ylabel('SDC')

        for i, (config, config_rows) in enumerate(sorted_df.groupby('config')):
            c, k, bitwidth = extract_configuration_info(config)
            short_name = f'{c}x{k}_int{bitwidth}'
            legend_names.append(short_name)
            legend_names.append(short_name + ' trend')

            x_values = config_rows[param]
            y_values = config_rows['susceptibility']

            ax.plot(x_values, y_values, 'o', x_values, calculate_trendline(x_values, y_values), "-", color=colors[i])

        ax.legend(legend_names, loc=0, fontsize=11)
        save_plt_figure(fig, savepath)
        plt.close()
            

def plot_vary_one_parameter(df: pd.DataFrame, outdir: str):
    """Groups the layers so that all parameters but one are fixed."""
    variable_params = ['padding', 'C', 'K', 'W', 'R', 'c_over_atomicc', 'k_over_atomick']

    for i in range(len(variable_params)):
        varying_param = variable_params[i]
        fixed_params = variable_params[:i] + variable_params[i+1:]

        savedir = os.path.join(outdir, varying_param)
        os.makedirs(savedir, exist_ok=True)

        for group_values, group_rows in df.groupby(fixed_params):
            if len(group_rows) < 2:     # avoid plotting if there is only 1 layer
                continue

            x_values = group_rows[varying_param]
            if len(x_values.unique()) < 2:      # avoid plotting if there are not at least 2 different parameter values
                continue

            y_values = group_rows['susceptibility']

            group_values_str = [value.item() for value in group_values]

            plt.clf()
            plt.plot(x_values, y_values, 'bo', x_values, calculate_trendline(x_values, y_values), "r-")
            plt.suptitle(f'Varying parameter {varying_param}')
            plt.xlabel(varying_param)
            plt.ylabel('SDC')

            savepath = os.path.join(savedir, f'{group_values_str}.png')
            plt.savefig(savepath)


def plot_raw_points(df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    
    colors = cm.rainbow(np.linspace(0, 1, 2)) # 2 colors, one for observability and one for susceptibility
    parameters = ['AtomicC','AtomicK','bitwidth','padding','C','K','W','R','TileSize','c_over_atomicc','k_over_atomick','NumTiles']

    y_observ = df['observability']
    y_suscep = df['susceptibility']

    for param in parameters:
        x_values = df[param]
        if len(x_values.unique()) < 2:
            continue

        fig, ax = plt.subplots()

        ax.plot(x_values, y_observ, 'o', x_values, calculate_trendline(x_values, y_observ), "-", color=colors[0])
        ax.plot(x_values, y_suscep, 'o', x_values, calculate_trendline(x_values, y_suscep), "-", color=colors[1])
        ax.set_title(f'Observability + Susceptibility')
        ax.set_xlabel(param)
        ax.legend(['Observability', 'Obs. Trend', 'Susceptibility', 'Susc. Trend'], loc=0, fontsize=11)

        savepath = os.path.join(outdir, f'{param}.png')
        save_plt_figure(fig, savepath)
        plt.close()


def save_plt_figure(fig, filename: str, sizex=1920, sizey=1080, dpi=100):
    """Saves a pyplot figure to file. Defaults to a size of 1920x1080."""
    sizex_inches = float(sizex / dpi)
    sizey_inches = float(sizey / dpi)
    fig.set_size_inches(sizex_inches, sizey_inches)
    fig.savefig(filename, dpi=dpi)


def main():
    os.makedirs(outdir, exist_ok=True)

    # load total results df
    # configs=[
    #     "nv_8x8_b1_dat-524288_wt-32768_int8",
    #     "nv_8x8_b1_dat-1048576_wt-65536_int16",
    #     "nv_8x16_b1_dat-2097152_wt-262144_int32",
    #     "nv_16x16_b1_dat-524288_wt-65536_int8",
    #     "nv_32x16_b1_dat-1048576_wt-131072_int16",
    #     "nv_32x8_b1_dat-2097152_wt-131072_int32",
    #     "nv_16x32_b1_dat-524288_wt-131072_int8",
    #     "nv_32x32_b1_dat-1048576_wt-262144_int16",
    #     "nv_32x32_b1_dat-2097152_wt-524288_int32",
    # ]
    # results_df = pd.read_csv(results_path)
    # results_df = df[df['config'].isin(configs)]
    
    # load raw (compact) layer results df
    raw_layer_df = pd.read_csv(raw_results_path)

    print('Plotting single layer obs. and suscept. across configurations, sorting by parameters')
    single_layers_dir = os.path.join(outdir, 'single_layers')
    plot_every_layer(raw_layer_df, single_layers_dir)
    every_layer_density_plot(raw_layer_df, single_layers_dir)

    print('Plotting heatmap columns, sorted by value')
    heatmap_cols_dir = os.path.join(outdir, 'heatmap_cols')
    plot_heatmap_cols(raw_layer_df, heatmap_cols_dir)

    print('Plotting layers grouped by configuration')
    plot_colored_configurations(raw_layer_df, os.path.join(outdir, 'grouped'))

    print('Plotting layers that differ by one parameter')
    plot_vary_one_parameter(raw_layer_df, os.path.join(outdir, 'vary_one_param'))

    # take the raw results and plot all points (related to each parameter), with both observability and susceptibility
    print('Plotting observability/susceptibility scatters')
    raw_df = pd.read_csv(raw_results_path)
    plot_raw_points(raw_df, os.path.join(outdir, 'observ+suscept'))


if __name__ == '__main__':
    main()