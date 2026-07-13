"""
Given the raw layer and units csv files, builds additional heatmaps, both for observability and susceptibility.
For the layers, 1 row per layer, 1 column per configuration.
For the units, 1 row per layer, 1 column per configuration, 1 heatmap per unit.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import utils

from argparse import ArgumentParser

#----- PLOT CONFIGURATION

# set plot sizes
plt.rc('font', size=9)          # controls default text sizes
plt.rc('axes', titlesize=13)    # fontsize of the axes title
plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)   # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=9)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

CMAP_RED_TO_GREEN = 'RdYlGn_r'  # SDC
CMAP_YB = 'managua'             # CLASS FREQUENCIES

CLASSES_NETWORKS = [
    'alexnet_cifar10',
    'deeplabv3_oxfordpet',
    'mobilenetv2-large_gtsrb',
    'res50_cifar10',
    'yolov11_coco',
]

#------ HEATMAP VARIABLES

LAYER_COLS = ['network', 'layer', 'config']
HYPER_COLS = ['K', 'C', 'W', 'R', 'padding'] # in order: (Channels_out, Channels_in, Input_size, Kernel_size, Padding)
CONFIG_COLS = ['AtomicC', 'AtomicK', 'bitwidth']

SORTING_PARAMS = ['AtomicC','AtomicK','bitwidth','padding','C','K','W','R','TileSize','c_over_atomicc','k_over_atomick','NumTiles']


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--layer_csv_path', required=True, help='Path to the raw_layer_results.csv.')
    parser.add_argument('--unit_csv_path', required=True, help='Path to the raw_unit_results.csv.')
    parser.add_argument('--other_results_dir', required=True, help='Directory of the additional raw network results.')
    parser.add_argument('--organized_plots_dir', required=True, help='Directory of the processed dataframes.')
    parser.add_argument('--show_annotations', action='store_true')
    parser.add_argument('--fig_pixels_x', type=int, required=False, default=1920)
    parser.add_argument('--fig_pixels_y', type=int, required=False, default=1080)
    return parser.parse_args()


def get_packed_hyperparameters_list(df: pd.DataFrame):
    """Packs all columns of the provided dataframe into a series of tuples which can be used as single-level index/axis label."""
    hyper_tuples: list[tuple] = []
    for _, row in df.iterrows():
        hyper_tuples.append(tuple(row))
    return hyper_tuples


def extract_configuration_info(config_id: str):
    pieces = config_id.split('_')
    C, K = pieces[1].split('x') # skip "nv_", then split on x in "CxK"
    precision = pieces[-1][3:] # strip "int" from last piece
    return int(C), int(K), int(precision)


def calculate_trendline(x_data, y_data):
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    return p(x_data)


# LEGACY-----------------------------------------------------------------------------------------------------------------

def get_parameter_lists_for_configs(configs: list[str]):
    """Given the list of configurations, builds three lists, [AtomicC], [AtomicK], [precision]. The order is preserved."""
    cs = []
    ks = []
    precisions = []
    for config in configs:
        c,k,precision = extract_configuration_info(config)
        cs.append(c)
        ks.append(k)
        precisions.append(precision)

    return cs, ks, precisions


def sort_configs_by(configs: list[str], cs, ks, precisions, by: str):
    if by is None:
        return configs

    match by:
        case 'AtomicC'  : selected_list = cs
        case 'AtomicK'  : selected_list = ks
        case 'precision': selected_list = precisions
        case _: raise ValueError(f'Unknown value for "by": {by}')
    
    zipped = zip(selected_list, configs)
    return [el[1] for el in sorted(zipped)]


def df_from_results_csv(results_path: str, configs: list[str]=None):
    results_df = pd.read_csv(results_path, index_col=0)

    # extract relevant columns
    results_df = results_df[LAYER_COLS + HYPER_COLS + ['Silent']]

    # add hyperparameter tuples
    results_df['tuples'] = get_packed_hyperparameters_list(results_df[HYPER_COLS])

    # get configuration names (if none are provided, get all of them)
    if configs is None:
        configs = utils.sort_configs(results_df['config'].unique())

    # sort by hyperparameters
    results_df = results_df.sort_values(by=HYPER_COLS)

    # build columns: one per configuration
    df_cols = []
    for config in configs:
        short_config = utils.config_reformat(config) + '_' + utils.detach_precision(config)[1]
        config_rows = results_df[results_df['config'] == config]

        hyper_sets = []
        sdc_values = []
        # average frequencies of layers with the same hyperparameters
        for hyper_set, rows in config_rows.groupby('tuples'):
            avg_sdc = rows['Silent'].mean()
            hyper_sets.append(hyper_set)
            sdc_values.append(avg_sdc)
        
        df_cols.append( pd.Series(sdc_values, name=short_config, index=hyper_sets) )

    return pd.concat(df_cols, axis=1)


def base_df_from_results_csv(results_path: str, configs: list[str]=None):
    """Extract requested configurations from the results file and returns it as-is, not removing or modifying columns."""
    results_df = pd.read_csv(results_path, index_col=0)

    if configs is None:
        # nothing to do, just return the df
        return results_df
    
    # extract the requested configurations
    return results_df[results_df['config'].isin(configs)]


def sort_df(df: pd.DataFrame, sort_index, sort_columns, cs, ks, precisions, configs: list[str]=None):
    sorted_df = df.sort_values(by=sort_index)   # sort rows
    sorted_configs = sort_configs_by(configs, cs, ks, precisions, sort_columns) # sort columns

    # build the heatmap df by extracting layers in the order specified by the sorted configurations
    heatmap_cols = []
    for config in sorted_configs:
        short_config = utils.config_reformat(config) + '_' + utils.detach_precision(config)[1]
        config_rows = sorted_df[sorted_df['config'] == config]

        parameter_values = []
        sdc_values = []
        # average frequencies of layers with the same parameter value
        for parameter_value, rows in config_rows.groupby(sort_index):
            avg_sdc = rows['Silent'].mean()
            parameter_values.append(parameter_value)
            sdc_values.append(avg_sdc)

        heatmap_cols.append( pd.Series(sdc_values, name=short_config, index=parameter_values) )

    return pd.concat(heatmap_cols, axis=1)

# -----------------------------------------------------------------------------------------------------------------------

def make_heatmap(
    df: pd.DataFrame,
    suptitle: str,
    xaxis_label: str,
    yaxis_label: str,
    save_path: str = None,
    add_colorbar: bool = True,
    colormap: str = CMAP_RED_TO_GREEN,
    show_cell_values: bool = True,
    show_values_as_percentages: bool = True,
    use_value_range: bool = False,
    num_decimal_digits: int = 0,
    fig_pixels_x: int = 1920,
    fig_pixels_y: int = 1080,
):
    df = df.astype(float)

    fig, ax = plt.subplots()
    fig.suptitle(suptitle)
    cbar_ax = fig.add_axes([.91, .3, .03, .4]) if add_colorbar else None

    if num_decimal_digits < 0:
        num_decimal_digits = 0
    #annot_format = f'.{num_decimal_digits}%' if show_values_as_percentages else f'.{num_decimal_digits}f'
    annot_format = f'.{num_decimal_digits}f'

    if show_values_as_percentages:
        df = df * 100.0

    if use_value_range:
        val_min = df.min().min()
        val_max = df.max().max()
    else:
        val_min = 0.0
        val_max = 100.0

    with sns.axes_style("white"):
        sns.heatmap(
            df, 
            vmin=val_min, vmax=val_max,
            cmap=colormap,
            ax=ax,
            cbar=add_colorbar, cbar_ax=cbar_ax,
            annot=show_cell_values,
            fmt=annot_format,
            yticklabels=df.index.tolist()
        )

    ax.patch.set_linewidth(1)
    ax.patch.set_edgecolor('black')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xticklabels(df.columns.tolist(), rotation=45, ha='center')
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)
    
    if save_path is not None:
        save_plt_figure(fig, save_path, sizex=fig_pixels_x, sizey=fig_pixels_y)
        plt.close()


def save_plt_figure(fig, filename: str, sizex=1920, sizey=1080, dpi=100):
    """Saves a pyplot figure to file. Defaults to a size of 1920x1080."""
    sizex_inches = float(sizex / dpi)
    sizey_inches = float(sizey / dpi)
    fig.set_size_inches(sizex_inches, sizey_inches)
    fig.savefig(filename, dpi=dpi)


def simple_scatter(x, y, suptitle: str, xlabel: str, ylabel: str, savepath: str):
    plt.clf()
    plt.plot(x, y, 'bo', x, calculate_trendline(x, y), "r-")
    plt.suptitle(suptitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(savepath)


def build_heatmap_df(df: pd.DataFrame, value_col: str):
    # sort by configuration parameters
    df = df.sort_values(CONFIG_COLS)
    # get all configurations
    configs = df['config'].unique()

    # add hyperparameter tuples
    df['tuples'] = get_packed_hyperparameters_list(df[HYPER_COLS])

    df_cols = []
    for config in configs:
        # get all rows for the configuration
        config_df = df.query('config == @config')

        # use the first row to build a short configuration name
        first_row = config_df.iloc[0]
        atomicC  = int(first_row['AtomicC'].item())
        atomicK  = int(first_row['AtomicK'].item())
        bitwidth = int(first_row['bitwidth'].item())
        short_name = f'{atomicC}x{atomicK}_int{bitwidth}'

        # sort layers according to the main hyperparameters
        config_df = config_df.sort_values(HYPER_COLS)

        parameter_values = []
        values = []
        # average frequencies of layers with the same parameter values
        for parameter_value, rows in config_df.groupby('tuples'):
            avg_value = rows[value_col].mean()
            parameter_values.append(parameter_value)
            values.append(avg_value)

        # build the configuration column
        df_cols.append( pd.Series(values, name=short_name, index=parameter_values) )

    return pd.concat(df_cols, axis=1)


def build_application_heatmap_df(df: pd.DataFrame, value_col: str):
    """Behaves similarly to the other function, but skips the hyperparameters and uses the layer names directly"""
    # get all configurations
    configs = df['config'].unique()

    df_cols = []
    for config in configs:
        # get all rows for the configuration
        config_df = df.query('config == @config')

        # use the first row to build a short configuration name
        first_row = config_df.iloc[0]
        atomicC  = int(first_row['AtomicC'].item())
        atomicK  = int(first_row['AtomicK'].item())
        bitwidth = int(first_row['bitwidth'].item())
        short_name = f'{atomicC}x{atomicK}_int{bitwidth}'

        # index is the layer names
        layer_names = config_df.index

        # values are the requested column
        values = config_df[value_col]

        # build the configuration column
        df_cols.append( pd.Series(values, name=short_name, index=layer_names) )

    return pd.concat(df_cols, axis=1)


def main():
    args = parse_arguments()

    outdir              = os.path.realpath(args.outdir)
    layer_csv_path      = os.path.realpath(args.layer_csv_path)
    unit_csv_path       = os.path.realpath(args.unit_csv_path)
    other_results_dir   = os.path.realpath(args.other_results_dir)
    organized_plots_dir = os.path.realpath(args.organized_plots_dir)
    show_annotations    = args.show_annotations
    fig_pixels_x        = args.fig_pixels_x
    fig_pixels_y        = args.fig_pixels_y

    os.makedirs(outdir, exist_ok=True)

    obs_dir                 = os.path.join(outdir, 'observability')
    suscept_dir             = os.path.join(outdir, 'susceptibility')
    classes_dir             = os.path.join(outdir, 'classes')
    network_sdc_dir         = os.path.join(outdir, 'network_SDC')
    network_criticality_dir = os.path.join(outdir, 'network_criticality')

    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(suscept_dir, exist_ok=True)
    os.makedirs(classes_dir, exist_ok=True)
    os.makedirs(network_sdc_dir, exist_ok=True)
    os.makedirs(network_criticality_dir, exist_ok=True)

    # heatmaps output directories
    obs_heatmap_dir                 = os.path.join(obs_dir, 'heatmaps')
    suscept_heatmap_dir             = os.path.join(suscept_dir, 'heatmaps')
    classes_heatmap_dir             = os.path.join(classes_dir, 'heatmaps')
    network_sdc_heatmap_dir         = os.path.join(network_sdc_dir, 'heatmaps')
    network_criticality_heatmap_dir = os.path.join(network_criticality_dir, 'heatmaps')

    os.makedirs(obs_heatmap_dir, exist_ok=True)
    os.makedirs(suscept_heatmap_dir, exist_ok=True)
    os.makedirs(classes_heatmap_dir, exist_ok=True)
    os.makedirs(network_sdc_heatmap_dir , exist_ok=True)
    os.makedirs(network_criticality_heatmap_dir , exist_ok=True) 

    # scatters output directories
    obs_scatters_dir                 = os.path.join(obs_dir, 'scatters')
    suscept_scatters_dir             = os.path.join(suscept_dir, 'scatters')
    classes_scatters_dir             = os.path.join(classes_dir, 'scatters')
    network_sdc_scatters_dir         = os.path.join(network_sdc_dir, 'scatters')
    network_criticality_scatters_dir = os.path.join(network_criticality_dir, 'scatters')

    os.makedirs(obs_scatters_dir, exist_ok=True)
    os.makedirs(suscept_scatters_dir, exist_ok=True)
    os.makedirs(classes_scatters_dir, exist_ok=True)
    os.makedirs(network_sdc_scatters_dir, exist_ok=True)
    os.makedirs(network_criticality_scatters_dir, exist_ok=True)

    # ----- HEATMAPS -----

    # BUILD UNIT OBSERVABILITY HEATMAPS
    unit_results_df = pd.read_csv(unit_csv_path)
    units = unit_results_df['unit'].unique()
    for unit in units:
        print(f'Making {unit} observability heatmap')
        unit_df = unit_results_df.query("unit == @unit")
        obs_unit_df = build_heatmap_df(unit_df, 'observability')
        make_heatmap(
            df                          = obs_unit_df,
            suptitle                    = f'{unit} observability (%)',
            xaxis_label                 = 'Configuration',
            yaxis_label                 = 'Layer hyperparameters',
            save_path                   = os.path.join(obs_heatmap_dir, f'{unit}_suscept.png'),
            show_cell_values            = show_annotations,
            show_values_as_percentages  = True,
            fig_pixels_x                = fig_pixels_x,
            fig_pixels_y                = fig_pixels_y,
        )

    # BUILD LAYER OBSERVABILITY AND SUSCEPTIBILITY HEATMAPS
    layer_results_df = pd.read_csv(layer_csv_path) 

    print('Making layer observability heatmap')
    obs_layer_df = build_heatmap_df(layer_results_df, 'observability')
    make_heatmap(
        df                          = obs_layer_df,
        suptitle                    = 'SDC layer observability (%)',
        xaxis_label                 = 'Configuration',
        yaxis_label                 = 'Layer hyperparameters',
        save_path                   = os.path.join(obs_heatmap_dir, 'sdc_layer_obs.png'),
        show_cell_values            = show_annotations,
        show_values_as_percentages  = True,
        fig_pixels_x                = fig_pixels_x,
        fig_pixels_y                = fig_pixels_y,
    )

    print('Making layer susceptibility heatmap')
    suscept_layer_df = build_heatmap_df(layer_results_df, 'susceptibility')
    make_heatmap(
        df                          = suscept_layer_df.fillna(0.0),
        suptitle                    = 'SDC layer susceptibility (%)',
        xaxis_label                 = 'Configuration',
        yaxis_label                 = 'Layer hyperparameters',
        save_path                   = os.path.join(suscept_heatmap_dir, 'sdc_layer_suscept.png'),
        show_cell_values            = show_annotations,
        show_values_as_percentages  = True,
        fig_pixels_x                = fig_pixels_x,
        fig_pixels_y                = fig_pixels_y,
    )

    # BUILD CLASS FREQUENCIES HEATMAPS
    # get column names that start with class-
    class_cols = list(filter(lambda x: x.startswith('class-'), layer_results_df.columns.tolist()))
    for class_name in class_cols:
        print(f'Making {class_name} frequency heatmap')
        class_freq_df = build_heatmap_df(layer_results_df, class_name)
        make_heatmap(
            df                          = class_freq_df,
            suptitle                    = f'{class_name} frequency (%)',
            xaxis_label                 = 'Configuration',
            yaxis_label                 = 'Layer hyperparameters',
            save_path                   = os.path.join(classes_dir, f'{class_name}_freq.png'),
            show_cell_values            = show_annotations,
            show_values_as_percentages  = True,
            fig_pixels_x                = fig_pixels_x,
            fig_pixels_y                = fig_pixels_y,
            colormap                    = CMAP_YB,
        )

    # BUILD LAYER SDC HEATMAPS (derived from CLASSES experiments) AND CLASS CRITICALITY HEATMAPS
    for network in CLASSES_NETWORKS:
        print(f'Making network SDC heatmap for {network}')
        network_results_path = os.path.join(other_results_dir, f'raw_{network}_results.csv')
        network_results_df = pd.read_csv(network_results_path, index_col=0)

        configs = network_results_df['config'].unique().tolist()

        network_sdc_df = build_application_heatmap_df(network_results_df, 'SDC')
        make_heatmap(
            df                          = network_sdc_df,
            suptitle                    = f'SDC for {network} (%)',
            xaxis_label                 = 'Configuration',
            yaxis_label                 = 'Layer',
            save_path                   = os.path.join(network_sdc_heatmap_dir, f'{network}.png'),
            show_cell_values            = show_annotations,
            show_values_as_percentages  = True,
            fig_pixels_x                = fig_pixels_x,
            fig_pixels_y                = fig_pixels_y,
        )

        # make the global scatters
        for sorting_param in SORTING_PARAMS:
            if sorting_param not in network_results_df.columns:
                continue

            print(f'Making global scatter, sorting by {sorting_param}')
            sorted_layer_df = network_results_df.sort_values(sorting_param)
            simple_scatter(
                x         = sorted_layer_df[sorting_param], 
                y         = sorted_layer_df['SDC'], 
                suptitle = f'{network} SDC vs {sorting_param}', 
                xlabel    = sorting_param, 
                ylabel    = 'SDC', 
                savepath  = os.path.join(network_sdc_scatters_dir, f'{network}_by_{sorting_param}.png'),
            )

            # for each configuration, load the applev_class_groups file, sum the columns to obtain total layer criticality
            # and plot the results
            for config in configs:
                print(f'Making total criticality scatter, sorting by {sorting_param} - configuration is {config}')
                sorted_config_df = sorted_layer_df.query('config == @config')

                c,k,bitwidth = extract_configuration_info(config)
                short_config = f'{c}x{k}_int{bitwidth}'
                applev_class_groups_path = os.path.join(organized_plots_dir, network, f'applev_class_groups_{short_config}.csv')
                total_criticality_df = pd.read_csv(applev_class_groups_path, index_col=0).sum(axis=1) # this is technically a Series

                simple_scatter(
                    x         = sorted_config_df[sorting_param], 
                    y         = total_criticality_df[sorted_config_df.index], # take values in the order specified by the sorted index 
                    suptitle = f'{network} total criticality vs {sorting_param} - {short_config}', 
                    xlabel    = sorting_param, 
                    ylabel    = 'Total criticality', 
                    savepath  = os.path.join(network_sdc_scatters_dir, f'{network}_total_criticality_{short_config}_by_{sorting_param}.png'),
                )


        for class_name in class_cols:
            print(f'Making network {class_name} heatmap')
            class_crit_df = build_application_heatmap_df(network_results_df, class_name + '_crit')
            make_heatmap(
                df                          = class_crit_df,
                suptitle                    = f'Criticality for {network} - {class_name} (%)',
                xaxis_label                 = 'Configuration',
                yaxis_label                 = 'Layer',
                save_path                   = os.path.join(network_criticality_heatmap_dir, f'{network}_{class_name}.png'),
                show_cell_values            = show_annotations,
                show_values_as_percentages  = True,
                fig_pixels_x                = fig_pixels_x,
                fig_pixels_y                = fig_pixels_y,
            )

            # make the global scatters
            for sorting_param in SORTING_PARAMS:
                if sorting_param not in network_results_df.columns:
                    continue

                print(f'Making global scatter for {class_name}, sorting by {sorting_param}')
                sorted_layer_df = network_results_df.sort_values(sorting_param)
                simple_scatter(
                    x         = sorted_layer_df[sorting_param], 
                    y         = sorted_layer_df[class_name + '_crit'], 
                    suptitle = f'{class_name} criticality for {network} vs {sorting_param}', 
                    xlabel    = sorting_param, 
                    ylabel    = class_name, 
                    savepath  = os.path.join(network_criticality_scatters_dir, f'{network}_{class_name}_by_{sorting_param}.png'),
                )


    # ----- SCATTERS -----

    # BUILD GLOBAL SCATTERS (all points, ordering by one parameter)
    for sorting_param in SORTING_PARAMS:
        # unit observability
        for unit in units:
            print(f'Making {unit} observability scatter by {sorting_param}')
            unit_df = unit_results_df.query("unit == @unit").sort_values(sorting_param)
            simple_scatter(
                x         = unit_df[sorting_param], 
                y         = unit_df['observability'], 
                suptitle = f'{unit} observability vs {sorting_param}', 
                xlabel    = sorting_param, 
                ylabel    = 'Observability', 
                savepath  = os.path.join(obs_scatters_dir, f'{unit}_by_{sorting_param}.png'),
            )
        
        sorted_layer_df = layer_results_df.sort_values(sorting_param)
        layer_x = sorted_layer_df[sorting_param]

        # layer observability
        print(f'Making layer observability scatter by {sorting_param}')
        simple_scatter(
                x         = layer_x, 
                y         = sorted_layer_df['observability'], 
                suptitle = f'Layer observability vs {sorting_param}', 
                xlabel    = sorting_param, 
                ylabel    = 'Observability', 
                savepath  = os.path.join(obs_scatters_dir, f'layers_by_{sorting_param}.png'),
        )

        # layer susceptibility
        print(f'Making layer susceptibility scatter by {sorting_param}')
        simple_scatter(
                x         = layer_x, 
                y         = sorted_layer_df['susceptibility'], 
                suptitle = f'Layer susceptibility vs {sorting_param}', 
                xlabel    = sorting_param, 
                ylabel    = 'Susceptibility', 
                savepath  = os.path.join(suscept_scatters_dir, f'layers_by_{sorting_param}.png'),
        )

        # class frequencies
        for class_name in class_cols:
            print(f'Making {class_name} frequency scatter by {sorting_param}')
            simple_scatter(
                x         = layer_x, 
                y         = sorted_layer_df[class_name], 
                suptitle = f'{class_name} frequency vs {sorting_param}', 
                xlabel    = sorting_param, 
                ylabel    = 'Frequency', 
                savepath  = os.path.join(classes_scatters_dir, f'{class_name}_by_{sorting_param}.png'),
            )


if __name__ == '__main__':
    main()