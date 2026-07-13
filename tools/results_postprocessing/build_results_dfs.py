import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import utils

from argparse import ArgumentParser
from args_parser import Args
from plot_heatmap import make_heatmap, CMAP_RED_TO_GREEN, CMAP_ALTERNATIVE


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('path_to_args_yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    return Args(args.path_to_args_yaml)


def collect_error_models_dfs(args: Args, output_dir: str):
    columns_to_drop = ['Masked', 'Crash+Hang']

    clean_error_models_dfs: list[pd.DataFrame] = []

    # reorganize the unique complete error model dfs (extract the relevant columns)
    for config_name, short_name in zip(args.configuration_ids, args.short_configuration_ids):
        error_models_df_path = os.path.join(args.error_models_base_path, config_name, 'unique_complete_df.xlsx')
        error_models_df = pd.read_excel(error_models_df_path, sheet_name=0, header=0, index_col=0)
        clean_df = error_models_df.drop(columns=columns_to_drop)

        # sort by the hyperparameters in the order they were listed
        clean_df = clean_df.sort_values(utils.hyperparameters)
        clean_error_models_dfs.append(clean_df)
        
        clean_output_path = os.path.join(output_dir, f'error_models_df_{short_name}.csv')
        clean_df.to_csv(clean_output_path, index=False)

    return clean_error_models_dfs


def get_packed_hyperparameters_list(df: pd.DataFrame, hyperparameter_cols_names: list[str]):
    """Extracts the first (hyperparameters) columns from the dataframe and builds a series of tuples which can be used as index."""
    hyper_tuples: list[tuple] = []
    hyper_df = df.loc[:, hyperparameter_cols_names].astype(int)
    for _, row in hyper_df.iterrows():
        hyper_tuples.append(tuple(row))
    
    return hyper_tuples


def save_architectural_df(
    args: Args,
    df: pd.DataFrame,
    output_dir: str,
    filename: str,
    excel_sheet_name: str,
    heatmap_title: str,
    heatmap_xaxis: str,
    heatmap_yaxis: str,
    alternative_column_names: list[str]=None,
    use_alternative_colormap: bool=False
):
    """Saves dataframe as csv file and also plots it as a heatmap, using the hyperparameters columns as labels."""
    output_path = os.path.join(output_dir, filename + '.csv')
    df.to_csv(output_path, index=False)

    output_path = os.path.join(output_dir, filename + '.png')
    hyperparameter_index = get_packed_hyperparameters_list(df, utils.hyperparameters)
    plot_df = df.drop(columns=utils.hyperparameters)
    plot_df.index = hyperparameter_index

    if alternative_column_names is not None:
        plot_df.columns = alternative_column_names

    colormap = CMAP_ALTERNATIVE if use_alternative_colormap else CMAP_RED_TO_GREEN

    make_heatmap(plot_df, heatmap_title, heatmap_xaxis, heatmap_yaxis, output_path, colormap=colormap)


def build_sdc_df(args: Args, error_models_dfs: list[pd.DataFrame], output_dir: str):
    rows = error_models_dfs[0].index
    cols = utils.hyperparameters + args.short_configuration_ids
    sdc_df = pd.DataFrame(index=rows, columns=cols)

    # get the hyperparameters columns from the first df
    sdc_df.loc[:, utils.hyperparameters] = error_models_dfs[0].loc[:, utils.hyperparameters]

    for short_config, df in zip(args.short_configuration_ids, error_models_dfs):
        sdc_col = df.loc[:, 'SDC']
        sdc_df.loc[:, short_config] = sdc_col

    save_architectural_df(
        args=args,
        df=sdc_df,
        output_dir=output_dir,
        filename='SDC_by_configuration',
        excel_sheet_name='SDC',
        heatmap_title='SDC across configurations',
        heatmap_xaxis='Configuration',
        heatmap_yaxis='Error model'
    )

    return sdc_df


def build_class_distribution_dfs(
    args: Args,
    error_models_df: list[pd.DataFrame],
    output_dir: str,
):
    hyperparameters_cols = error_models_df[0].loc[:, utils.hyperparameters]
    class_group_names = utils.class_group_names

    for short_config, df in zip(args.short_configuration_ids, error_models_df):
        class_distribution_df = pd.DataFrame(0.0, index=df.index, columns=class_group_names)
        
        # weigh spatial class frequencies with single and multi channel frequencies and group
        for class_name in utils.spatial_classes:
            if class_name in df.columns:
                class_name_snake = utils.class_pascal_to_snake(class_name)
                class_col = df.loc[:, class_name]
                single_channel_col = df.loc[:, class_name_snake + '-single']
                multi_channel_col = df.loc[:, class_name_snake + '-multi']

                single_group_name, multi_group_name = utils.spatial_class_to_group[class_name]

                class_distribution_df.loc[:, single_group_name] += class_col * single_channel_col
                class_distribution_df.loc[:, multi_group_name] += class_col * multi_channel_col

        # append the hyper columns to both dataframes and save
        class_distribution_df = pd.concat([hyperparameters_cols, class_distribution_df], axis=1)
        save_architectural_df(
            args=args,
            df=class_distribution_df,
            output_dir=output_dir,
            filename=f'class_distribution_df_{short_config}',
            excel_sheet_name='class_distrib',
            heatmap_title=f'Class frequencies for {short_config}',
            heatmap_xaxis='Spatial class',
            heatmap_yaxis='Error model',
            use_alternative_colormap=True,
            alternative_column_names=utils.short_class_group_names,
        )




def load_applev_dicts(args: Args):
    applev_dicts = {}

    for network_id in args.network_dataset_ids:
        applev_dicts[network_id] = {}
        network_dir = os.path.join(args.experiments_base_path, f'exp_{network_id}')
        for configuration in args.configuration_ids:
            config_dir = os.path.join(network_dir, configuration)
            filenames = os.listdir(config_dir)

            applev_filename = find_default_file(filenames, 'applev')
            applev_filepath = os.path.join(config_dir, applev_filename)

            with open(applev_filepath) as f:
                applev_dict = yaml.load(f, yaml.SafeLoader)
                applev_dicts[network_id][configuration] = applev_dict

            # load the SDC dataframe
            sdc_freq_filename = find_default_file(filenames, 'SDC')
            sdc_freq_path = os.path.join(config_dir, sdc_freq_filename)
            sdc_freq_df = pd.read_excel(sdc_freq_path, sheet_name=0, index_col=0, header=0)
            
            applev_dicts[network_id]['SDC'] = sdc_freq_df
    
    return applev_dicts


def find_default_file(candidate_filenames: list[str], beginning_token: str):
    possible_files = list(filter(lambda x: x.startswith(beginning_token), candidate_filenames))
    if len(possible_files) != 1:
        raise FileNotFoundError(f'Found either 0 or more than 1 possible "{beginning_token}" files in base directory. Provide a specific path.')

    return possible_files[0]


def save_application_df(
    df: pd.DataFrame,
    output_dir: str,
    filename: str,
    excel_sheet_name: str,
    heatmap_title: str,
    heatmap_xaxis: str,
    heatmap_yaxis: str,
    alternative_column_names: list[str]=None,
    use_alternative_colormap: bool=False,
    show_values_as_percentages: bool=True,
    num_decimal_digits: int=0,
):
    output_path = os.path.join(output_dir, filename + '.csv')
    df.to_csv(output_path)

    output_path = os.path.join(output_dir, filename + '.png')
    plot_df = df.copy()
    if alternative_column_names is not None:
        plot_df.columns = alternative_column_names
    colormap = CMAP_ALTERNATIVE if use_alternative_colormap else CMAP_RED_TO_GREEN
    make_heatmap(plot_df, heatmap_title, heatmap_xaxis, heatmap_yaxis, output_path, colormap=colormap, show_values_as_percentages=show_values_as_percentages, num_decimal_digits=num_decimal_digits)


def build_application_class_distrib_dfs(args: Args, applev_dicts: dict, app_output_dir: str):
    class_group_names = utils.class_group_names

    critical_by_spatial_class = {} # one per configuration for each network

    for network_id, network_dict in applev_dicts.items():
        critical_by_spatial_class[network_id] = {}

        network_output_dir = os.path.join(app_output_dir, network_id)
        os.makedirs(network_output_dir, exist_ok=True)

        # get the frequencies dataframe for the network
        freqs_df: pd.DataFrame = network_dict['SDC']

        # SingleFullChannel is taken from external csv: load it here
        if network_id.startswith('mobilenetv2'):
            sfc_filepath = os.path.join('application', f'single_full_channels_mobilenetv2_gtsrb.csv')
        else:
            sfc_filepath = os.path.join('application', f'single_full_channels_{network_id}.csv')
        sfc_df = pd.read_csv(sfc_filepath, index_col=0)

        # for each network, build a <num_layers>x<num_configs> dataframe: each cell value contains the total criticality
        # value for a layer in a configuration. The columns are obtained by summing over the columns of the dataframes
        # constructed below
        total_criticality_df = pd.DataFrame(columns=args.short_configuration_ids)

        # for each configuration, build a <num_layers>x<num_class_groups> dataframe; a cell value is obtained by
        # multiplying the critical error frequencies from applev by the respective class frequency and then aggregating
        # the results into the class groups
        for config_id, short_config in zip(args.configuration_ids, args.short_configuration_ids):
            applev_dict: dict = network_dict[config_id]
            
            # use the layers as indices and the class groups as columns
            layers = list(applev_dict.keys())
            group_distribution_df = pd.DataFrame(0.0, index=layers, columns=class_group_names, dtype=float)

            # get critical error frequencies for all layers and multiply by class frequencies
            for layer_name, layer_dict in applev_dict.items():
                # filter out non-spatial-classes
                class_names = [key for key in layer_dict if not key.startswith('prob')]
                # convert to pascal
                class_names_pascal = [utils.snakecase_to_pascalcase(name) for name in class_names]

                # for each spatial class, if an entry is in the layer, take the critical frequency and multiply
                for spatial_class in utils.spatial_classes:
                    if spatial_class in class_names_pascal:
                        class_name_snakecase = class_names[class_names_pascal.index(spatial_class)]
                        crit_freq = float(layer_dict[class_name_snakecase]['sdc_critical'])

                        class_freq = freqs_df.at[layer_name, spatial_class]
                        multi_channel_val  = freqs_df.at[layer_name, class_name_snakecase + '-multi']
                        single_channel_val = freqs_df.at[layer_name, class_name_snakecase + '-single']

                        single_group_name, multi_group_name = utils.spatial_class_to_group[spatial_class]

                        # SingleFullChannel is taken from different files: take the critical frequency from it
                        if single_group_name == 'class-SingleFullChannel':
                            group_distribution_df.at[layer_name, single_group_name] += sfc_df.at[layer_name, config_id] * class_freq * single_channel_val
                        else:
                            group_distribution_df.at[layer_name, single_group_name] += crit_freq * class_freq * single_channel_val

                        group_distribution_df.at[layer_name, multi_group_name]  += crit_freq * class_freq * multi_channel_val
            
            critical_by_spatial_class[network_id][short_config] = group_distribution_df

            # save dataframe for this configuration
            save_application_df(
                df=group_distribution_df,
                output_dir=network_output_dir,
                filename=f'applev_class_groups_{short_config}',
                excel_sheet_name='critical_by_class',
                heatmap_title=f'Critical frequencies * class distribution - {short_config}',
                heatmap_xaxis='Spatial class',
                heatmap_yaxis='Layer',
                alternative_column_names=utils.short_class_group_names,
                # modify number of significant digits here if necessary
            )

            # sum over the columns and add the resulting column to the total criticality dataframe
            total_criticality_df[short_config] = group_distribution_df.sum(axis=1)

        # save total criticality dataframe
        save_application_df(
            df=total_criticality_df,
            output_dir=network_output_dir,
            filename='total_crit',
            excel_sheet_name='total_crit_csv',
            heatmap_title=f'Total criticality values for {network_id}',
            heatmap_xaxis='Configs',
            heatmap_yaxis='Layer',
        )


def build_application_fit_dfs(args: Args, applev_dicts: dict, app_output_dir: str):
    # final fit dataframe
    network_fit_df = pd.DataFrame(0.0, index=args.network_dataset_ids, columns=args.short_configuration_ids)

    for network_id in args.network_dataset_ids:
        network_output_dir = os.path.join(app_output_dir, network_id)

        layers = list(applev_dicts[network_id][args.configuration_ids[0]].keys())
        
        # all layers fit dataframe
        layer_fit_df = pd.DataFrame(0.0, index=layers, columns=args.short_configuration_ids)

        # load final report file
        if network_id.startswith('mobilenetv2'):    # mobilenet special case
            final_report_path = os.path.join(args.final_reports_base_path, 'mobilenetv2_gtsrb_final.yaml')
        else:
            final_report_path = os.path.join(args.final_reports_base_path, network_id + '_final.yaml')
        with open(final_report_path) as f:
            report_dict = yaml.load(f, yaml.SafeLoader)
        
        for config_id, short_config in zip(args.configuration_ids, args.short_configuration_ids):
            config_dict = report_dict[config_id]['layers']

            for layer_name, layer_dict in config_dict.items():
                layer_fit_df.at[layer_name, short_config] = layer_dict['FIT']
            
            # add total fit to final df
            network_fit_df.at[network_id, short_config] = report_dict[config_id]['total']['FIT']
        
        # save layer fit dataframe and heatmap
        save_application_df(
            df=layer_fit_df,
            output_dir=network_output_dir,
            filename=f'layer_fit_values',
            excel_sheet_name='fit values',
            heatmap_title=f'Layer FIT values - {network_id}',
            heatmap_xaxis='Configuration',
            heatmap_yaxis='Layer',
            show_values_as_percentages=False,
            num_decimal_digits=2,
        )
    
    # save final network fit dataframe
    save_application_df(
        df=network_fit_df,
        output_dir=app_output_dir,
        filename='network_fit_values',
        excel_sheet_name='fit values',
        heatmap_title='Network FIT values',
        heatmap_xaxis='Configuration',
        heatmap_yaxis='Network',
        show_values_as_percentages=False,
        num_decimal_digits=2,
    )


def build_total_criticality_rankings(args: Args, app_output_dir: str):
    for network_id in args.network_dataset_ids:
        network_output_dir = os.path.join(app_output_dir, network_id)

        # load total criticality dataframe and average over rows to obtain mean network criticality
        total_crit_df = pd.read_csv(os.path.join(network_output_dir, 'total_crit.csv'), index_col=0)
        mean_crit_df = total_crit_df.mean().sort_values(ascending=False).T

        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.set_title(network_id)

        table_data = [[config, value.item()] for config, value in zip(mean_crit_df.index, mean_crit_df.values)]

        colors = [[entry, entry] for entry in cm.tab10(np.linspace(0, 1, len(mean_crit_df)))]
        the_table = ax.table(cellText=table_data, rowLabels=None, colLabels=['Config', 'Crit'], cellColours=colors, loc='center')
        # the_table.scale(1,0.6)

        plot_output_path = os.path.join(network_output_dir, 'total_crit_rankings.png')
        fig.savefig(plot_output_path, dpi=100)
        plt.close()


def delta_crit_graph(delta_df: pd.DataFrame, mean_layer_crit: pd.Series, network_id: str, layer: str, graphs_dir: str, center_on_mean=True, error_bars=False):
    fig, ax = plt.subplots(3, 1)
    fig.suptitle(f'Delta criticality for {network_id} - {layer}')

    for i, param in enumerate(['C', 'K', 'BW']):
        # sort the df according to the parameter row
        sorted_delta_df = delta_df.sort_values(by=param, axis=1)
        
        ax[i].set_title(f'Sorted by {param}')

        x = sorted_delta_df.columns
        y = sorted_delta_df.loc[layer]
        bottom = mean_layer_crit[layer] if center_on_mean else 0.0

        ax[i].bar(x, y, bottom=bottom)

        if error_bars:
            ax[i].errorbar(x, y, yerr=sorted_delta_df.iloc[:-4, :].std(axis=0), fmt='o', color='r')

        ax[i].hlines(y=bottom, xmin=x[0], xmax=x[-1], label=bottom)

    plt.subplots_adjust(hspace=0.5)
    fig.set_size_inches(19.2, 10.8)
    fig.savefig(os.path.join(graphs_dir, f'{layer}.png'))
    plt.close()


def build_delta_criticality_graphs(args: Args, app_output_dir: str):
    for network_id in args.network_dataset_ids:
        print(f'Building delta crit graphs for {network_id}')
        network_output_dir = os.path.join(app_output_dir, network_id)

        graphs_dir = os.path.join(network_output_dir, 'delta_criticality')
        os.makedirs(graphs_dir, exist_ok=True)

        # load total criticality dataframe and get layer averages
        total_crit_df = pd.read_csv(os.path.join(network_output_dir, 'total_crit.csv'), index_col=0)

        # get parameters from configuration ids
        config_ids = total_crit_df.columns.tolist()
        cs = []
        ks = []
        bws = []
        for config_id in config_ids:
            pieces = config_id.split('_')
            c, k = pieces[0].split('x')
            bw = pieces[-1][3:]
            cs.append(int(c))
            ks.append(int(k))
            bws.append(int(bw))

        # obtain delta criticality by subtracting the mean column from all the columns
        mean_crit_df = total_crit_df.mean(axis=1)
        delta_crit_df = total_crit_df.sub(mean_crit_df, axis=0)

        # append the parameter rows and the average rows
        delta_crit_df.loc['NetworkMean', :] = delta_crit_df.mean(axis=0)
        delta_crit_df.loc['C', :] = cs
        delta_crit_df.loc['K', :] = ks
        delta_crit_df.loc['BW', :] = bws

        # for each layer, plot the displacement graphs
        for layer in total_crit_df.index:
            delta_crit_graph(delta_crit_df, mean_crit_df, network_id, layer, graphs_dir)

        # also plot the graphs for the whole network mean
        delta_crit_graph(delta_crit_df, mean_crit_df, network_id, 'NetworkMean', graphs_dir, center_on_mean=False, error_bars=True)


def main():
    args = parse_arguments()
    
    # ARCHITECTURAL LEVEL
    architectural_output_dir = os.path.join(args.output_dir, 'sdc_and_classes')
    os.makedirs(architectural_output_dir, exist_ok=True)

    error_models_dfs = collect_error_models_dfs(args, architectural_output_dir)
    sdc_df = build_sdc_df(args, error_models_dfs, architectural_output_dir)
    
    build_class_distribution_dfs(args, error_models_dfs, architectural_output_dir)

    # APPLICATION LEVEL
    app_output_dir = os.path.join(args.output_dir, 'cross_layer_aggregates')
    os.makedirs(app_output_dir, exist_ok=True)

    applev_dicts = load_applev_dicts(args)
    build_application_class_distrib_dfs(args, applev_dicts, app_output_dir)
    build_application_fit_dfs(args, applev_dicts, app_output_dir)
    build_total_criticality_rankings(args, app_output_dir)
    build_delta_criticality_graphs(args, app_output_dir)


if __name__ == '__main__':
    main()