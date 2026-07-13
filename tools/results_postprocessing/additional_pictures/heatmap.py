import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import utils

from argparse import ArgumentParser

# set plot sizes
plt.rc('font', size=9)          # controls default text sizes
plt.rc('axes', titlesize=13)    # fontsize of the axes title
plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)   # fontsize of the tick labels
plt.rc('ytick', labelsize=8)   # fontsize of the tick labels
plt.rc('legend', fontsize=9)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

CMAP_RED_TO_GREEN = 'RdYlGn_r'

#------

LAYER_COLS = ['benchmark', 'layer', 'config']
HYPER_COLS = ['K', 'C', 'W', 'R', 'padding-x'] # in order: (Channels_out, Channels_in, Input_size, Kernel_size, Padding)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--show_annotations', action='store_true')
    parser.add_argument('--fig_pixels_x', type=int, required=False, default=1920)
    parser.add_argument('--fig_pixels_y', type=int, required=False, default=1080)
    return parser.parse_args()


def get_packed_hyperparameters_list(df: pd.DataFrame):
    """Packs all columns of the provided dataframe into a series of tuples which can be used as single-level index."""
    hyper_tuples: list[tuple] = []
    for _, row in df.iterrows():
        hyper_tuples.append(tuple(row))
    return hyper_tuples


def df_from_results_csv(results_path: str):
    results_df = pd.read_csv(results_path, index_col=0)

    # extract relevant columns
    results_df = results_df[LAYER_COLS + HYPER_COLS + ['Silent']]

    # get configuration names
    configs = utils.sort_configs(results_df['config'].unique())

    # sort by hyperparameters
    results_df = results_df.sort_values(by=HYPER_COLS)
    # add hyperparameter tuples
    results_df['tuples'] = get_packed_hyperparameters_list(results_df[HYPER_COLS])

    # build columns: one per configuration
    df_cols = []
    for config in configs:
        short_config = utils.config_reformat(config) + '_' + utils.detach_precision(config)[1]
        config_rows = results_df[results_df['config'] == config]

        hyper_sets = []
        sdc_values = []
        # average frequencies of layers with the same hyperparameters
        for hyper_set, rows in config_rows.groupby('tuples'):
            # avg_sdc = rows['Silent'].mean() if rows['Silent'] != {} else 0
            avg_sdc = pd.to_numeric(rows['Silent'], errors="coerce").mean() or 0
            hyper_sets.append(hyper_set)
            sdc_values.append(avg_sdc)
        
        df_cols.append( pd.Series(sdc_values, name=short_config, index=hyper_sets) )

    return pd.concat(df_cols, axis=1)


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

    val_min = df.min().min()
    val_max = df.max().max()

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


def main():
    units = [
        'top',
        # 'top.conv.csb',
        # 'top.conv.csc',
        # 'top.conv.dl',
        # 'top.conv.wl',
        # 'top.conv.cdma',
        # 'top.conv.cbuf',
        # 'top.conv.cmac',
        # 'top.conv.cacc',
        # 'top.conv.dbuf',
        # 'top.sdp',
    ]

    args      = parse_arguments()
    outdir    = os.path.realpath(args.outdir)
    save_dir  = os.path.join(outdir, 'pictures', 'heatmaps')
    os.makedirs(save_dir, exist_ok=True)

    for unit in units:

        print(f'-I: Generating Heatmap for {unit}...')

        if unit == 'top':
            suffix = ''
        else:
            suffix = f'_{unit}'

        results_csv_path    = os.path.join(outdir, 'reports', f'results{suffix}.csv')
        show_annotations    = args.show_annotations
        fig_pixels_x        = args.fig_pixels_x
        fig_pixels_y        = args.fig_pixels_y

        save_path = os.path.join(outdir, 'pictures', 'heatmaps', f'pvf_heatmap{suffix}.png')
        
        sdc_df = df_from_results_csv(results_csv_path)
        make_heatmap(
            df = sdc_df,
            suptitle = 'SDC values by configuration (%)',
            xaxis_label = 'Configuration',
            yaxis_label = 'Layer hyperparameters',
            save_path = save_path,
            show_cell_values = show_annotations,
            show_values_as_percentages = True,
            fig_pixels_x = fig_pixels_x,
            fig_pixels_y = fig_pixels_y,
        )


if __name__ == '__main__':
    main()
