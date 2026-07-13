import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# set plot sizes
plt.rc('font', size=9)          # controls default text sizes
plt.rc('axes', titlesize=13)    # fontsize of the axes title
plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)   # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=9)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

CMAP_RED_TO_GREEN = 'RdYlGn_r'
CMAP_ALTERNATIVE = 'managua'

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
):
    """If save_path is None, the heatmap is shown, not saved as an image."""

    df = df.astype(float)

    fig, ax = plt.subplots()
    fig.suptitle(suptitle)
    cbar_ax = fig.add_axes([.91, .3, .03, .4]) if add_colorbar else None

    if num_decimal_digits < 0:
        num_decimal_digits = 0
    annot_format = f'.{num_decimal_digits}%' if show_values_as_percentages else f'.{num_decimal_digits}f'

    if show_values_as_percentages:
        val_min = 0
        val_max = 1
    else:
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
    ax.tick_params(axis='x', labelrotation=0)
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)
    
    if save_path is None:
        plt.show()
    else:
        save_plt_figure(fig, save_path)
        plt.close()


def save_plt_figure(fig, filename: str, sizex=19.2, sizey=10.8, dpi=100):
    """Saves a pyplot figure to file. Defaults to a resolution of 1920x1080."""
    fig.set_size_inches(sizex, sizey)
    fig.savefig(filename, dpi=dpi)