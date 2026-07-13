"""
Given the CSV file containing, for each combination of layer and configuration, the aggregation coefficients (both time and area)
used to combine the individual units' injection results into a single SDC value, builds bar charts to compare the impact of
each coefficient across configurations.
Produces a figure for each layer, containing a chart for each configuration; each chart contains a bar for every unit: the bar's width
represents the time factor (liveness) of the unit, while its height represents the area factor. Both factors are in the range [0,1].
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REGIME_THRESHOLD = 0.8

aggregation_csv_path = '/home/miele/WORKSPACE/results-storage/network_reports/aggregation_pieces.csv'
output_dir = '/home/miele/WORKSPACE/results-storage/pictures/test_plots/aggregation_coefficients'


def make_layer_charts(layer_id, layer_rows: pd.DataFrame, time_cols: list[str], area_cols: list[str], units: np.ndarray, outdir: str):
    network, layer = layer_id
    layer_name = f'{network}_{layer}'
    output_path = os.path.join(outdir, f'{layer_name}.png')
    print(f'Making chart for {layer_name}')

    num_configs = len(layer_rows)
    # arrange plots in a square grid
    nsize = np.ceil(np.sqrt(num_configs)).astype(np.int64).item()

    # get layer-specific parameters from the first row
    first_row = layer_rows.iloc[0]
    C = first_row['C']
    K = first_row['K']

    fig = plt.figure()
    fig.suptitle(f'{layer_name} | C = {C}, K = {K}')
    fig.set_size_inches(19.20, 10.8)

    for i, (_, config_row) in enumerate(layer_rows.iterrows()):
        ax = fig.add_subplot(nsize, nsize, i+1)

        # get configuration parameters
        AC = config_row['atomic-c']
        AK = config_row['atomic-k']
        BW = config_row['bitwidth']
        CAC = config_row['c_over_atomicc']

        # determine regime
        regime = 'memory' if config_row['fetch_over_compute'] > REGIME_THRESHOLD else 'compute'

        ax.set_title(f'{regime=}, {AC=}, {CAC=}, {AK=}, {BW=}')
        # ax.title.set_size(7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Area')

        # x-axis: time factors
        widths = config_row[time_cols].to_numpy()
        # y-axis: area factors
        heights = config_row[area_cols].to_numpy()
        # total coefficients
        products = widths * heights
        total_weight = products.sum()

        # filter out units for which at least one coefficient is too small
        use_widths_mask = (widths > 1e-3)
        use_heights_mask = (heights > 0.1)

        use_mask = use_widths_mask & use_heights_mask
        no_use_mask = ~use_mask

        use_units = units[use_mask]

        use_widths = widths[use_mask]
        use_heights = heights[use_mask]

        use_products = products[use_mask]
        product_weights = use_products / total_weight

        leftover_product_sum = products[no_use_mask].sum()
        leftover_weight = leftover_product_sum / total_weight

        bars = ax.barh(use_units, width=use_widths, height=use_heights)

        # add the product labels: compute the products of area and time, truncate to 2 decimal digits
        product_labels = [f'{product:.2f}' for product in use_products]
        ax.bar_label(bars, labels=product_labels)

        # enlarge x-axis to fit labels
        ax.set_xlim(right=1.1)
        
        # prepare table dataframe
        table_df = pd.DataFrame(columns=['Area', 'Time', 'Product', 'Relative weight'])
        for i, unit in enumerate(use_units):
            table_df.loc[unit] = np.array([use_heights[i], use_widths[i], use_products[i], product_weights[i]])
        table_df.loc['Others'] = np.array([0, 0, leftover_product_sum, leftover_weight])
        table_df = table_df.round(decimals=2)

        the_table = ax.table(cellText=table_df, loc='bottom', fontsize=7)
        #the_table.auto_set_font_size(False)
        #the_table.auto_set_column_width(col=list(range(len(table_df.columns))))
        # the_table.scale(1,0.6)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    
    plt.close()


def main():
    aggregation_df = pd.read_csv(aggregation_csv_path)

    os.makedirs(output_dir, exist_ok=True)

    # extract names of time columns and area columns
    time_cols = list(filter(lambda x: x.startswith('time-'), aggregation_df.columns))
    area_cols = list(filter(lambda x: x.startswith('area-'), aggregation_df.columns))
    # get names of the units
    units = np.array([col_name[5:] for col_name in time_cols])

    for layer_id, layer_rows in aggregation_df.groupby(['benchmark', 'layer']):
        make_layer_charts(layer_id, layer_rows, time_cols, area_cols, units, output_dir)


if __name__ == '__main__':
    main()