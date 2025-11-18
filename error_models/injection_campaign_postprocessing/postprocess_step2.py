import os
import json
import shutil
import pandas as pd

import error_models.injection_campaign_postprocessing.postprocessing_utils as utils


def merge_matching_layers(complete_df: pd.DataFrame, initial_models_dir: str, merged_models_dir: str):
    """Groups layers with the same hyperparameters and averages all the frequencies to obtain the description of their merged
    model. Builds a new layer dataframe with each group replaced by its merged row. At the same time, the error models
    corresponding to the groups are also merged."""
    merged_model_index = 0
    new_df_rows = []
    freq_cols = utils.macroscopic_results + utils.spatial_classes

    for group_layers, group in complete_df.groupby(by=utils.hyperparameters):
        if len(group_layers) == 1:
            # just one row, use it as-is
            new_df_rows.append(group)
            # simply copy the model file to the output directory
            src_path = os.path.join(initial_models_dir, group_layers[0] + '.json')
            dst_path = os.path.join(merged_models_dir, group_layers[0] + '.json')
            shutil.copyfile(src_path, dst_path)
        else:
            new_model_name = pd.Series({'Layer': f'merge{merged_model_index}'})
            merged_model_index += 1

            # extract hyperparameter columns from one of the group rows
            hyper_values = group.iloc[0][utils.hyperparameters]

            freq_df = group[freq_cols]
            freq_averages = freq_df.mean()

            new_df_rows.append(pd.concat([new_model_name, hyper_values, freq_averages]))

            # merge the error models of the group into a single one
            merge_error_models(initial_models_dir, list(group_layers), merged_models_dir, new_model_name)
    
    # build unique complete dataframe
    return pd.concat(new_df_rows, axis=1)


def merge_error_models(src_models_dir: str, model_names: list[str], output_models_dir: str, merged_model_name: str):
    # load models
    final_tensor_count = 0
    spatial_classes: dict[str, list] = {}

    for model_name in model_names:
        source_path = os.path.join(src_models_dir, model_name + '.json')
        if not os.path.exists(source_path):
            raise ValueError(f'Error model at {source_path} does not exist.')
        
        with open(source_path) as f:
            model: dict = json.load(f)

        final_tensor_count += model['_tensor_count']
        # iterate through the model fields and add the spatial classes to the dictionary
        for name, field in model.items():
            if name.startswith('_'):
                continue

            if name in spatial_classes:
                spatial_classes[name].append(field)
            else:
                spatial_classes[name] = [field]

    # build the final model
    final_model = {
        '_tensor_count': final_tensor_count,
        '_categories_count': len(spatial_classes),
    }

    # iterate through the spatial classes lists: for each list, build a merged version
    for class_name, class_list in spatial_classes.items():
        num_instances = len(class_list)

        total_count = 0
        frequency_sum = 0.0
        num_categories = 0
        new_domain_classes = []
        new_parameters = []

        # collect all model pieces
        for class_instance in class_list:
            total_count         += class_instance['count']
            frequency_sum       += class_instance['frequency']
            num_categories      += class_instance['categories_count']
            new_domain_classes  += class_instance['domain_classes']
            new_parameters      += class_instance['parameters']

        final_frequency = frequency_sum / num_instances # average the frequencies

        # rescale domain class and spatial parameter frequencies
        for domain_class in new_domain_classes:
            domain_class['frequency'] = domain_class['frequency'] / num_instances
        for parameter in new_parameters:
            parameter['conditional_frequency'] = parameter['conditional_frequency'] / num_instances
            parameter['overall_frequency'] = parameter['conditional_frequency'] * final_frequency

        # add to the final model
        final_model[class_name] = {
            'count': total_count,
            'frequency': final_frequency,
            'categories_count': num_categories,
            'domain_classes': new_domain_classes,
            'parameters': new_parameters
        }
    
    # save the final model
    if not merged_model_name.endswith('.json'):
        merged_model_name += '.json'
    output_path = os.path.join(output_models_dir, merged_model_name)

    with open(output_path, 'w') as f:
        json.dump(final_model, f)


def model_reconstruction_test(unique_complete_df: pd.DataFrame, num_neighbors=5):
    """Takes each row of the unique dataframe describing each error model and tries to reconstruct it by averaging over the
    'num_neighbors' nearest neighbors. The absolute difference of the resulting row and the original row is used as a row in
    a new reconstruction distance dataframe.
    For the Masked column only, the Z-score and IQR tests are performed: if both fail, the corresponding error model
    is considered unreconstructable and is signaled. The SDC column is almost exactly 1 - Masked, so the tests for one
    column also give information on the other."""
    # normalize hyperparameters
    model_df = unique_complete_df.astype(float)
    for i, row in model_df.iterrows():
        hypers = row[utils.hyperparameters]
        h_min: float = hypers.min()
        h_max: float = hypers.max()
        normalized_hypers = (hypers - h_min) / (h_max - h_min)
        model_df.loc[i, utils.hyperparameters] = normalized_hypers

    # save frequency distance rows to build an Excel file
    frequency_distance_rows = []

    # start reconstruction test
    for i, model_row in model_df.iterrows():
        rest_of_df = model_df.drop(i)

        # extract hyperparameters
        model_hyper = model_row[utils.hyperparameters]
        rest_of_df_hyper = rest_of_df.loc[:, utils.hyperparameters]

        # compute all distances and take the smallest ones
        distances = ((rest_of_df_hyper - model_hyper) ** 2).sum(axis=1)
        min_distance_indices = distances.argsort()[:num_neighbors]

        # select the corresponding rows
        closest_rows = rest_of_df.iloc[min_distance_indices]

        # extract frequencies
        model_freq = model_row[utils.macroscopic_results + utils.spatial_classes]
        rest_of_df_freq = closest_rows.loc[:, utils.macroscopic_results + utils.spatial_classes]

        # build average model
        avg_freq = rest_of_df_freq.mean()

        # compute distances from the row
        freq_distances = (model_freq - avg_freq).abs()
        frequency_distance_rows.append(pd.concat([
            pd.Series({'Layer': i}),
            model_hyper,
            freq_distances
        ]))

    # build frequency distance dataframe
    distance_df = pd.concat(frequency_distance_rows)
    
    # get Masked distances
    masked_col = distance_df['Masked']

    # compute zscores and check where the absolute value is greater than 3
    zscore_col = (masked_col - masked_col.mean()) / masked_col.std()
    zscore_fail_col = (zscore_col.abs() > 3.0)

    # get first and third quartiles, compute IQR and corresponding bounds; check where Masked is outside the interval
    quartile1 = masked_col.quantile(q=0.25)
    quartile3 = masked_col.quantile(q=0.75)
    iqr = quartile3 - quartile1
    iqr_lower = quartile1 - 1.5 * iqr
    iqr_upper = quartile3 + 1.5 * iqr
    iqr_fail_col = (masked_col < iqr_lower) | (masked_col > iqr_upper)

    # check where both tests fail
    both_fail_col = zscore_fail_col & iqr_fail_col

    # add the three test columns to the distance dataframe
    distance_df.loc[:, 'Masked_Zscore'] = zscore_col
    distance_df.loc[:, 'Masked_outside_iqr_range'] = iqr_fail_col
    distance_df.loc[:, 'Masked_failed_both'] = both_fail_col

    return distance_df



if __name__ == '__main__':
    config_dict = utils.load_config_dict()
    output_dir = config_dict['final_output_dir']

    step1_output_path = os.path.join(output_dir, utils.step1_output_filename)
    initial_models_dir = os.path.join(output_dir, utils.initial_models_dirname)

    merged_models_dir = os.path.join(output_dir, utils.merged_models_dirname)
    os.makedirs(merged_models_dir, exist_ok=True)

    # load complete layer dataframe from step 1
    complete_df = pd.read_excel(step1_output_path, sheet_name=0, header=0)

    # merge layers with the same hyperparameters and build final models dataframe
    unique_complete_df = merge_matching_layers(complete_df, initial_models_dir, merged_models_dir)
    step2_output_path = os.path.join(output_dir, utils.step2_output_filename)
    unique_complete_df.to_excel(step2_output_path, sheet_name='unique_complete_layers')

    # perform reconstruction test on unique df
    reconstruction_distance_df = model_reconstruction_test(unique_complete_df)
    step2_reconstruction_output_path = os.path.join(output_dir, utils.step2_reconstruction_output_filename)
    reconstruction_distance_df.to_excel(step2_reconstruction_output_path, sheet_name='reconstruction_test')

    print('STEP 2 DONE')
    print(f'The final merged layers dataframe has been saved as {step2_output_path} and is now available for CLASSES simulation.')
    print(f'The merged error models are in {merged_models_dir}.')
    print(f'The model reconstruction test results are in {step2_reconstruction_output_path}. Review the file and ' \
          'consider removing models that fail both tests. If you choose to remove a model, delete its json file from ' \
          f'{merged_models_dir} and delete the corresponding entry in the dataframe file.')