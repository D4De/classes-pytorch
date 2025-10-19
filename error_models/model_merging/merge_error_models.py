import os
import json

from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('output_path', help='Filepath to save the merged model to.')
    parser.add_argument('source_paths', nargs='+', help='List of filepaths to the error models to be merged.')
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_arguments()
    output_path: str = args['output_path']
    source_paths: list[str] = args['source_paths']

    if len(source_paths) < 2:
        raise ValueError('Expected at least 2 source models.')
    
    # load models
    final_tensor_count = 0
    spatial_classes: dict[str, list] = {}

    for source_path in source_paths:
        source_path = os.path.realpath(source_path)
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
    if not output_path.endswith('.json'):
        output_path += '.json'

    with open(output_path, 'w') as f:
        json.dump(final_model, f)