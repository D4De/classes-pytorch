def merge_error_models(error_model_dicts: list[dict]):
    final_tensor_count = 0
    spatial_classes: dict[str, list] = {}

    for error_model_dict in error_model_dicts:
        final_tensor_count += error_model_dict['_tensor_count']
        # iterate through the model fields and add the spatial classes to the dictionary
        for name, field in error_model_dict.items():
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
    
    # create merged model
    return final_model