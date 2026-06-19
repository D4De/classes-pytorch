"""
Modifies dictionaries in the simulation output applev files to ensure that every spatial class is represented.
Missing spatial classes are introduced in each layer as entries whose occurrence frequencies are all 0.0.
From the `experiments` directory, this can be run with:

python _fix_applevs.py `find exp* -name "applev*"`

to target every applev found in all directories whose name starts with exp.
"""

import sys
import yaml

snakecase_spatial_classes = [
  'full_channels',
  'single',
  'single_block',
  'bullet_wake',
  'single_channel_random',
  'multi_channel_block',
  'shattered_channel',
  'quasi_shattered_channel',
  'rectangles',
  'same_row',
  'skip_4',
  'single_channel_alternated_blocks',
  'multiple_channels_uncategorized',
]

paths = sys.argv[1:]
for path in paths:
    modified = False

    with open(path) as f:
        applev_dict = yaml.load(f, yaml.SafeLoader)

    applev_copy = applev_dict.copy()

    for layer_name, layer_dict in applev_dict.items():
        for spatial_class in snakecase_spatial_classes:
            if spatial_class not in layer_dict.keys():
                applev_copy[layer_name][spatial_class] = {
                    'masked'       : 0.0,
                    'sdc_safe'     : 0.0,
                    'sdc_critical' : 0.0,
                }
                modified = True

    if modified:
        print(f'Modifying for {path}')
        save_path = path[:-5] + '_fixed.yaml'
        with open(save_path, 'w') as f:
            yaml.dump(applev_copy, f, sort_keys=False)