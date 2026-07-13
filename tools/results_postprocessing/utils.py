import re

def snakecase_to_pascalcase(snake: str):
    """Utility class function to convert spatial class names from snake to pascal."""
    tokens = snake.split('_')
    return ''.join(token.capitalize() for token in tokens)

def pascal_to_snake(s: str) -> str:
    s = re.sub(r'(?<!^)([A-Z])', r'_\1', s)
    return s.lower()

def class_pascal_to_snake(name: str) -> str:
    if name == 'Skip4':
        return 'skip_4'
    else:
        return pascal_to_snake(name)


# List of spatial classes to consider (if there are other ones in the results, they will be ignored); names in Pascal case
spatial_classes = [
    'Single',
    'FullChannels',
    'MultiChannelBlock',
    'BulletWake',
    'Rectangles',
    'ShatteredChannel',
    'QuasiShatteredChannel',
    'SameRow',
    'SingleBlock',
    'Skip4',
    'SingleChannelRandom',
]

spatial_classes_snakecase = [class_pascal_to_snake(class_name) for class_name in spatial_classes]

# List of abbreviated spatial class names (for labelling)
short_spatial_classes = [
    'Sgl',
    'FllCh',
    'MltChBl',
    'BltWk',
    'Rect',
    'ShtrCh',
    'QShtrCh',
    'SmRow',
    'SglBl',
    'Skip4',
    'SglChRnd',
]

# List of hyperparameters that characterize error models
hyperparameters = [
    'Channels_out',
    'Channels_in',
    'Input_size',
    'Kernel_size',
    'Padding',
]

# mapping dictionary: each spatial class is mapped to its group
spatial_class_to_group = {
    'Single'                        : ('class-Single'             , 'class-BulletWake'),
    'Skip4'                         : ('class-SingleChannelRandom', 'class-MultiChannelRandom'),
    'SingleBlock'                   : ('class-SingleChannelBlock' , 'class-MultiChannelBlock'),
    'SingleChannelAlternatedBlocks' : ('class-SingleChannelRandom', 'class-MultiChannelRandom'),
    'SameRow'                       : ('class-SingleChannelBlock' , 'class-MultiChannelBlock'),
    'FullChannels'                  : ('class-SingleFullChannel'  , 'class-MultiFullChannels'),
    'Rectangles'                    : ('class-SingleChannelBlock' , 'class-MultiChannelBlock'),
    'SingleChannelRandom'           : ('class-SingleChannelRandom', 'class-MultiChannelRandom'),
    'MultiChannelBlock'             : ('class-SingleChannelBlock' , 'class-MultiChannelBlock'),
    'BulletWake'                    : ('class-Single'             , 'class-BulletWake'),
    'ShatteredChannel'              : ('class-SingleChannelRandom', 'class-MultiChannelRandom'),
    'QuasiShatteredChannel'         : ('class-SingleChannelRandom', 'class-MultiChannelRandom'),
    'MultipleChannelsUncategorized' : ('class-SingleChannelRandom', 'class-MultiChannelRandom'),
}

class_group_names = [
    'class-Single',
    'class-BulletWake',
    'class-SingleChannelRandom',
    'class-MultiChannelRandom',
    'class-SingleChannelBlock',
    'class-MultiChannelBlock',
    'class-SingleFullChannel',
    'class-MultiFullChannels',
]

short_class_group_names = [
    'Sgl',
    'BltWk',
    'SglChRnd',
    'MltChRnd',
    'SglChBlk',
    'MltChBlk',
    'SglFllCh',
    'MltFllCh',
]