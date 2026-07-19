import os

def select_index_from_list_interactive(choices: list, prompt_msg: str, allow_none=False):
    """
    Given a list of alternatives, asks the user to select one and returns the
    chosen index.

    If 'allow_none' is True, the user may input 'none' to select nothing. In that case,
    None is returned.
    """
    while True:
        print(prompt_msg)
        print('Pick from the following (input the index):')

        if allow_none:
            print('--NOTE: you may skip this selection and move on by inputting \'none\'')

        for i, choice in enumerate(choices):
            print(f'\t{i:<3} {choice}')

        selection = input('>> ').strip().lower()

        if allow_none and selection == 'none':
            return None

        try:
            selection = int(selection)
        except:
            print('--> Input is not a number <--\n')
            continue

        if selection < 0 or selection >= len(choices):
            print('--> Index out of range <--\n')
            continue
        
        return selection


def read_string_non_empty_interactive(prompt_msg: str=''):
    """
    Gets a string from stdin, displaying the provided prompt and continuing until a non-empty string is entered.
    """
    while True:
        print(prompt_msg)
        response = input('>> ').strip()

        if response != '':
            return response
        # add an empty line and try again
        print()
        

def read_separated_string_list_interactive(prompt_msg: str, separator=','):
    """
    Reads a non-empty string from stdin and splits it according to the provided separator. Returns the resulting list.
    """
    response = read_string_non_empty_interactive(prompt_msg)
    return [item.strip() for item in response.split(separator)]


def yes_no_choice_interactive(question: str):
    """
    Asks the user to respond to the provided question with 'y' or 'n'. Returns the answer as a bool.
    """
    while True:
        response = read_string_non_empty_interactive(question + ' [y/n]').lower()
        if response == 'y':
            return True
        if response == 'n':
            return False
        # add empty line and try again
        print()


def read_existing_path_interactive(prompt_msg: str, must_be_dir=False):
    """
    Asks the user to provide a path to a file that must exist. If 'must_be_dir' is True, the path must be that of a directory.
    """
    while True:
        response = os.path.realpath(read_string_non_empty_interactive(prompt_msg))

        if must_be_dir:
            if os.path.isdir(response):
                return response
            print('--> This path does not correspond to an existing directory <--\n')
        else:
            if os.path.exists(response):
                return response
            print('--> This path does not correspond to an existing file <--\n')


def read_number_interactive(prompt_msg: str, type: int|float=int, force_positive=False) -> int|float:
    while True:
        response = read_string_non_empty_interactive(prompt_msg)

        try:
            number = type(response)
        except:
            print(f'--> Not a number or wrong type (should be {type}) <--\n')
            continue
        
        if force_positive and number <= 0:
            print('--> Number must be positive <--\n')
            continue

        return number


def read_number_in_range_interactive(prompt_msg: str, type: int|float, min, max) -> int|float:
    while True:
        response = read_string_non_empty_interactive(prompt_msg + f'\nPick a number in [{min}, {max}].')

        try:
            number = type(response)
        except:
            print(f'--> Not a number or wrong type (should be {type}) <--\n')
            continue

        if number < min or number > max:
            print('Number is out of range.')
            continue

        return number


def select_multiple_indices_from_list_interactive(choices: list, prompt_msg: str, allow_none=False):
    chosen: list[int] = []

    while True:
        print(prompt_msg)
        print('Pick from the following by entering an index or multiple indices separated by commas.')
        print('Make as many selections as you want. When you\'re done, just enter \'done\'.\n')
        print(f'\tCurrent selection is: {chosen}')

        for i, choice in enumerate(choices):
            print(f'{i}\t{choice}')

        response = read_string_non_empty_interactive().split(',')

        # try to end selection
        if len(response) == 1 and response[0] == 'done':
            if allow_none or len(chosen) != 0:
                return chosen
        
            print('--> You must select at least one index <--\n')
            continue

        for index in response:
            try:
                index = int(index)
                if index < 0 or index >= len(choices):
                    print(f'--> {index} is out of range. Skipping. <--')
                    continue
                if index not in chosen:
                    chosen.append(index)
            except:
                print(f'--> {index} is not a number. Skipping. <--')


def print_console_separator(sep='-', amount=40):
    print(sep * amount)