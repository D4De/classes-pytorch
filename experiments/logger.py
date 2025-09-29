import logging

def create_experiment_logger(logger_name: str, logfile_path: str):
    logger = logging.getLogger(logger_name)
    logger.setLevel('DEBUG')

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "{levelname} - {message}",
        style='{',
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel('DEBUG')

    file_handler = logging.FileHandler(logfile_path, mode='a', encoding='utf-8')
    file_formatter = logging.Formatter(
        "{asctime} - {levelname} - {message}",
        style='{',
        datefmt='%Y-%m-%d %H.%M.%S',
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel('INFO')

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger