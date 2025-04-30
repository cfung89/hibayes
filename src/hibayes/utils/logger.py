import logging


def init_logger(
    log_level: int = logging.INFO, handler: logging.Handler | None = None
) -> logging.Logger:
    """
    Initialise the logger for the application. Can be run as many times as needed.

    Args:
        log_level (int): The logging level. Default is logging.INFO.
        handler (logging.Handler | None): A custom logging handler e.g the rich handles, see ui

    """

    # set the global logger
    logging.getLogger().setLevel(log_level)
    if handler:
        logging.getLogger().addHandler(handler)

    return logging.getLogger()
