import logging


class LogCaptureHandler(logging.Handler):
    """Captures logs and sends them to a Rich display component."""

    def __init__(self, display_callback, log_level=logging.INFO):
        super().__init__(level=log_level)
        self.display_callback = display_callback
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
            )
        )

    def emit(self, record):
        log_entry = self.format(record)
        self.display_callback(log_entry)
