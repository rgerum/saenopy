import warnings


class suppress_warnings:
    def __init__(self, warning_type):
        self.warning_type = warning_type

    def __enter__(self):
        warnings.filterwarnings("ignore", category=self.warning_type)

    def __exit__(self, warn_type, value, traceback):
        warnings.filterwarnings("default", category=self.warning_type)
