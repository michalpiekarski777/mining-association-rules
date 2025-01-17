import logging
from datetime import datetime
from pathlib import Path

from config import ROOT_DIR


class Logger(logging.Logger):
    """Custom logger class with StreamHandler and FileHandler saving logs in the logs directory"""

    @staticmethod
    def _get_path() -> Path:
        """Returns path to a file used for logging. Creates logs directory if necessary."""
        path = Path(ROOT_DIR) / "logs"
        path.mkdir(exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d_%H%M_%s")

        return path / filename

    def _add_handlers(self):
        filename = self._get_path()
        self.addHandler(logging.FileHandler(filename))
        self.addHandler(logging.StreamHandler())

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self._add_handlers()
