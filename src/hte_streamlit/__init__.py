from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

from .fit import *
from .file_processing import *
from .experiments_database import *