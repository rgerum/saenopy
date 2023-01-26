from . import numbaOverload
from .load import *
from .save import *
from .solver import Solver, load, save, load_results
from .result_file import get_stacks, Result
from .getDeformations import get_displacements_from_stacks
from .solver import substract_reference_state, interpolate_mesh
from .examples import loadExample
import importlib.metadata

__version__ = importlib.metadata.metadata('saenopy')['version']
