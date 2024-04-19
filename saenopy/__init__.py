from .solver import Solver, load, load_results
from .result_file import get_stacks, Result
from .get_deformations import get_displacements_from_stacks
from .solver import subtract_reference_state, interpolate_mesh
from .examples import load_example
import importlib.metadata
from .gui.solver.modules.exporter.Exporter import render_image

from . import pyTFM as pyTFM

__version__ = importlib.metadata.metadata('saenopy')['version']
