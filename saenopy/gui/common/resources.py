import os
from pathlib import Path
from qtpy import QtGui


def resource_path(relative):
    path = os.path.join(
        str(Path(__file__).parent.parent.parent / "img"),
        relative
    )
    return path


def resource_icon(name):
    return QtGui.QIcon(resource_path(name))
