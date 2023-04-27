from pathlib import Path
import shutil
import tifffile
import numpy as np
import pytest
import uuid
import os
import appdirs


class MockDir:
    def __init__(self, structure, callback=None):
        self.structure = structure
        self.callback = callback

    def __enter__(self):
        def mock_dir(structure, parent=None):
            if parent is None:
                parent = Path(".")
            if isinstance(structure, list):
                for file in structure:
                    if self.callback:
                        self.callback(parent / file)
                    else:
                        (parent / file).touch(exist_ok=True)
            else:
                for key in structure:
                    folder = parent / key
                    folder.mkdir(exist_ok=True)
                    mock_dir(structure[key], folder)

        mock_dir(self.structure)

    def __exit__(self, exc_type, exc_val, exc_tb):
        def remove_mock(structure, parent=None):
            if parent is None:
                parent = Path(".")
            if isinstance(structure, list):
                for file in structure:
                    (parent / file).unlink(missing_ok=True)
            else:
                for key in structure:
                    folder = parent / key
                    remove_mock(structure[key], folder)
                    #folder.rmdir()
                    shutil.rmtree(folder, ignore_errors=False, onerror=None)
        remove_mock(self.structure)


def mock_dir(structure, parent=None, callback=None):
    if parent is None:
        parent = Path(".")
    if isinstance(structure, list):
        for file in structure:
            if callback:
                callback(parent / file)
            else:
                (parent / file).touch(exist_ok=True)
    else:
        for key in structure:
            folder = parent / key
            folder.mkdir(exist_ok=True)
            mock_dir(structure[key], folder, callback)


def create_tif(filename, y=20, x=10, z=1, rgb=None):
    with tifffile.TiffWriter(filename) as tif:
        for i in range(z):
            if rgb is None:
                tif.write((np.random.rand(y, x)*255).astype(np.uint8))
            else:
                tif.write((np.random.rand(y, x, rgb)*255).astype(np.uint8))


def sf4(x):
    if isinstance(x, float):
        x = float(np.format_float_positional(x, precision=4, unique=False, fractional=False,trim='k'))
        return x
    return [sf4(xx) for xx in x]


@pytest.fixture
def random_path(tmp_path, monkeypatch):
    target_path = Path(tmp_path) / str(uuid.uuid4())
    target_path.mkdir(exist_ok=True)
    os.chdir(target_path)

    monkeypatch.setattr(appdirs, "user_data_dir", lambda *args: Path(tmp_path) / "saenopy" / "rgerum")