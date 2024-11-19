#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import uuid

import pytest
from qtpy import QtWidgets, QtGui
import numpy as np
from mock_dir import mock_dir, create_tif, random_path
from pathlib import Path
import sys
import os
np.random.seed(1234)


@pytest.fixture
def catch_popup_error(monkeypatch):
    def do_raise(_, name, desc):
        raise QMessageBoxCritical(f"{name}: {desc}")
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", do_raise)


app = QtWidgets.QApplication(sys.argv)

def init_app():
    from saenopy.gui.gui_master import MainWindow
    window: MainWindow = MainWindow()  # gui_master.py:MainWindow
    window.setTab(1)

    # switch to the Solver part
    from saenopy.gui.solver.gui_solver import MainWindowSolver
    solver: MainWindowSolver = window.solver  # modules.py:MainWindow

    # get the Evaluate part
    from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluateBase
    batch_evaluate: BatchEvaluateBase = solver.deformations  # modules/BatchEvaluateBase.py:BatchEvaluate

    return app, window, solver, batch_evaluate


def in_random_dir():
    def wrapper(func):
        def wrapped(monkeypatch, random_path, catch_popup_error, *args, **kwargs):
            # create a new test dir
            test_dir = Path(str(uuid.uuid4()))
            test_dir.mkdir(exist_ok=True)
            test_dir = test_dir.absolute()
            os.chdir(test_dir)
            # call the function
            func(monkeypatch, random_path, catch_popup_error, *args, **kwargs)
            # back to the parent dir and remove the test dir
            os.chdir(test_dir.parent)
            shutil.rmtree(test_dir)
        return wrapped
    return wrapper

@in_random_dir()
def test_download_example(monkeypatch, random_path, catch_popup_error):
    from saenopy.examples import load_example, get_examples
    examples = get_examples()
    load_example("OrganoidTFM")
