#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from qtpy import QtWidgets
import numpy as np
from mock_dir import mock_dir, create_tif, sf4
from saenopy.gui.gui_master import MainWindow
from saenopy import get_stacks
import sys
import os
from pathlib import Path
np.random.seed(1234)


@pytest.fixture
def files(tmp_path):
    file_structure = {
        "tmp": {
            "run-1": [f"Pos004_S001_z{z:03d}_ch00.tif" for z in range(50)],
            "run-1-reference": [f"Pos004_S001_z{z:03d}_ch00.tif" for z in range(50)],
        }
    }
    os.chdir(tmp_path)
    mock_dir(file_structure, callback=lambda file: create_tif(file, x=50, y=50))


def test_stack(files, monkeypatch):
    app = QtWidgets.QApplication(sys.argv)
    # start the main gui
    from saenopy.gui.gui_master import MainWindow
    window: MainWindow = MainWindow()  # gui_master.py:MainWindow
    window.changedTab(1)

    # switch to the Solver part
    from saenopy.gui.solver.gui_solver import MainWindow
    solver: MainWindow = window.solver  # modules.py:MainWindow

    # get the Evaluate part
    from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate
    batch_evaluate: BatchEvaluate = solver.deformations  # modules/BatchEvaluate.py:BatchEvaluate

    # get the input
    results = get_stacks("tmp/run-1/Pos004_S001_z{z}_ch00.tif", "tmp/run-1", [1, 1, 1],
                         reference_stack="tmp/run-1-reference/Pos004_S001_z{z}_ch00.tif")
    # bundle the images
    results[0].stack[0].pack_files()
    results[0].stack_reference.pack_files()

    # add them to the gui and select them
    batch_evaluate.list.addData(results[0].output, True, results[0], "#FF0000")
    batch_evaluate.list.setCurrentRow(0)

    # change some parameters
    batch_evaluate.sub_module_deformation.setParameter("win_um", 30)
    batch_evaluate.sub_module_regularize.setParameter("i_max", 10)
    print("params A", getattr(batch_evaluate.sub_module_deformation.result, batch_evaluate.sub_module_deformation.params_name + "_tmp"))
    print("params C", getattr(batch_evaluate.sub_module_regularize.result, batch_evaluate.sub_module_regularize.params_name + "_tmp"))

    # schedule to run all
    batch_evaluate.run_all()

    # wait until they are processed
    while batch_evaluate.current_task_id < len(batch_evaluate.tasks):
        app.processEvents()

    # check the result
    M = results[0].solver[0]
    print(M.U[M.reg_mask])
    print(results[0].solver[0].U.shape)
    #assert sf4(M.U[M.reg_mask][0]) == sf4([-2.01259036e-38, -1.96865342e-38, -4.92921492e-38])
    #91.64216076e-38 -3.15079497e-39  3.19069614e-39

    # apply the monkeypatch for requests.get to mock_get
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: "tmp.py")
    batch_evaluate.generate_code()


if __name__ == "__main__":
    def files(tmp_path):
        file_structure = {
            "tmp": {
                "run-1": [f"Pos004_S001_z{z:03d}_ch00.tif" for z in range(50)],
                "run-1-reference": [f"Pos004_S001_z{z:03d}_ch00.tif" for z in range(50)],
            }
        }
        os.chdir(tmp_path)
        mock_dir(file_structure, callback=lambda file: create_tif(file, x=50, y=50))
    from pathlib import Path
    Path("tmp").mkdir(exist_ok=True)
    class MonkeyPatch:
        @staticmethod
        def setattr(obj, name, func):
            obj.name = func
    monkeypatch = MonkeyPatch()

    test_stack(files("tmp"), monkeypatch)
