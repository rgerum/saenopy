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
import appdirs
from pathlib import Path
np.random.seed(1234)


@pytest.fixture
def files(tmp_path, monkeypatch):
    file_structure = {
        "tmp": {
            "run-1": [f"Pos004_S001_z{z:03d}_ch00.tif" for z in range(50)],
            "run-1-reference": [f"Pos004_S001_z{z:03d}_ch00.tif" for z in range(50)],
            "run-2-time": [f"Pos004_S001_z{z:03d}_t{t:02d}.tif" for z in range(50) for t in range(3)],
            "run-2-reference": [f"Pos004_S001_z{z:03d}_t00.tif" for z in range(50)],
        },
        "saenopy": {"rgerum": []}
    }
    os.chdir(tmp_path)
    mock_dir(file_structure, callback=lambda file: create_tif(file, x=50, y=50))

    monkeypatch.setattr(appdirs, "user_data_dir", lambda *args: Path(tmp_path) / "saenopy" / "rgerum")


class QMessageBoxException(Exception):
    pass


class QMessageBoxCritical(QMessageBoxException):
    pass


@pytest.fixture
def catch_popup_error(monkeypatch):
    def do_raise(_, name, desc):
        raise QMessageBoxCritical(f"{name}: {desc}")
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", do_raise)


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

    from saenopy.gui.solver.modules.load_measurement_dialog import AddFilesDialog
    def handle_files(self: AddFilesDialog):
        # set the data stack
        self.stack_data.input_filename.setValue("tmp/run-1/Pos004_S001_z000_ch00.tif", send_signal=True)

        # set the reference stack
        self.reference_choice.setValue(1, send_signal=True)
        self.stack_reference.input_filename.setValue("tmp/run-1/Pos004_S001_z000_ch00.tif", send_signal=True)

        # add voxels
        self.stack_crop.input_voxel_size.setValue("1, 1, 1", send_signal=True)

        # set the output
        self.outputText.setValue("tmp/output")

        # click "ok"
        self.accept_new()
        return True

    monkeypatch.setattr(AddFilesDialog, "exec", handle_files)
    batch_evaluate.show_files()

    # add them to the gui and select them
    batch_evaluate.list.setCurrentRow(0)

    results = [batch_evaluate.sub_module_deformation.result]
    assert batch_evaluate.sub_module_deformation.result != None

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


def test_measurement_load(files, monkeypatch, catch_popup_error):
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

    from saenopy.gui.solver.modules.load_measurement_dialog import AddFilesDialog, FileExistsDialog

    # raise for forgot voxel size
    with pytest.raises(QMessageBoxCritical, match="voxel size"):
        def handle_files(self: AddFilesDialog):
            # set the data stack
            self.stack_data.input_filename.setValue("tmp/run-1/Pos004_S001_z000_ch00.tif", send_signal=True)

            # set the reference stack
            self.reference_choice.setValue(1, send_signal=True)
            self.stack_reference.input_filename.setValue("tmp/run-1/Pos004_S001_z000_ch00.tif", send_signal=True)

            # add voxels
            # self.stack_crop.input_voxel_size.setValue("1, 1, 1", send_signal=True)

            # set the output
            self.outputText.setValue("tmp/output")

            # click "ok"
            self.accept_new()

        monkeypatch.setattr(AddFilesDialog, "exec", handle_files)
        batch_evaluate.show_files()

    # raise for no reference stack
    with pytest.raises(QMessageBoxCritical, match="reference stack.*time"):
        def handle_files(self: AddFilesDialog):
            # set the data stack
            self.stack_data.input_filename.setValue("tmp/run-1/Pos004_S001_z000_ch00.tif", send_signal=True)

            # set the reference stack
            #self.reference_choice.setValue(1, send_signal=True)
            #self.stack_reference.input_filename.setValue("tmp/run-1/Pos004_S001_z000_ch00.tif", send_signal=True)

            # add voxels
            self.stack_crop.input_voxel_size.setValue("1, 1, 1", send_signal=True)

            # set the output
            self.outputText.setValue("tmp/output")

            # click "ok"
            self.accept_new()

        monkeypatch.setattr(AddFilesDialog, "exec", handle_files)
        batch_evaluate.show_files()

    # raise for time stack but selected reference stack
    with pytest.raises(QMessageBoxCritical, match="reference state"):
        def handle_files(self: AddFilesDialog):
            # set the data stack
            self.stack_data.input_filename.setValue("tmp/run-2-time/Pos004_S001_z000_t00.tif", send_signal=True)
            self.stack_data.selectors[1].t_prop.setValue("t", send_signal=True)

            # set the reference stack
            self.reference_choice.setValue(1, send_signal=True)
            # self.stack_reference.input_filename.setValue("tmp/run-1/Pos004_S001_z000_ch00.tif", send_signal=True)

            # add voxels
            self.stack_crop.input_voxel_size.setValue("1, 1, 1", send_signal=True)

            # set the output
            self.outputText.setValue("tmp/output")

            # click "ok"
            self.accept_new()

        monkeypatch.setattr(AddFilesDialog, "exec", handle_files)
        batch_evaluate.show_files()

    # raise for no data stack
    with pytest.raises(QMessageBoxCritical, match="deformed state"):
        def handle_files(self: AddFilesDialog):
            # set the data stack
            #self.stack_data.input_filename.setValue("tmp/run-1/Pos004_S001_z000_ch00.tif", send_signal=True)

            # set the reference stack
            self.reference_choice.setValue(1, send_signal=True)
            self.stack_reference.input_filename.setValue("tmp/run-1/Pos004_S001_z000_ch00.tif", send_signal=True)

            # add voxels
            self.stack_crop.input_voxel_size.setValue("1, 1, 1", send_signal=True)

            # set the output
            self.outputText.setValue("tmp/output")

            # click "ok"
            self.accept_new()

        monkeypatch.setattr(AddFilesDialog, "exec", handle_files)
        batch_evaluate.show_files()

    # raise for no time delta
    with pytest.raises(QMessageBoxCritical, match="time"):
        def handle_files(self: AddFilesDialog):
            # set the data stack
            self.stack_data.input_filename.setValue("tmp/run-2-time/Pos004_S001_z000_t00.tif", send_signal=True)
            self.stack_data.selectors[1].t_prop.setValue("t", send_signal=True)

            # set the reference stack
            #self.reference_choice.setValue(0, send_signal=True)
            self.stack_reference.input_filename.setValue("tmp/run-1/Pos004_S001_z000_ch00.tif", send_signal=True)

            # add voxels
            self.stack_crop.input_voxel_size.setValue("1, 1, 1", send_signal=True)

            # set the output
            self.outputText.setValue("tmp/output")

            # click "ok"
            self.accept_new()

        monkeypatch.setattr(AddFilesDialog, "exec", handle_files)
        batch_evaluate.show_files()

    # load successfully
    def handle_files(self: AddFilesDialog):
        # set the data stack
        self.stack_data.input_filename.setValue("tmp/run-1/Pos004_S001_z000_ch00.tif", send_signal=True)

        # set the reference stack
        self.reference_choice.setValue(1, send_signal=True)
        self.stack_reference.input_filename.setValue("tmp/run-1/Pos004_S001_z000_ch00.tif", send_signal=True)

        # add voxels
        self.stack_crop.input_voxel_size.setValue("1, 1, 1", send_signal=True)

        # set the output
        self.outputText.setValue("tmp/output")

        # click "ok"
        self.accept_new()
        return True

    monkeypatch.setattr(AddFilesDialog, "exec", handle_files)
    batch_evaluate.show_files()

    def handle_overwrite_dialog(self: FileExistsDialog):
        self.accept_overwrite()
    monkeypatch.setattr(FileExistsDialog, "exec", handle_overwrite_dialog)
    batch_evaluate.show_files()

    def handle_overwrite_dialog_read(self: FileExistsDialog):
        self.accept_read()
    monkeypatch.setattr(FileExistsDialog, "exec", handle_overwrite_dialog_read)
    batch_evaluate.show_files()

    existing_file = list(Path("tmp/output").glob("*.npz"))[0]
    def handle_load_existing(self: AddFilesDialog):
        # select the existing file tab
        self.tabs.setCurrentIndex(1)

        # set the data stack
        self.outputText3.setValue(existing_file, send_signal=True)

        # click "ok"
        self.accept_existing()
        return True

    monkeypatch.setattr(AddFilesDialog, "exec", handle_load_existing)
    batch_evaluate.show_files()

    def handle_download_example(self: AddFilesDialog):
        # select the existing file tab
        self.tabs.setCurrentIndex(2)

        # set the data stack
        self.example_buttons[2].clicked.emit()

        # click "ok"
        #self.accept_existing()
        return True

    monkeypatch.setattr(AddFilesDialog, "exec", handle_download_example)
    batch_evaluate.show_files()


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
