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
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
import os
from saenopy.gui.gui_master import MainWindow
from saenopy.gui.solver.gui_solver import MainWindowSolver
from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate
from saenopy.gui.solver.modules.load_measurement_dialog import AddFilesDialog, FileExistsDialog
np.random.seed(1234)


class QMessageBoxException(Exception):
    pass


class QMessageBoxCritical(QMessageBoxException):
    pass


@pytest.fixture
def catch_popup_error(monkeypatch):
    def do_raise(_, name, desc):
        raise QMessageBoxCritical(f"{name}: {desc}")
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", do_raise)


app = QtWidgets.QApplication(sys.argv)
font = QtGui.QFont("Arial", 10)
app.setFont(font)

def init_app():
    from saenopy.gui.gui_master import MainWindow
    window: MainWindow = MainWindow()  # gui_master.py:MainWindow
    window.setTab(1)

    # switch to the Solver part
    from saenopy.gui.solver.gui_solver import MainWindowSolver
    solver: MainWindowSolver = window.solver  # modules.py:MainWindow

    # get the Evaluate part
    from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate
    batch_evaluate: BatchEvaluate = solver.deformations  # modules/BatchEvaluate.py:BatchEvaluate

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

@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,), deadline=None)
@given(use_time=st.booleans(), use_reference=st.booleans())
@in_random_dir()
def test_run_example(monkeypatch, random_path, catch_popup_error, use_time, use_reference):
    # either time or reference must be used (or both)
    if not use_time and not use_reference:
        return
    use_channels = use_time

    app, window, solver, batch_evaluate = init_app()

    if use_time:
        if use_reference:
            file_structure = {
                "run-1": [f"Pos{pos:03d}_S001_z{z:03d}_t{t:02d}_ch{ch:02d}.tif" for z in range(50) for pos in range(4,7) for t in range(3) for ch in range(2)],
                "run-1-reference": [f"Pos{pos:03d}_S001_z{z:03d}_ch{ch:02d}.tif" for z in range(50) for pos in range(4,7)  for ch in range(2)],
            }
        else:
            file_structure = {
                "run-1": [f"Pos{pos:03d}_S001_z{z:03d}_t{t:02d}_ch{ch:02d}.tif" for z in range(50) for pos in range(4,7)  for t in range(3) for ch in range(2)],
            }
    else:
        file_structure = {
            "run-1": [f"Pos{pos:03d}_S001_z{z:03d}_ch{ch:02d}.tif" for z in range(50) for pos in range(4,7) for ch in range(2)],
            "run-1-reference": [f"Pos{pos:03d}_S001_z{z:03d}_ch{ch:02d}.tif" for z in range(50) for pos in range(4,7) for ch in range(2)],
        }
    mock_dir(file_structure, callback=lambda file: create_tif(file, x=50, y=50))

    def handle_overwrite_dialog(self: FileExistsDialog):
        self.accept_overwrite()
        return True

    monkeypatch.setattr(FileExistsDialog, "exec", handle_overwrite_dialog)

    def handle_files(self: AddFilesDialog):
        nonlocal pos
        # set the data stack
        if use_time:
            self.stack_data.input_filename.setValue(f"run-1/Pos{pos}_S001_z000_t00_ch00.tif", send_signal=True)
            self.stack_data.active.t_prop.setValue("t", send_signal=True)
            self.stack_crop.input_time_dt.setValue("1", send_signal=True)
        else:
            self.stack_data.input_filename.setValue(f"run-1/Pos{pos}_S001_z000_ch00.tif", send_signal=True)

        if use_channels:
            self.stack_data.active.c_prop.setValue("ch", send_signal=True)

        if use_reference:
            # set the reference stack
            self.reference_choice.setValue(1, send_signal=True)
            self.stack_reference.input_filename.setValue(f"run-1-reference/Pos{pos}_S001_z000_ch00.tif", send_signal=True)
            if use_channels:
                self.stack_reference.active.c_prop.setValue("ch", send_signal=True)

        # add voxels
        self.stack_crop.input_voxel_size.setValue("1, 1, 1", send_signal=True)

        # set the output
        self.outputText.setValue(f"output_{pos}")

        # click "ok"
        self.accept_new()
        return True

    monkeypatch.setattr(AddFilesDialog, "exec", handle_files)
    pos = "004"
    batch_evaluate.add_measurement()
    pos = "005"
    batch_evaluate.add_measurement()
    pos = "006"
    batch_evaluate.add_measurement()

    # use a thread instead of a process
    batch_evaluate.sub_module_deformation.use_thread = True

    # uncheck the other items
    batch_evaluate.list.item(1).setCheckState(False)
    batch_evaluate.list.item(2).setCheckState(False)

    # add them to the gui and select them
    batch_evaluate.list.setCurrentRow(0)

    results = [batch_evaluate.sub_module_deformation.result]
    results[0].stacks[0].pack_files()
    if use_time:
        results[0].stacks[1].pack_files()
    if use_reference:
        results[0].stack_reference.pack_files()
    assert batch_evaluate.sub_module_deformation.result != None

    # change some parameters
    batch_evaluate.sub_module_deformation.parameter_mappings[0].setParameter("window_size", 30)
    batch_evaluate.sub_module_regularize.parameter_mappings[1].setParameter("max_iterations", 10)
    print("params A", getattr(batch_evaluate.sub_module_deformation.result,
                              batch_evaluate.sub_module_deformation.params_name + "_tmp"))
    print("params C", getattr(batch_evaluate.sub_module_regularize.result,
                              batch_evaluate.sub_module_regularize.params_name + "_tmp"))

    # schedule to run all
    batch_evaluate.run_all()
    # wait until they are processed
    while batch_evaluate.current_task_id < len(batch_evaluate.tasks):
        app.processEvents()

    # check the result
    M = results[0].solvers[0]
    print(M.mesh.displacements[M.mesh.regularisation_mask])
    print(results[0].solvers[0].mesh.displacements.shape)
    # assert sf4(M.U[M.reg_mask][0]) == sf4([-2.01259036e-38, -1.96865342e-38, -4.92921492e-38])
    # 91.64216076e-38 -3.15079497e-39  3.19069614e-39

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: "tmp")
    batch_evaluate.generate_code()

    # open all the tabs
    for i in range(10):
        batch_evaluate.tabs.setCurrentIndex(i)

    """ functions of stack_display """
    batch_evaluate.sub_module_stacks.button_display_single.setValue(True, send_signal=True)
    batch_evaluate.sub_module_stacks.button_display_single.setValue(False, send_signal=True)
    if use_channels:
        batch_evaluate.sub_module_stacks.channel_select.setValue(1, send_signal=True)

    batch_evaluate.sub_module_stacks.button_z_proj.setValue(1)
    batch_evaluate.sub_module_stacks.button_z_proj.setValue(2)
    batch_evaluate.sub_module_stacks.button_z_proj.setValue(3)

    batch_evaluate.sub_module_stacks.contrast_enhance.setValue(1)
    batch_evaluate.sub_module_stacks.button.clicked.emit()

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: "export.tif")
    batch_evaluate.sub_module_stacks.button2.clicked.emit()

    if use_time:
        batch_evaluate.sub_module_stacks.t_slider.setValue(1)

    batch_evaluate.sub_module_stacks.z_slider.setValue(1)

    batch_evaluate.list.setCurrentRow(0)
    batch_evaluate.list.setCurrentRow(1)
    batch_evaluate.list.setCurrentRow(2)

    # remove from list
    batch_evaluate.list.setCurrentRow(2)
    batch_evaluate.list.act_delete.triggered.emit()

    batch_evaluate.list.setCurrentRow(1)
    batch_evaluate.list.act_delete.triggered.emit()

    batch_evaluate.list.setCurrentRow(0)
    batch_evaluate.list.act_delete.triggered.emit()


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,), deadline=None)
@given(use_time=st.booleans())
def test_path_editor(monkeypatch, random_path, catch_popup_error, use_time):
    from saenopy.gui.solver.modules.path_editor import PathEditor, PathChanger
    """ path changer unit test """
    pc = PathChanger("/tmp/sub/pos1_z{z}_t{t}_c{c:00}.tif", "/tmp/pos2_z{z}_t{t}_c{c:00}.tif")
    assert pc.change_path(Path("/tmp/sub/pos1_z00_t00_c00.tif")) == Path("/tmp/pos2_z00_t00_c00.tif")
    assert pc.change_path("/tmp/sub/pos1_z10_t02_c00.tif") == "/tmp/pos2_z10_t02_c00.tif"
    with pytest.raises(ValueError):
        assert pc.change_path(Path("/tmp/subb/pos1_z00_t00_c00.tif")) == Path("/tmp/pos2_z00_t00_c00.tif")

    app, window, solver, batch_evaluate = init_app()

    file_structure = {
        "tmp_path_editor": {
            "run-1": [f"Pos{pos:03d}_S001_z{z:03d}_t{t:03d}_ch00.tif" for z in range(3) for pos in range(4,7) for t in range(3)],
            "run-1-reference": [f"Pos{pos:03d}_S001_z{z:03d}_ch00.tif" for z in range(3) for pos in range(4,7)],
            "run-2": [f"Pos{pos:03d}_S001_z{z:03d}_t{t:03d}_ch00.tif" for z in range(3) for pos in range(4,7) for t in range(3)],
            "run-2-reference": [f"Pos{pos:03d}_S001_z{z:03d}_ch00.tif" for z in range(3) for pos in range(4, 7)],
        },
        "saenopy": {"rgerum": []}
    }
    mock_dir(file_structure, callback=lambda file: create_tif(file, x=10, y=10))

    def handle_files(self: AddFilesDialog):
        # set the data stack
        self.stack_data.input_filename.setValue("tmp_path_editor/run-1/Pos004_S001_z000_t000_ch00.tif", send_signal=True)

        if use_time:
            self.stack_data.active.t_prop.setValue("t", send_signal=True)
            self.stack_crop.input_time_dt.setValue("1", send_signal=True)
        else:
            # set the reference stack
            self.reference_choice.setValue(1, send_signal=True)
            self.stack_reference.input_filename.setValue("tmp_path_editor/run-1-reference/Pos004_S001_z000_ch00.tif", send_signal=True)

        # add voxels
        self.stack_crop.input_voxel_size.setValue("1, 1, 1", send_signal=True)

        # set the output
        self.outputText.setValue("tmp_path_editor/output")

        # click "ok"
        self.accept_new()
        return True

    def handle_overwrite_dialog(self: FileExistsDialog):
        self.accept_overwrite()
        return True

    monkeypatch.setattr(FileExistsDialog, "exec", handle_overwrite_dialog)
    monkeypatch.setattr(AddFilesDialog, "exec", handle_files)
    batch_evaluate.add_measurement()

    # add them to the gui and select them
    batch_evaluate.list.setCurrentRow(0)

    """ click on cancel """
    def path_editor_exec(self: PathEditor):
        self.reject()
        return False

    monkeypatch.setattr(PathEditor, "exec", path_editor_exec)
    batch_evaluate.path_editor()

    """ select new paths """
    def path_editor_exec(self: PathEditor):
        self.input_folder.setValue("tmp_path_editor/run-2/Pos004_S001_z{z}_t000_ch00.tif")
        if not use_time:
            self.input_folder2.setValue("tmp_path_editor/run-2-reference/Pos004_S001_z{z}_ch00.tif")
        self.input_save.setValue(True)
        self.input_pack.setValue(True)
        self.accept()
        return True

    monkeypatch.setattr(PathEditor, "exec", path_editor_exec)
    batch_evaluate.path_editor()

def test_copy_paste_params(monkeypatch, random_path, catch_popup_error):
    app, window, solver, batch_evaluate = init_app()

    file_structure = {
        "tmp_copy_paste_params": {
            "run-1": [f"Pos{pos:03d}_S001_z{z:03d}_ch00.tif" for z in range(50) for pos in range(4,7)],
            "run-1-reference": [f"Pos{pos:03d}_S001_z{z:03d}_ch00.tif" for z in range(50) for pos in range(4,7)],
        },
        "saenopy": {"rgerum": []}
    }
    mock_dir(file_structure, callback=lambda file: create_tif(file, x=50, y=50))

    def handle_files(self: AddFilesDialog):
        # set the data stack
        self.stack_data.input_filename.setValue("tmp_copy_paste_params/run-1/Pos004_S001_z000_ch00.tif", send_signal=True)
        for prop in self.stack_data.active.property_selectors:
            if prop.name == "Pos":
                prop.check.setValue(True)
                prop.check.valueChanged.emit(True)

        # set the reference stack
        self.reference_choice.setValue(1, send_signal=True)
        self.stack_reference.input_filename.setValue("tmp_copy_paste_params/run-1-reference/Pos004_S001_z000_ch00.tif", send_signal=True)
        for prop in self.stack_reference.active.property_selectors:
            if prop.name == "Pos":
                prop.check.setValue(True)
                prop.check.valueChanged.emit(True)

        # add voxels
        self.stack_crop.input_voxel_size.setValue("1, 1, 1", send_signal=True)

        # set the output
        self.outputText.setValue("tmp_copy_paste_params/output")

        # click "ok"
        self.accept_new()
        return True

    def handle_overwrite_dialog(self: FileExistsDialog):
        self.accept_read()
        return True

    monkeypatch.setattr(AddFilesDialog, "exec", handle_files)
    monkeypatch.setattr(FileExistsDialog, "exec", handle_overwrite_dialog)
    batch_evaluate.add_measurement()

    # add them to the gui and select them
    batch_evaluate.list.setCurrentRow(0)

    # copy the params
    batch_evaluate.copy_params()

    # assert that we can past the params
    assert batch_evaluate.allow_paste()

    # paste the params to the next measurement
    batch_evaluate.list.setCurrentRow(1)
    batch_evaluate.paste_params()

    """ same with loaded measurement """
    # remove from list
    batch_evaluate.list.act_delete.triggered.emit()

    monkeypatch.setattr(AddFilesDialog, "exec", handle_files)
    monkeypatch.setattr(FileExistsDialog, "exec", handle_overwrite_dialog)
    batch_evaluate.add_measurement()

    # add them to the gui and select them
    batch_evaluate.list.setCurrentRow(0)

    # copy the params
    batch_evaluate.copy_params()

    # assert that we can past the params
    assert batch_evaluate.allow_paste()

    # paste the params to the next measurement
    batch_evaluate.list.setCurrentRow(1)
    batch_evaluate.paste_params()

    """ with problems """

    # add them to the gui and select them
    batch_evaluate.list.setCurrentRow(0)
    batch_evaluate.sub_module_deformation.result.piv_parameter_tmp = None

    # copy the params
    batch_evaluate.copy_params()

    cb = QtGui.QGuiApplication.clipboard()
    cb.setText("test", mode=cb.Clipboard)

    # assert that we can past the params
    assert not batch_evaluate.allow_paste()

    # paste the params to the next measurement
    batch_evaluate.list.setCurrentRow(1)
    assert not batch_evaluate.paste_params()

    cb = QtGui.QGuiApplication.clipboard()
    cb.setText('{"a": 5}', mode=cb.Clipboard)

    # assert that we can past the params
    assert not batch_evaluate.allow_paste()

    # paste the params to the next measurement
    batch_evaluate.list.setCurrentRow(1)
    assert not batch_evaluate.paste_params()


def test_loading(monkeypatch, catch_popup_error, random_path):
    app, window, solver, batch_evaluate = init_app()

    file_structure = {
        "tmp": {
            "run-1": [f"Pos{pos:03d}_S001_z{z:03d}_ch00.tif" for z in range(50) for pos in range(4,7)],
            "run-1-reference": [f"Pos004_S001_z{z:03d}_ch00.tif" for z in range(50)],
            "run-2-time": [f"Pos004_S001_z{z:03d}_t{t:02d}.tif" for z in range(50) for t in range(3)],
            "run-2-reference": [f"Pos004_S001_z{z:03d}_t00.tif" for z in range(50)],
        },
        "saenopy": {"rgerum": []}
    }
    mock_dir(file_structure, callback=lambda file: create_tif(file, x=50, y=50))

    """ cancel measurement dialog """
    monkeypatch.setattr(AddFilesDialog, "exec", lambda *args: False)
    batch_evaluate.add_measurement()

    """ raise for forgot voxel size """
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
        batch_evaluate.add_measurement()

    """ raise for no reference stack """
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
        batch_evaluate.add_measurement()

    """ raise for time stack but selected reference stack """
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
        batch_evaluate.add_measurement()

    """ raise for no data stack"""
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
        batch_evaluate.add_measurement()

    """ raise for no time delta """
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
        batch_evaluate.add_measurement()

    """ load successfully """
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
    batch_evaluate.add_measurement()

    """ overwrite dialog cancel """
    monkeypatch.setattr(FileExistsDialog, "exec", lambda self: False)
    batch_evaluate.add_measurement()

    """ overwrite dialog overwrite """
    def handle_overwrite_dialog(self: FileExistsDialog):
        self.accept_overwrite()
        return True
    monkeypatch.setattr(FileExistsDialog, "exec", handle_overwrite_dialog)
    batch_evaluate.add_measurement()

    """ overwrite dialog read """
    def handle_overwrite_dialog_read(self: FileExistsDialog):
        self.accept_read()
        return True
    monkeypatch.setattr(FileExistsDialog, "exec", handle_overwrite_dialog_read)
    batch_evaluate.add_measurement()

    """ open existing files """
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
    batch_evaluate.add_measurement()

    """ download examples """
    def handle_download_example(self: AddFilesDialog):
        # select the existing file tab
        self.tabs.setCurrentIndex(2)

        # set the data stack
        self.example_buttons[2].clicked.emit()

        # click "ok"
        #self.accept_existing()
        return True

    monkeypatch.setattr(AddFilesDialog, "exec", handle_download_example)
    batch_evaluate.add_measurement()


def test_fit(monkeypatch, catch_popup_error, random_path):
    import pandas as pd
    from saenopy.gui.material_fit.gui_fit import MainWindowFit
    window = MainWindowFit()
    data0_6 = np.array(
        [[4.27e-06, -2.26e-03], [1.89e-02, 5.90e-01], [3.93e-02, 1.08e+00], [5.97e-02, 1.57e+00], [8.01e-02, 2.14e+00],
         [1.00e-01, 2.89e+00], [1.21e-01, 3.83e+00], [1.41e-01, 5.09e+00], [1.62e-01, 6.77e+00], [1.82e-01, 8.94e+00],
         [2.02e-01, 1.17e+01], [2.23e-01, 1.49e+01], [2.43e-01, 1.86e+01], [2.63e-01, 2.28e+01], [2.84e-01, 2.71e+01]])
    np.savetxt("6.txt", data0_6.T)
    window.add_file("6.txt")

    # run empty
    window.run()

    # set the one data
    window.list.setCurrentRow(0)
    window.input_type.setValue("shear rheometer", send_signal=True)
    window.input_transpose.setValue(True, send_signal=True)

    data1_2 = np.array(
        [[1.22e-05, -1.61e-01], [1.71e-02, 2.57e+00], [3.81e-02, 4.69e+00], [5.87e-02, 6.34e+00], [7.92e-02, 7.93e+00],
         [9.96e-02, 9.56e+00], [1.20e-01, 1.14e+01], [1.40e-01, 1.35e+01], [1.61e-01, 1.62e+01], [1.81e-01, 1.97e+01],
         [2.02e-01, 2.41e+01], [2.22e-01, 2.95e+01], [2.42e-01, 3.63e+01], [2.63e-01, 4.43e+01], [2.83e-01, 5.36e+01],
         [3.04e-01, 6.37e+01], [3.24e-01, 7.47e+01], [3.44e-01, 8.61e+01], [3.65e-01, 9.75e+01], [3.85e-01, 1.10e+02],
         [4.06e-01, 1.22e+02], [4.26e-01, 1.33e+02]])
    pd.DataFrame(data1_2).to_csv("2.csv")
    window.add_file("2.csv")

    window.list.setCurrentRow(1)
    window.input_type.setValue("shear rheometer", send_signal=True)
    window.input_col1.setValue(1, send_signal=True)
    window.input_col2.setValue(2, send_signal=True)

    stretch = np.array(
        [[9.33e-01, 1.02e+00], [9.40e-01, 1.01e+00], [9.47e-01, 1.02e+00], [9.53e-01, 1.02e+00], [9.60e-01, 1.02e+00],
         [9.67e-01, 1.01e+00], [9.73e-01, 1.01e+00], [9.80e-01, 1.01e+00], [9.87e-01, 1.01e+00], [9.93e-01, 1.00e+00],
         [1.00e+00, 1.00e+00], [1.01e+00, 9.89e-01], [1.01e+00, 9.70e-01], [1.02e+00, 9.41e-01], [1.03e+00, 9.00e-01],
         [1.03e+00, 8.46e-01], [1.04e+00, 7.76e-01], [1.05e+00, 6.89e-01], [1.05e+00, 6.02e-01], [1.06e+00, 5.17e-01],
         [1.07e+00, 4.39e-01], [1.07e+00, 3.74e-01], [1.08e+00, 3.17e-01], [1.09e+00, 2.72e-01], [1.09e+00, 2.30e-01],
         [1.10e+00, 2.02e-01]])
    np.savetxt("stretch.txt", stretch)

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *args: "stretch.txt")
    window.add_measurement()

    window.list.setCurrentRow(2)
    window.input_type.setValue("stretch thinning", send_signal=True)
    window.input_params.setValue("k4, d_01, lambda_s1, d_s1", send_signal=True)
    window.input_params.setValue("invalid", send_signal=True)

    data_extension = np.array(
        [[1.005, 8.11503391], [1.015, 4.73271782], [1.025, 12.39500323], [1.035, 4.78868658], [1.045, 13.4463538],
         [1.055, 25.3804035], [1.065, 25.04892348], [1.075, 27.15489932], [1.085, 39.09152633], [1.095, 51.33645119],
         [1.105, 73.54898657], [1.115, 92.71062764], [1.125, 110.42613606], [1.135, 151.62593391],
         [1.145, 198.30184441], [1.155, 251.72810034]])
    np.savetxt("data_extension.txt", data_extension)
    window.add_file("data_extension.txt")

    window.list.setCurrentRow(3)
    window.input_type.setValue("extensional rheometer", send_signal=True)
    window.input_params.setValue("k4, d_01, lambda_s1, d_s1", send_signal=True)
    window.input_params.setValue("invalid", send_signal=True)

    window.list.setCurrentRow(2)
    window.all_params.param_inputs[1][0].bool2.setValue(True, send_signal=True)
    window.all_params.param_inputs[1][0].bool.setValue(True, send_signal=True)
    window.all_params.param_inputs[1][0].bool.setValue(False, send_signal=True)
    window.all_params.param_inputs[1][1].bool.setValue(True, send_signal=True)
    window.all_params.param_inputs[1][1].bool2.setValue(True, send_signal=True)
    window.all_params.param_inputs[1][1].bool2.setValue(False, send_signal=True)
    window.run()

    with pytest.raises(QMessageBoxCritical, match="k1"):
        window.all_params.param_inputs[0][0].input.setValue("invalid", send_signal=True)
        window.run()
        window.all_params.param_inputs[0][0].input.setValue("k1, d_01, lambda_s1, d_s1", send_signal=True)

    with pytest.raises(QMessageBoxCritical, match="column"):
        window.input_col1.setValue(10, send_signal=True)
        window.input_col2.setValue(10, send_signal=True)
        window.run()
        window.input_col1.setValue(0, send_signal=True)
        window.input_col2.setValue(1, send_signal=True)

    #window.show()
    #app.exec_()


if __name__ == "__main__":
    class monk():
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_run_example(monk(), None, None, use_time=False, use_reference=True)