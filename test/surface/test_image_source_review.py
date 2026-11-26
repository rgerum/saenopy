"""Regression checks for copying local choices and browsing Leica sources."""
import json
from types import MethodType, SimpleNamespace

import pytest

from test_image_source_gui import app, matrix_result


@pytest.mark.parametrize("paste_all", [False, True])
@pytest.mark.parametrize("has_local_choice", [False, True])
def test_parameter_paste_keeps_destination_local_sources(app, tmp_path, monkeypatch,
                                                       paste_all, has_local_choice):
    from qtpy import QtGui
    from saenopy.gui.common.BatchEvaluateBase import BatchEvaluateBase
    from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate

    targets = [matrix_result(tmp_path), matrix_result(tmp_path)]
    if has_local_choice:
        for result in targets:
            result.mesh_parameters = {"custom_mask": "local:3"}
            result.solve_parameters = {"seg_channel": "local:7"}
    payload = dict(mesh_parameters=dict(custom_mask="local:0", element_size=22,
                                       mesh_size=[100, 100, 100]),
                   solve_parameters=dict(seg_channel="local:1", alpha=1234))
    monkeypatch.setattr(QtGui.QGuiApplication, "clipboard", lambda: SimpleNamespace(
        text=lambda: json.dumps(payload)))
    emitted = []
    window = SimpleNamespace(result_params=BatchEvaluate.result_params,
        result_local_params=BatchEvaluate.result_local_params,
        list=SimpleNamespace(data=[[None, None, target] for target in targets], currentRow=lambda: 0),
        set_current_result=SimpleNamespace(emit=emitted.append))
    window._set_pasted_params = MethodType(BatchEvaluateBase._set_pasted_params, window)
    (BatchEvaluateBase.paste_params2 if paste_all else BatchEvaluateBase.paste_params)(window)
    for target in targets if paste_all else targets[:1]:
        assert target.mesh_parameters_tmp.get("custom_mask") == ("local:3" if has_local_choice else None)
        assert target.solve_parameters_tmp.get("seg_channel") == ("local:7" if has_local_choice else None)
        assert target.mesh_parameters_tmp["element_size"] == 22
        assert target.solve_parameters_tmp["alpha"] == 1234
        assert not target.image_sources
    assert emitted == [targets[0]]
    if paste_all:
        targets[0].mesh_parameters_tmp["mesh_size"][0] = 99
        assert targets[1].mesh_parameters_tmp["mesh_size"] == [100, 100, 100]
    # Ordinary scalar choices still copy, including disabling a local mask.
    payload["mesh_parameters"]["custom_mask"] = None
    payload["solve_parameters"]["seg_channel"] = 1
    window._set_pasted_params(targets[0], payload)
    assert targets[0].mesh_parameters_tmp["custom_mask"] is None
    assert targets[0].solve_parameters_tmp["seg_channel"] == 1


def test_parameter_paste_preserves_pending_local_choice_and_other_pipelines():
    from saenopy.gui.common.BatchEvaluateBase import BatchEvaluateBase
    from saenopy.gui.solver.modules.BatchEvaluate import BatchEvaluate
    target = SimpleNamespace(mesh_parameters={"custom_mask": "local:0"},
                             mesh_parameters_tmp={"custom_mask": "local:2"})
    window = SimpleNamespace(result_params=["mesh_parameters"],
                             result_local_params=BatchEvaluate.result_local_params)
    data = {"mesh_parameters": {"custom_mask": "local:0"}}
    BatchEvaluateBase._set_pasted_params(window, target, data)
    assert target.mesh_parameters_tmp["custom_mask"] == "local:2"
    # Other GUI pipelines have no result-local parameter policy.
    window.result_local_params = BatchEvaluateBase.result_local_params
    BatchEvaluateBase._set_pasted_params(window, target, data)
    assert target.mesh_parameters_tmp == data["mesh_parameters"]


def test_path_editor_browses_leica_file_and_keeps_manual_templates(app, tmp_path, monkeypatch):
    from qtpy import QtWidgets
    from saenopy.gui.common import QtShortCuts
    from saenopy.gui.solver.modules.path_editor import PathChanger, PathEditor
    result = matrix_result(tmp_path)
    leica_template = str(tmp_path / "original{f:2}{c:1}{t:0}.lif")
    result.image_sources = [SimpleNamespace(name="Leica", template=leica_template),
                            SimpleNamespace(name="TIFF", template=str(tmp_path / "cell_z{z}.tif"))]
    dialog = PathEditor(None, result)
    leica, tiff = dialog.input_sources
    assert isinstance(leica, QtShortCuts.QInputFilename) and leica.existing
    assert isinstance(tiff, QtShortCuts.QInputFolder)
    assert leica.line.isEnabled() and tiff.line.isEnabled()
    destination = str(tmp_path / "relocated.lif")
    def choose_file(*args, **kwargs):
        assert args[3] == "Leica (*.lif)"
        return destination, ""
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", choose_file)
    leica._openDialog()
    assert leica.value() == destination
    assert PathChanger(leica_template, leica.value()).change_path(leica_template).endswith(
        "relocated{f:2}{c:1}{t:0}.lif")
    edited = str(tmp_path / "manual{f:2}{c:1}{t:0}.lif")
    leica.line.setText(edited)
    leica.line.editingFinished.emit()
    assert leica.value() == edited
    dialog.close()
