"""Mode switching retains measurements and shows only the relevant result tabs."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import traceback
import numpy as np
import pytest
import tifffile


@pytest.fixture
def window(tmp_path, monkeypatch):
    from qtpy import QtCore, QtWidgets
    original_settings = QtCore.QSettings
    monkeypatch.setattr(QtCore, "QSettings", lambda *a, **k: original_settings(
        str(tmp_path / "settings.ini"), original_settings.IniFormat))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda typ, value, tb: errors.append("".join(traceback.format_exception(typ,value,tb))))
    from saenopy.gui.orientation.gui_orientation import MainWindow
    widget = MainWindow()
    widget.resize(1600, 900)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()
    assert not errors, errors


@pytest.fixture
def make_result(tmp_path):
    from saenopy.gui.orientation.modules.result import ResultOrientation
    count = 0
    def create(mode):
        nonlocal count
        count += 1
        fiber = tmp_path / "fiber.tif"
        tifffile.imwrite(fiber, np.arange(80*120, dtype=np.uint16).reshape(80,120))
        result = ResultOrientation(output=str(tmp_path / f"{count}.saenopyOrientation"),
            image_fiber=str(fiber), image_cell=None if mode == "axis_orientation_only" else str(fiber),
            pixel_size=.5, analysis_mode=mode,
            orientation_parameters=dict(edge=4, sigma_tensor=7., sigma_tensor_type="µm"),
            angle_to_x_map=np.full((72,112),45.), angle_to_y_map=np.full((72,112),-45.),
            orientation_map=np.zeros((72,112)), coherence_map=np.ones((72,112)))
        if mode != "full":
            bins = np.linspace(-90,90,37)
            result.axis_histogram = np.column_stack([bins[:-1], bins[1:], np.zeros((36,2))])
            result.axis_histogram[27,2] = result.axis_histogram[9,3] = 8064
            result.results_total = [{"Valid pixels":8064, "Alignment x":0., "Alignment y":0.}]
        else:
            from saenopy.gui.orientation.modules.result import DistanceResult
            result.results_total = [{"Orientation":.4,"Mean Angle":27.}]
            result.results_distance = [{key:0. for key in DistanceResult.__annotations__}]
        if mode == "axis_orientation_masked":
            result.validity_mask = np.ones((72,112),dtype=bool)
        result.save()
        return result
    return create


def axis_tabs(window):
    batch = window.deformations
    return [batch.tabs.isTabVisible(batch.tabs.indexOf(t.tab.parent()))
            for t in (batch.tab2,)]


MODES = ["full", "axis_orientation_only", "axis_orientation_masked"]


@pytest.mark.parametrize("mode", ["full", "axis"])
def test_horizontal_choices_only_at_start(window, mode):
    from saenopy.gui.orientation.modules.AddFilesDialog import AddFilesDialog
    buttons = list(window.workflow_choice.buttons.values())
    assert len({button.y() for button in buttons}) == 1
    assert max(b.width() for b in buttons)-min(b.width() for b in buttons) <= 1
    for left,right in zip(buttons,buttons[1:]):
        assert left.x()+left.width() <= right.x()
    window.workflow_choice.buttons[mode].click()
    assert not window.workflow_choice.isVisible()
    assert window.tabs.isVisible()
    assert window.deformations.analysis_mode == ("full" if mode == "full" else "axis_orientation_only")
    dialog = AddFilesDialog(window.deformations,window.deformations.settings)
    assert dialog.selected_analysis_mode() == ("full" if mode == "full" else "axis_orientation_only")
    dialog.close()


def test_import_mode_switches_inputs_and_validation(window, make_result, tmp_path):
    from saenopy.gui.orientation.modules.AddFilesDialog import AddFilesDialog
    result = make_result("full")
    dialog = AddFilesDialog(window.deformations,window.deformations.settings)
    dialog.fiberText.setValue(result.image_fiber)
    dialog.outputText.setValue(str(tmp_path/"output"))
    dialog.cellText.setValue("")
    dialog.show()
    for _ in range(2):
        for mode in MODES:
            dialog.analysis_mode_selector.buttons["full" if mode == "full" else "axis"].click()
            dialog.mask_source.setValue("separate_image" if mode == "axis_orientation_masked" else "whole_image", send_signal=True)
            assert dialog.selected_analysis_mode() == mode
            whole = mode == "axis_orientation_only"
            assert dialog.cellText.isVisible() == (not whole)
            assert dialog.button_addList1.isEnabled() == whole
            assert sum(b.isChecked() for b in dialog.analysis_mode_selector.buttons.values()) == 1
    dialog.cellText.setValue(result.image_fiber)
    assert dialog.button_addList1.isEnabled()
    assert dialog.cellText.label.text() == "Mask Images"
    dialog.analysis_mode_selector.buttons["full"].click()
    assert dialog.cellText.label.text() == "Cell Images"
    assert dialog.button_addList1.isEnabled()
    dialog.close()


def test_import_records_selected_mode_per_measurement(window, make_result, tmp_path, monkeypatch):
    from saenopy.gui.orientation.modules.AddFilesDialog import AddFilesDialog
    source = make_result("full")
    batch = window.deformations
    for mode in MODES:
        def accept(dialog):
            dialog.analysis_mode_selector.buttons["full" if mode == "full" else "axis"].click()
            dialog.mask_source.setValue("separate_image" if mode == "axis_orientation_masked" else "whole_image", send_signal=True)
            dialog.fiberText.setValue(source.image_fiber)
            dialog.cellText.setValue(source.image_cell)
            dialog.outputText.setValue(str(tmp_path/mode))
            dialog.accept_new()
            return 1
        monkeypatch.setattr(AddFilesDialog,"exec",accept)
        batch.add_measurement()
        result = batch.data[-1][2]
        assert result.analysis_mode == mode
        assert (result.image_cell is None) == (mode == "axis_orientation_only")
        assert all(axis_tabs(window))
    assert len(batch.data) == 3
    assert not window.workflow_choice.isVisible()


@pytest.mark.parametrize("source", ["whole_image", "same_image", "separate_image"])
def test_axis_region_source_is_imported_and_restored(window, tmp_path, monkeypatch, source):
    from saenopy.gui.orientation.modules.AddFilesDialog import AddFilesDialog
    from saenopy.gui.orientation.modules.result import ResultOrientation
    fiber = tmp_path / "fiber.tif"
    other = tmp_path / "actinin.tif"
    tifffile.imwrite(fiber, np.zeros((80, 120), dtype=np.uint16))
    tifffile.imwrite(other, np.ones((80, 120), dtype=np.uint16))
    batch = window.deformations

    def accept(dialog):
        dialog.show()
        dialog.analysis_mode_selector.buttons["axis"].click()
        dialog.fiberText.setValue(str(fiber))
        dialog.cellText.setValue(str(tmp_path / "missing-old-selection.tif"))
        dialog.outputText.setValue(str(tmp_path / "output"))
        for _ in range(2):
            for choice, button in dialog.mask_source.buttons.items():
                button.click()
                assert dialog.mask_source.value() == choice
                assert dialog.cellText.isVisible() == (choice == "separate_image")
                assert dialog.button_addList1.isEnabled() == (choice != "separate_image")
                assert sum(b.isChecked() for b in dialog.mask_source.buttons.values()) == 1
        buttons = list(dialog.mask_source.buttons.values())
        assert len({button.y() for button in buttons}) == 1
        dialog.mask_source.buttons[source].click()
        if source == "separate_image":
            dialog.cellText.setValue(str(other))
        assert dialog.button_addList1.isEnabled()
        dialog.accept_new()
        return 1

    monkeypatch.setattr(AddFilesDialog, "exec", accept)
    batch.add_measurement()
    result = batch.data[-1][2]
    assert result.mask_source == source
    expected = {"whole_image": None, "same_image": str(fiber), "separate_image": str(other)}[source]
    assert result.image_cell == expected
    loaded = ResultOrientation.load(result.output)
    assert loaded.mask_source == source
    # No new serialized field is needed: the source remains clear in old and
    # relocated files whose cell/fiber paths point to the same image.
    if source == "same_image":
        loaded.image_cell = "../fiber.tif"
        loaded.image_fiber = str(fiber)
        assert loaded.mask_source == source
    reopened = AddFilesDialog(batch, batch.settings)
    assert reopened.mask_source.value() == source
    reopened.close()


def test_selecting_measurements_updates_tabs_without_changing_results(window, make_result):
    from pathlib import Path
    batch = window.deformations
    results = [make_result(mode) for mode in MODES]
    saved = {r.output:Path(r.output).read_bytes() for r in results}
    for result in results:
        batch.add_data(result)
    for _ in range(3):
        for row, result in enumerate(results):
            batch.list.setCurrentRow(row)
            assert batch.sub_module_deformation.result is result
            assert all(axis_tabs(window))
            assert not window.workflow_choice.isVisible()
            if result.analysis_mode != "full":
                batch.tabs.setCurrentWidget(batch.tab2.tab.parent())
    assert {r.output:Path(r.output).read_bytes() for r in results} == saved
    assert [r.analysis_mode for r in results] == MODES


def test_switch_from_angle_tab_to_uncomputed_measurement(window, make_result):
    batch = window.deformations
    axis = make_result("axis_orientation_only")
    cell = make_result("full")
    cell.angle_to_x_map = cell.angle_to_y_map = cell.orientation_map = cell.coherence_map = None
    batch.add_data(axis)
    batch.tabs.setCurrentWidget(batch.tab2.tab.parent())
    batch.add_data(cell)
    assert all(axis_tabs(window))
    assert not batch.tabs.isTabEnabled(batch.tabs.indexOf(batch.tab2.tab.parent()))


def test_group_means_histograms_and_export(window, make_result, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from saenopy.gui.common.PlottingWindowBase import ExportDialog
    a, b, cell = [make_result(mode) for mode in MODES[1:]+["full"]]
    for result, count, index, alignment in [(a,10,18,1.), (b,10000,0,-1.)]:
        result.results_total = [{"Valid pixels":count,"Alignment x":alignment,"Alignment y":-alignment}]
        result.axis_histogram[:,2:] = 0
        result.axis_histogram[index,2] = count
        result.axis_histogram[(index+18)%36,3] = count
        result.save()
    plotter = window.plotting_window
    plotter.add_files([r.output for r in (a,b,cell)])
    for axis in ("x","y"):
        plotter.type.setValue(f"mean absolute deviation {axis}",send_signal=True)
        table = plotter.getAllCurrentPandasData()
        assert len(table) == 2
        assert table[f"Mean absolute deviation {axis}"].mean() == 45
        plotter.type.setValue(f"angle histogram {axis}",send_signal=True)
        hist = plotter.get_histogram_data()
        assert hist.filename.nunique() == 2
        np.testing.assert_allclose(hist.groupby("filename").fraction.sum(),1)
        average = hist.groupby("angle").fraction.mean().to_numpy()
        np.testing.assert_allclose(average[average>0],[.5,.5])
        assert plotter.canvas.axes.get_ylabel() == "Probability"
        np.testing.assert_allclose(plotter.canvas.axes.lines[0].get_ydata(), average)
        # Exercise the real Export handler, including its external CSV mode.
        destination = tmp_path/f"histogram_{axis}.py"
        def accept(dialog):
            dialog.inputText.setValue(str(destination))
            dialog.include_df.setValue(axis=="x")
            return 1
        monkeypatch.setattr(ExportDialog,"exec",accept)
        plotter.export()
        code = destination.read_text(encoding="utf-8")
        monkeypatch.setattr(plt,"show",lambda:None)
        plt.figure(figsize=(8,4))
        exec(compile(code,str(destination),"exec"),{})
        assert plt.gca().get_ylabel() == "Probability"
        np.testing.assert_allclose(plt.gca().lines[0].get_ydata(), average)
        assert (tmp_path/f"histogram_{axis}_data.csv").exists() == (axis=="y")
    plotter.type.setValue("mean cell-relative deviation",send_signal=True)
    assert list(plotter.getAllCurrentPandasData()["Mean Angle"]) == [27.]
    plotter.type.setValue("angle histogram x",send_signal=True)
    plotter.type.setValue("orientation over distance",send_signal=True)
    assert plotter.current_plot_func == plotter.plot_groups


def test_angle_and_alignment_switch_axes_without_changing_result(window, make_result, monkeypatch):
    import importlib
    result = make_result("axis_orientation_masked")
    result.angle_to_x_map[:] = 20.
    result.angle_to_y_map[:] = -70.
    batch = window.deformations
    seen = []

    def render(parameters, measurement):
        seen.append((parameters["maps"], measurement.get_map_data(parameters["maps"]).copy()))
        return np.zeros((72, 112, 3), dtype=np.uint8)

    for name in ("TabAngle", "TabOrientationField"):
        monkeypatch.setattr(importlib.import_module(f"saenopy.gui.orientation.modules.{name}"), "render_2d", render)
    batch.add_data(result)
    titles = [batch.tabs.tabText(i) for i in range(batch.tabs.count())]
    assert titles.count("Angle") == 1
    assert "Axis histogram" not in titles
    assert not any(title.startswith("Angle (") for title in titles)
    for _ in range(2):
        for axis, expected_angle, expected_alignment in (("x", 20., 1-20/45), ("y", -70., -(1-20/45))):
            batch.tab2.axis.buttons[axis].click()
            assert seen[-1][0] == f"orientation_{axis}_axis"
            np.testing.assert_allclose(seen[-1][1], expected_angle)
            batch.tab4.axis.buttons[axis].click()
            assert seen[-1][0] == f"alignment_{axis}_axis"
            np.testing.assert_allclose(seen[-1][1], expected_alignment)
    np.testing.assert_allclose(result.angle_to_x_map, 20.)
    np.testing.assert_allclose(result.angle_to_y_map, -70.)


def wait_for_preview(module, predicate):
    from qtpy import QtCore
    loop = QtCore.QEventLoop()
    poll = QtCore.QTimer()
    poll.setInterval(20)
    finished = []
    def check():
        if not module._preview_running and not module._preview_timer.isActive() and predicate():
            finished.append(True)
            loop.quit()
    poll.timeout.connect(check)
    timeout = QtCore.QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    poll.start()
    timeout.start(8000)
    loop.exec()
    poll.stop()
    timeout.stop()
    assert finished, f"Preview did not finish: {module.preview_status.text()}"


@pytest.mark.parametrize("source", ["same_image", "separate_image"])
def test_threshold_slider_previews_mask_without_saving(window, make_result, tmp_path, source):
    from pathlib import Path
    from CompactionAnalyzer.CompactionFunctions import create_axis_mask
    result = make_result("axis_orientation_masked")
    if source == "separate_image":
        mask_image = tmp_path / "mask-image.tif"
        tifffile.imwrite(mask_image, result.get_image(0))
        result.image_cell = str(mask_image)
        result.save()
    before = Path(result.output).read_bytes()
    old_mask = result.validity_mask.copy()
    batch = window.deformations
    batch.add_data(result)
    module = batch.sub_module_segmentation
    assert not module.mask_threshold_slider.isHidden()
    assert module.seg_threshold_slider.isHidden()
    for position in (30, 40, 45):
        module.mask_threshold_slider.setValue(position)
    assert module.mask_threshold.value() == .45
    assert result.segmentation_parameters_tmp["mask_threshold"] == .45
    wait_for_preview(module, lambda: getattr(result, "mask_preview", None) is not None)
    expected = create_axis_mask(result.get_image(0)[4:-4,4:-4], threshold=.45)
    np.testing.assert_array_equal(result.mask_preview, expected)
    np.testing.assert_array_equal(result.validity_mask, old_mask)
    assert batch.tab1.mask_overlay.isVisible()
    module.mask_threshold.setValue(.72, send_signal=True)
    assert module.mask_threshold_slider.value() == 72
    wait_for_preview(module, lambda: np.array_equal(result.mask_preview,
        create_axis_mask(result.get_image(0)[4:-4,4:-4], threshold=.72)))
    assert Path(result.output).read_bytes() == before
    assert result.segmentation_parameters is None
    batch.add_data(make_result("axis_orientation_only"))
    assert module.isHidden()
    assert not module._preview_timer.isActive()


def test_classic_threshold_slider_only_changes_live_preview(window, make_result):
    from pathlib import Path
    result = make_result("full")
    y, x = np.mgrid[:80, :120]
    cell = (60000 * np.exp(-((y-40)**2/625 + (x-60)**2/1225))).astype(np.uint16)
    tifffile.imwrite(result.image_cell, cell)
    before = Path(result.output).read_bytes()
    batch = window.deformations
    batch.add_data(result)
    module = batch.sub_module_segmentation
    assert not module.seg_threshold_slider.isHidden()
    assert module.mask_threshold_slider.isHidden()
    module.seg_threshold_slider.setValue(90)
    assert module.seg_thresh.value() == .9
    wait_for_preview(module, lambda: getattr(result, "segmentation_preview", None) is not None)
    assert result.segmentation_preview["mask"].shape == (80,120)
    assert result.segmentation_preview["mask"].sum() > 0
    assert batch.tab1.mask_overlay.isVisible()
    assert result.segmentation is None and result.segmentation_parameters is None
    assert Path(result.output).read_bytes() == before


def test_live_preview_debounces_and_discards_previous_measurement(window, make_result, monkeypatch):
    import threading
    from qtpy import QtTest
    from saenopy.gui.orientation.modules import SegmentationPreview
    result = make_result("axis_orientation_masked")
    other = make_result("axis_orientation_masked")
    batch = window.deformations
    batch.add_data(result)
    module = batch.sub_module_segmentation
    calls = []
    started, release = threading.Event(), threading.Event()
    def slow_preview(snapshot, parameters, orientation):
        calls.append(parameters["mask_threshold"])
        started.set()
        release.wait(3)
        return {"mask": np.zeros((72,112), dtype=bool), "edge": 4}
    monkeypatch.setattr(SegmentationPreview, "calculate_preview", slow_preview)
    try:
        for value in (30,40,50,60):
            module.mask_threshold_slider.setValue(value)
        QtTest.QTest.qWait(100)
        assert calls == []
        for _ in range(100):
            QtTest.QTest.qWait(20)
            if started.is_set():
                break
        assert started.is_set() and calls == [.6]
        batch.add_data(other)
        release.set()
        wait_for_preview(module, lambda: True)
        assert getattr(result, "mask_preview", None) is None
        assert getattr(other, "mask_preview", None) is None
    finally:
        release.set()
        module._preview_pool.waitForDone(5000)


def test_axis_switch_slides_with_mouse_and_keyboard(window, make_result):
    from qtpy import QtCore, QtTest
    batch = window.deformations
    batch.add_data(make_result("axis_orientation_masked"))
    for tab in (batch.tab2, batch.tab4):
        batch.tabs.setCurrentWidget(tab.tab.parent())
        switch = tab.axis
        switch.setValue("x")
        button = switch.buttons["x"]
        start = button.rect().center()
        end = QtCore.QPoint(start.x() + button.width(), start.y())
        QtTest.QTest.mousePress(button, QtCore.Qt.LeftButton, pos=start)
        QtTest.QTest.mouseMove(button, end)
        QtTest.QTest.mouseRelease(button, QtCore.Qt.LeftButton, pos=end)
        for _ in range(50):
            QtTest.QTest.qWait(20)
            if switch.track.animation.state() == QtCore.QAbstractAnimation.Stopped:
                break
        assert switch.value() == "y"
        assert switch.track.position == pytest.approx(1.)
        switch.buttons["x"].setFocus()
        QtTest.QTest.keyClick(switch.buttons["x"], QtCore.Qt.Key_Space)
        for _ in range(50):
            QtTest.QTest.qWait(20)
            if switch.track.animation.state() == QtCore.QAbstractAnimation.Stopped:
                break
        assert switch.value() == "x"
        assert switch.track.position == pytest.approx(0.)


def test_mask_fill_matches_preview_and_clears_for_whole_image(window, make_result):
    result = make_result("axis_orientation_masked")
    result.validity_mask[36:] = False
    before = result.validity_mask.copy()
    batch = window.deformations
    batch.add_data(result)
    tab = batch.tab1
    im = tab.mask_overlay.pixmap().toImage()
    assert (im.width(), im.height()) == (112, 72)
    assert 0 < im.pixelColor(5, 5).alpha() < 128
    assert im.pixelColor(5, 5).red() > im.pixelColor(5, 5).green()
    assert im.pixelColor(5, 50).alpha() == 0
    result.mask_preview = ~result.validity_mask
    result.mask_preview_edge = 4
    tab.update_display()
    im = tab.mask_overlay.pixmap().toImage()
    assert im.pixelColor(5, 5).alpha() == 0
    assert im.pixelColor(5, 50).alpha() > 0
    np.testing.assert_array_equal(result.validity_mask, before)
    batch.add_data(make_result("axis_orientation_only"))
    assert not tab.mask_overlay.isVisible()
    assert tab.mask_overlay.pixmap().isNull()


def test_metrics_follow_checked_measurement_types(window, make_result):
    plotter = window.plotting_window
    axis = make_result("axis_orientation_only")
    cell = make_result("full")

    def enabled(key):
        return plotter.type.combobox.model().item(plotter.type.values.index(key)).isEnabled()

    plotter.add_files([axis.output])
    assert enabled("alignment x") and enabled("angle histogram y")
    assert not enabled("global orientation") and not enabled("orientation over distance")
    assert plotter.type.valueName() == "Alignment to X axis"
    plotter.add_files([cell.output])
    assert enabled("global orientation") and enabled("alignment x")
    plotter.data_folders[0][2][0][1] = False
    plotter.replot()
    assert not enabled("alignment x") and not enabled("angle histogram x")
    assert enabled("global orientation")
    assert plotter.type.value() == "global orientation"
    plotter.data_folders[0][1] = False
    plotter.replot()
    assert not plotter.type.combobox.isEnabled()
    assert not plotter.button_export.isEnabled()
    assert plotter.export_data is None


@pytest.mark.parametrize("mode", MODES)
def test_coherency_group_mean_and_export(window, make_result, tmp_path, monkeypatch, mode):
    """Existing results give each image one vote, independent of mask size/shells."""
    import hashlib
    from pathlib import Path
    import matplotlib.pyplot as plt
    from saenopy.gui.common.PlottingWindowBase import ExportDialog

    results = [make_result(mode), make_result(mode)]
    for index, (result, mean) in enumerate(zip(results, (.2, .8))):
        if mode == "full":
            result.results_total[0]["Mean Coherency"] = mean
            result.results_distance *= 1 + index * 4
        else:
            # Old files have no mean column; it must come from the saved map.
            assert "Mean Coherency" not in result.results_total[0]
            result.coherence_map[:] = mean
            if mode == "axis_orientation_masked":
                result.validity_mask[:] = False
                result.validity_mask[:2 + index * 40] = True
                result.coherence_map[~result.validity_mask] = 1 - mean
        result.save()
    hashes = [hashlib.sha256(Path(r.output).read_bytes()).hexdigest() for r in results]
    plotter = window.plotting_window
    plotter.add_files([r.output for r in results])
    assert "coherency" in plotter.update_metric_availability()
    plotter.type.setValue("coherency", send_signal=True)
    assert plotter.type.valueName() == "Coherency"
    assert not plotter.button_run.isEnabled()  # Image mean, no distance course.
    np.testing.assert_allclose(plotter.canvas.axes.get_ylim(), [0, 1])
    assert plotter.canvas.axes.get_ylabel() == "Coherency"
    assert plotter.canvas.axes.patches[0].get_height() == pytest.approx(.5)
    table = plotter.export_data[1][0]
    np.testing.assert_allclose(table["Mean Coherency"], [.2, .8])
    assert "All measurement modes" in plotter.metric_note.text()

    destination = tmp_path / f"coherency_{mode}.py"
    def accept(dialog):
        dialog.inputText.setValue(str(destination))
        dialog.include_df.setValue(mode != "axis_orientation_masked")
        return 1
    monkeypatch.setattr(ExportDialog, "exec", accept)
    plotter.export()
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append((plt.gca().get_ylim(),
        plt.gca().patches[0].get_height(), plt.gca().get_ylabel())))
    figure = plt.figure()
    exec(compile(destination.read_text(encoding="utf-8"), str(destination), "exec"), {})
    assert shown[0][0] == (0., 1.)
    assert shown[0][1] == pytest.approx(.5)
    assert shown[0][2] == "Coherency"
    plt.close(figure)
    assert [hashlib.sha256(Path(r.output).read_bytes()).hexdigest() for r in results] == hashes


def test_coherency_mixed_modes_and_missing_values(window, make_result):
    """Coherency spans modes; empty selections stay unavailable, zero stays valid."""
    from saenopy.gui.orientation.analyze.PlottingWindow import PlottingWindow
    cell = make_result("full")
    cell.results_total[0]["Mean Coherency"] = .6
    cell.save()
    axis = make_result("axis_orientation_only")
    axis.coherence_map[:] = np.nan
    axis.coherence_map[0, :4] = [0, .2, .4, np.inf]
    axis.angle_to_x_map[:] = np.nan
    axis.save()
    masked = make_result("axis_orientation_masked")
    masked.validity_mask[:] = False
    masked.validity_mask[0, 0] = True
    masked.coherence_map[0, 0] = 0
    masked.save()
    empty = make_result("axis_orientation_masked")
    empty.validity_mask[:] = False
    empty.save()
    missing = make_result("axis_orientation_only")
    missing.coherence_map = None
    missing.save()
    # Keep the established standalone reader usable without constructing a GUI.
    assert PlottingWindow.load_file(None, axis.output).resulting_data["Mean Coherency"][0] == pytest.approx(.2)
    plotter = window.plotting_window
    plotter.add_files([r.output for r in (cell, axis, masked, empty, missing)])
    plotter.type.setValue("coherency", send_signal=True)
    table = plotter.export_data[1][0]
    np.testing.assert_allclose(table["Mean Coherency"], [.6, .2, 0])
    assert plotter.canvas.axes.patches[0].get_height() == pytest.approx(.8 / 3)
    for row in plotter.data_folders[0][2][:3]:
        row[1] = False
    plotter.replot()
    assert "coherency" not in plotter.update_metric_availability()
    assert not plotter.type.combobox.model().item(plotter.type.values.index("coherency")).isEnabled()


@pytest.mark.parametrize("axis", ["x", "y"])
def test_alignment_limits_survive_python_export(window, make_result, tmp_path, monkeypatch, axis):
    import matplotlib.pyplot as plt
    from saenopy.gui.common.PlottingWindowBase import ExportDialog
    result = make_result("axis_orientation_only")
    result.results_total[0].update({"Alignment x": .2, "Alignment y": -.2})
    result.save()
    plotter = window.plotting_window
    plotter.add_files([result.output])
    plotter.type.setValue(f"alignment {axis}", send_signal=True)
    np.testing.assert_allclose(plotter.canvas.axes.get_ylim(), [-1, 1])
    assert plotter.canvas.axes.get_ylabel() == f"Alignment to {axis.upper()} axis"
    destination = tmp_path / f"alignment_{axis}.py"

    def accept(dialog):
        dialog.inputText.setValue(str(destination))
        dialog.include_df.setValue(axis == "x")
        return 1

    monkeypatch.setattr(ExportDialog, "exec", accept)
    plotter.export()
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gca().get_ylim()))
    plt.figure()
    exec(compile(destination.read_text(encoding="utf-8"), str(destination), "exec"), {})
    assert shown == [(-1., 1.)]
    plotter.type.setValue(f"mean absolute deviation {axis}", send_signal=True)
    assert plotter.type.valueName() == f"Angular deviation to {axis.upper()} axis"
    np.testing.assert_allclose(plotter.canvas.axes.get_ylim(), [0, 90])


def test_empty_histogram_does_not_reuse_previous_export(window, make_result):
    result = make_result("axis_orientation_only")
    plotter = window.plotting_window
    plotter.add_files([result.output])
    plotter.type.setValue("angle histogram x",send_signal=True)
    assert plotter.export_data is not None
    plotter.data_folders[0][2][0][1] = False
    plotter.replot()
    assert plotter.get_histogram_data().empty
    assert plotter.export_data is None


def test_mask_preview_preserves_saved_parameters(window, make_result):
    from pathlib import Path
    result = make_result("axis_orientation_masked")
    result.segmentation_parameters = dict(mask_sigma=5.,mask_threshold=.6,mask_invert=False)
    result.save()
    before = Path(result.output).read_bytes()
    batch = window.deformations
    batch.add_data(result)
    detector = batch.sub_module_segmentation
    detector.process_thread(result,{"segmentation_parameters":dict(result.segmentation_parameters,mask_invert=True)})
    assert result.mask_preview is not None
    assert result.segmentation_parameters["mask_invert"] is False
    assert detector.input_button.isEnabled()
    assert Path(result.output).read_bytes() == before
    errors = []
    detector.processing_error.disconnect()
    detector.processing_error.connect(errors.append)
    detector.process_thread(result,{"segmentation_parameters":dict(result.segmentation_parameters,mask_sigma=-1)})
    assert len(errors) == 1
    assert result.segmentation_parameters["mask_sigma"] == 5
    assert detector.input_button.isEnabled()
    assert Path(result.output).read_bytes() == before


def test_detection_can_rerun_after_parameter_changes(window, make_result):
    from saenopy.gui.orientation.modules.result import ResultOrientation
    result = make_result("axis_orientation_only")
    batch = window.deformations
    batch.add_data(result)
    detector = batch.sub_module_deformation
    previous_map = result.orientation_map.copy()
    for edge in (4, 8):
        parameters = dict(result.orientation_parameters, edge=edge, sigma_first_blur=.5,
                          SaveNumpy=False, plotting=False)
        detector.process_thread(result, {"orientation_parameters": parameters})
        assert detector.get_result_state(result) != "failed"
        assert result.orientation_map.shape == (80 - 2*edge, 120 - 2*edge)
        assert result.orientation_map.shape != previous_map.shape or not np.allclose(result.orientation_map, previous_map)
        loaded = ResultOrientation.load(result.output)
        assert loaded.orientation_parameters["edge"] == edge
        np.testing.assert_allclose(loaded.orientation_map, result.orientation_map)
        assert detector.input_button.isEnabled()
        assert detector.input_button.text() == "find orientations"
        previous_map = result.orientation_map.copy()
