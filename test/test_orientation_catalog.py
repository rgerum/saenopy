"""Catalog mode labels and reference examples reuse the classic download safely."""
from pathlib import Path
import numpy as np
import pytest
import tifffile
from test_orientation_workflow import window


@pytest.mark.parametrize('name,mode', [('PointReference', 'point_orientation'), ('LineReference', 'line_orientation')])
def test_reference_example_reuses_data_and_can_be_clicked_and_reloaded(window, tmp_path, monkeypatch, name, mode):
    from saenopy import examples
    from saenopy.gui.orientation.modules.example_data import get_orientation_example_files
    from saenopy.gui.orientation.modules.result import ResultOrientation
    from saenopy.gui.orientation.modules.analysis import analyze_orientation
    monkeypatch.setattr(examples.appdirs, 'user_data_dir', lambda *a: str(tmp_path))
    monkeypatch.setattr(examples, 'urlretrieve', lambda *a: pytest.fail('Classic data already cached'))
    folder = tmp_path/'Orientation_CellsCollagen/data/Cell_1'
    folder.mkdir(parents=True)
    yy, xx = np.indices((160, 200))
    tifffile.imwrite(folder/'C003.tif', (20000+18000*np.cos(xx*.3+yy*.17)).astype(np.uint16))
    configs = examples.get_examples_orientation()
    config = configs[name]
    assert config['url'] == configs['CellFibers']['url']
    assert config['pixel_size'] == configs['CellFibers']['pixel_size'] == .318
    examples.load_example(name)
    result, = get_orientation_example_files(config)
    assert result.analysis_mode == mode and result.image_cell is None
    assert config['output_path'] != configs['CellFibers']['output_path']
    window.deformations.add_data(result)
    module = window.deformations.sub_module_deformation
    assert not module.check_available(result)
    tab = window.deformations.tab1
    if mode == 'point_orientation':
        tab.set_reference_point(100, 80)
    else:
        tab.set_reference_line((50, 110), (150, 50))
    assert module.check_available(result)
    analyze_orientation(result, result.orientation_parameters_tmp)
    restored = ResultOrientation.load(result.output)
    assert restored.analysis_mode == mode
    assert np.isfinite(restored.orientation_map).any()
    assert restored.results_total[0]['Alignment ' + mode.split('_')[0]] == pytest.approx(np.nanmean(restored.orientation_map))
    assert examples.get_examples_orientation()[name]['orientation_parameters'].get('reference_x') is None
    assert examples.get_examples_orientation()[name]['orientation_parameters'].get('line_x1') is None


def test_example_categories_and_mode_first_titles(window):
    from qtpy import QtWidgets
    from saenopy.gui.orientation.modules.AddFilesDialog import AddFilesDialog
    dialog = AddFilesDialog(window.deformations, window.deformations.settings)
    sections = [w.objectName() for w in dialog.findChildren(QtWidgets.QFrame)
                if w.objectName().startswith('exampleCategory_')]
    assert sections == ['exampleCategory_cell', 'exampleCategory_axis', 'exampleCategory_reference']
    titles = [w.text() for w in dialog.findChildren(QtWidgets.QLabel) if w.objectName() == 'exampleTitle']
    assert titles == ['Cell-relative orientation', 'General orientation — whole image',
                      'General orientation — same-image mask', 'General orientation — second-image mask',
                      'Orientation towards a point', 'Orientation towards a point — spheroid',
                      'Orientation relative to a line', 'Orientation relative to a line — cardiomyocytes']
    cards = {w.property('exampleName'): w for w in dialog.findChildren(QtWidgets.QFrame) if w.objectName() == 'exampleCard'}
    for name in ('PointReference', 'LineReference'):
        buttons = cards[name].findChildren(QtWidgets.QPushButton)
        assert buttons[0].isEnabled() and not buttons[1].isEnabled()
    for name in ('ActininMask', 'SpheroidPoint', 'CardiomyocyteLine'):
        buttons = cards[name].findChildren(QtWidgets.QPushButton)
        assert len(buttons) == 2 and all(button.isEnabled() for button in buttons)
    dialog.close()
