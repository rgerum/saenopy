"""The Python calls displayed in the help panel run against the analysis API."""
import numpy as np
import pytest
import tifffile
from test_orientation_workflow import window


@pytest.mark.parametrize('mode', ['full', 'axis_orientation_only', 'axis_orientation_masked',
                                   'point_orientation', 'line_orientation'])
def test_displayed_call_runs_and_help_starts_with_gui_instructions(window, mode, tmp_path, monkeypatch):
    from saenopy.gui.orientation.modules.api_help import analysis_api_example
    yy, xx = np.indices((480, 640))
    tifffile.imwrite(tmp_path/'orientation.tif', (20000+18000*np.cos((xx+yy)*.3)).astype(np.uint16))
    tifffile.imwrite(tmp_path/'cell.tif', (((xx-320)**2+(yy-240)**2 < 45**2)*60000).astype(np.uint16))
    monkeypatch.chdir(tmp_path)
    window.select_analysis_mode(mode)
    window.show_analysis_help(mode)  # Repeated selection must not duplicate the API section.
    text = window.analysis_description.toPlainText()
    assert text.count('Python API') == 1
    assert text.index('Python API') > text.index('find orientations' if mode != 'full' else 'Parameters')
    assert window.analysis_description.verticalScrollBar().value() == 0
    code = analysis_api_example(mode)
    assert code in text
    namespace = {}
    files_before = set(tmp_path.iterdir())
    exec(code, namespace)
    assert isinstance(namespace['value'], float)
    assert np.isfinite(namespace['value'])
    assert -1 <= namespace['value'] <= 1
    assert set(tmp_path.iterdir()) == files_before
