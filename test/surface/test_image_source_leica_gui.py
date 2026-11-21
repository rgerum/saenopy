"""Leica source selection follows the dialog's single/series choice."""
from types import SimpleNamespace

import numpy as np
import pytest

from test_image_source_gui import app, matrix_result


@pytest.mark.parametrize("kind,depth", [("surface", 5), ("mask", 1)])
def test_leica_dialog_keeps_selected_channel_and_time_mode(app, tmp_path, monkeypatch, kind, depth):
    from qtpy import QtWidgets
    from saenopy import get_image_source
    from saenopy.gui.common import lif_reader
    from saenopy.gui.solver.modules.image_source_dialog import ImageSourceDialog

    class Image:
        info = {"path": "", "name": "measurement"}
        scale = (0.5, 0.75, 2., 1.)
        dims = SimpleNamespace(y=8, x=12, z=depth, t=2)
        channels = 2
        nt = 2

        def get_frame(self, z, t, c):
            return np.full((8, 12), 100 * t + 10 * c + z + 1, np.uint16)

    class Reader:
        num_images = 1

        def __init__(self, filename):
            pass

        def get_iter_image(self):
            return iter([Image()])

        def get_image(self, folder):
            assert int(folder) == 0
            return Image()

    monkeypatch.setattr(lif_reader, "LifFile", Reader)
    path = tmp_path / "measurement.lif"
    path.touch()
    for mode in ("single", "series"):
        result = matrix_result(tmp_path)
        dialog = ImageSourceDialog(None, result, kind)
        dialog.stack_selector.file_changed(str(path))
        selector = dialog.stack_selector.active
        selector.channel.setValue(1, send_signal=True)
        selector.time.setValue(1, send_signal=True)
        assert "{c:1}{t:1}.lif" in dialog.pattern.text()
        assert not selector.time.isHidden()
        if mode == "series":
            dialog.input_time.setValue("series", send_signal=True)
            assert "{t:" not in dialog.pattern.text()
            assert selector.time.isHidden()
            dialog.input_time.setValue("single", send_signal=True)
            assert "{c:1}{t:1}.lif" in dialog.pattern.text()
            dialog.input_time.setValue("series", send_signal=True)
        dialog.accept()
        assert dialog.result() == QtWidgets.QDialog.Accepted
        source = result.image_sources[0]
        assert source.time_mode == mode
        assert len(source.stacks) == (1 if mode == "single" else 2)
        first, second = [get_image_source(result, dialog.source_key, i) for i in range(2)]
        assert first.channels == second.channels == ["1"]
        assert first.shape == second.shape == (8, 12, depth, 1)
        if mode == "single":
            assert first is second
        np.testing.assert_array_equal(first[:, :, 0, :, 0][0, 0],
                                      np.arange(depth) + (111 if mode == "single" else 11))
        np.testing.assert_array_equal(second[:, :, 0, :, 0][0, 0], np.arange(depth) + 111)
        dialog.close()
