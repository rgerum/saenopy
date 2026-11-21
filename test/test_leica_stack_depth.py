"""A Leica surface source must retain every Z plane, selected time and channel."""
from types import SimpleNamespace

import numpy as np

import saenopy
from saenopy.stack import Stack


def test_leica_full_stack_and_source_keep_depth_time_and_channel(monkeypatch):
    from saenopy.gui.common import lif_reader
    from saenopy import result_file

    frames = []

    class Image:
        dims = SimpleNamespace(y=3, x=4, z=5)
        channels = 3

        def get_frame(self, z, t, c):
            frames.append((z, t, c))
            return np.full((3, 4), 100 * t + 10 * c + z, np.uint16)

    monkeypatch.setattr(lif_reader, "LifFile", lambda path: SimpleNamespace(get_image=lambda folder: Image()))
    template = "example{f:0}{c:2}{t:3}.lif"
    stack = Stack(template, (1, 1, 2))
    volume = stack[:, :, 0, :, 0]
    assert volume.shape == (3, 4, 5)
    np.testing.assert_array_equal(volume[0, 0], np.arange(320, 325))
    assert frames == [(z, 3, 2) for z in range(5)]
    np.testing.assert_array_equal(stack[:, :, 0, 2:, 0][0, 0], np.arange(322, 325))

    # process_line supplies the chosen channel but omits a fixed time from its
    # template; the additional-source loader restores it from the user path.
    monkeypatch.setattr(result_file, "process_line", lambda *args: [
        {"filename": "example{f:0}{c:2}.lif"}])
    result = saenopy.Result(stack=[stack], stack_reference=stack)
    key = saenopy.add_image_source(result, template)
    source = saenopy.get_image_source(result, key, 0)
    assert source.channels == ["2"]
    assert source.leica_time == 3
    np.testing.assert_array_equal(source[:, :, 0, :, 0], volume)
