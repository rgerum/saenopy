from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from saenopy.gui.tfm2d.modules import draw


class PixmapTarget:
    pixmap = None

    def setPixmap(self, pixmap):
        self.pixmap = pixmap


class Signal:
    emitted = False

    def emit(self):
        self.emitted = True


def test_cursor_size_is_normalized_to_positive_integer():
    assert draw._normalize_cursor_size(12.0) == 12
    assert type(draw._normalize_cursor_size(12.0)) is int
    assert draw._normalize_cursor_size(12.6) == 13
    assert draw._normalize_cursor_size(0) == 1
    assert draw._normalize_cursor_size(-1) == 1


def test_draw_line_accepts_float_cursor_size():
    palette = np.zeros((256, 4), dtype=np.uint8)
    palette[1] = [255, 0, 0, 255]
    signal = Signal()
    target = SimpleNamespace(
        cursor_size=12.0,
        color=1,
        full_image=Image.new("I", (64, 64)),
        palette=palette,
        pixmap_mask=PixmapTarget(),
        changeOpacity=lambda _value: None,
        signal_mask_drawn=signal,
    )

    with (
        patch.object(draw, "array2qimage", side_effect=lambda image: image),
        patch.object(draw.QtGui, "QPixmap", side_effect=lambda image: image),
    ):
        draw.DrawWindow.DrawLine(target, 10, 30, 10, 30)

    assert target.full_image.getpixel((20, 20)) == 1
    assert target.pixmap_mask.pixmap is not None
    assert signal.emitted
