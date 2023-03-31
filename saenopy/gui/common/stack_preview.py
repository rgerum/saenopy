import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np

from qimage2ndarray import array2qimage
from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView
from saenopy.gui.solver.modules.QTimeSlider import QTimeSlider
from saenopy.gui.solver.modules.StackDisplay import ModuleScaleBar


def crop(im, z, t, cropped):
    if "x" not in cropped:
        cropped["x"] = (None, None)
    minx = cropped["x"][0] or 0
    maxx = cropped["x"][1] or im.shape[1]
    if "y" not in cropped:
        cropped["y"] = (None, None)
    miny = cropped["y"][0] or 0
    maxy = cropped["y"][1] or im.shape[0]
    if "z" not in cropped:
        cropped["z"] = (None, None)
    minz = cropped["z"][0] or 0
    maxz = cropped["z"][1] or z+1
    if "t" not in cropped:
        cropped["t"] = (None, None)
    mint = cropped["t"][0] or 0
    maxt = cropped["t"][1] or t+1
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.tile(im, (1, 1, 4))
        im[:, :, 3] = 255
    if im.shape[2] == 3:
        im = np.tile(im, (1, 1, 4))
        im = np.dstack([im[:, :, 0], im[:, :, 1], im[:, :, 2], np.ones_like(im[:, :, 2])*255])
    if minz <= z < maxz and mint <= t < maxt:
        im[:, :minx, 3] = 128
        im[:, maxx:, 3] = 128
        im[:miny, :, 3] = 128
        im[maxy:, :, 3] = 128
    else:
        im[:, :, 3] = 128
    return im


class StackPreview(QtWidgets.QWidget):
    view_single = False
    view_single_timer = None
    view_single_switch = 0

    def __init__(self, layout, reference_choice, stack_reference, stack):
        super().__init__()
        layout.addWidget(self)
        self.stack_reference = stack_reference
        self.stack = stack
        self.reference_choice = reference_choice

        with QtShortCuts.QHBoxLayout(self) as layout:
            with QtShortCuts.QVBoxLayout() as layout:
                with QtShortCuts.QHBoxLayout() as layout:
                    self.label1 = QtWidgets.QLabel("reference").addToLayout()
                    layout.addStretch()

                    self.button = QtWidgets.QPushButton(qta.icon("fa5s.home"), "").addToLayout()
                    self.button.setToolTip("reset view")
                    self.button.clicked.connect(lambda x: (self.view1.fitInView(), self.view2.fitInView()))

                self.view1 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                self.view1.setMinimumWidth(300)
                self.pixmap1 = QtWidgets.QGraphicsPixmapItem(self.view1.origin)
                self.scale1 = ModuleScaleBar(self, self.view1)

                self.label2 = QtWidgets.QLabel("active").addToLayout()
                self.view2 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                # self.label2.setMinimumWidth(300)
                self.pixmap2 = QtWidgets.QGraphicsPixmapItem(self.view2.origin)
                self.scale2 = ModuleScaleBar(self, self.view2)

                self.views = [self.view1, self.view2]
                self.pixmaps = [self.pixmap1, self.pixmap2]

                self.t_slider = QTimeSlider(connected=self.z_slider_value_changed).addToLayout()

            self.z_slider = QTimeSlider("z", self.z_slider_value_changed, "set z position",
                                        QtCore.Qt.Vertical).addToLayout()

        self.view1.link(self.view2)
        self.current_tab_selected = True

        self.stack.glob_string_changed.connect(self.update)
        self.stack.stack_changed.connect(self.update)
        self.stack_reference.glob_string_changed.connect(self.update)
        self.stack_reference.stack_changed.connect(self.update)
        self.reference_choice.valueChanged.connect(self.update)

    def update(self, x=None, y=None):
        try:
            z_count = self.stack.get_z_count()
            t_count = self.stack.get_t_count()
        except IndexError:
            z_count = self.stack_reference.get_z_count()
            t_count = 0
        if self.reference_choice.value() == 0:
            t_count -= 1
        if self.z_slider.t_slider.maximum() != z_count-1:
            self.z_slider.setRange(0, z_count-1)
            self.z_slider.setValue(z_count // 2)
        if self.t_slider.t_slider.maximum() != t_count-1:
            self.t_slider.setRange(0, t_count-1)

        z = self.z_slider.value()
        t = self.t_slider.value()

        if self.reference_choice.value() == 0 or self.stack_reference.get_t_count() == 0:
            cropped = self.stack.get_crop()
            if t_count <= 1:
                im = self.stack.get_image(t, z, 0)
                if im is None:
                    return
                self.pixmaps[1].setPixmap(QtGui.QPixmap(array2qimage(crop(im, z, t, cropped))))
                self.views[1].setExtend(im.shape[0], im.shape[1])

                im = np.zeros([im.shape[0], im.shape[1], 3])
                im[:, :, 0] = 255
                im[:, :, 2] = 255
                self.pixmaps[0].setPixmap(QtGui.QPixmap(array2qimage(crop(im, z, t, cropped))))
                self.views[0].setExtend(im.shape[0], im.shape[1])
            else:

                im = self.stack.get_image(t, z, 0)
                self.pixmaps[0].setPixmap(QtGui.QPixmap(array2qimage(crop(im, z, t, cropped))))
                self.views[0].setExtend(im.shape[0], im.shape[1])

                im = self.stack.get_image(t+1, z, 0)
                self.pixmaps[1].setPixmap(QtGui.QPixmap(array2qimage(crop(im, z, t+1, cropped))))
                self.views[1].setExtend(im.shape[0], im.shape[1])

        else:
            cropped = self.stack.get_crop()
            im = self.stack_reference.get_image(0, z, 0)
            self.pixmaps[0].setPixmap(QtGui.QPixmap(array2qimage(crop(im, z, t+1, cropped))))
            self.views[0].setExtend(im.shape[0], im.shape[1])

            if self.stack.get_t_count() == 0:
                im = np.zeros([im.shape[0], im.shape[1], 3])
                im[:, :, 0] = 255
                im[:, :, 2] = 255
            else:
                im = self.stack.get_image(t, z, 0)
            self.pixmaps[1].setPixmap(QtGui.QPixmap(array2qimage(crop(im, z, t, cropped))))
            self.views[1].setExtend(im.shape[0], im.shape[1])

        self.scale1.setScale(self.stack.getVoxelSize())
        self.scale2.setScale(self.stack.getVoxelSize())

        if t_count <= 1:
            self.label2.setText(f"active")
        else:
            self.label2.setText(f"active ({t * self.stack.getTimeDelta()}s)")

    def z_slider_value_changed(self):
        return self.update(0, 0)
