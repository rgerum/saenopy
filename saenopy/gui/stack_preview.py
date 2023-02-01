import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np

from qimage2ndarray import array2qimage
from saenopy.gui import QtShortCuts, QExtendedGraphicsView
from saenopy.gui_solver.QTimeSlider import QTimeSlider
from saenopy.gui_solver.StackDisplay import ModuleScaleBar

def crop(im, minx, maxx, miny, maxy, z, minz, maxz, t, mint, maxt):
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
        self.stack_reference.glob_string_changed.connect(self.update)
        self.reference_choice.valueChanged.connect(self.update)
        self.stack.selectors[1].input_cropx.valueChanged.connect(self.update)
        self.stack.selectors[1].input_cropy.valueChanged.connect(self.update)
        self.stack.selectors[1].input_cropz.valueChanged.connect(self.update)

    def update(self, x, y=None):
        try:
            z_count = np.array(self.stack.selectors[1].stack_obj[0].image_filenames).shape[0]
            t_count = len(self.stack.selectors[1].stack_obj)
        except IndexError:
            z_count = np.array(self.stack_reference.selectors[1].stack_obj[0].image_filenames).shape[0]
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
        if self.reference_choice.value() == 0 or len(self.stack_reference.selectors[1].stack_obj) == 0:
            minx, maxx = self.stack.selectors[1].input_cropx.value()
            miny, maxy = self.stack.selectors[1].input_cropy.value()
            minz, maxz = self.stack.selectors[1].input_cropz.value()
            mint, maxt = self.stack.selectors[1].input_cropt.value()
            if t_count <= 1:
                im = self.stack.selectors[1].stack_obj[self.t_slider.value()][:, :, :, self.z_slider.value(), 0]
                self.pixmaps[1].setPixmap(QtGui.QPixmap(array2qimage(crop(im, minx, maxx, miny, maxy, z, minz, maxz, t, mint, maxt))))
                self.views[1].setExtend(im.shape[0], im.shape[1])

                im = np.zeros([im.shape[0], im.shape[1], 3])
                im[:, :, 0] = 255
                im[:, :, 2] = 255
                self.pixmaps[0].setPixmap(QtGui.QPixmap(array2qimage(crop(im, minx, maxx, miny, maxy, z, minz, maxz, t, mint, maxt))))
                self.views[0].setExtend(im.shape[0], im.shape[1])
            else:

                im = self.stack.selectors[1].stack_obj[self.t_slider.value()][:, :, :, z, 0]
                self.pixmaps[0].setPixmap(QtGui.QPixmap(array2qimage(crop(im, minx, maxx, miny, maxy, z, minz, maxz, t, mint, maxt))))
                self.views[0].setExtend(im.shape[0], im.shape[1])

                im = self.stack.selectors[1].stack_obj[self.t_slider.value()+1][:, :, :, z, 0]
                self.pixmaps[1].setPixmap(QtGui.QPixmap(array2qimage(crop(im, minx, maxx, miny, maxy, z, minz, maxz, t+1, mint, maxt))))
                self.views[1].setExtend(im.shape[0], im.shape[1])

            self.scale1.setScale(self.stack.selectors[1].getVoxelSize())
            self.scale2.setScale(self.stack.selectors[1].getVoxelSize())
        else:
            minx, maxx = self.stack_reference.selectors[1].input_cropx.value()
            miny, maxy = self.stack_reference.selectors[1].input_cropy.value()
            minz, maxz = self.stack_reference.selectors[1].input_cropz.value()
            im = self.stack_reference.selectors[1].stack_obj[0][:, :, :, z, 0]
            self.pixmaps[0].setPixmap(QtGui.QPixmap(array2qimage(crop(im, minx, maxx, miny, maxy, z, minz, maxz, 0, 0, 1))))
            self.views[0].setExtend(im.shape[0], im.shape[1])

            minx, maxx = self.stack.selectors[1].input_cropx.value()
            miny, maxy = self.stack.selectors[1].input_cropy.value()
            minz, maxz = self.stack.selectors[1].input_cropz.value()
            mint, maxt = self.stack.selectors[1].input_cropt.value()
            if len(self.stack.selectors[1].stack_obj) == 0:
                im = np.zeros([im.shape[0], im.shape[1], 3])
                im[:, :, 0] = 255
                im[:, :, 2] = 255
            else:
                im = self.stack.selectors[1].stack_obj[t][:, :, :, z, 0]
            self.pixmaps[1].setPixmap(QtGui.QPixmap(array2qimage(crop(im, minx, maxx, miny, maxy, z, minz, maxz, t, mint, maxt))))
            self.views[1].setExtend(im.shape[0], im.shape[1])

            self.scale1.setScale(self.stack_reference.selectors[1].getVoxelSize())
            self.scale2.setScale(self.stack.selectors[1].getVoxelSize())

    def z_slider_value_changed(self):
        return self.update(0, 0)
