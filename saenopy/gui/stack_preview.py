import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import numpy as np

from qimage2ndarray import array2qimage
from saenopy.gui import QtShortCuts, QExtendedGraphicsView
from saenopy.gui_solver.QTimeSlider import QTimeSlider
from saenopy.gui_solver.StackDisplay import ModuleScaleBar


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

    def update(self, x, y):
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

        if self.reference_choice.value() == 0 or len(self.stack_reference.selectors[1].stack_obj) == 0:
            if t_count == 0 or len(self.stack_reference.selectors[1].stack_obj) == 0:
                im = self.stack.selectors[1].stack_obj[self.t_slider.value()][:, :, :, self.z_slider.value(), 0]
                self.pixmaps[1].setPixmap(QtGui.QPixmap(array2qimage(im)))
                self.views[1].setExtend(im.shape[0], im.shape[1])

                im = np.zeros([im.shape[0], im.shape[1], 3])
                im[:, :, 0] = 255
                im[:, :, 2] = 255
                self.pixmaps[0].setPixmap(QtGui.QPixmap(array2qimage(im)))
                self.views[0].setExtend(im.shape[0], im.shape[1])
            else:

                im = self.stack.selectors[1].stack_obj[self.t_slider.value()][:, :, :, self.z_slider.value(), 0]
                self.pixmaps[0].setPixmap(QtGui.QPixmap(array2qimage(im)))
                self.views[0].setExtend(im.shape[0], im.shape[1])

                im = self.stack.selectors[1].stack_obj[self.t_slider.value()+1][:, :, :, self.z_slider.value(), 0]
                self.pixmaps[1].setPixmap(QtGui.QPixmap(array2qimage(im)))
                self.views[1].setExtend(im.shape[0], im.shape[1])

            self.scale1.setScale(self.stack.selectors[1].getVoxelSize())
            self.scale2.setScale(self.stack.selectors[1].getVoxelSize())
        else:
            im = self.stack_reference.selectors[1].stack_obj[0][:, :, :, self.z_slider.value(), 0]
            self.pixmaps[0].setPixmap(QtGui.QPixmap(array2qimage(im)))
            self.views[0].setExtend(im.shape[0], im.shape[1])

            if len(self.stack.selectors[1].stack_obj) == 0:
                im = np.zeros([im.shape[0], im.shape[1], 3])
                im[:, :, 0] = 255
                im[:, :, 2] = 255
            else:
                im = self.stack.selectors[1].stack_obj[self.t_slider.value()][:, :, :, self.z_slider.value(), 0]
            self.pixmaps[1].setPixmap(QtGui.QPixmap(array2qimage(im)))
            self.views[1].setExtend(im.shape[0], im.shape[1])

            self.scale1.setScale(self.stack_reference.selectors[1].getVoxelSize())
            self.scale2.setScale(self.stack.selectors[1].getVoxelSize())

    def z_slider_value_changed(self):
        return self.update(0, 0)
