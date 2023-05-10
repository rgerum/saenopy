import os
from qtpy import QtWidgets, QtCore, QtGui
from saenopy.gui.common.stack_selector_leica import StackSelectorLeica
from saenopy.gui.common.stack_selector_tif import StackSelectorTif
from saenopy.gui.common import QtShortCuts
from qimage2ndarray import array2qimage
import tifffile
import imageio
from pathlib import Path


class StackSelector(QtWidgets.QWidget):
    active = None
    stack_crop = None
    stack_changed = QtCore.Signal()
    glob_string_changed = QtCore.Signal(str, object)

    def __init__(self, layout, name, partner=None, use_time=False):
        super().__init__()
        self.name = name
        self.partner = partner

        self.settings = QtCore.QSettings("Saenopy", "Saenopy")


        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self._open_dir = self.settings.value("_open_dir" + self.name)
        if self._open_dir is None:
            self._open_dir = ""

        self.input_filename = QtShortCuts.QInputFilename(main_layout, name, self._open_dir, existing=True,
                                                         settings=self.settings, settings_key=self.name,
                                                         file_type="images (*.tif *.jpg *.png *.lif)")

        self.input_filename.valueChanged.connect(self.file_changed)

        self.selectors = []
        for selector in [StackSelectorLeica, StackSelectorTif]:
            selector_instance = selector(self, use_time=use_time)
            self.selectors.append(selector_instance)
            main_layout.addWidget(selector_instance)

        layout_image = QtWidgets.QHBoxLayout()
        main_layout.addLayout(layout_image)
        main_layout.addStretch()

        layout.addWidget(self)

    def export(self):
        new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Images", os.getcwd())
        # if we got one, set it
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)
            new_path = new_path.strip(".gif").strip("_relaxed.tif").strip("_deformed.tif")
            new_path = Path(new_path)
            tifffile.imsave(new_path.parent / (new_path.stem + "_relaxed.tif"), self.im)
            tifffile.imsave(new_path.parent / (new_path.stem + "_deformed.tif"), self.partner.im)
            imageio.mimsave(new_path.parent / (new_path.stem + ".gif"), [self.partner.im, self.im], fps=2)

    def setImage(self, im):
        self.im = im
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.view.setExtend(im.shape[1], im.shape[0])

    def file_changed(self, filename):
        for selector_instance in self.selectors:
            selector_instance.setVisible(False)

        for selector_instance in self.selectors:
            if selector_instance.checkAcceptFilename(filename):
                self.active = selector_instance
                selector_instance.setImage(filename)

    def getStack(self):
        return self.active.getStack()

    def getVoxelSize(self):
        if self.stack_crop is None:
            return None
        return self.stack_crop.getVoxelSize()

    def getTimeDelta(self):
        return self.stack_crop.getTimeDelta()

    def get_crop(self):
        if self.stack_crop is None:
            return {}
        return self.stack_crop.get_crop()

    def getStackParameters(self):
        return []

    def get_image(self, t, z, c=0):
        if self.active is None:
            return None
        return self.active.get_image(t, z, c)

    def get_t_count(self):
        if self.active is None:
            return 0
        return self.active.get_t_count()

    def get_z_count(self):
        if self.active is None:
            return 0
        return self.active.get_z_count()
