# Setting the Qt bindings for QtPy
import os
import sys
import numpy as np
from qtpy import QtWidgets
from saenopy.gui import QtShortCuts


class StackSelectorLeica(QtWidgets.QWidget):
    no_update = False
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setVisible(False)
        self.input_folder = QtShortCuts.QInputChoice(main_layout, "folder")
        self.input_folder.valueChanged.connect(self.setFolder)
        self.channel = QtShortCuts.QInputChoice(main_layout, "channel")
        self.channel.valueChanged.connect(self.showImage)
        self.time = QtShortCuts.QInputChoice(main_layout, "time")
        self.time.valueChanged.connect(self.showImage)

        self.label = QtWidgets.QLabel()
        main_layout.addWidget(self.label)

    def checkAcceptFilename(self, filename):
        return filename.endswith(".lif")

    def setImage(self, filename):
        from readlif.reader import LifFile
        from saenopy.gui import patch_lifreader
        new = LifFile(filename)
        self.new = new
        print([(im.info["path"], im.info["name"], im.channels, im.dims) for im in new.get_iter_image()])
        self.setVisible(True)
        self.input_folder.setValues(list(np.arange(new.num_images)), [im.info["path"]+ im.info["name"] for im in new.get_iter_image()])

    def setFolder(self, index):
        print("setFolder")
        self.im = self.new.get_image(index)
        try:
            self.no_update = True
            self.channel.setValues(list(np.arange(self.im.channels)))
            self.time.setValues(list(np.arange(self.im.nt)))
            self.parent.setZCount(self.im.nz)
        finally:
            self.no_update = False
        print("showImage")
        self.showImage()

    def showImage(self):
        print("showing Image", self.no_update)
        if self.no_update is True:
            return

        if self.parent.z_slider.active_range != 0:
            im = np.max(self.im[self.time.value(), self.parent.getZRange(), self.channel.value()], axis=0)
            self.parent.setImage(im)
        else:
            im = self.im[self.time.value(), self.parent.getZ(), self.channel.value()]
            self.parent.setImage(im)

        self.label.setText(f"Voxel {self.im.scale[0]:.3}µm x {self.im.scale[1]:.3}µm x {self.im.scale[2]:.3}µm ({im.shape[1]}, {im.shape[0]}, {self.im.nz})")

    def getVoxelSize(self):
        return self.im.scale[0:2]

    def getStack(self):
        return self.im[self.time.value(), :, self.channel.value()].transpose(1, 2, 0)