import numpy as np
from qtpy import QtWidgets
from saenopy.gui.common import QtShortCuts


class StackSelectorLeica(QtWidgets.QWidget):
    no_update = False
    def __init__(self, parent, use_time=False):
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

    def validator(self, value=None):
        return True

    def checkAcceptFilename(self, filename):
        return filename.endswith(".lif")

    def setImage(self, filename):
        from .lif_reader import LifFile
        #from saenopy.gui import patch_lifreader
        new = LifFile(filename)
        self.filename = filename
        self.new = new
        print([(im.info["path"], im.info["name"], im.channels, im.dims) for im in new.get_iter_image()])
        self.setVisible(True)
        self.input_folder.setValues(list(np.arange(new.num_images)), [im.info["path"]+ im.info["name"] for im in new.get_iter_image()])
        self.setFolder(0)

    def setFolder(self, index):
        self.im = self.new.get_image(index)
        self.folder_index = index
        try:
            self.no_update = True
            self.channel.setValues(list(np.arange(self.im.channels)))
            self.time.setValues(list(np.arange(self.im.nt)))
        finally:
            self.no_update = False
        self.showImage()

    def showImage(self):
        self.parent.stack_changed.emit()

        self.target_glob = f"{self.filename[:-4]}{{f:{self.folder_index}}}{{c:{self.channel.value()}}}.lif"
        self.parent.glob_string_changed.emit('getstack', self.target_glob)
        if self.no_update is True:
            return

        im = self.get_image(0, 0, 0)
        # if there is no z channel
        if self.im.scale[2] is None:
            self.label.setText(
                f"Voxel {self.im.scale[0]:.3}µm x {self.im.scale[1]:.3}µm x {0}µm ({im.shape[1]}, {im.shape[0]}, {self.im.dims.z})")
        else:
            self.label.setText(f"Voxel {self.im.scale[0]:.3}µm x {self.im.scale[1]:.3}µm x {self.im.scale[2]:.3}µm ({im.shape[1]}, {im.shape[0]}, {self.im.dims.z})")

    def getVoxelSize(self):
        return self.im.scale[0:3]

    def getStack(self):
        return self.im[self.time.value(), :, self.channel.value()].transpose(1, 2, 0)

    def get_image(self, t, z, c=0):
        return np.asarray(self.im.get_frame(z, t, self.channel.value()))

    def get_t_count(self):
        return self.im.dims.t

    def get_z_count(self):
        return self.im.dims.z

    def get_crop(self):
        return {}
