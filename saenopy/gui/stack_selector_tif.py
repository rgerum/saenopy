import os
import sys
import numpy as np
import pandas as pd
import re
from pathlib import Path
import imageio
import tifffile
from qtpy import QtWidgets, QtCore, QtGui
from saenopy.gui import QtShortCuts
from saenopy.gui.stack_selector_leica import StackSelectorLeica


class StackSelectorTif(QtWidgets.QWidget):
    loading_process = QtCore.Signal(int, np.ndarray)
    loading_finished = QtCore.Signal()

    def __init__(self, layout):
        super().__init__()

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")

        main_layout = QtWidgets.QVBoxLayout(self)
        return

        #self.name = name

        self._open_dir = self.settings.value("_open_dir"+self.name)
        if self._open_dir is None:
            self._open_dir = ""

        self.button_load = QtWidgets.QPushButton("load")
        main_layout.addWidget(self.button_load)
        def doLoad():
            self.window = StackSelectorWindow(self)
            self.window.show()
        self.button_load.clicked.connect(doLoad)

        self.input_filename = QtShortCuts.QInputFilename(main_layout, name, self._open_dir, existing=True,
                                                         file_type="images (*.tif *.jpg *.png)")
        self.input_filename.line.setEnabled(True)
        self.input_filename.button_load = QtWidgets.QPushButton("load")
        self.input_filename.layout().addWidget(self.input_filename.button_load)
        self.input_filename.button_load.clicked.connect(self.load)
        self.input_label = QtWidgets.QLabel("")
        main_layout.addWidget(self.input_label)

        self.progressbar = QtWidgets.QProgressBar()
        self.progressbar.setOrientation(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.progressbar)

        self.view = QExtendedGraphicsView.QExtendedGraphicsView(self)
        main_layout.addWidget(self.view)
        self.pixmap = QtWidgets.QGraphicsPixmapItem(self.view.origin)

        layout.addWidget(self)
        self.loading_process.connect(lambda i, im: (self.progressbar.setValue(i), self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im))), self.view.setExtend(im.shape[1], im.shape[0])))
        self.loading_finished.connect(self.loadingFinished)

    def checkAcceptFilename(self, filename):
        return filename.endswith(".tif")

    def load(self):
        self.settings.setValue("_open_dir"+self.name, self.input_filename.value())
        self.settings.sync()

        self.files = glob.glob(self.input_filename.value())

        self.progressbar.setRange(0, len(self.files))
        self.thread = threading.Thread(target=self.loading)
        self.thread.start()

        self.input_filename.button_load.setEnabled(False)

    def loading(self):
        self.input_label.setText(f"{len(self.files)} images")
        images = []
        for i, file in enumerate(self.files):
            im = imageio.mimread(file)
            if len(im[0].shape) == 3:
                im[0] = im[0][:, :, 0]
            images.extend(im)
            self.loading_process.emit(i+1, images[-1])
        self.images = np.array(images, images[0].dtype)
        self.loading_finished.emit()

    def loadingFinished(self):
        self.input_label.setText(f"{self.images.shape} images")
        self.input_filename.button_load.setEnabled(True)

    def getStack(self):
        return self.images



class StackSelectorTif(QtWidgets.QWidget):
    no_update = False
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setVisible(False)

        self.property_selectors = []
        self.property_layout = QtWidgets.QVBoxLayout(self)
        self.property_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(self.property_layout)

        self.z_prop = QtShortCuts.QInputChoice(main_layout, "property to use for z")
        self.z_prop.valueChanged.connect(self.showImage)

        layout_voxel = QtWidgets.QHBoxLayout()
        main_layout.addLayout(layout_voxel)
        self.input_voxel_size = QtShortCuts.QInputString(layout_voxel, "Voxel size (xyz) (µm)", "1, 1, 1")

        self.label = QtWidgets.QLabel()
        layout_voxel.addWidget(self.label)

    def checkAcceptFilename(self, filename):
        return filename.endswith(".tif")

    def setImage(self, filename):
        # get the list of similar filenames
        filename = Path(filename)
        filenames = list(filename.parent.glob(re.sub("\d+", "*", filename.name)))

        selected_prop = {key: value for key, value in re.findall(r"_([^_\d]*)(\d+)", filename.name)}

        properties = []
        for file in filenames:
            prop = {key:value for key, value in re.findall(r"_([^_\d]*)(\d+)", file.name)}
            prop["filename"] = file
            properties.append(prop)
        df = pd.DataFrame(properties)
        for col in df.columns:
            if len(df[col].unique()) == 1:
                df.drop(col, inplace=True, axis=1)

        for prop in self.property_selectors:
            prop.setParent(None)
        self.property_selectors = []

        properties = []
        z_name = None
        last_name = None
        for col in df.columns:
            if col == "filename":
                continue
            last_name = col
            print(col, selected_prop[col], df[col].unique())
            prop = QtShortCuts.QInputChoice(self.property_layout, col, str(selected_prop[col]), [str(i) for i in df[col].unique()])
            prop.valueChanged.connect(self.propertiesChanged)
            prop.name = col
            self.property_selectors.append(prop)
            properties.append(col)
            if col == "z":
                z_name = col
        if z_name is None:
            z_name = last_name

        self.z_prop.setValues(properties)
        self.z_prop.setValue(z_name)

        self.setVisible(True)
        self.df = df
        self.propertiesChanged()

    def propertiesChanged(self):
        z_prop_name = self.z_prop.value()
        d = self.df
        for prop in self.property_selectors:
            if prop.name == z_prop_name:
                prop.setEnabled(False)
                continue
            else:
                prop.setEnabled(True)
            print(prop.name, prop.value())
            d = d[d[prop.name] == prop.value()]

        self.parent.setZCount(len(d))
        self.d = d
        im = tifffile.imread(d.iloc[0].filename)
        if len(im.shape) == 3:
            im = im[:, :, 0]
        self.stack = np.zeros((im.shape[0], im.shape[1], len(d)), dtype=im.dtype)
        self.stack_initialized = np.zeros(len(d), dtype=np.bool)
        self.stack_initialized[0] = True
        self.stack[:, :, 0] = im

        self.showImage()

    def check_initialized(self, index):
        if isinstance(index, slice):
            index = range(index.start, index.stop)
        else:
            index = [index]

        for i in index:
            if self.stack_initialized[i] == False:
                print(i)
                im = tifffile.imread(self.d.iloc[i].filename)
                if len(im.shape) == 3:
                    im = im[:, :, 0]
                self.stack[:, :, i] = im
                self.stack_initialized[i] = True

    def showImage(self):
        print("showing Image", self.no_update)
        if self.no_update is True:
            return

        #d = d[d[z_prop_name] == self.parent.getZ()]
        if self.parent.z_slider.active_range != 0:
            self.check_initialized(self.parent.getZRange())
            im = np.max(self.stack[:, :, self.parent.getZRange()], axis=2)
            self.parent.setImage(im)
        else:
            self.check_initialized(self.parent.getZ())
            im = self.stack[:, :, self.parent.getZ()]
            self.parent.setImage(im)
        self.label.setText(f"({im.shape[1]}, {im.shape[0]}, {self.parent.z_count})")
        #self.label.setText(f"Voxel {self.im.scale[0]:.3}µm x {self.im.scale[1]:.3}µm x {self.im.scale[2]:.3}µm ({im.shape[1]}, {im.shape[0]}, {self.im.nz})")

    def getVoxelSize(self):
        return [float(x.strip("[,]")) for x in self.input_voxel_size.value().split()]

    def getStack(self):
        return self.stack
