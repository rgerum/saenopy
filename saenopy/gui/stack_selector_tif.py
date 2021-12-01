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
    stack = None

    def __init__(self, parent, use_time=False):
        super().__init__()
        self.parent = parent
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setVisible(False)

        self.property_selectors = []
        self.property_layout = QtWidgets.QVBoxLayout()
        self.property_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(self.property_layout)

        self.z_prop = QtShortCuts.QInputChoice(main_layout, "property to use for z")
        self.z_prop.valueChanged.connect(self.showImage)

        self.use_time = use_time
        if use_time is True:
            self.t_prop = QtShortCuts.QInputChoice(main_layout, "property to use for t")
            self.t_prop.valueChanged.connect(self.showImage)

        layout_voxel = QtWidgets.QHBoxLayout()
        main_layout.addLayout(layout_voxel)

        self.input_voxel_size = QtShortCuts.QInputString(layout_voxel, "Voxel size (xyz) (µm)", "0, 0, 0", validator=self.validator)
        self.input_voxel_size.valueChanged.connect(self.input_voxel_size_changed)

        if self.use_time:
            self.input_time_dt = QtShortCuts.QInputString(layout_voxel, "Time Delta", "0",
                                                             validator=self.validator_time)
            self.input_tbar_unit = QtShortCuts.QInputChoice(self.input_time_dt.layout(), None, "s",
                                                            ["s", "min", "h"])

        self.label = QtWidgets.QLabel()
        layout_voxel.addWidget(self.label)

        self.stack_initialized = None

    def validator_time(self, value=None):
        try:
            if value is None:
                value = self.input_time_dt.value()
            size = float(value)

            if size <= 1e-10:
                return False
            return True
        except ValueError:
            return False

    def validator(self, value=None):
        try:
            size = self.getVoxelSize(value)

            if len(size) != 3 or np.any(np.array(size) == 0):
                return False
            return True
        except ValueError:
            return False

    def checkAcceptFilename(self, filename):
        return filename.endswith(".tif") or filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")

    def setImage(self, filename):
        # get the list of similar filenames
        filename = Path(filename)
        filenames = list(filename.parent.glob(re.sub("\d+", "*", filename.name)))

        self.format_template = str(filename)

        selected_prop = {key: value for _, key, value in re.findall(r"(_|^)([^_\d]*)(\d+)", filename.name)}

        properties = []
        for file in filenames:
            prop = {key:value for _, key, value in re.findall(r"(_|^)([^_\d]*)(\d+)", file.name)}
            prop["filename"] = file
            properties.append(prop)
        df = pd.DataFrame(properties)
        for col in df.columns:
            if len(df[col].unique()) == 1:
                df.drop(col, inplace=True, axis=1)

        for key, value in selected_prop.items():
            if key in df.columns:
                self.format_template = self.format_template.replace(key+value, key+"{"+key+"}")

        for prop in self.property_selectors:
            prop.setParent(None)
        self.property_selectors = []

        properties = []
        z_name = None
        t_name = None
        names = []
        for col in df.columns:
            if col == "filename":
                continue
            names.append(col)

            prop = QtShortCuts.QInputChoice(self.property_layout, col, str(selected_prop[col]), [str(i) for i in df[col].unique()])
            prop.valueChanged.connect(self.propertiesChanged)
            prop.name = col
            prop.check = QtShortCuts.QInputBool(prop.layout(), "all", False)
            prop.check.valueChanged.connect(self.propertiesChanged)
            self.property_selectors.append(prop)
            properties.append(col)
            if col == "z":
                z_name = col
            if col == "t":
                t_name = col
        if z_name is None:
            z_name = names.pop()
        if t_name is None:
            t_name = names.pop()

        self.z_prop.setValues(properties)
        self.z_prop.setValue(z_name)
        if self.use_time:
            self.t_prop.setValues(properties)
            self.t_prop.setValue(t_name)

        self.setVisible(True)
        self.df = df
        self.propertiesChanged()

    def propertiesChanged(self):
        z_prop_name = self.z_prop.value()
        d = self.df
        selected_props_dict = {z_prop_name: "{z}"}
        if self.use_time:
            t_prop_name = self.t_prop.value()
            selected_props_dict[t_prop_name] = "{t}"
        for prop in self.property_selectors:
            if prop.name == z_prop_name or (self.use_time and prop.name == t_prop_name):
                prop.setEnabled(False)
                continue
            else:
                prop.setEnabled(True)
            d = d[d[prop.name] == prop.value()]
            if prop.check.value() is True:
                selected_props_dict[prop.name] = "*"
            else:
                selected_props_dict[prop.name] = str(prop.value())

        self.target_glob = self.format_template.format(**selected_props_dict)
        self.parent.glob_string_changed.emit('getstack', self.target_glob)

        self.parent.setZCount(len(d))
        self.d = d
        if str(d.iloc[0].filename).endswith(".tif"):
            im = tifffile.imread(str(d.iloc[0].filename))
        else:
            im = imageio.imread(str(d.iloc[0].filename))
        if len(im.shape) == 3:
            im = im[:, :, 0]
        self.stack = np.zeros((im.shape[0], im.shape[1], len(d)), dtype=im.dtype)
        self.stack_initialized = np.zeros(len(d), dtype=np.bool)
        self.stack_initialized[0] = True
        self.stack[:, :, 0] = im

        self.showImage()

    def input_voxel_size_changed(self):
        if not self.validator(None):
            return
        voxel_size = self.getVoxelSize()
        im = self.im
        shape = (im.shape[1], im.shape[0], self.parent.z_count)
        shape2 = np.array(shape) * np.array(voxel_size)
        self.label.setText(f"{shape2}µm")

    def check_initialized(self, index):
        if self.stack_initialized is False:
            return False
        if isinstance(index, slice):
            index = range(index.start, index.stop)
        else:
            index = [index]

        for i in index:
            if self.stack_initialized is not None and self.stack_initialized[i] == False:
                if str(self.d.iloc[i].filename).endswith(".tif"):
                    im = tifffile.imread(str(self.d.iloc[i].filename))
                else:
                    im = imageio.imread(str(self.d.iloc[i].filename))
                if len(im.shape) == 3:
                    im = im[:, :, 0]
                self.stack[:, :, i] = im
                self.stack_initialized[i] = True

    def showImage(self):
        if self.no_update is True or self.stack is None:
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
        self.im = im
        self.label.setText(f"({im.shape[1]}, {im.shape[0]}, {self.parent.z_count})px")
        self.input_voxel_size_changed()
        #self.label.setText(f"Voxel {self.im.scale[0]:.3}µm x {self.im.scale[1]:.3}µm x {self.im.scale[2]:.3}µm ({im.shape[1]}, {im.shape[0]}, {self.im.nz})")

    def getVoxelSize(self, value=None):
        if value is None:
            value = self.input_voxel_size.value()
        return [float(x) for x in re.split(r"[\[\](), ]+", value) if x != ""]

    def getTimeDelta(self):
        factor = 1
        if self.input_tbar_unit.value() == "h":
            factor = 60 * 60
        elif self.input_tbar_unit.value() == "min":
            factor = 60
        return factor * self.input_time_dt.value()

    def getStack(self):
        return self.stack

