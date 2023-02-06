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
import appdirs
from typing import List


voxel_size_file = Path(appdirs.user_data_dir("saenopy", "rgerum")) / 'voxel_sizes.txt'
time_delta_file = Path(appdirs.user_data_dir("saenopy", "rgerum")) / 'time_deltas.txt'
def get_last_voxel_sizes() -> List:
    # if the file does not exist we have no recent voxel sizes
    if not voxel_size_file.exists():
        return []
    # read the file
    with open(voxel_size_file, "r") as fp:
        sizes = []
        # collect the voxel sizes of each line
        for line in fp:
            line = line.strip()
            if re.match(r"(\d*\.\d+|\d+.?),\s*(\d*\.\d+|\d+.?),\s*(\d*\.\d+|\d+.?)", line):
                if line not in sizes:
                    sizes.append(line)
    return sizes


def add_last_voxel_size(size: List) -> None:
    # convert voxel size to string
    size = ", ".join(str(s) for s in size)

    # get the last voxel sizes
    sizes = get_last_voxel_sizes()

    # create a list of the 5 most recent distinct voxel sizes, starting with this one
    new_sizes = [size.strip()]
    for s in sizes:
        # avoid having the same voxel size twice in the list
        if s.strip() != size.strip():
            new_sizes.append(s.strip())

    # make sure the folder exists
    Path(voxel_size_file).parent.mkdir(parents=True, exist_ok=True)

    # write the file with the last 5 voxel sizes
    with open(voxel_size_file, "w") as fp:
        for s in new_sizes[:5]:
            fp.write(s)
            fp.write("\n")
          
                
def get_last_time_deltas() -> List:
    if not time_delta_file.exists():
        return []
    with open(time_delta_file, "r") as fp:
        sizes = []
        for line in fp:
            line = line.strip()
            if re.match(r"(\d*\.\d+|\d+.?)", line):
                if line not in sizes:
                    sizes.append(line)
    return sizes

def add_last_time_delta(size: float) -> None:
    size = str(size)
    sizes = get_last_voxel_sizes()
    new_sizes = [size.strip()]
    for s in sizes:
        if s.strip() != size.strip():
            new_sizes.append(s.strip())

    # make sure the folder exists
    Path(time_delta_file).parent.mkdir(parents=True, exist_ok=True)

    # write the file with the last 5 time deltas
    with open(time_delta_file, "w") as fp:
        for s in new_sizes[:5]:
            fp.write(s)
            fp.write("\n")


class StackSelectorTif(QtWidgets.QWidget):
    no_update = False
    stack = None
    df = None
    target_glob = ""

    input_time_dt = None

    def __init__(self, parent: "StackSelector", use_time=False):
        super().__init__()
        self.parent_selector = parent

        with QtShortCuts.QVBoxLayout(self) as main_layout:
            main_layout.setContentsMargins(0, 0, 0, 0)
            self.setVisible(False)

            self.property_selectors = []
            with QtShortCuts.QVBoxLayout() as self.property_layout:
                self.property_layout.setContentsMargins(0, 0, 0, 0)

            QtShortCuts.QHLine()

            self.z_prop = QtShortCuts.QInputChoice(None, "property to use for z")
            self.z_prop.valueChanged.connect(self.propertiesChanged)

            self.input_cropx = QtShortCuts.QRangeSlider(None, "crop x", 0, 200)
            self.input_cropy = QtShortCuts.QRangeSlider(None, "crop y", 0, 200)
            self.input_cropz = QtShortCuts.QRangeSlider(None, "crop z", 0, 200)
            self.input_cropx.valueChanged.connect(self.z_moved)
            self.input_cropy.valueChanged.connect(self.z_moved)
            self.input_cropz.valueChanged.connect(self.z_moved)
            self.input_cropz.initialized = False

            self.input_voxel_size = QtShortCuts.QInputString(None, "Voxel size (xyz) (µm)", "0, 0, 0", validator=self.validator)
            self.input_voxel_size.valueChanged.connect(self.input_voxel_size_changed)
            self.completer = QtWidgets.QCompleter(get_last_voxel_sizes(), self)
            #self.completer.setCompletionMode(QtWidgets.QCompleter.UnfilteredPopupCompletion)
            self.input_voxel_size.line_edit.setCompleter(self.completer)

            with QtShortCuts.QHBoxLayout():
                self.label = QtWidgets.QLabel().addToLayout()
                self.label2 = QtWidgets.QLabel().addToLayout()

            QtShortCuts.QHLine()

            self.c_prop = QtShortCuts.QInputChoice(None, "property to use for channel")
            self.c_prop.valueChanged.connect(self.propertiesChanged)

            self.use_time = use_time
            if use_time is True:
                QtShortCuts.QHLine()
                self.t_prop = QtShortCuts.QInputChoice(None, "property to use for t")
                self.t_prop.valueChanged.connect(self.propertiesChanged)

                with QtShortCuts.QHBoxLayout(main_layout):
                    self.input_time_dt = QtShortCuts.QInputString(None, "Time Delta", "0",
                                                                     validator=self.validator_time, type=float)
                    self.input_tbar_unit = QtShortCuts.QInputChoice(self.input_time_dt.layout(), None, "s",
                                                                    ["s", "min", "h"])
                    self.completer2 = QtWidgets.QCompleter(get_last_time_deltas(), self)
                    self.input_time_dt.line_edit.setCompleter(self.completer2)

                self.input_cropt = QtShortCuts.QRangeSlider(None, "crop t", 0, 200)
                self.input_cropt.setRange(0, 1)
                self.input_cropt.setValue((0, 1))
                self.input_cropt.initialized = False
                self.input_cropt.valueChanged.connect(self.z_moved)
            else:
                self.input_cropt = None

        self.stack_initialized = None
        from saenopy.getDeformations import Stack
        self.stack_obj = []

    def validator_time(self, value=None):
        if getattr(self, "input_time_dt", None) and not self.input_time_dt.isEnabled():
            True
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

        self.format_template = str(filename.parent) + "/" + re.sub("\d+", "*", filename.name)

        # find data according to regular expression pattern
        regexpr = re.compile(r"([^_\d]*|[^_\d]+_)(\d+)")

        def filename_to_prop_dict(regexpr, filename):
            properties = {}
            for index, (key, value) in enumerate(regexpr.findall(filename.name)):
                if key == "":
                    key = f"Unnamed_{index}"
                properties[key.strip("_")] = value
            return properties

        selected_prop = filename_to_prop_dict(regexpr, filename)

        properties = []
        for file in filenames:
            prop = filename_to_prop_dict(regexpr, file)
            prop["filename"] = file
            properties.append(prop)
        df = pd.DataFrame(properties)

        for key, value in selected_prop.items():
            if key in df.columns:
                self.format_template = self.format_template.replace("*", "{"+key+"}", 1)

        for col in df.columns:
            if len(df[col].unique()) == 1:
                self.format_template = self.format_template.replace("{" + col + "}", df[col].unique()[0], 1)
                df.drop(col, inplace=True, axis=1)

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
        if z_name is None and len(names):
            z_name = names.pop()
        if t_name is None and len(names):
            t_name = names.pop()

        self.z_prop.setValues(properties)
        self.z_prop.setValue(z_name)
        self.c_prop.setValues(["None"]+properties)
        self.c_prop.setValue("None")
        if self.use_time:
            self.t_prop.setValues(["None"]+properties)
            self.t_prop.setValue("None")

        self.setVisible(True)
        self.df = df
        self.propertiesChanged()

    def get_crop(self):
        def value(widget, index):
            if widget.value()[index] != widget.range()[index]:
                return widget.value()[index]
            return None

        crop = {}
        for name, widget in {
            "x": self.input_cropx,
            "y": self.input_cropy,
            "z": self.input_cropz,
            "t": self.input_cropt,
        }.items():
            if (widget is None) or (name == "t" and widget.isEnabled() is False):
                continue
            a, b = value(widget, 0), value(widget, 1)
            if a != None or b != None:
                crop[name] = (a, b)

        return crop

    def z_moved(self):
        self.parent_selector.stack_changed.emit()
        return
        def value(widget, index):
            if widget.value()[index] != widget.range()[index]:
                return widget.value()[index]
            return ""

        x1, x2 = value(self.input_cropx, 0), value(self.input_cropx, 1)
        y1, y2 = value(self.input_cropy, 0), value(self.input_cropy, 1)
        z1, z2 = value(self.input_cropz, 0), value(self.input_cropz, 1)
        if 0:
            if self.input_cropz.value() != self.input_cropz.range():
                a, b = self.input_cropz.value()
                self.target_glob = re.sub(r"\{z(?::\d+-\d+)?\}", f"{{z:{a}-{b}}}", self.target_glob)
                self.parent_selector.glob_string_changed.emit('getstack', self.target_glob)
            else:
                self.target_glob = re.sub(r"\{z(?::\d+-\d+)?\}", "{z}", self.target_glob)
                self.parent_selector.glob_string_changed.emit('getstack', self.target_glob)

        if x1 == "" and x2 == "" and y1 == "" and y2 == "" and z1 == "" and z2 == "":
            self.target_glob = re.sub(f"\.tif(?:\[\d*:\d*,\s*\d*:\d*,\s*\d*:\d*\])?$", ".tif", self.target_glob)
        else:
            self.target_glob = re.sub(f"\.tif(?:\[\d*:\d*,\s*\d*:\d*,\s*\d*:\d*\])?$", f".tif[{y1}:{y2},{x1}:{x2},{z1}:{z2}]", self.target_glob)
        if self.use_time:
            t1, t2 = value(self.input_cropt, 0), value(self.input_cropt, 1)
            if t1 == "" and t2 == "":
                self.target_glob = re.sub(r"\{t(?:\[\d*:\d*\])?\}", "{t}", self.target_glob)
            else:
                self.target_glob = re.sub(r"\{t(?:\[\d*:\d*\])?\}", f"{{t[{t1}:{t2}]}}", self.target_glob)
        self.parent_selector.glob_string_changed.emit('getstack', self.target_glob)

    def propertiesChanged(self):
        if self.df is None:
            return
        z_prop_name = self.z_prop.value()
        c_prop_name = self.c_prop.value()
        d = self.df
        selected_props_dict = {z_prop_name: "{z}"}
        if c_prop_name != "None":
            selected_props_dict[c_prop_name] = "{c}"
        if self.use_time:
            t_prop_name = self.t_prop.value()
            if t_prop_name != "None":
                selected_props_dict[t_prop_name] = "{t}"
                self.input_tbar_unit.setDisabled(False)
                self.input_time_dt.setDisabled(False)
                self.input_cropt.setDisabled(False)
            else:
                self.input_tbar_unit.setDisabled(True)
                self.input_time_dt.setDisabled(True)
                self.input_cropt.setDisabled(True)
        for prop in self.property_selectors:
            if prop.name == z_prop_name or (self.use_time and prop.name == t_prop_name):
                prop.setEnabled(False)
                continue
            else:
                prop.setEnabled(True)
            d = d[d[prop.name] == prop.value()]
            if c_prop_name == prop.name:
                selected_props_dict[prop.name] = "{c:"+str(prop.value())+"}"
                continue
            if prop.check.value() is True:
                selected_props_dict[prop.name] = "*"
            else:
                selected_props_dict[prop.name] = str(prop.value())

        self.stack_obj = []
        from saenopy.getDeformations import Stack
        if not self.use_time or t_prop_name == "None":
            d = d.sort_values(z_prop_name)
            self.stack_obj = [Stack()]
            self.stack_obj[0].image_filenames = [[str(f)] for f in d.filename]
        else:
            d = d.sort_values([t_prop_name, z_prop_name])
            self.stack_obj = []
            for t, dd in d.groupby(t_prop_name):
                s = Stack()
                s.image_filenames = [[str(f)] for f in dd.filename]
                self.stack_obj.append(s)
        self.input_cropz.setRange(0, len(self.stack_obj[0].image_filenames))
        if self.input_cropz.initialized != z_prop_name:
            im = tifffile.TiffFile(d.iloc[0].filename).pages[0].shape
            self.input_cropx.setRange(0, im[1])
            self.input_cropy.setRange(0, im[0])

            self.input_cropz.setValue((0, len(self.stack_obj[0].image_filenames)))
            self.input_cropx.setValue((0, im[1]))
            self.input_cropy.setValue((0, im[0]))
            self.input_cropz.initialized = z_prop_name
        if self.use_time and self.input_cropt.initialized != t_prop_name and t_prop_name != "None":
            self.input_cropt.setRange(0, len(self.stack_obj))
            self.input_cropt.setValue((0, len(self.stack_obj)))
            self.input_cropt.initialized = t_prop_name


        self.target_glob = self.format_template.format(**selected_props_dict)
        self.z_moved()
        self.parent_selector.glob_string_changed.emit('getstack', self.target_glob)

        #self.parent_selector.setZCount(len(d))
        self.d = d
        if str(d.iloc[0].filename).endswith(".tif"):
            im = tifffile.imread(str(d.iloc[0].filename))
        else:
            im = imageio.imread(str(d.iloc[0].filename))
        if len(im.shape) == 3:
            im = im[:, :, 0]
        self.stack = np.zeros((im.shape[0], im.shape[1], len(d)), dtype=im.dtype)
        self.stack_initialized = np.zeros(len(d), dtype=bool)
        self.stack_initialized[0] = True
        self.stack[:, :, 0] = im

        self.showImage()

    def input_voxel_size_changed(self):
        if not self.validator(None):
            return
        voxel_size = self.getVoxelSize()
        im = self.get_image(0, 0, 0)
        shape = (im.shape[1], im.shape[0], self.get_z_count())
        shape2 = np.array(shape) * np.array(voxel_size)
        self.label2.setText(f"{shape2}µm")

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

        im = self.get_image(0, 0, 0)
        self.label.setText(f"Stack: ({im.shape[1]}, {im.shape[0]}, {self.get_z_count()})px")
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

    def get_image(self, t, z, c=0):
        return self.stack_obj[t][:, :, :, z, c]

    def get_t_count(self):
        t_count = len(self.stack_obj)
        return t_count

    def get_z_count(self):
        z_count = np.array(self.stack_obj[0].image_filenames).shape[0]
        return z_count
