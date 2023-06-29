import sys
import numpy as np
import re
from pathlib import Path
from qtpy import QtWidgets
from saenopy.gui.common import QtShortCuts
import appdirs


voxel_size_file = Path(appdirs.user_data_dir("saenopy", "rgerum")) / 'voxel_sizes.txt'
time_delta_file = Path(appdirs.user_data_dir("saenopy", "rgerum")) / 'time_deltas.txt'
def get_last_voxel_sizes():
    if not voxel_size_file.exists():
        return []
    with open(voxel_size_file, "r") as fp:
        sizes = []
        for line in fp:
            line = line.strip()
            if re.match(r"(\d*\.\d+|\d+.?),\s*(\d*\.\d+|\d+.?),\s*(\d*\.\d+|\d+.?)", line):
                if line not in sizes:
                    sizes.append(line)
    return sizes

def add_last_voxel_size(size):
    size = ", ".join(str(s) for s in size)
    sizes = get_last_voxel_sizes()
    new_sizes = [size.strip()]
    for s in sizes:
        if s.strip() != size.strip():
            new_sizes.append(s.strip())
    with open(voxel_size_file, "w") as fp:
        for s in new_sizes[:5]:
            fp.write(s)
            fp.write("\n")

def get_last_time_deltas():
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

def add_last_time_delta(size):
    size = str(size)
    sizes = get_last_voxel_sizes()
    new_sizes = [size.strip()]
    for s in sizes:
        if s.strip() != size.strip():
            new_sizes.append(s.strip())
    with open(time_delta_file, "w") as fp:
        for s in new_sizes[:5]:
            fp.write(s)
            fp.write("\n")


class StackSelectorCrop(QtWidgets.QWidget):
    set_voxel_size = None

    def __init__(self, parent: "StackSelector", reference_choice, parent2: "StackSelector", use_time=False):
        super().__init__()
        self.parent_selector = parent
        self.reference_choice = reference_choice

        with QtShortCuts.QHBoxLayout(self) as main_layout:
            main_layout.setContentsMargins(0, 0, 0, 0)
            with QtShortCuts.QVBoxLayout():
                self.input_voxel_size = QtShortCuts.QInputString(None, "Voxel size (xyz) (µm)", "0, 0, 0", validator=self.validator)
                self.input_voxel_size.valueChanged.connect(self.input_voxel_size_changed)
                self.completer = QtWidgets.QCompleter(get_last_voxel_sizes(), self)
                #self.completer.setCompletionMode(QtWidgets.QCompleter.UnfilteredPopupCompletion)
                self.input_voxel_size.line_edit.setCompleter(self.completer)

                with QtShortCuts.QHBoxLayout():
                    self.label = QtWidgets.QLabel().addToLayout()
                    self.label2 = QtWidgets.QLabel().addToLayout()

                self.input_cropx = QtShortCuts.QRangeSlider(None, "crop x", 0, 1)
                self.input_cropy = QtShortCuts.QRangeSlider(None, "crop y", 0, 1)
                self.input_cropz = QtShortCuts.QRangeSlider(None, "crop z", 0, 1)
                self.input_cropx.valueChanged.connect(self.z_moved)
                self.input_cropy.valueChanged.connect(self.z_moved)
                self.input_cropz.valueChanged.connect(self.z_moved)
                self.input_cropx.setDisabled(True)
                self.input_cropy.setDisabled(True)
                self.input_cropz.setDisabled(True)
                self.input_voxel_size.setDisabled(True)

                QtShortCuts.current_layout.addStretch()

            with QtShortCuts.QVBoxLayout():
                with QtShortCuts.QHBoxLayout():
                    self.input_time_dt = QtShortCuts.QInputString(None, "Time Delta", "0",
                                                                     validator=self.validator_time, type=float)
                    self.input_time_dt.setEnabled(False)
                    self.input_time_dt.emitValueChanged()
                    self.input_tbar_unit = QtShortCuts.QInputChoice(self.input_time_dt.layout(), None, "s",
                                                                    ["s", "min", "h"])
                    self.completer2 = QtWidgets.QCompleter(get_last_time_deltas(), self)
                    self.input_time_dt.line_edit.setCompleter(self.completer2)

                self.input_cropt = QtShortCuts.QRangeSlider(None, "crop t", 0, 200)
                self.input_cropt.setRange(0, 1)
                self.input_cropt.setValue((0, 1))
                self.input_cropt.valueChanged.connect(self.z_moved)
                self.input_t_label = QtWidgets.QLabel().addToLayout()
                QtShortCuts.current_layout.addStretch()

                self.input_cropt.setDisabled(True)
                self.input_tbar_unit.setDisabled(True)
                self.input_time_dt.setDisabled(True)

        self.parent_selector.stack_changed.connect(self.update_ranges)
        self.parent2 = parent2
        parent2.stack_changed.connect(self.update_ranges)

    def update_ranges(self):
        im = self.parent_selector.get_image(0, 0)
        z_max = self.parent_selector.get_z_count()
        t_max = self.parent_selector.get_t_count()
        try:
            voxel_size = self.parent_selector.active.getVoxelSize()
        except AttributeError:
            voxel_size = None
        if im is None:
            im = self.parent2.get_image(0, 0)
            z_max = self.parent2.get_z_count()
            t_max = self.parent2.get_t_count()
            try:
                voxel_size = self.parent2.active.getVoxelSize()
            except AttributeError:
                voxel_size = None
        if im is None:
            return
        y_max = im.shape[0]
        x_max = im.shape[1]

        if x_max != self.input_cropx.range()[1] or\
            y_max != self.input_cropy.range()[1] or\
            z_max != self.input_cropz.range()[1] or\
            t_max != self.input_cropt.range()[1]:

            self.input_cropx.setRange(0, x_max)
            self.input_cropy.setRange(0, y_max)
            self.input_cropz.setRange(0, z_max)
            self.input_cropt.setRange(0, t_max)

            self.input_cropz.setValue((0, z_max))
            self.input_cropx.setValue((0, x_max))
            self.input_cropy.setValue((0, y_max))
            self.input_cropt.setValue((0, t_max))

            self.input_cropx.setDisabled(False)
            self.input_cropy.setDisabled(False)
            self.input_cropz.setDisabled(False)
            if voxel_size is None:
                self.input_voxel_size.setDisabled(False)
            else:
                self.input_voxel_size.setDisabled(True)
                self.input_voxel_size.setValue(", ".join([f"{x:.3f}" for x in voxel_size]))
            self.set_voxel_size = voxel_size

            if t_max == 1:
                self.input_time_dt.setDisabled(True)
                self.input_tbar_unit.setDisabled(True)
                self.input_cropt.setDisabled(True)
                if self.parent2.get_t_count():
                    self.input_t_label.setText(f"1 time step - 1 reference state\n1 differences")
                else:
                    self.input_t_label.setText(f"1 time step - no reference state\ninvalid")
            else:
                self.input_time_dt.setDisabled(False)
                self.input_tbar_unit.setDisabled(False)
                self.input_cropt.setDisabled(False)
                if self.parent2.get_t_count():
                    self.input_t_label.setText(f"{t_max} time steps - 1 reference state\n{t_max} differences")
                else:
                    self.input_t_label.setText(f"{t_max} time steps - no reference state\n{t_max-1} differences")
            self.input_time_dt.emitValueChanged()

            self.label.setText(f"Stack: ({x_max}, {y_max}, {z_max})px")
        self.update_timesteps_text()
        self.reference_choice.valueChanged.connect(self.update_timesteps_text)

    def validator_time(self, value=None):
        if getattr(self, "input_time_dt", None) and not self.input_time_dt.isEnabled():
            return True
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

    def update_timesteps_text(self):
        if self.reference_choice.value():
            t_max_ref = min(self.parent2.get_t_count(), 1)
        else:
            t_max_ref = 0
        t_max = self.input_cropt.value()[1] - self.input_cropt.value()[0]
        num_diff = t_max + t_max_ref - 1
        self.input_t_label.setText(f"{t_max_ref if t_max_ref else 'no'} reference state, {t_max} time step{'s' if t_max != 1 else ''}\n→ {num_diff} difference{'s' if num_diff != 1 else ''} {'(invalid)' if num_diff == 0 else ''}")

    def z_moved(self):
        self.parent_selector.stack_changed.emit()
        self.update_timesteps_text()
        return

    def input_voxel_size_changed(self):
        if not self.validator(None):
            return
        voxel_size = self.getVoxelSize()
        shape = (self.input_cropx.range()[1], self.input_cropy.range()[1], self.input_cropz.range()[1])
        shape2 = np.array(shape) * np.array(voxel_size)
        self.label2.setText(f"{shape2}µm")

    def getVoxelSize(self, value=None):
        if self.set_voxel_size is not None:
            return self.set_voxel_size
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
