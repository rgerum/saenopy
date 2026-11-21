"""Local image sources for segmentation and target-displacement masks."""
from qtpy import QtCore, QtWidgets

from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.stack_selector import StackSelector
from saenopy.gui.common.stack_selector_tif import StackSelectorTif
from saenopy.image_sources import add_image_source


class ImageSourceChoice(QtShortCuts.QInputChoice):
    """A choice with an import action which never becomes a stored parameter."""
    add_requested = QtCore.Signal()
    ADD = "__add_local_images__"

    def _doSetValue(self, value):
        super()._doSetValue(value)
        if value != self.ADD:
            self._selected = self.value()

    def _valueChangedEvent(self, value):
        if self.no_signal:
            return
        if value == self.ADD:
            self.setValue(getattr(self, "_selected", self.values[0]))
            self.add_requested.emit()
            return
        super()._valueChangedEvent(value)


def source_choices(result, kind):
    sources = getattr(result, "image_sources", None) or []
    return [(f"local:{index}", source.name) for index, source in enumerate(sources)
            if source.kind == kind]


class SourceStackSelectorTif(StackSelectorTif):
    """One channel/position, with only the required Z and time dimensions."""
    def __init__(self, parent, kind):
        self.is_mask = kind == "mask"
        self.series = False
        self.time_property = None
        super().__init__(parent, use_time=True)
        self.c_prop.hide()
        self.t_prop.hide()
        self.z_prop.setLabel("Z slices")
        self.t_prop.setLabel("time index")
        for line in self.findChildren(QtWidgets.QFrame):
            if line.frameShape() == QtWidgets.QFrame.HLine:
                line.hide()
        if self.is_mask:
            self.z_prop.hide()

    def propertiesChanged(self):
        if self.df is None:
            return
        if "tiff pages" in self.df:
            self.df["tiff pages"] = self.df["tiff pages"].astype(str)
        fixed = {"c", "ch", "channel", "pos", "position", "well", "fov"}
        if self.is_mask:
            # Sorting one XY plane by filename reuses the existing parser.
            self.z_prop.setValues(["filename"], ["2D mask (all Z)"])
            self.z_prop.setValue("filename")
        else:
            z_values = [name for name in self.df.columns
                        if name != "filename" and name.lower() not in fixed | {"t", "time"}]
            selected_z = self.z_prop.value()
            self.z_prop.setValues(z_values or ["filename"])
            self.z_prop.setValue(selected_z if selected_z in z_values else (z_values or ["filename"])[-1])
        z_property = self.z_prop.value()
        candidates = [name for name in self.df.columns
                      if name not in ("filename", z_property) and name.lower() not in fixed]
        selected = self.t_prop.value()
        if selected in candidates:
            self.time_property = selected
        if self.time_property not in candidates:
            self.time_property = next((name for name in candidates
                                       if name.lower() in ("t", "time", "frame")), "None")
        self.t_prop.setValues(["None"] + candidates, ["select time index"] + candidates)
        self.t_prop.setValue(self.time_property if self.series else "None")
        self.t_prop.setVisible(self.series and bool(candidates))
        self.c_prop.setValue("None")
        for prop in self.property_selectors:
            prop.check.setValue(False)
            prop.check.hide()
            prop.setVisible(prop.name not in (z_property, self.t_prop.value()))
        super().propertiesChanged()
        self.label2.setText(f"{self.get_t_count()} time point(s), one channel")


class ImageSourceDialog(QtWidgets.QDialog):
    def __init__(self, parent, result, kind):
        super().__init__(parent)
        self.measurement = result
        self.kind = kind
        self.source_key = None
        is_mask = kind == "mask"
        self.setWindowTitle("Add custom mask" if is_mask else "Add local images")
        layout = QtWidgets.QVBoxLayout(self)
        explanation = QtWidgets.QLabel(
            "Nonzero mask pixels exclude target deformations through the entire Z column."
            if is_mask else "Load a 3D cell-image stack for surface segmentation.")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        counts = sorted(({len(result.stacks), len(result.mesh_piv)} if is_mask else {len(result.stacks)}) - {0})
        self.input_time = QtShortCuts.QInputChoice(
            layout, "use images", "single", ["single", "series"],
            ["one image for all time points" if is_mask else "one 3D stack for all time points",
             "one image per time point" if is_mask else "one 3D stack per time point"])
        self.requirements = QtWidgets.QLabel(
            f"This measurement: {len(result.stacks)} matrix time point(s). "
            + ("A mask series needs " if is_mask else "A surface series needs ")
            + " or ".join(map(str, counts)) + " time point(s). Select one channel and one position.")
        self.requirements.setWordWrap(True)
        layout.addWidget(self.requirements)
        self.stack_selector = StackSelector(layout, "mask images" if is_mask else "surface images",
                                            use_time=True)
        for selector in list(self.stack_selector.selectors):
            if isinstance(selector, StackSelectorTif):
                replacement = SourceStackSelectorTif(self.stack_selector, kind)
                self.stack_selector.layout().replaceWidget(selector, replacement)
                self.stack_selector.selectors[self.stack_selector.selectors.index(selector)] = replacement
                selector.setParent(None)
                selector.deleteLater()
        self.pattern = QtWidgets.QLineEdit()
        self.pattern.setPlaceholderText("Image path / pattern with {z} and/or {t}")
        self.pattern.setToolTip("Select one measurement. Use {t} for time and {z} for 3D stacks.")
        layout.addWidget(self.pattern)
        self.stack_selector.glob_string_changed.connect(self._set_pattern)
        self.input_time.valueChanged.connect(self._update_time_mode)
        self.input_name = QtShortCuts.QInputString(layout, "name", "Ignore mask" if is_mask else "Surface images")
        voxel_row = QtWidgets.QHBoxLayout()
        layout.addLayout(voxel_row)
        self.input_same_voxel_size = QtShortCuts.QInputBool(
            voxel_row, "Use matrix voxel size", True,
            tooltip="Use the voxel size of the original matrix images.")
        self.voxel_inputs = [QtShortCuts.QInputNumber(
            voxel_row, axis, value, min=0, float=True, decimals=6, unit="µm")
            for axis, value in zip("XYZ", result.stacks[0].voxel_size)]
        if is_mask:
            self.voxel_inputs[2].hide()
        self.input_same_voxel_size.valueChanged.connect(self._update_voxel_inputs)
        self._update_voxel_inputs()
        note = QtWidgets.QLabel("Images are aligned by their centers. Time series are matched in natural filename order.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(760, 360)

    def _set_pattern(self, _, pattern):
        # The generic TIFF picker emits [{t}] for pages used as time; the
        # existing image loader uses [t]. Leica needs an explicit fixed time.
        pattern = pattern.replace("[{t}]", "[t]")
        active = self.stack_selector.active
        if pattern.endswith(".lif") and active is not None:
            single = self.input_time.value() == "single"
            active.time.setVisible(single)
            if single:
                pattern = pattern.replace(".lif", f"{{t:{active.time.value()}}}.lif")
        self.pattern.setText(pattern)

    def _update_time_mode(self):
        for selector in self.stack_selector.selectors:
            if isinstance(selector, SourceStackSelectorTif):
                selector.series = self.input_time.value() == "series"
        active = self.stack_selector.active
        if isinstance(active, SourceStackSelectorTif):
            active.propertiesChanged()
        elif active is not None:
            active.showImage()

    def _update_voxel_inputs(self):
        for widget in self.voxel_inputs:
            widget.setEnabled(not self.input_same_voxel_size.value())

    def accept(self):
        voxel_size = None if self.input_same_voxel_size.value() else tuple(
            widget.value() for widget in self.voxel_inputs)
        try:
            self.source_key = add_image_source(
                self.measurement, self.pattern.text().strip(), voxel_size=voxel_size,
                name=self.input_name.value().strip(), kind=self.kind, time_mode=self.input_time.value())
        except (ValueError, OSError, IndexError) as exc:
            QtWidgets.QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        super().accept()
