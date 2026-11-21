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


class MaskStackSelectorTif(StackSelectorTif):
    """Use the existing wildcard selector, with one XY image per time point."""
    def propertiesChanged(self):
        if self.df is None:
            return
        # The ordinary selector requires a Z sort column. Sorting by filename
        # keeps its channel/time/wildcard handling without introducing a Z axis.
        self.z_prop.setValues(["filename"], ["2D mask (all Z)"])
        self.z_prop.setValue("filename")
        self.z_prop.hide()
        super().propertiesChanged()


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
        self.stack_selector = StackSelector(layout, "mask images" if is_mask else "surface images",
                                            use_time=True)
        if is_mask:
            for selector in list(self.stack_selector.selectors):
                if isinstance(selector, StackSelectorTif):
                    replacement = MaskStackSelectorTif(self.stack_selector, use_time=True)
                    self.stack_selector.layout().replaceWidget(selector, replacement)
                    self.stack_selector.selectors[self.stack_selector.selectors.index(selector)] = replacement
                    selector.setParent(None)
                    selector.deleteLater()
        self.pattern = QtWidgets.QLineEdit()
        self.pattern.setPlaceholderText("Image path / pattern with {z} and/or {t}")
        self.pattern.setToolTip("Select one measurement. Use {t} for time and {z} for 3D stacks.")
        layout.addWidget(self.pattern)
        self.stack_selector.glob_string_changed.connect(lambda _, pattern: self.pattern.setText(pattern))
        self.input_name = QtShortCuts.QInputString(layout, "name", "Ignore mask" if is_mask else "Surface images")
        self.input_time = QtShortCuts.QInputChoice(
            layout, "time points", "single", ["single", "series"],
            ["one image for all time points" if is_mask else "one stack for all time points", "time series"])
        self.input_time.setEnabled(len(result.stacks) > 1)
        self.input_same_voxel_size = QtShortCuts.QInputBool(layout, "Use same voxel size", True)
        voxel_row = QtWidgets.QHBoxLayout()
        layout.addLayout(voxel_row)
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
        self.resize(620, 500)

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
