import numpy as np
import copy
from typing import Tuple
import matplotlib.pyplot as plt
import jointforces as jf
from pathlib import Path

from saenopy.gui.orientation.modules.result import ResultOrientation, SegmentationParametersDict
from qtpy import QtCore, QtWidgets, QtGui
from qimage2ndarray import array2qimage
from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView

from saenopy.gui.common.gui_classes import QHLine, CheckAbleGroup
from saenopy.gui.common.PipelineModule import PipelineModule
from saenopy.gui.common.PipelineModule import StateEnum
from saenopy.gui.common.QTimeSlider import QTimeSlider
from saenopy.gui.common.resources import resource_icon
from saenopy.gui.common.code_export import get_code, export_as_string
from saenopy.gui.common.ModuleScaleBar import ModuleScaleBar
from saenopy.gui.common.ModuleColorBar import ModuleColorBar


class SegmentationDetector(PipelineModule):
    pipeline_name = "segmentation"
    use_thread = False
    progress_signal = QtCore.Signal(int, str)
    result: ResultOrientation = None

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "segmentation").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout():
                    with QtShortCuts.QHBoxLayout():
                        self.seg_thresh = QtShortCuts.QInputString(None, "thresh", "1.0",
                                                                   type=float,
                                                                   settings=self.settings,
                                                                   settings_key="orientation/segmentation/thresh")
                        self.seg_invert = QtShortCuts.QInputBool(None, "invert", False, settings=self.settings,
                                                                 settings_key="orientation/segmentation/invert",
                                                                 tooltip="Tick if cell is dark on white background.")
                    # self.segmention_thres.valueChanged.connect(self.listSelected)
                    self.seg_threshold_slider = self.create_threshold_slider("Cell segmentation threshold")
                    with QtShortCuts.QHBoxLayout():
                        self.seg_gauss1 = QtShortCuts.QInputString(None, "gauss1", "0.5", type=float,
                                                                   settings=self.settings,
                                                                   settings_key="orientation/segmentation/gauss1")
                        # self.seg_gaus1.valueChanged.connect(self.listSelected)
                        self.seg_gauss2 = QtShortCuts.QInputString(None, "gauss2", "100", type=float,
                                                                   settings=self.settings,
                                                                   settings_key="orientation/segmentation/gauss2")
                        # self.seg_gaus2.valueChanged.connect(self.listSelected)

                    self.mask_sigma = QtShortCuts.QInputString(None, "mask blur", 5., type=float, unit="px")
                    self.mask_threshold = QtShortCuts.QInputString(None, "mask threshold", 0.5, type=float,
                        tooltip="Otsu threshold on the blurred image, scaled by this factor. Lower values include dimmer cell regions; higher values select brighter regions.")
                    self.mask_threshold_slider = self.create_threshold_slider("Mask threshold")
                    self.mask_invert = QtShortCuts.QInputBool(None, "dark regions", False)
                    self.mask_note = QtWidgets.QLabel("Mask selects valid pixels; it does not define a cell centre.").addToLayout()
                    self.mask_note.setWordWrap(True)
                    self.preview_status = QtWidgets.QLabel("Automatic mask preview").addToLayout()
                    self.preview_status.setWordWrap(True)
                    self.input_button = QtShortCuts.QPushButton(None, "preview segmentation", self.start_process)

        self.setParameterMapping("segmentation_parameters", {
            "thresh": self.seg_thresh,
            "gauss1": self.seg_gauss1,
            "gauss2": self.seg_gauss2,
            "invert": self.seg_invert,
            "mask_sigma": self.mask_sigma,
            "mask_threshold": self.mask_threshold,
            "mask_invert": self.mask_invert,
        })

        self._setting_result = False
        self._preview_revision = 0
        self._preview_running = False
        self._preview_waiting = False
        self._preview_target = None
        self._preview_worker = None
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(350)
        self._preview_timer.timeout.connect(self.start_live_preview)
        self._preview_pool = QtCore.QThreadPool(self)
        self._preview_pool.setMaxThreadCount(1)
        self.parent.result_changed.connect(self.analysis_result_changed)
        for field, slider in ((self.seg_thresh, self.seg_threshold_slider),
                              (self.mask_threshold, self.mask_threshold_slider)):
            slider.valueChanged.connect(lambda value, field=field: field.setValue(value / 100., send_signal=True))
            field.valueChanged.connect(lambda value, slider=slider: self.sync_threshold_slider(slider, value))
            self.sync_threshold_slider(slider, field.value())
        for field in (self.seg_thresh, self.seg_invert, self.seg_gauss1, self.seg_gauss2,
                      self.mask_sigma, self.mask_threshold, self.mask_invert):
            field.valueChanged.connect(self.request_live_preview)

    def create_threshold_slider(self, name):
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal).addToLayout()
        slider.setRange(1, 200)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        slider.setAccessibleName(name)
        slider.setToolTip("Adjust the threshold; the mask preview updates after a short pause.")
        return slider

    def sync_threshold_slider(self, slider, value):
        try:
            value = float(value)
            if not np.isfinite(value):
                return
            position = round(value * 100)
        except (TypeError, ValueError):
            return
        # Keep explicitly entered values; the slider must not clamp the text field.
        with QtCore.QSignalBlocker(slider):
            slider.setMaximum(max(200, slider.maximum(), position))
            slider.setValue(position)

    def preview_is_available(self):
        if not self.check_available(self.result):
            return False
        return not any(getattr(self.result, name + "_state", "") in (StateEnum.running, StateEnum.scheduled)
                       for name in ("segmentation_parameters", "orientation_parameters"))

    def analysis_result_changed(self, result):
        if result is self.result:
            self._preview_revision += 1
            self._preview_timer.stop()
            self._preview_waiting = False

    def hideEvent(self, event):
        if hasattr(self, "_preview_timer"):
            self._preview_revision += 1
            self._preview_timer.stop()
            self._preview_waiting = False
        super().hideEvent(event)

    def request_live_preview(self, *_):
        if self._setting_result or not self.preview_is_available():
            return
        self._preview_revision += 1
        self.parent.tabs.setCurrentWidget(self.parent.tab1.tab.parent())
        self.preview_status.setText("Updating preview…")
        self._preview_timer.start(350)

    def start_live_preview(self):
        if not self.preview_is_available():
            return
        if self._preview_running:
            self._preview_waiting = True
            return
        from .SegmentationPreview import PreviewWorker
        self._preview_running = True
        self._preview_target = self.result
        # A copy keeps image-shape initialization and preview data off the result.
        worker = PreviewWorker(self._preview_revision, copy.copy(self.result),
            dict(self.result.segmentation_parameters_tmp),
            dict(getattr(self.result, "orientation_parameters_tmp", None) or self.result.orientation_parameters or {}))
        self._preview_worker = worker
        worker.signals.finished.connect(self.finish_live_preview)
        self._preview_pool.start(worker)

    @QtCore.Slot(int, object, object)
    def finish_live_preview(self, revision, preview, error):
        target = self._preview_target
        self._preview_running = False
        self._preview_worker = None
        if target is self.result and revision == self._preview_revision and self.preview_is_available():
            if error:
                target.mask_preview = None
                target.mask_preview_edge = 0
                target.segmentation_preview = None
                self.preview_status.setText("Preview unavailable — adjust the threshold or blur.")
                self.preview_status.setToolTip(error)
                self.parent.tab1.update_display()
            else:
                if target.analysis_mode == "full":
                    target.segmentation_preview = preview["segmentation"]
                else:
                    target.mask_preview = preview["mask"]
                    target.mask_preview_edge = preview["edge"]
                self.preview_status.setText("Live preview")
                self.preview_status.setToolTip("Run Find orientations to update the analysis with these settings.")
                # Only repaint the mask; do not mark the analysis complete or save.
                self.parent.tab1.update_display()
        if self._preview_waiting:
            self._preview_waiting = False
            if not self._preview_timer.isActive():
                self._preview_timer.start(0)

    def check_available(self, result: ResultOrientation) -> bool:
        return result is not None and (result.analysis_mode == "full" or result.uses_mask)

    def state_changed(self, result):
        super().state_changed(result)
        if result is self.result:
            busy = self.get_result_state(result) in (StateEnum.running, StateEnum.scheduled)
            self.seg_threshold_slider.setEnabled(not busy)
            self.mask_threshold_slider.setEnabled(not busy)
            if busy:
                self._preview_revision += 1
                self._preview_timer.stop()
                self._preview_waiting = False
        if result is self.result and self.get_result_state(result) in (StateEnum.finished, StateEnum.failed):
            self.input_button.setEnabled(self.check_available(result))
            self.input_button.setText("preview segmentation" if result.analysis_mode == "full" else "preview mask")

    def setResult(self, result: ResultOrientation):
        self._setting_result = True
        self._preview_revision += 1
        self._preview_timer.stop()
        self._preview_waiting = False
        super().setResult(result)
        self.sync_threshold_slider(self.seg_threshold_slider, self.seg_thresh.value())
        self.sync_threshold_slider(self.mask_threshold_slider, self.mask_threshold.value())
        self.preview_status.setText("Automatic mask preview")
        self.preview_status.setToolTip("")
        self._setting_result = False
        full = result is None or result.analysis_mode == "full"
        self.group.label.setText("cell segmentation" if full else "orientation mask")
        if result is not None and result.uses_mask:
            if result.mask_source == "same_image":
                self.mask_note.setText("Mask from the orientation image: adjust blur and threshold to select the area, then preview. "
                                       "Orientation is calculated from the original image.")
            else:
                self.mask_note.setText("The mask image selects the area. Orientation is calculated only from the orientation image.")
        self.input_button.setText("preview segmentation" if full else "preview mask")
        self.setVisible(result is None or (result.analysis_mode == "full" or result.uses_mask))
        for widget in (self.seg_thresh, self.seg_threshold_slider, self.seg_invert, self.seg_gauss1, self.seg_gauss2):
            widget.setVisible(full)
        for widget in (self.mask_sigma, self.mask_threshold, self.mask_threshold_slider, self.mask_invert, self.mask_note):
            widget.setVisible(not full)

    def process_thread(self, result, params):
        if not result.uses_mask:
            return super().process_thread(result, params)
        # A mask preview must not overwrite the parameters of the last analysis.
        # The detection step commits parameters together with its new results.
        self.parent.progressbar.setRange(0, 0)
        self.set_result_state(result, StateEnum.running)
        self.processing_state_changed.emit(result)
        try:
            self.process(result, **params)
        except Exception as error:
            import traceback
            traceback.print_exc()
            self.set_result_state(result, StateEnum.failed)
            self.processing_state_changed.emit(result)
            self.processing_error.emit(str(error))
        else:
            self.set_result_state(result, StateEnum.finished)
            self.parent.result_changed.emit(result)
            self.processing_finished.emit()

    def process(self, result: ResultOrientation, segmentation_parameters: SegmentationParametersDict):
        from CompactionAnalyzer.CompactionFunctions import segment_cell, normalize
        from skimage import color

        if result.analysis_mode != "full" and not result.uses_mask:
            return  # Batch runs need no segmentation for whole-image analysis.
        im_cell = result.get_image(0)
        ## if 3 channels convert to grey
        if len(im_cell.shape) == 3:
            im_cell = color.rgb2gray(im_cell)
        if result.uses_mask:
            try:
                from CompactionAnalyzer.CompactionFunctions import create_axis_mask
            except ImportError as error:
                raise RuntimeError("Mask preview requires CompactionAnalyzer >= 1.0.8. Update CompactionAnalyzer in this Python environment and restart Saenopy.") from error
            orientation = getattr(result, "orientation_parameters_tmp", None) or result.orientation_parameters or {}
            edge = orientation.get("edge", 40)
            if not isinstance(edge, (int, np.integer)) or edge < 0 or 2*edge >= min(im_cell.shape):
                raise ValueError("edge must leave a nonempty mask image")
            crop = slice(edge, -edge) if edge else slice(None)
            # Preview is independent of the analyzed validity mask. Changing
            # a threshold must not silently relabel already computed results.
            preview = create_axis_mask(im_cell[crop,crop], segmentation_parameters["mask_sigma"],
                                       segmentation_parameters["mask_threshold"], segmentation_parameters["mask_invert"])
            result.mask_preview = preview
            result.mask_preview_edge = edge
            return
        im_cell_n = normalize(im_cell, 1, 99)

        segmentation = segment_cell(im_cell_n, thres=segmentation_parameters["thresh"],
                                  seg_gaus1=segmentation_parameters["gauss1"],
                                  seg_gaus2=segmentation_parameters["gauss2"],
                                  show_segmentation=False,
                                  seg_invert=segmentation_parameters["invert"],
                                  seg_iter=1,
                                  segmention_method="otsu",
                                  regional_max_correction=True,
                                  segmention_min_area=1000)

        result.segmentation = segmentation
        result.segmentation_preview = None
        result.save()

    def get_code(self) -> Tuple[str, str]:
        # Detection exports the complete analysis, including segmentation.
        return "", ""
