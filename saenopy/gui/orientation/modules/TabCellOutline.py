import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import jointforces as jf
from pathlib import Path

from saenopy.gui.common.TabModule import TabModule
from saenopy.gui.orientation.modules.result import ResultOrientation, SegmentationParametersDict
from qtpy import QtCore, QtWidgets, QtGui
from qimage2ndarray import array2qimage
from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView

from saenopy.gui.common.gui_classes import QHLine, CheckAbleGroup
from saenopy.gui.common.PipelineModule import PipelineModule
from saenopy.gui.common.QTimeSlider import QTimeSlider
from saenopy.gui.common.resources import resource_icon
from saenopy.gui.common.code_export import get_code
from saenopy.gui.common.ModuleScaleBar import ModuleScaleBar
from saenopy.gui.common.ModuleColorBar import ModuleColorBar
from .ReferenceOverlay import ReferenceOverlay, parameter_line, calculated_line, LINE_KEYS


class TabCellOutline(TabModule):
    result: ResultOrientation = None

    def __init__(self, parent: "BatchEvaluate"):
        super().__init__(parent)
        self._line_start = None
        self._reference_press = None
      #  self.update_display()
        with self.parent.tabs.createTab("Segmentation / mask") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                #self.label_tab = QtWidgets.QLabel(
                #    "The deformations from the piv algorithm at every window where the crosscorrelation was evaluated.").addToLayout()

                self.point_controls = QtWidgets.QWidget()
                controls = QtWidgets.QHBoxLayout(self.point_controls)
                controls.setContentsMargins(0, 0, 0, 0)
                self.choose_point = QtWidgets.QPushButton("Choose reference point")
                self.choose_point.setCheckable(True)
                self.point_status = QtWidgets.QLabel()
                self.point_status.setWordWrap(True)
                self.point_status.setMinimumHeight(3 * self.point_status.fontMetrics().height())
                controls.addWidget(self.choose_point)
                controls.addWidget(self.point_status, 1)
                layout.addWidget(self.point_controls)
                self.point_controls.hide()
                with QtShortCuts.QHBoxLayout() as layout:
                    #self.plotter = QtInteractor(self, auto_update=False)  # , theme=pv.themes.DocumentTheme())
                    #self.tab.parent().plotter = self.plotter
                    #self.plotter.set_background("black")
                    #layout.addWidget(self.plotter.interactor)
                    self.label = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    self.scale1 = ModuleScaleBar(self, self.label)
                    self.color1 = ModuleColorBar(self, self.label)
                    #self.label.setMinimumWidth(300)
                    self.pixmap = QtWidgets.QGraphicsPixmapItem(self.label.origin)
                    self.mask_overlay = QtWidgets.QGraphicsPixmapItem(self.label.origin)
                    self.mask_overlay.setZValue(1)
                    self.contour = QtWidgets.QGraphicsPathItem(self.label.origin)
                    self.contour.setZValue(2)
                    pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
                    pen.setCosmetic(True)
                    pen.setWidth(2)
                    self.contour.setPen(pen)
                    self.center = QtWidgets.QGraphicsEllipseItem(self.label.origin)
                    self.center.setZValue(3)
                    brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
                    self.center.setBrush(brush)
                    self.reference_overlay = ReferenceOverlay(self.label.origin)
                    self.point_marker = self.reference_overlay.markers[0]
                    self.label.viewport().installEventFilter(self)
                    self.label.viewport().setMouseTracking(True)
                    self.choose_point.toggled.connect(self.picking_changed)

    def checkTabEnabled(self, result: ResultOrientation) -> bool:
            # whether the tab should be enabled for this Result object
            return result is not None


    def setResult(self, result: ResultOrientation):
        self.choose_point.setChecked(False)
        self._line_start = None
        point = result is not None and result.analysis_mode == "point_orientation"
        line = result is not None and result.analysis_mode == "line_orientation"
        self.point_controls.setVisible(point or line)
        self.choose_point.setText("Choose reference line" if line else "Choose reference point")
        title = "Reference line" if line else ("Reference point" if point else ("Image" if result is not None and result.analysis_mode == "axis_orientation_only" else "Segmentation / mask"))
        self.parent.tabs.setTabText(self.parent.tabs.indexOf(self.tab.parent()), title)
        super().setResult(result)
        self.update_display()

    def update_display(self, *, plotter=None):
        # if self.result is None:
        #     return
        #if self.current_tab_selected is False:
        #    return

        self.reference_overlay.hide()
        self.mask_overlay.setPixmap(QtGui.QPixmap())
        self.mask_overlay.setVisible(False)
        if self.result is None:
            return
        reference_mode = self.result.analysis_mode in ("point_orientation", "line_orientation")
        im = self.result.get_image(1 if reference_mode else 0)
        segmentation = getattr(self.result, "segmentation_preview", None) if self.result.analysis_mode == "full" else None
        if segmentation is None:
            segmentation = self.result.segmentation
        axis_mask = None
        if self.result.uses_mask:
            axis_mask = getattr(self.result, "mask_preview", None)
            edge = getattr(self.result, "mask_preview_edge", 0)
            if axis_mask is None:
                axis_mask = self.result.validity_mask
                edge = (self.result.orientation_parameters or {}).get("edge", 0)
            if axis_mask is not None and edge:
                if reference_mode:
                    # Keep click coordinates in the original image, including
                    # when the displayed mask came from an edge-cropped preview.
                    axis_mask = np.pad(axis_mask, edge, constant_values=False)
                else:
                    im = im[edge:-edge, edge:-edge]
        preview_range = False
        if self.result.analysis_mode in ("point_orientation", "line_orientation"):
            low, high = np.nanpercentile(im, [1, 99])
            if high > low:
                preview_range = (low, high)
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im, normalize=preview_range)))
        self.label.setExtend(im.shape[1], im.shape[0])

        if axis_mask is not None or segmentation is not None:
            from skimage import measure
            mask = np.asarray(axis_mask if axis_mask is not None else segmentation["mask"], dtype=bool)
            overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
            overlay[mask] = [255, 40, 40, 70]
            self.mask_overlay.setPixmap(QtGui.QPixmap(array2qimage(overlay)))
            self.mask_overlay.setVisible(True)
            # Find contours at a constant value of 0.8
            contours = measure.find_contours(mask, 0.5)

            path = QtGui.QPainterPath()
            for c in contours:
                path.moveTo(c[0][1],  c[0][0])
                for cc in c:
                    path.lineTo(cc[1],  cc[0])
            self.contour.setPath(path)
            self.center.setVisible(axis_mask is None)
            if axis_mask is None:
                x, y = segmentation["centroid"]
                self.center.setRect(x-3, y-3, 6, 6)
        else:
            path = QtGui.QPainterPath()
            self.contour.setPath(path)
            self.center.setVisible(False)
        if self.result.analysis_mode == "point_orientation":
            parameters = getattr(self.result, "orientation_parameters_tmp", None) or self.result.orientation_parameters or {}
            x, y = parameters.get("reference_x"), parameters.get("reference_y")
            if x is not None and y is not None:
                self.point_marker.setPos(x + .5, y + .5)
                self.point_marker.show()
                previous = (self.result.results_total or [{}])[0]
                calculated = (previous.get("Reference x (px)"), previous.get("Reference y (px)")) == (x, y)
                self.point_status.setText(f"X: {x:g}, Y: {y:g} px" +
                    (" — analyzed reference" if calculated else " — run find orientations to apply"))
            else:
                self.point_status.setText("Choose a point in this image before running the analysis.")
        elif self.result.analysis_mode == "line_orientation":
            parameters = getattr(self.result, "orientation_parameters_tmp", None) or self.result.orientation_parameters or {}
            line = parameter_line(parameters)
            self.reference_overlay.show(line, im.shape)
            if line is not None:
                from CompactionAnalyzer.plotting import line_reference_angle
                try:
                    angle = line_reference_angle(line, im.shape)
                except ValueError:
                    self.reference_overlay.show(None, im.shape)
                    self.point_status.setText("Choose two distinct points inside this image before running the analysis.")
                    return
                calculated = np.array_equal(line, calculated_line(self.result))
                self.point_status.setText(f"Line: ({line[0,0]:g}, {line[0,1]:g}) to ({line[1,0]:g}, {line[1,1]:g}) px; {angle:.1f}° to X" +
                    (" — analyzed reference" if calculated else " — run find orientations to apply"))
            else:
                self.point_status.setText("Choose two distinct points in this image before running the analysis.")
        return None

    def picking_changed(self, checked):
        self._line_start = None
        self._reference_press = None
        self.label.viewport().setCursor(QtCore.Qt.CrossCursor if checked else QtCore.Qt.ArrowCursor)
        if checked:
            if self.result is None or getattr(self.result, "orientation_parameters_state", "") in ("running", "scheduled"):
                self.choose_point.setChecked(False)
                return
            line = self.result.analysis_mode == "line_orientation"
            self.point_status.setText(("Click the first of two points." if line else "Click the reference point.") +
                                     " Esc, right-click or the button cancels.")
            self.label.viewport().setFocus()
            self.parent.sub_module_deformation.input_button.setEnabled(False)
        else:
            self.update_display()
            module = self.parent.sub_module_deformation
            module.input_button.setEnabled(module.check_available(self.result))

    def eventFilter(self, watched, event):
        # Ignore paint/destruction events before touching sibling widgets;
        # PySide can deliver them while the window is being torn down.
        if event.type() not in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease,
                                QtCore.QEvent.MouseMove, QtCore.QEvent.KeyPress):
            return False
        if (watched is self.label.viewport() and self.choose_point.isChecked()
                and self.result is not None and self.result.analysis_mode in ("point_orientation", "line_orientation")):
            if event.type() == QtCore.QEvent.KeyPress:
                if event.key() == QtCore.Qt.Key_Escape:
                    self.choose_point.setChecked(False)
                    return True
                return False
            if event.type() == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.RightButton:
                self.choose_point.setChecked(False)
                return True
            pos = self.label.origin.mapFromScene(self.label.mapToScene(event.pos()))
            point = (int(np.floor(pos.x())), int(np.floor(pos.y())))
            if event.type() == QtCore.QEvent.MouseMove:
                if self._line_start is not None:
                    self.reference_overlay.show([self._line_start, point], self.result.get_image(1).shape)
                return True
            if event.button() == QtCore.Qt.LeftButton:
                if event.type() == QtCore.QEvent.MouseButtonPress:
                    self._reference_press = event.pos()
                elif self._reference_press is not None:
                    distance = (event.pos() - self._reference_press).manhattanLength()
                    self._reference_press = None
                    if distance < QtWidgets.QApplication.startDragDistance():
                        if self.result.analysis_mode == "line_orientation":
                            self.pick_line_point(*point)
                        else:
                            self.set_reference_point(*point)
                return True
        return super().eventFilter(watched, event)

    def set_reference_point(self, x, y):
        if self.result is None or self.result.analysis_mode != "point_orientation":
            return
        if getattr(self.result, "orientation_parameters_state", "") in ("running", "scheduled"):
            return
        height, width = self.result.get_image(1).shape[:2]
        if not 0 <= x < width or not 0 <= y < height:
            return
        module = self.parent.sub_module_deformation
        module.parameter_mappings[0].ensure_tmp_params_initialized(self.result)
        self.result.orientation_parameters_tmp.update(reference_x=float(x), reference_y=float(y))
        self.choose_point.setChecked(False)
        self.update_display()
        module.input_button.setText("find orientations")
        module.input_button.setEnabled(True)

    def pick_line_point(self, x, y):
        height, width = self.result.get_image(1).shape[:2]
        if not 0 <= x < width or not 0 <= y < height:
            return
        if self._line_start is None:
            self._line_start = (x, y)
            self.reference_overlay.show([self._line_start], (height, width))
            self.point_status.setText("Click a different second point. Esc or right-click cancels.")
        else:
            self.set_reference_line(self._line_start, (x, y))

    def set_reference_line(self, start, end):
        if self.result is None or self.result.analysis_mode != "line_orientation":
            return
        if getattr(self.result, "orientation_parameters_state", "") in ("running", "scheduled"):
            return
        from CompactionAnalyzer.plotting import line_reference_angle
        try:
            line_reference_angle((start, end), self.result.get_image(1).shape)
        except ValueError:
            self.point_status.setText("Choose two different points inside the image.")
            return
        module = self.parent.sub_module_deformation
        module.parameter_mappings[0].ensure_tmp_params_initialized(self.result)
        self.result.orientation_parameters_tmp.update(zip(LINE_KEYS, map(float, (*start, *end))))
        self.choose_point.setChecked(False)
        self.update_display()
        module.input_button.setText("find orientations")
        module.input_button.setEnabled(True)
