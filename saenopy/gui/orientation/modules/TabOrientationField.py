from saenopy.gui.common.TabModule import TabModule
from saenopy.gui.orientation.modules.result import ResultOrientation
from qtpy import QtCore, QtWidgets, QtGui
from qimage2ndarray import array2qimage
from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView

from saenopy.gui.common.ModuleScaleBar import ModuleScaleBar
from saenopy.gui.common.ModuleColorBar import ModuleColorBar
from saenopy.gui.solver.modules.exporter.ExporterRender2D import render_2d
from .AxisSelector import AxisSelector
from .ReferenceOverlay import ReferenceOverlay, calculated_line


class TabOrientationField(TabModule):
    result: ResultOrientation = None

    def __init__(self, parent: "BatchEvaluate"):
        super().__init__(parent)

        with self.parent.tabs.createTab("Orientation (cell center)") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.axis = AxisSelector()
                layout.addWidget(self.axis)
                self.axis.valueChanged.connect(lambda *_: self.update_display())
                #self.label_tab = QtWidgets.QLabel(
                #    "The deformations from the piv algorithm at every window where the crosscorrelation was evaluated.").addToLayout()

                with QtShortCuts.QHBoxLayout() as layout:
                    self.label = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
                    self.scale1 = ModuleScaleBar(self, self.label)
                    self.color1 = ModuleColorBar(self, self.label)

                    self.pixmap = QtWidgets.QGraphicsPixmapItem(self.label.origin)
                    self.contour = QtWidgets.QGraphicsPathItem(self.label.origin)
                    pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
                    pen.setCosmetic(True)
                    pen.setWidth(2)
                    self.contour.setPen(pen)
                    self.reference_overlay = ReferenceOverlay(self.label.origin)
                    self.point_marker = self.reference_overlay.markers[0]

    def setResult(self, result):
        full = result is None or result.analysis_mode == "full"
        point = result is not None and result.analysis_mode == "point_orientation"
        line = result is not None and result.analysis_mode == "line_orientation"
        if line != getattr(self, "_line_mode", False):
            self.axis.set_choices(([('line', 'Line')] if line else []) + [('x', 'X axis'), ('y', 'Y axis')],
                                  'line' if line else 'x')
        self._line_mode = line
        self.axis.setVisible(not full and not point)
        index = self.parent.tabs.indexOf(self.tab.parent())
        self.parent.tabs.setTabText(index, "Orientation (cell center)" if full else ("Orientation (point)" if point else ("Alignment (line)" if line else "Axis alignment")))
        super().setResult(result)

    def checkTabEnabled(self, result: ResultOrientation) -> bool:
        # whether the tab should be enabled for this Result object
        return result is not None and result.orientation_map is not None

    def update_display(self, *, plotter=None):
        if not self.checkTabEnabled(self.result):
            return
        self.reference_overlay.hide()
        im = render_2d({
            "image": {
                "scale": 1,
                "scale_overlay": 1,
                "antialiase": 1,
                "logo_size": 0,
            },
            "stack": {
                "image": 1,
                "channel": "cells" if self.result.analysis_mode == "full" else "fibers",
                "colormap": "gray",
                "alpha": 1,
                "channel_B": "fibers",
                "colormap_B": "gray",
                "alpha_B": 1 if self.result.analysis_mode == "full" else 0,
                "use_reference_stack": False,
                "contrast_enhance": None,
                "z_proj": False,
                "z": 0,
                "use_contrast_enhance": True,
            },
            "time": {
                "t": 0
            },
            "maps": "orientation_cell_center" if self.result.analysis_mode == "full" else (
                "orientation_point" if self.result.analysis_mode == "point_orientation" else (
                "orientation_line" if self.axis.value() == "line" else f"alignment_{self.axis.value()}_axis")),
            "maps_cmap": "coolwarm",
            "maps_alpha": 0.5,

            "colorbar": {
                "hide": True,
            },

            "scalebar": {
                "hide": True,
            },
            "theme": "dark",
        }, self.result)
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.label.setExtend(im.shape[1], im.shape[0])
        self.scale1.setScale([self.result.pixel_size])
        self.color1.setScale(-1, 1, "coolwarm")
        if self.result.analysis_mode == "point_orientation" and self.result.results_total:
            row = self.result.results_total[0]
            edge = self.result.orientation_parameters["edge"]
            self.point_marker.setPos(row["Reference x (px)"] - edge + .5, row["Reference y (px)"] - edge + .5)
            self.point_marker.show()
        elif self.result.analysis_mode == "line_orientation":
            self.reference_overlay.show(calculated_line(self.result), im.shape,
                                        edge=self.result.orientation_parameters['edge'])
