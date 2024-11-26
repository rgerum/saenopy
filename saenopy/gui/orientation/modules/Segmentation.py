import numpy as np
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
from saenopy.gui.common.QTimeSlider import QTimeSlider
from saenopy.gui.common.resources import resource_icon
from saenopy.gui.common.code_export import get_code
from saenopy.gui.common.ModuleScaleBar import ModuleScaleBar
from saenopy.gui.common.ModuleColorBar import ModuleColorBar


class SegmentationDetector(PipelineModule):
    pipeline_name = "find deformations"
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
                    with QtShortCuts.QHBoxLayout():
                        self.seg_gauss1 = QtShortCuts.QInputString(None, "gauss1", "0.5", type=float,
                                                                   settings=self.settings,
                                                                   settings_key="orientation/segmentation/gauss1")
                        # self.seg_gaus1.valueChanged.connect(self.listSelected)
                        self.seg_gauss2 = QtShortCuts.QInputString(None, "gauss2", "100", type=float,
                                                                   settings=self.settings,
                                                                   settings_key="orientation/segmentation/gauss2")
                        # self.seg_gaus2.valueChanged.connect(self.listSelected)

                    self.input_button = QtShortCuts.QPushButton(None, "detect deformations", self.start_process)

        with self.parent.tabs.createTab("Segmentation Deformations") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                #self.label_tab = QtWidgets.QLabel(
                #    "The deformations from the piv algorithm at every window where the crosscorrelation was evaluated.").addToLayout()

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
                    self.contour = QtWidgets.QGraphicsPathItem(self.label.origin)
                    pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
                    pen.setCosmetic(True)
                    pen.setWidth(2)
                    self.contour.setPen(pen)
                    self.center = QtWidgets.QGraphicsEllipseItem(self.label.origin)
                    brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
                    self.center.setBrush(brush)

                    #self.z_slider = QTimeSlider("z", self.z_slider_value_changed, "set z position",
                    #                            QtCore.Qt.Vertical).addToLayout()
                    #self.z_slider.t_slider.valueChanged.connect(
                    #    lambda value: parent.shared_properties.change_property("z_slider", value, self))
                    #parent.shared_properties.add_property("z_slider", self)

                #self.vtk_toolbar = VTK_Toolbar(self.plotter, self.update_display, "deformation", z_slider=self.z_slider, shared_properties=self.parent.shared_properties).addToLayout()
                with QtShortCuts.QHBoxLayout() as layout0:
                    layout0.setContentsMargins(0, 0, 0, 0)

                    self.auto_scale = QtShortCuts.QInputBool(None, "", icon=[
                        resource_icon("autoscale0.ico"),
                        resource_icon("autoscale1.ico"),
                    ], group=False, tooltip="Automatically choose the maximum for the color scale.")
                    # self.auto_scale = QtShortCuts.QInputBool(None, "auto color", True, tooltip="Automatically choose the maximum for the color scale.")
                    self.auto_scale.setValue(True)
                    self.auto_scale.valueChanged.connect(self.scale_max_changed)
                    self.scale_max = QtShortCuts.QInputString(None, "", 10,# if self.is_force_plot else 10,
                                                              type=float,
                                                              tooltip="Set the maximum of the color scale.")
                    self.scale_max.valueChanged.connect(self.scale_max_changed)
                    self.scale_max.setDisabled(self.auto_scale.value())

                    self.window_scale = QtWidgets.QWidget()
                    self.window_scale.setWindowTitle("Saenopy - Arrow Scale")
                    with QtShortCuts.QVBoxLayout(self.window_scale):
                        self.arrow_scale = QtShortCuts.QInputNumber(None, "arrow scale", 1, 0.01, 100, use_slider=True,
                                                                    log_slider=True)
                        self.arrow_scale.valueChanged.connect(self.update_display)
                        #self.arrow_scale.valueChanged.connect(
                        #    lambda value: shared_properties.change_property("arrow_scale" + addition, value, self))
                        #shared_properties.add_property("arrow_scale" + addition, self)

                        QtWidgets.QLabel("Colormap for arrows").addToLayout()
                        self.colormap_chooser = QtShortCuts.QDragableColor("turbo").addToLayout()
                        self.colormap_chooser.valueChanged.connect(self.update_display)

                        #self.colormap_chooser.valueChanged.connect(
                        #    lambda value: shared_properties.change_property("colormap_chooser" + addition, value, self))

                        QtWidgets.QLabel("Colormap for image").addToLayout()
                        self.colormap_chooser2 = QtShortCuts.QDragableColor("gray").addToLayout()
                        self.colormap_chooser2.valueChanged.connect(self.update_display)

                        #self.colormap_chooser2.valueChanged.connect(
                        #    lambda value: shared_properties.change_property("colormap_chooser2" + addition, value,
                        #                                                    self))
                        #shared_properties.add_property("colormap_chooser2" + addition, self)
                    self.button_arrow_scale = QtShortCuts.QPushButton(None, "", lambda x: self.window_scale.show())
                    self.button_arrow_scale.setIcon(resource_icon("arrowscale.ico"))

                    self.show_seg = QtShortCuts.QInputBool(None, "seg", value=True).addToLayout()
                    self.show_seg.valueChanged.connect(self.update_display)

                self.t_slider = QTimeSlider(connected=self.update_display).addToLayout()
                self.tab.parent().t_slider = self.t_slider

        self.setParameterMapping("segmentation_parameters", {
            "thresh": self.seg_thresh,
            "gauss1": self.seg_gauss1,
            "gauss2": self.seg_gauss2,
            "invert": self.seg_invert,
        })

        #self.progress_signal.connect(self.progress_callback)

    def scale_max_changed(self):
        self.scale_max.setDisabled(self.auto_scale.value())
        scalebar_max = self.getScaleMax()
        #print(scalebar_max, self.plotter.auto_value, type(self.plotter.auto_value))
        #if scalebar_max is None:
        #    self.plotter.update_scalar_bar_range([0, self.plotter.auto_value])
        #else:
        #    self.plotter.update_scalar_bar_range([0, scalebar_max])
        self.update_display()

    def getScaleMax(self):
        if self.auto_scale.value():
            return None
        return self.scale_max.value()

    def check_available(self, result: ResultOrientation) -> bool:
        return True

    def check_evaluated(self, result: ResultOrientation) -> bool:
        return True

    def setResult(self, result: ResultOrientation):
        super().setResult(result)
        self.update_display()

    def update_display(self, *, plotter=None):
        if self.result is None:
            return
        #if self.current_tab_selected is False:
        #    return

        im = self.result.get_image(0)
        print(im.dtype, im.max())
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.label.setExtend(im.shape[1], im.shape[0])

        if self.result.segmentation is not None:
            from skimage import measure
            # Find contours at a constant value of 0.8
            contours = measure.find_contours(self.result.segmentation["mask"], 0.5)

            path = QtGui.QPainterPath()
            for c in contours:
                path.moveTo(c[0][1],  c[0][0])
                for cc in c:
                    path.lineTo(cc[1],  cc[0])
            self.contour.setPath(path)
            x, y = self.result.segmentation["centroid"]
            self.center.setRect(x-3, y-3, 6, 6)
            self.center.setVisible(True)
        else:
            path = QtGui.QPainterPath()
            self.contour.setPath(path)
            self.center.setVisible(False)

        return None

    def process(self, result: ResultOrientation, segmentation_parameters: SegmentationParametersDict):
        from CompactionAnalyzer.CompactionFunctions import segment_cell, normalize
        from skimage import color

        im_cell = result.get_image(0)
        ## if 3 channels convert to grey
        if len(im_cell.shape) == 3:
            im_cell = color.rgb2gray(im_cell)
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
        result.save()

    def get_code(self) -> Tuple[str, str]:
        import_code = "from saenopy.gui.spheroid.modules.result import get_stacks_spheroid\nimport jointforces as jf\n"

        def code(filename, output1, pixel_size1, result_file, piv_parameters1):  # pragma: no cover
            # load the relaxed and the contracted stack
            # {t} is the placeholder for the time points
            # use * to load multiple stacks for batch processing
            # load_existing=True allows to load an existing file of these stacks if it already exists
            results = get_stacks_spheroid(
                filename,
                output_path=output1,
                pixel_size=pixel_size1,
                load_existing=True)
            # or if you want to explicitly load existing results files
            # use * to load multiple result files for batch processing
            # results = saenopy.load_results(result_file)

            for result in results:
                result.piv_parameters = piv_parameters1
                jf.piv.compute_displacement_series_gui(result,
                                                       n_max=result.piv_parameters["n_max"],
                                                       n_min=result.piv_parameters["n_min"],
                                                       continous_segmentation=result.piv_parameters["continuous_segmentation"],
                                                       thres_segmentation=result.piv_parameters["thresh_segmentation"],
                                                       window_size=result.piv_parameters["window_size"],
                                                       )
                result.save()

        data = dict(
            filename=self.result.get_absolute_path(),
            output1=str(Path(self.result.output).parent),
            pixel_size1=self.result.pixel_size,
            result_file=str(self.result.output),
            piv_parameters1=self.result.piv_parameters,
        )

        code = get_code(code, data)
        return import_code, code
