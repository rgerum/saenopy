import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import jointforces as jf
from pathlib import Path

from saenopy.gui.spheroid.modules.result import ResultSpheroid
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


class DeformationDetector(PipelineModule):
    pipeline_name = "find deformations"
    use_thread = False
    progress_signal = QtCore.Signal(int, str)
    result: ResultSpheroid = None

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "find deformations (piv)").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QVBoxLayout() as layout2:
                        self.window_size = QtShortCuts.QInputNumber(layout2, "window size", 50,
                                                                    float=False, name_post='px',
                                                                    settings=self.settings,
                                                                    settings_key="spheroid/deformation/window_size")
                        self.overlap = QtShortCuts.QInputNumber(layout2, "overlap", 50,
                                                                    float=False, name_post='%',
                                                                    settings=self.settings,
                                                                    settings_key="spheroid/deformation/overlap")

                        QHLine().addToLayout()
                        with QtShortCuts.QHBoxLayout(None):
                            self.n_min = QtShortCuts.QInputString(None, "n_min", "None", allow_none=True, type=int,
                                                                  settings=self.settings,
                                                                  settings_key="spheroid/deformation/n_min")
                            self.n_max = QtShortCuts.QInputString(None, "n_max", "None", allow_none=True, type=int,
                                                                  settings=self.settings, name_post='frames',
                                                                  settings_key="spheroid/deformation/n_max")

                        self.thresh_segmentation = QtShortCuts.QInputNumber(None, "segmentation threshold", 0.9,
                                                                            float=True,
                                                                            min=0.2, max=1.5, step=0.1,
                                                                            use_slider=False,
                                                                            settings=self.settings,
                                                                            settings_key="spheroid/deformation/thres_segmentation2")
                        #self.thresh_segmentation.valueChanged.connect(
                        #    lambda: self.param_changed("thres_segmentation", True))
                        self.continuous_segmentation = QtShortCuts.QInputBool(None, "continous_segmentation", False,
                                                                              settings=self.settings,
                                                                              settings_key="spheroid/deformation/continous_segemntation")
                        #self.continuous_segmentation.valueChanged.connect(
                        #    lambda: self.param_changed("continous_segmentation", True))
                        #self.n_min.valueChanged.connect(lambda: self.param_changed("n_min"))
                        #self.n_max.valueChanged.connect(lambda: self.param_changed("n_max"))
                        self.thresh_segmentation.valueChanged.connect(self.update_display)
                        self.continuous_segmentation.valueChanged.connect(self.update_display)

                    self.input_button = QtShortCuts.QPushButton(None, "detect deformations", self.start_process)

        with self.parent.tabs.createTab("PIV Deformations") as self.tab:
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

        self.setParameterMapping("piv_parameters", {
            "window_size": self.window_size,
            "overlap": self.overlap,
            "n_min": self.n_min,
            "n_max": self.n_max,
            "thresh_segmentation": self.thresh_segmentation,
            "continuous_segmentation": self.continuous_segmentation,
        })

        self.progress_signal.connect(self.progress_callback)

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

    def check_available(self, result: ResultSpheroid) -> bool:
        return True

    def check_evaluated(self, result: ResultSpheroid) -> bool:
        return True

    def setResult(self, result: ResultSpheroid):
        super().setResult(result)
        self.update_display()

    def update_display(self, *, plotter=None):
        if self.result is None:
            return
        #if self.current_tab_selected is False:
        #    return

        import imageio
        t = self.t_slider.value()
        im = self.result.get_image_data(t).astype(float)
        #im = imageio.v2.imread(self.result.images[t]).astype(float)
        im0 = im.copy()
        im = im - im.min()
        im = (im / im.max() * 255).astype(np.uint8)
        im = np.squeeze(im)

        colormap2 = self.colormap_chooser2.value()
        #print(im.shape)
        if len(im.shape) == 2 and colormap2 is not None and colormap2 != "gray":
            cmap = plt.get_cmap(colormap2)
            im = (cmap(im)[:, :, :3] * 255).astype(np.uint8)
            #print(im.shape, im.dtype, im[100, 100])

        from saenopy.gui.solver.modules.exporter.ExporterRender2D import render_2d_arrows
        im_scale = 1
        aa_scale = 1
        display_image = [im, [1, 1], 0]
        from PIL import Image
        pil_image = Image.fromarray(im).convert("RGB")
        pil_image = pil_image.resize(
            [int(pil_image.width * im_scale * aa_scale), int(pil_image.height * im_scale * aa_scale)])
        #print(self.auto_scale.value(), self.getScaleMax())
        pil_image, disp_params = render_2d_arrows({
            'arrows': 'deformation',
            'deformation_arrows': {
                "autoscale": self.auto_scale.value(),
                "scale_max": self.getScaleMax(),
                "colormap": self.colormap_chooser.value(),
                "skip": 1,
                "arrow_opacity": 1,
                "arrow_scale": 10*self.arrow_scale.value(),
            },
            "time": {"t": t},
            '2D_arrows': {'width': 2.0, 'headlength': 5.0, 'headheight': 5.0},
        }, self.result, pil_image, im_scale, aa_scale, display_image, return_scale=True)

        im = np.asarray(pil_image)
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.label.setExtend(im.shape[1], im.shape[0])
        self.scale1.setScale([self.result.pixel_size])
        self.color1.setScale(0, disp_params["scale_max"] if disp_params else None, self.colormap_chooser.value())

        if self.show_seg.value():
            thresh_segmentation = self.thresh_segmentation.value()
            if not self.continuous_segmentation.value():
                im0 = imageio.v2.imread(self.result.get_image_data(0, return_filename=True)).astype(float)
            seg0 = jf.piv.segment_spheroid(im0, True, thresh_segmentation)
            from skimage import measure
            # Find contours at a constant value of 0.8
            contours = measure.find_contours(seg0["mask"], 0.5)

            path = QtGui.QPainterPath()
            for c in contours:
                path.moveTo(c[0][1], im.shape[0] - c[0][0])
                for cc in c:
                    path.lineTo(cc[1], im.shape[0] - cc[0])
            self.contour.setPath(path)

    def process(self, result: ResultSpheroid, piv_parameters: dict):
        jf.piv.compute_displacement_series_gui(result,
                                               n_max=piv_parameters["n_max"],
                                               n_min=piv_parameters["n_min"],
                                               continous_segmentation=piv_parameters["continuous_segmentation"],
                                               thres_segmentation=piv_parameters["thresh_segmentation"],
                                               window_size=piv_parameters["window_size"],
                                               overlap=piv_parameters["overlap"]/100,
                                               #cmap="turbo",
                                               callback=lambda ii, nn: self.progress_signal.emit(ii, result.output),
                                               #                                                  nn)
                                               )
        result.save()

    def progress_callback(self, i, result):
        if result == self.result.output:
            self.t_slider.setValue(i)
        #self.progressbar.setRange(0, result.n_images)
        #self.progressbar.setValue(i)

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
