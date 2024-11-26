import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import jointforces as jf
from pathlib import Path

from saenopy.gui.orientation.modules.result import ResultOrientation, OrientationParametersDict
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

def im_process(im, cmap, vmin=None, vmax=None, alpha=1, add_to=None):
    if vmin is not None and vmax is not None:
        im -= vmin
        im /= (vmax-vmin)
        im = np.clip(im, 0, 1)*255
    im = plt.get_cmap(cmap)(im)
    if add_to is not None:
        im = add_to * (1-alpha) + im * alpha
    else:
        im *= alpha
    return im

def plot_fancy_overlay(fiber, cell, field,label="field",dpi=300,cmap_cell="Greys_r",cmap_fiber="Greys_r",
                       cmap_angle="viridis", alpha_ori =0.8,  alpha_cell = 0.4, alpha_fiber = 0.4,   # example cmaps: viridis/inferno/coolwarm
                       omin=-1, omax=1, scale=None):

    im = np.ones([field.shape[0], field.shape[1], 4])
    im = im_process(field, cmap=cmap_angle, vmin=omin, vmax=omax, alpha=alpha_ori)
    im = im_process(fiber, cmap=cmap_fiber, alpha=alpha_fiber, add_to=im)
    if cell is not None:
        im = im_process(cell, cmap=cmap_cell, alpha=alpha_cell, add_to=im)
    return im

    fig =plt.figure()
    plt.imshow(field,cmap=cmap_angle,vmin=omin,vmax=omax,alpha=alpha_ori);cbar =plt.colorbar()
    plt.imshow(fiber, cmap=cmap_fiber,alpha=alpha_fiber)
    if cell is not None:
        plt.imshow(cell, cmap=cmap_cell,alpha=alpha_cell)
    plt.axis('off');
    cbar.set_label(label,fontsize=12)
    if scale is not None:
        pass
        #scalebar = ScaleBar(scale, "um", length_fraction=0.1, location="lower right", box_alpha=0 ,
        #            color="k")
        #plt.gca().add_artist(scalebar)
    plt.tight_layout()
    #plt.savefig(path_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
    return fig

class DeformationDetector(PipelineModule):
    pipeline_name = "find deformations"
    use_thread = False
    progress_signal = QtCore.Signal(int, str)
    result: ResultOrientation = None

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "find orientations").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout() as layout:
                    with QtShortCuts.QHBoxLayout():
                        self.sigma_tensor = QtShortCuts.QInputString(None, "sigma_tensor", "7.0", type=float,
                                                                     settings=self.settings,
                                                                     settings_key="orientation/sigma_tensor")
                        self.sigma_tensor_type = QtShortCuts.QInputChoice(None, "", "um", ["um", "pixel"],
                                                                          settings=self.settings,
                                                                          settings_key="orientation/sigma_tensor_unit")
                        self.sigma_tensor_button = QtShortCuts.QPushButton(None, "detect",
                                                                           self.sigma_tensor_button_clicked)
                        self.sigma_tensor_button.setDisabled(True)
                    with QtShortCuts.QHBoxLayout():
                        self.edge = QtShortCuts.QInputString(None, "edge", "40", type=int, settings=self.settings,
                                                             allow_none=False,
                                                             settings_key="orientation/edge",
                                                             tooltip="How many pixels to cut at the edge of the image.")
                        QtWidgets.QLabel("px").addToLayout()
                        self.max_dist = QtShortCuts.QInputString(None, "max_dist", "None", type=int,
                                                                 settings=self.settings,
                                                                 settings_key="orientation/max_dist",
                                                                 tooltip="Optional: specify the maximal distance around the cell center",
                                                                 none_value=None)
                        QtWidgets.QLabel("px").addToLayout()

                    self.ignore_cell = QtShortCuts.QInputBool(None, "ignore_cell_outline", False, settings=self.settings,
                                                                 settings_key="orientation/ignore_cell_outline",)

                    with QtShortCuts.QHBoxLayout():
                        self.sigma_first_blur = QtShortCuts.QInputString(None, "sigma_first_blur", "0.5",
                                                                         type=float, settings=self.settings,
                                                                         settings_key="orientation/sigma_first_blur")
                        QtWidgets.QLabel("px").addToLayout()
                    with QtShortCuts.QHBoxLayout():
                        self.angle_sections = QtShortCuts.QInputString(None, "angle_sections", "5", type=int,
                                                                       settings=self.settings,
                                                                       settings_key="orientation/angle_sections")
                        QtWidgets.QLabel("deg").addToLayout()

                    with QtShortCuts.QHBoxLayout():
                        self.shell_width = QtShortCuts.QInputString(None, "shell_width", "5", type=float,
                                                                    settings=self.settings,
                                                                    settings_key="orientation/shell_width")
                        self.shell_width_type = QtShortCuts.QInputChoice(None, "", "um", ["um", "pixel"],
                                                                         settings=self.settings,
                                                                         settings_key="orientation/shell_width_type")

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

        self.setParameterMapping("orientation_parameters", {
            "sigma_tensor": self.sigma_tensor,
            "sigma_tensor_type": self.sigma_tensor_type,
            "edge": self.edge,
            "max_dist": self.max_dist,
            "ignore_cell_outline": self.ignore_cell,
            "sigma_first_blur": self.sigma_first_blur,
            "angle_sections": self.angle_sections,
            "shell_width": self.shell_width,
            "shell_width_type": self.shell_width_type,
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

        im_fiber = self.result.get_image(1)
        im_cell = self.result.get_image(0)

        print("orientation_map", self.result.orientation_map)
        if self.result.orientation_map is not None:
            print("plot oriengation")
            edge = self.result.orientation_parameters["edge"]
            im = plot_fancy_overlay(im_fiber[edge:-edge, edge:-edge], im_cell[edge:-edge, edge:-edge], self.result.orientation_map, omin=-1, omax=1)
            print(im.shape, im.dtype)
            im = (im*255).astype(np.uint8)
            print(im[:10,:10])
        else:
            print("plot fibers")
            im = im_fiber

        print(im.dtype, im.max())
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.label.setExtend(im.shape[1], im.shape[0])
        return None

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

    def process(self, result: ResultOrientation, orientation_parameters: OrientationParametersDict):
        segmentation_parameters = result.segmentation_parameters
        sigma_tensor = orientation_parameters["sigma_tensor"]
        if orientation_parameters["sigma_tensor_type"] == "um":
            sigma_tensor /= result.pixel_size
        shell_width = orientation_parameters["shell_width"]
        if orientation_parameters["shell_width_type"] == "um":
            shell_width /= result.pixel_size

        from CompactionAnalyzer.CompactionFunctions import StuctureAnalysisMain
        StuctureAnalysisMain(fiber_list=[result.image_fiber],
                             cell_list=[result.image_cell],
                             out_list=[result.output[:-len(".saenopyOrientation")]],
                             scale=result.pixel_size,
                             sigma_tensor=sigma_tensor,
                             edge=orientation_parameters["edge"],
                             max_dist=orientation_parameters["max_dist"],
                             ignore_cell_outline=orientation_parameters["ignore_cell_outline"],
                             segmention_thres=segmentation_parameters["thresh"],
                             seg_gaus1=segmentation_parameters["gauss1"],
                             seg_gaus2=segmentation_parameters["gauss2"],
                             seg_invert=segmentation_parameters["invert"],
                             sigma_first_blur=orientation_parameters["sigma_first_blur"],
                             angle_sections=orientation_parameters["angle_sections"],
                             shell_width=shell_width,
                             SaveNumpy=True,
                             )

        result.orientation_map = np.load(Path(result.output).parent / Path(result.output).stem / "NumpyArrays" / "OrientationMap.npy")

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

    def sigma_tensor_button_clicked(self):
        parent = self

        class SigmaRange(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Determine Sigma Tensor")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.output_folder = QtShortCuts.QInputFolder(None, "output folder", settings=settings,
                                                                  settings_key="orientation/sigma_tensor_range_output")
                    self.label_scale = QtWidgets.QLabel(f"Scale is {parent.scale.value()} px/um").addToLayout(layout)
                    with QtShortCuts.QHBoxLayout() as layout2:
                        self.sigma_tensor_min = QtShortCuts.QInputString(None, "min", "1.0", type=float,
                                                                         settings=settings,
                                                                         settings_key="orientation/sigma_tensor_min")
                        self.sigma_tensor_max = QtShortCuts.QInputString(None, "max", "15", type=float,
                                                                         settings=settings,
                                                                         settings_key="orientation/sigma_tensor_max")
                        self.sigma_tensor_step = QtShortCuts.QInputString(None, "step", "1", type=float,
                                                                          settings=settings,
                                                                          settings_key="orientation/sigma_tensor_step")
                        self.sigma_tensor_type = QtShortCuts.QInputChoice(None, "", "um", ["um", "pixel"],
                                                                          settings=settings,
                                                                          settings_key="orientation/sigma_tensor_unit")

                    self.progresss = QtWidgets.QProgressBar().addToLayout(layout)

                    self.canvas = MatplotlibWidget(self)
                    layout.addWidget(self.canvas)
                    layout.addWidget(NavigationToolbar(self.canvas, self))

                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList1 = QtShortCuts.QPushButton(None, "run", self.run)

            def run(self):
                from natsort import natsorted
                fiber, cell, output, attr = parent.data[parent.list.currentRow()][2]

                output_folder = self.output_folder.value()

                sigma_tensor_min = self.sigma_tensor_min.value()
                sigma_tensor_max = self.sigma_tensor_max.value()
                sigma_tensor_step = self.sigma_tensor_step.value()
                if self.sigma_tensor_type.value() == "um":
                    sigma_tensor_min /= parent.scale.value()
                    sigma_tensor_max /= parent.scale.value()
                    sigma_tensor_step /= parent.scale.value()
                shell_width = parent.shell_width.value()
                if parent.shell_width_type.value() == "um":
                    shell_width /= parent.scale.value()

                sigma_list = np.arange(sigma_tensor_min, sigma_tensor_max + sigma_tensor_step, sigma_tensor_step)
                self.progresss.setRange(0, len(sigma_list))

                from CompactionAnalyzer.CompactionFunctions import StuctureAnalysisMain, generate_lists
                for index, sigma in enumerate(sigma_list):
                    sigma = float(sigma)
                    self.progresss.setValue(index)
                    app.processEvents()
                    # Create outputfolder
                    output_sub = os.path.join(output_folder,
                                              rf"Sigma{str(sigma * parent.scale.value()).zfill(3)}")  # subpath to store results
                    fiber_list, cell_list, out_list = generate_lists(fiber, cell,
                                                                     output_main=output_sub)

                    StuctureAnalysisMain(fiber_list=fiber_list,
                                         cell_list=cell_list,
                                         out_list=out_list,
                                         scale=parent.scale.value(),
                                         sigma_tensor=sigma,
                                         edge=parent.edge.value(),
                                         max_dist=parent.max_dist.value(),
                                         segmention_thres=parent.segmention_thres.value() if attr[
                                                                                                 "segmention_thres"] is None else
                                         attr["segmention_thres"],
                                         seg_gaus1=parent.seg_gaus1.value() if attr["seg_gaus1"] is None else attr[
                                             "seg_gaus1"],
                                         seg_gaus2=parent.seg_gaus2.value() if attr["seg_gaus2"] is None else attr[
                                             "seg_gaus2"],
                                         sigma_first_blur=parent.sigma_first_blur.value(),
                                         angle_sections=parent.angle_sections.value(),
                                         shell_width=shell_width,
                                         regional_max_correction=True,
                                         seg_iter=1,
                                         SaveNumpy=False,
                                         plotting=True,
                                         dpi=100
                                         )
                    self.progresss.setValue(index + 1)

                    ### plot results
                    # read in all creates result folder
                    result_folders = natsorted(
                        glob.glob(os.path.join(output_folder, "Sigma*", "*", "results_total.xlsx")))

                    import yaml
                    sigmas = []
                    orientation = []
                    for folder in result_folders:
                        with (Path(folder).parent / "parameters.yml").open() as fp:
                            parameters = yaml.load(fp, Loader=yaml.SafeLoader)["Parameters"]
                            sigmas.append(parameters["sigma_tensor"][0] * parameters["scale"][0])
                        orientation.append(pd.read_excel(folder)["Orientation (weighted by intensity and coherency)"])

                    self.canvas.setActive()
                    plt.cla()
                    plt.axis("auto")

                    plt.plot(sigmas, orientation, "o-")
                    plt.ylabel("Orientation", fontsize=12)
                    plt.xlabel("Windowsize (μm)", fontsize=12)
                    plt.grid()
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, "Results.png"), dpi=500)
                    self.canvas.draw()

        dialog = SigmaRange(self)
        if not dialog.exec():
            return