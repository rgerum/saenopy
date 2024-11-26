import numpy as np
import matplotlib.pyplot as plt
import jointforces as jf

from saenopy.gui.common.TabModule import TabModule
from saenopy.gui.orientation.modules.result import ResultOrientation
from qtpy import QtCore, QtWidgets, QtGui
from qimage2ndarray import array2qimage
from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView

from saenopy.gui.common.QTimeSlider import QTimeSlider
from saenopy.gui.common.resources import resource_icon
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

class TabOrientationField(TabModule):
    result: ResultOrientation = None

    def __init__(self, parent: "BatchEvaluate"):
        super().__init__(parent)

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

    def check_evaluated(self, result: ResultOrientation) -> bool:
        # whether the tab should be enabled to this Result object
        return True

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
