import os
from pathlib import Path
import qtawesome as qta
from qtpy import QtWidgets, QtGui
import pyvista as pv
from pyvistaqt import QtInteractor
from saenopy.gui import QtShortCuts
from .ResultView import result_view


vtk_toolbars = []
class VTK_Toolbar(QtWidgets.QWidget):
    theme_values = [pv.themes.DefaultTheme(), pv.themes.ParaViewTheme(),
                                                          pv.themes.DarkTheme(), pv.themes.DocumentTheme()]
    def __init__(self, plotter, update_display, scalbar_type="deformation", center=False, z_slider=None, channels=None, shared_properties=None):
        super().__init__()
        self.plotter = plotter
        self.update_display = update_display
        self.z_slider = z_slider
        vtk_toolbars.append(self)

        self.is_force_plot = center

        with QtShortCuts.QHBoxLayout(self) as layout0:
            layout0.setContentsMargins(0, 0, 0, 0)
            self.theme = QtShortCuts.QInputChoice(None, "Theme", value=self.theme_values[2],
                                                  values=self.theme_values,
                                                  value_names=["default", "paraview", "dark", "document"],
                                                  tooltip="Set a color theme for the 3D view.")

            self.auto_scale = QtShortCuts.QInputBool(None, "", icon=[
                QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "autoscale0.ico")),
                QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "autoscale1.ico")),
            ], group=False, tooltip="Automatically choose the maximum for the color scale.")
            #self.auto_scale = QtShortCuts.QInputBool(None, "auto color", True, tooltip="Automatically choose the maximum for the color scale.")
            self.auto_scale.setValue(True)
            self.auto_scale.valueChanged.connect(self.scale_max_changed)
            self.scale_max = QtShortCuts.QInputString(None, "", 1000 if self.is_force_plot else 10, type=float, tooltip="Set the maximum of the color scale.")
            self.scale_max.valueChanged.connect(self.scale_max_changed)
            self.scale_max.setDisabled(self.auto_scale.value())
            if self.is_force_plot is True:
                self.auto_scale.valueChanged.connect(
                    lambda value: shared_properties.change_property("auto_scale_force", value, self))
                shared_properties.add_property("auto_scale_force", self)
                self.scale_max.valueChanged.connect(lambda value: shared_properties.change_property("scale_max_force", value, self))
                shared_properties.add_property("scale_max_force", self)
            else:
                self.auto_scale.valueChanged.connect(
                    lambda value: shared_properties.change_property("auto_scale", value, self))
                shared_properties.add_property("auto_scale", self)
                self.scale_max.valueChanged.connect(
                    lambda value: shared_properties.change_property("scale_max", value, self))
                shared_properties.add_property("scale_max", self)

            self.window_scale = QtWidgets.QWidget()
            self.window_scale.setWindowTitle("Saenopy - Arrow Scale")
            with QtShortCuts.QVBoxLayout(self.window_scale):
                self.arrow_scale = QtShortCuts.QInputNumber(None, "arrow scale", 1, 0.1, 10, use_slider=True, log_slider=True)
                self.arrow_scale.valueChanged.connect(self.update_display)
                addition = ""
                if self.is_force_plot:
                    addition = "_force"
                self.arrow_scale.valueChanged.connect(
                    lambda value: shared_properties.change_property("arrow_scale"+addition, value, self))
                shared_properties.add_property("arrow_scale"+addition, self)

                self.colormap_chooser = QtShortCuts.QDragableColor("turbo").addToLayout()
                self.colormap_chooser.valueChanged.connect(self.update_display)

                self.colormap_chooser.valueChanged.connect(
                    lambda value: shared_properties.change_property("colormap_chooser"+addition, value, self))
                shared_properties.add_property("colormap_chooser"+addition, self)
            self.button_arrow_scale = QtShortCuts.QPushButton(None, "", lambda x: self.window_scale.show())
            self.button_arrow_scale.setIcon(QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "arrowscale.ico")))

            self.use_nans = QtShortCuts.QInputBool(None, "", icon=[
                QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "nan0.ico")),
                QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "nan1.ico")),
            ], group=False, tooltip="Display nodes which do not have values associated as gray dots.")

            self.use_nans.valueChanged.connect(self.update_display)
            self.show_grid = QtShortCuts.QInputBool(None, "", True,
                                                     tooltip="Display a grid or a bounding box.", icon=[
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "grid.ico")),
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "grid2.ico")),
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "grid3.ico")),
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "grid3.ico")),
                ])
            self.show_grid.valueChanged.connect(self.update_display)
            self.show_grid.valueChanged.connect(lambda value: shared_properties.change_property("show_grid", value, self))
            shared_properties.add_property("show_grid", self)


            if center is True:
                self.use_center = QtShortCuts.QInputBool(None, "", icon=[
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "center0.ico")),
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "center1.ico")),
                ], group=False, tooltip="Display the center of the force field.")
                self.use_center.valueChanged.connect(self.update_display)

            layout0.addStretch()

            self.show_image = QtShortCuts.QInputBool(None, "", True,
                                                   tooltip="Display the stack image in the stack or at the bottom.", icon=[
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "show_image3.ico")),
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "show_image.ico")),
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "show_image2.ico")),
                ])
            self.show_image.valueChanged.connect(self.update_display)
            self.show_image.valueChanged.connect(
                lambda value: shared_properties.change_property("show_image", value, self))
            shared_properties.add_property("show_image", self)

            self.channel_select = QtShortCuts.QInputChoice(None, "", 0, [0], ["       "], tooltip="From which channel to display the stack image.")
            self.channel_select.valueChanged.connect(self.update_display)
            self.channel_select.valueChanged.connect(
                lambda value: shared_properties.change_property("channel_select", value, self))
            shared_properties.add_property("channel_select", self)

            self.button_z_proj = QtShortCuts.QInputBool(None, "", icon=[
                QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "slice0.ico")),
                QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "slice1.ico")),
                QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "slice2.ico")),
                QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "slice_all.ico")),
            ], group=False, tooltip=["Show only the current z slice",
                                     "Show a maximum intensity projection over +-5 z slices",
                                     "Show a maximum intensity projection over +-10 z slices",
                                     "Show a maximum intensity projection over all z slices"])
            self.button_z_proj.valueChanged.connect(self.update_display)
            #self.button_z_proj.valueChanged.connect(lambda value: self.setZProj([0, 5, 10, 1000][value]))
            self.button_z_proj.valueChanged.connect(
                lambda value: shared_properties.change_property("button_z_proj", value, self))
            shared_properties.add_property("button_z_proj", self)

            self.contrast_enhance = QtShortCuts.QInputBool(None, "", icon=[
                QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "contrast0.ico")),
                QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "contrast1.ico")),
            ], group=False, tooltip="Toggle contrast enhancement")
            self.contrast_enhance.valueChanged.connect(self.update_display)
            self.contrast_enhance.valueChanged.connect(
                lambda value: shared_properties.change_property("contrast_enhance", value, self))
            shared_properties.add_property("contrast_enhance", self)

            QtShortCuts.QVLine()

            self.theme.valueChanged.connect(lambda x: self.new_plotter(x))

            self.button = QtWidgets.QPushButton(qta.icon("fa5s.home"), "").addToLayout()
            self.button.setToolTip("reset view")
            self.button.clicked.connect(lambda x: self.plotter.isometric_view())

            def save():
                if 1:
                    new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Images", os.getcwd())
                    # if we got one, set it
                    if new_path:
                        if isinstance(new_path, tuple):
                            new_path = new_path[0]
                        else:
                            new_path = str(new_path)
                        print(new_path)
                        self.plotter.screenshot(new_path)
                    return
                outer_self = self

                class PlotDialog(QtWidgets.QDialog):
                    def __init__(self, parent):
                        super().__init__(parent)
                        with QtShortCuts.QVBoxLayout(self) as layout:
                            self.plotter = QtInteractor(self, theme=outer_self.plotter.theme, auto_update=False)
                            layout.addWidget(self.plotter)
                            outer_self.update_display(self.plotter)
                            #showVectorField(self.plotter, outer_self.result.mesh_piv, "U_measured")
                            self.button2 = QtWidgets.QPushButton(qta.icon("mdi.floppy"), "").addToLayout()
                            self.button2.setToolTip("save")
                            self.button2.clicked.connect(self.save)

                    def save(self):
                        new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Images", os.getcwd())
                        # if we got one, set it
                        if new_path:
                            if isinstance(new_path, tuple):
                                new_path = new_path[0]
                            else:
                                new_path = str(new_path)
                            print(new_path)
                            self.plotter.screenshot(new_path)
                        self.plotter.close()
                        self.close()

                    def close(self):
                        self.plotter.close()

                plot_diaolog = PlotDialog(self)
                plot_diaolog.show()

            self.button2 = QtWidgets.QPushButton(qta.icon("mdi.floppy"), "").addToLayout()
            self.button2.setToolTip("save")
            self.button2.clicked.connect(save)

    def property_changed(self, name, value):
        if name == "show_grid":
            if value != self.show_grid.value():
                self.show_grid.setValue(value)
                self.update_display()
        if name == "show_image":
            if value != self.show_image.value():
                self.show_image.setValue(value)
                self.update_display()
        if name == "channel_select":
            if value != self.channel_select.value():
                if value < len(self.channel_select.value_names):
                    self.channel_select.setValue(value)
                else:
                    self.channel_select.setValue(0)
                self.update_display()
        if name == "button_z_proj":
            if value != self.button_z_proj.value():
                self.button_z_proj.setValue(value)
                self.update_display()
        if name == "contrast_enhance":
            if value != self.contrast_enhance.value():
                self.contrast_enhance.setValue(value)
                self.update_display()

        addition = ""
        if self.is_force_plot:
            addition = "_force"

        if name == "auto_scale"+addition:
            if value != self.auto_scale.value():
                self.auto_scale.setValue(value)
                self.scale_max.setDisabled(self.auto_scale.value())
                self.update_display()
        if name == "scale_max"+addition:
            if value != self.scale_max.value():
                self.scale_max.setValue(value)
                self.update_display()
        if name == "colormap_chooser"+addition:
            if value != self.colormap_chooser.value():
                self.colormap_chooser.setValue(value)
                self.update_display()
        if name == "arrow_scale"+addition:
            if value != self.arrow_scale.value():
                self.arrow_scale.setValue(value)
                self.update_display()

    def scale_max_changed(self):
        self.scale_max.setDisabled(self.auto_scale.value())
        scalebar_max = self.getScaleMax()
        print(scalebar_max, self.plotter.auto_value, type(self.plotter.auto_value))
        if scalebar_max is None:
            self.plotter.update_scalar_bar_range([0, self.plotter.auto_value])
        else:
            self.plotter.update_scalar_bar_range([0, scalebar_max])
        self.update_display()

    def getScaleMax(self):
        if self.auto_scale.value():
            return None
        return self.scale_max.value()

    def new_plotter(self, x, no_recursion=False):
        if self.plotter.theme == x:
            return
        if no_recursion is False:
            for widget in vtk_toolbars:
                if widget is not self:
                    widget.theme.setValue(x)
                    widget.new_plotter(x, no_recursion=True)
        self.plotter.theme = x
        self.plotter.set_background(self.plotter._theme.background)
        self.update_display()


