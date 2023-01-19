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

        with QtShortCuts.QHBoxLayout(self) as layout0:
            layout0.setContentsMargins(0, 0, 0, 0)
            self.theme = QtShortCuts.QInputChoice(None, "Theme", value=self.theme_values[2],
                                                  values=self.theme_values,
                                                  value_names=["default", "paraview", "dark", "document"])

            self.auto_scale = QtShortCuts.QInputBool(None, "auto color", True, tooltip="Automatically choose the maximum for the color scale.")
            self.auto_scale.valueChanged.connect(self.scale_max_changed)
            self.scale_max = QtShortCuts.QInputString(None, "max color", 1e-6, type=float, tooltip="Set the maximum of the color scale.")
            self.scale_max.valueChanged.connect(self.scale_max_changed)
            self.use_nans = QtShortCuts.QInputBool(None, "nans", True, tooltip="Display nodes which do not have values associated as gray dots.")
            self.use_nans.valueChanged.connect(self.update_display)
            self.show_grid = QtShortCuts.QInputBool(None, "", True,
                                                     tooltip="Display a grid or a bounding box.", icon=[
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "grid.ico")),
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "grid2.ico")),
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "grid3.ico")),
                ])
            self.show_grid.valueChanged.connect(self.update_display)
            self.show_grid.valueChanged.connect(lambda value: shared_properties.change_property("show_grid", value, self))
            shared_properties.add_property("show_grid", self)


            self.show_image = QtShortCuts.QInputBool(None, "", True,
                                                   tooltip="Display the stack image in the stack or at the bottom.", icon=[
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "show_image3.ico")),
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "show_image.ico")),
                    QtGui.QIcon(str(Path(__file__).parent.parent / "img" / "show_image2.ico"))])
            self.show_image.valueChanged.connect(self.update_display)
            self.show_image.valueChanged.connect(
                lambda value: shared_properties.change_property("show_image", value, self))
            shared_properties.add_property("show_image", self)

            self.channel_select = QtShortCuts.QInputChoice(None, "", 0, [0], [""])
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

            if center is True:
                self.use_center = QtShortCuts.QInputBool(None, "center", True,
                                                       tooltip="Display the center of the force field.")
                self.use_center.valueChanged.connect(self.update_display)

            self.theme.valueChanged.connect(lambda x: self.new_plotter(x))

            layout0.addStretch()
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
            self.show_grid.setValue(value)
            self.update_display()
        if name == "show_image":
            self.show_image.setValue(value)
            self.update_display()
        if name == "channel_select":
            self.channel_select.setValue(value)
            self.update_display()
        if name == "button_z_proj":
            self.button_z_proj.setValue(value)
            self.update_display()

    def scale_max_changed(self):
        self.scale_max.setDisabled(self.auto_scale.value())
        scalebar_max = self.getScaleMax()
        print(scalebar_max, self.plotter.auto_value, type(self.plotter.auto_value))
        if scalebar_max is None:
            self.plotter.update_scalar_bar_range([0, self.plotter.auto_value])
        else:
            self.plotter.update_scalar_bar_range([0, scalebar_max])

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


