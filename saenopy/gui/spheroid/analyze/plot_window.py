import sys

# Setting the Qt bindings for QtPy
import os

import pandas as pd
import qtawesome as qta

os.environ["QT_API"] = "pyqt5"

from qtpy import QtCore, QtWidgets, QtGui

import numpy as np
from natsort import natsorted


from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import imageio
import threading
import glob


from matplotlib.figure import Figure
import jointforces as jf
import urllib
from pathlib import Path

import ctypes

from saenopy.gui.spheroid.modules.ListWidget import ListWidget
from saenopy.gui.spheroid.modules.MatplotlibWidget import MatplotlibWidget, NavigationToolbar
settings = QtCore.QSettings("Saenopy", "Saenopy")


class PlottingWindow(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int, int, int)
    finished_signal = QtCore.Signal()
    thread = None

    def __init__(self, parent=None):
        super().__init__(parent)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", "Saenopy")

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Evaluation")

        self.images = []
        self.data_folders = []
        self.current_plot_func = lambda: None

        with QtShortCuts.QVBoxLayout(self) as main_layout0:
         with QtShortCuts.QHBoxLayout() as main_layout00:
             self.button_save = QtShortCuts.QPushButton(None, "save", self.save)
             self.button_load = QtShortCuts.QPushButton(None, "load", self.load)
             main_layout00.addStretch()
         with QtShortCuts.QHBoxLayout() as main_layout:
            with QtShortCuts.QVBoxLayout() as layout:
                with QtShortCuts.QGroupBox(None, "Groups") as (_, layout2):
                    layout2.setContentsMargins(0, 3, 0, 1)
                    self.list = ListWidget(layout2, True, add_item_button="add group", color_picker=True)
                    self.list.setStyleSheet("QListWidget{border: none}")
                    self.list.itemSelectionChanged.connect(self.listSelected)
                    self.list.itemChanged.connect(self.replot)
                    self.list.itemChanged.connect(self.update_group_name)
                    self.list.addItemClicked.connect(self.addGroup)

                with QtShortCuts.QGroupBox(layout, "Group") as (self.box_group, layout2):
                    layout2.setContentsMargins(0, 3, 0, 1)
                    self.list2 = ListWidget(layout2, add_item_button="add files")
                    self.list2.setStyleSheet("QListWidget{border: none}")
                    self.list2.itemSelectionChanged.connect(self.run2)
                    self.list2.itemChanged.connect(self.replot)
                    self.list2.addItemClicked.connect(self.addFiles)

                    self.setAcceptDrops(True)

            with QtShortCuts.QGroupBox(main_layout, "Plot Forces") as (_, layout):
                self.type = QtShortCuts.QInputChoice(None, "type", "Pressure", ["Pressure", "Contractility"])
                self.type.valueChanged.connect(self.replot)
                self.dt = QtShortCuts.QInputString(None, "dt", 2, unit="min", type=float)
                self.dt.valueChanged.connect(self.replot)
                self.input_tbar = QtShortCuts.QInputString(None, "Comparison Time", 2, type=float)
                self.input_tbar_unit = QtShortCuts.QInputChoice(self.input_tbar.layout(), None, "min", ["steps", "min", "h"])
                self.input_tbar_unit.valueChanged.connect(self.replot)
                self.input_tbar.valueChanged.connect(self.replot)

                self.canvas = MatplotlibWidget(self)
                layout.addWidget(self.canvas)
                layout.addWidget(NavigationToolbar(self.canvas, self))

                with QtShortCuts.QHBoxLayout() as layout2:
                    self.button_export = QtShortCuts.QPushButton(layout2, "Export", self.export)
                    layout2.addStretch()
                    self.button_run = QtShortCuts.QPushButton(layout2, "Single Time Course", self.run2)
                    self.button_run2 = QtShortCuts.QPushButton(layout2, "Grouped Time Courses", self.plot_groups)
                    self.button_run3 = QtShortCuts.QPushButton(layout2, "Grouped Bar Plot", self.barplot)
                    self.plot_buttons = [self.button_run, self.button_run2, self.button_run3]
                    for button in self.plot_buttons:
                        button.setCheckable(True)

        self.list.setData(self.data_folders)
        self.addGroup()
        self.current_plot_func = self.run2

    def save(self):
        new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Session", os.getcwd(), "JSON File (*.json)")
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)
            list_new = []
            for item in self.list.data:
                list_new.append({"name": item[0], "selected": item[1], "color": item[3], "paths": []})
                for item2 in item[2]:
                    list_new[-1]["paths"].append({"path": item2[0], "selected": item[1]})
            import json
            with open(new_path, "w") as fp:
                json.dump(list_new, fp, indent=2)

    def load(self):
        new_path = QtWidgets.QFileDialog.getOpenFileName(None, "Save Session", os.getcwd(), "JSON File (*.json)")
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)
            import json
            with open(new_path, "r") as fp:
                list_new = json.load(fp)
            self.list.clear()
            self.list.setData([[i["name"], i["selected"], [], i["color"]] for i in list_new])
            self.data_folders = self.list.data

            for i, d in enumerate(list_new):
                self.list.setCurrentRow(i)
                self.list.listSelected()
                self.listSelected()
                self.list2.data = self.list.data[i][2]
                self.add_files([d_0["path"] for d_0 in d["paths"]])

                for ii, d_0 in enumerate(d["paths"]):
                    self.list2.data[ii][1] = d_0["selected"]

    def update_group_name(self):
        if self.list.currentItem() is not None:
            self.box_group.setTitle(f"Files for '{self.list.currentItem().text()}'")
            self.box_group.setEnabled(True)
        else:
            self.box_group.setEnabled(False)

    def addGroup(self):
        import matplotlib as mpl
        text = f"Group{1+len(self.data_folders)}"
        item = self.list.addData(text, True, [], mpl.colors.to_hex(f"C{len(self.data_folders)}"))
        self.list.setCurrentItem(item)
        self.list.editItem(item)


    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        # accept url lists (files by drag and drop)
        for url in event.mimeData().urls():
            # if str(url.toString()).strip().endswith(".npz"):
            event.accept()
            return
        event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QtCore.QEvent):
        urls = []
        for url in event.mimeData().urls():
            print(url)
            url = url.toLocalFile()
            if url[0] == "/" and url[2] == ":":
                url = url[1:]
            print(url)
            if url.endswith("result.xlsx"):
                urls += [url]
            else:
                urls += glob.glob(url + "/**/result.xlsx", recursive=True)
        self.add_files(urls)


    def addFiles(self):
        settings = self.settings
        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Add Files")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.label = QtWidgets.QLabel("Select a path as an input wildcard. Use * to specify a placeholder. All paths that match the wildcard will be added.")
                    layout.addWidget(self.label)
                    def checker(filename):
                        return filename + "/**/result.xlsx"
                    self.inputText = QtShortCuts.QInputFolder(None, None, settings=settings, filename_checker=checker,
                                                                settings_key="batch_eval/wildcard", allow_edit=True)
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList = QtShortCuts.QPushButton(None, "ok", self.accept)

        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        text = os.path.normpath(dialog.inputText.value())
        print(text)
        files = glob.glob(text, recursive=True)
        print(files)

        self.add_files(files)

    def add_files(self, files):
        current_group = self.list2.data
        current_files = [d[0] for d in current_group]
        for file in files:
            if file in current_files:
                #print("File already in list", file)
                continue
            try:
                #print("Add file", file)
                res = self.getPandasData(file)
                if self.list2.data is current_group:
                    self.list2.addData(file, True, res)
                    #print("replot")
                    self.replot()
                app.processEvents()
            except FileNotFoundError:
                continue

    def getPandasData(self, file):
        res = pd.read_excel(file)
        res["filename"] = file
        res["index"] = res["Unnamed: 0"]
        del res["Unnamed: 0"]
        res["group"] = file
        return res

    def listSelected(self):
        try:
            data = self.data_folders[self.list.currentRow()]
        except IndexError:
            return
        self.update_group_name()
        self.list2.setData(data[2])

    def getAllCurrentPandasData(self):
        results = []
        for name, checked, files, color in self.data_folders:
            if checked != 0:
                for name2, checked2, res, color in files:
                    if checked2 != 0:
                        res["group"] = name
                        results.append(res)
        res = pd.concat(results)
        res["t"] = res["index"] * self.dt.value() / 60
        res.to_csv("tmp_pandas.csv")
        return res

    def replot(self):
        if self.current_plot_func is not None:
            self.current_plot_func()

    def get_comparison_index(self):
        if self.input_tbar.value() is None:
            return None
        if self.input_tbar_unit.value() == "steps":
            index = int(np.floor(self.input_tbar.value() + 0.5))
        elif self.input_tbar_unit.value() == "min":
            index = int(np.floor(self.input_tbar.value() / self.dt.value() + 0.5))
        else:
            index = int(np.floor(self.input_tbar.value() * 60 / self.dt.value() + 0.5))
        return index

    def barplot(self):
        for button in self.plot_buttons:
            button.setChecked(False)
        self.button_run3.setChecked(True)
        self.current_plot_func = self.barplot
        self.canvas.setActive()
        plt.cla()
        if self.type.value() == "Contractility":
            mu_name = 'Mean Contractility (µN)'
            y_label = 'Contractility (µN)'
        else:
            mu_name = 'Mean Pressure (Pa)'
            y_label = 'Pressure (Pa)'

        # get all the data as a pandas dataframe
        res = self.getAllCurrentPandasData()

        # limit the dataframe to the comparison time
        index = self.get_comparison_index()
        res = res[res.index == index]

        code_data = [res, ["t", "group", mu_name, "filename"]]

        color_dict = {d[0]: d[3] for d in self.data_folders}

        def plot(res, mu_name, y_label, color_dict2):
            # define the colors
            color_dict = color_dict2

            # iterate over the groups
            for name, data in res.groupby("group", sort=False)[mu_name]:
                # add the bar with the mean value and the standard error as errorbar
                plt.bar(name, data.mean(), yerr=data.sem(), error_kw=dict(capsize=5), color=color_dict[name])
                # add the number of averaged points
                plt.text(name, data.mean() + data.sem(), f"n={data.count()}", ha="center", va="bottom")

            # add ticks and labels
            plt.ylabel(y_label)
            # despine the axes
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.tight_layout()
            # show the plot
            self.canvas.draw()

        code = execute(plot, code_data[0][code_data[1]], mu_name=mu_name, y_label=y_label, color_dict2=color_dict)

        self.export_data = [code, code_data]

    def plot_groups(self):
        for button in self.plot_buttons:
            button.setChecked(False)
        self.button_run2.setChecked(True)
        self.current_plot_func = self.plot_groups
        if self.type.value() == "Contractility":
            mu_name = 'Mean Contractility (µN)'
            std_name = 'St.dev. Contractility (µN)'
            y_label = 'Contractility (µN)'
        else:
            mu_name = 'Mean Pressure (Pa)'
            std_name = 'St.dev. Pressure (Pa)'
            y_label = 'Pressure (Pa)'

        self.canvas.setActive()
        plt.cla()
        res = self.getAllCurrentPandasData()

        code_data = [res, ["t", "group", mu_name, "filename"]]

        # add a vertical line where the comparison time is
        if self.input_tbar.value() is not None:
            comp_h = self.get_comparison_index() * self.dt.value() / 60
            plt.axvline(comp_h, color="k")

        color_dict = {d[0]: d[3] for d in self.data_folders}

        def plot(res, mu_name, y_label, color_dict2):
            # define the colors
            color_dict = color_dict2

            # iterate over the groups
            for group_name, data in res.groupby("group", sort=False):
                # get the mean and sem
                x = data.groupby("t")[mu_name].agg(["mean", "sem", "count"])
                # plot the mean curve
                p, = plt.plot(x.index, x["mean"], color=color_dict[group_name], lw=2, label=f"{group_name} (n={int(x['count'].mean())})")
                # add a shaded area for the standard error
                plt.fill_between(x.index, x["mean"] - x["sem"], x["mean"] + x["sem"], facecolor=p.get_color(), lw=0, alpha=0.5)

            # add a grid
            plt.grid(True)
            # add labels
            plt.xlabel('Time (h)')
            plt.ylabel(y_label)
            plt.tight_layout()
            plt.legend()

            # show
            self.canvas.draw()

        code = execute(plot, code_data[0][code_data[1]], mu_name=mu_name, y_label=y_label, color_dict2=color_dict)

        self.export_data = [code, code_data]
        return

    def run2(self):
        for button in self.plot_buttons:
            button.setChecked(False)
        self.button_run.setChecked(True)
        self.current_plot_func = self.run2
        if self.type.value() == "Contractility":
            mu_name = 'Mean Contractility (µN)'
            std_name = 'St.dev. Contractility (µN)'
            y_label = 'Contractility (µN)'
        else:
            mu_name = 'Mean Pressure (Pa)'
            std_name = 'St.dev. Pressure (Pa)'
            y_label = 'Pressure (Pa)'

        try:
            res = self.data_folders[self.list.currentRow()][2][self.list2.currentRow()][2]
        except IndexError:
            return

        #plt.figure(figsize=(6, 3))
        code_data = [res, ["t", mu_name, std_name]]

        res["t"] = res.index * self.dt.value() / 60

        self.canvas.setActive()
        plt.cla()

        def plot(res, mu_name, std_name, y_label, plot_color):
            mu = res[mu_name]
            std = res[std_name]

            # plot time course of mean values
            p, = plt.plot(res.t, mu, lw=2, color=plot_color)
            # add standard deviation area
            plt.fill_between(res.t, mu - std, mu + std, facecolor=p.get_color(), lw=0, alpha=0.5)

            # add grid
            plt.grid(True)
            # add labels
            plt.xlabel('Time (h)')
            plt.ylabel(y_label)
            plt.tight_layout()

            # show the plot
            self.canvas.draw()

        code = execute(plot, code_data[0][code_data[1]], mu_name=mu_name, std_name=std_name, y_label=y_label, plot_color=self.data_folders[self.list.currentRow()][3])

        self.export_data = [code, code_data]

    def export(self):
        settings = self.settings
        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Export Plot")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.label = QtWidgets.QLabel("Select a path to export the plot script with the data.")
                    layout.addWidget(self.label)
                    self.inputText = QtShortCuts.QInputFilename(None, None, file_type="Python Script (*.py)", settings=settings,
                                                                settings_key="batch_eval/export_plot", existing=False)
                    self.strip_data = QtShortCuts.QInputBool(None, "export only essential data columns", True, settings=settings, settings_key="batch_eval/export_complete_df")
                    self.include_df = QtShortCuts.QInputBool(None, "include dataframe in script", True, settings=settings, settings_key="batch_eval/export_include_df")
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList = QtShortCuts.QPushButton(None, "ok", self.accept)

        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        with open(str(dialog.inputText.value()), "wb") as fp:
            code = ""
            code += "import matplotlib.pyplot as plt\n"
            code += "import pandas as pd\n"
            code += "import io\n"
            code += "\n"
            code += "# the data for the plot\n"
            res, columns = self.export_data[1]
            if dialog.strip_data.value() is False:
                columns = None
            if dialog.include_df.value() is True:
                code += "csv_data = r'''" + res.to_csv(columns=columns) + "'''\n"
                code += "# load the data as a DataFrame\n"
                code += "res = pd.read_csv(io.StringIO(csv_data))\n\n"
            else:
                csv_file = str(dialog.inputText.value()).replace(".py", "_data.csv")
                res.to_csv(csv_file, columns=columns)
                code += "# load the data from file\n"
                code += f"res = pd.read_csv('{csv_file}')\n\n"
            code += self.export_data[0]
            fp.write(code.encode("utf8"))

