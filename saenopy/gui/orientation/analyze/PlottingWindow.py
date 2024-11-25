# Setting the Qt bindings for QtPy
import os

import pandas as pd
from qtpy import QtCore, QtWidgets, QtGui
from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.gui_classes import Spoiler, CheckAbleGroup, QHLine, QVLine, MatplotlibWidget, NavigationToolbar, execute, kill_thread, ListWidget
import matplotlib.pyplot as plt
import imageio
import glob
from pathlib import Path

settings = QtCore.QSettings("FabryLab", "CompactionAnalyzer")

class PlottingWindow(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int, int, int)
    finished_signal = QtCore.Signal()
    thread = None

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Evaluation")

        self.images = []
        self.data_folders = []
        self.current_plot_func = lambda: None

        with QtShortCuts.QHBoxLayout(self) as main_layout:
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

            with QtShortCuts.QGroupBox(main_layout, "Plot Forces") as (_, layout):
                self.type = QtShortCuts.QInputChoice(None, "type", "global orientation", ["global orientation", "normalized intensity (first shell)", "orientation over distance", "normed intensity over distance"])
                self.type.valueChanged.connect(self.replot)

                self.canvas = MatplotlibWidget(self)
                layout.addWidget(self.canvas)
                layout.addWidget(NavigationToolbar(self.canvas, self))

                with QtShortCuts.QHBoxLayout() as layout2:
                    self.button_export = QtShortCuts.QPushButton(layout2, "Export", self.export)
                    layout2.addStretch()
                    self.button_run = QtShortCuts.QPushButton(layout2, "Single Plot", self.run2)
                    self.button_run2 = QtShortCuts.QPushButton(layout2, "Grouped Plot", self.plot_groups)
                    self.plot_buttons = [self.button_run, self.button_run2]
                    for button in self.plot_buttons:
                        button.setCheckable(True)

        self.list.setData(self.data_folders)
        self.addGroup()
        self.current_plot_func = self.run2

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

    def addFiles(self):
        class AddFilesDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Add Files")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.label = QtWidgets.QLabel("Select a path as an input wildcard. Use * to specify a placeholder. All paths that match the wildcard will be added.")
                    layout.addWidget(self.label)
                    def checker(filename):
                        return filename + "/**/results_total.xlsx"
                    self.inputText = QtShortCuts.QInputFolder(None, None, settings=settings, filename_checker=checker,
                                                                settings_key="batch_eval/wildcard", allow_edit=True)
                    with QtShortCuts.QHBoxLayout() as layout3:
                        # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                        layout3.addStretch()
                        self.button_addList0 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                        self.button_addList1 = QtShortCuts.QPushButton(None, "ok", self.accept)

        dialog = AddFilesDialog(self)
        if not dialog.exec():
            return

        text = os.path.normpath(dialog.inputText.value())
        files = glob.glob(text, recursive=True)

        current_group = self.list2.data
        current_files = [d[0] for d in current_group]
        for file in files:
            if file in current_files:
                print("File already in list", file)
                continue
            if self.list2.data is current_group:
                data = {"results_total": pd.read_excel(file),
                        "results_distance": pd.read_excel(Path(file).parent / "results_distance.xlsx"),
                        "image": Path(file).parent / "Figures" / "overlay2.png"}
                self.list2.addData(file, True, data)

    def listSelected(self):
        try:
            data = self.data_folders[self.list.currentRow()]
        except IndexError:
            return
        self.update_group_name()
        self.list2.setData(data[2])

    def getAllCurrentPandasData(self, key, only_first_line=False):
        results = []
        for name, checked, files, color in self.data_folders:
            if checked != 0:
                for name2, checked2, res, color in files:
                    if checked2 != 0:
                        res[key]["group"] = name
                        if only_first_line is True:
                            results.append(res[key].iloc[0:1])
                        else:
                            results.append(res[key])
        res = pd.concat(results)
        res.to_csv("tmp_pandas.csv")
        return res

    def replot(self):
        if self.current_plot_func is not None:
            self.current_plot_func()

    def plot_groups(self):
        for button in self.plot_buttons:
            button.setChecked(False)
        self.button_run2.setChecked(True)
        self.current_plot_func = self.plot_groups

        self.button_export.setDisabled(False)

        self.canvas.setActive()
        plt.cla()
        plt.axis("auto")
        if self.type.value() == "global orientation":
            res = self.getAllCurrentPandasData("results_total")
            code_data = [res, ["group", 'Orientation (weighted by intensity and coherency)']]
            def plot(res, color_dict2):
                # define the colors
                color_dict = color_dict2

                # iterate over the groups
                for name, d in res.groupby("group")['Orientation (weighted by intensity and coherency)']:
                    plt.bar(name, d.mean(), yerr=d.sem(), color=color_dict[name])

                # add ticks and labels
                plt.ylabel("orientation")
                # despine the axes
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                plt.tight_layout()
                # show the plot
                self.canvas.draw()

        elif self.type.value() == "normalized intensity (first shell)":

            res = self.getAllCurrentPandasData("results_distance", only_first_line=True)
            code_data = [res, ["group", 'Intensity Norm (individual)']]

            def plot(res, color_dict2):
                # define the colors
                color_dict = color_dict2

                # iterate over the groups
                for name, d in res.groupby("group")['Intensity Norm (individual)']:
                    plt.bar(name, d.mean(), yerr=d.sem(), color=color_dict[name])

                # add ticks and labels
                plt.ylabel("normalized intensity")
                # despine the axes
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                plt.tight_layout()
                # show the plot
                self.canvas.draw()

        elif self.type.value() == "orientation over distance":
            res = self.getAllCurrentPandasData("results_distance")
            code_data = [res, ["group", "Shell_mid (µm)", "Orientation (individual)"]]

            def plot(res, color_dict2):
                # define the colors
                color_dict = color_dict2

                # iterate over the groups
                for name, d in res.groupby("group"):
                    d = d.groupby("Shell_mid (µm)")["Orientation (individual)"].agg(["mean", "sem"])
                    plt.plot(d.index,d["mean"], color=color_dict[name])
                    plt.fill_between(d.index, d["mean"]-d["sem"], d["mean"]+d["sem"], color=color_dict[name], alpha=0.5)

                # add ticks and labels
                plt.xlabel("shell mid (µm)")
                plt.ylabel("individual orientation")
                # despine the axes
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                plt.tight_layout()
                # show the plot
                self.canvas.draw()

        elif self.type.value() == "normed intensity over distance":
            res = self.getAllCurrentPandasData("results_distance")
            code_data = [res, ["group", "Shell_mid (µm)", 'Intensity Norm (individual)']]

            def plot(res, color_dict2):
                # define the colors
                color_dict = color_dict2

                # iterate over the groups
                for name, d in res.groupby("group"):
                    d = d.groupby("Shell_mid (µm)")['Intensity Norm (individual)'].agg(["mean", "sem"])
                    plt.plot(d.index, d["mean"], color=color_dict[name])
                    plt.fill_between(d.index, d["mean"] - d["sem"], d["mean"] + d["sem"], color=color_dict[name], alpha=0.5)

                # add ticks and labels
                plt.xlabel("shell mid (µm)")
                plt.ylabel("normalized intensity")
                # despine the axes
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                plt.tight_layout()
                # show the plot
                self.canvas.draw()

        self.canvas.setActive()
        plt.cla()

        color_dict = {d[0]: d[3] for d in self.data_folders}

        code = execute(plot, code_data[0][code_data[1]], color_dict2=color_dict)

        self.export_data = [code, code_data]
        return

    def run2(self):
        for button in self.plot_buttons:
            button.setChecked(False)
        self.button_run.setChecked(True)
        self.current_plot_func = self.run2

        self.button_export.setDisabled(True)

        data = self.list2.data[self.list2.currentRow()][2]
        im = imageio.v2.imread(data["image"])

        plot_color = self.list.data[self.list.currentRow()][3]

        self.canvas.setActive()
        plt.cla()
        plt.axis("auto")
        if self.type.value() == "global orientation":
            plt.imshow(im)
            plt.title(f"Orientation {data['results_total']['Orientation (weighted by intensity and coherency)'].iloc[0]:.3f}")
            plt.axis("off")
        elif self.type.value() == "normalized intensity (first shell)":
            plt.imshow(im)
            plt.title(f"Normalized Intensity {data['results_distance']['Intensity Norm (individual)'].iloc[0]:.3f}")
            plt.axis("off")
        elif self.type.value() == "normed intensity over distance":
            plt.plot(data["results_distance"]["Shell_mid (µm)"],
                     data["results_distance"]["Intensity Norm (individual)"], color=plot_color)
            plt.xlabel("shell mid (µm)")
            plt.ylabel("intensity norm")
        elif self.type.value() == "orientation over distance":
            plt.plot(data["results_distance"]["Shell_mid (µm)"],
                     data["results_distance"]["Orientation (individual)"], color=plot_color)
            plt.xlabel("shell mid (µm)")
            plt.ylabel("individual orientation")
        # despine the axes
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.tight_layout()
        self.canvas.draw()

    def export(self):
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