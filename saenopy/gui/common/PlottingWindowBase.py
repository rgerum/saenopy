import glob
import json
import os
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from qtpy import QtWidgets, QtCore, QtGui
import qtawesome as qta

from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.code_export import export_as_string
from saenopy.gui.common.gui_classes import ListWidget, MatplotlibWidget, execute


class AddFilesDialog(QtWidgets.QDialog):

    def __init__(self, parent, settings, file_extension):
        super().__init__(parent)
        self.setWindowTitle("Add Files")
        with QtShortCuts.QVBoxLayout(self) as layout:
            self.label = QtWidgets.QLabel(
                "Select a path as an input wildcard. Use * to specify a placeholder. All paths that match the wildcard will be added.")
            layout.addWidget(self.label)

            def checker(filename):
                return filename + "/**/*"+file_extension

            self.inputText = QtShortCuts.QInputFolder(None, None, settings=settings, filename_checker=checker,
                                                      settings_key="batch_eval/analyse_force_wildcard", allow_edit=True)
            with QtShortCuts.QHBoxLayout() as layout3:
                # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                layout3.addStretch()
                self.button_addList = QtShortCuts.QPushButton(None, "cancel", self.reject)
                self.button_addList = QtShortCuts.QPushButton(None, "ok", self.accept)


class ExportDialog(QtWidgets.QDialog):
    def __init__(self, parent, settings):
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


class PlottingWindowBase(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int, int, int, int)
    finished_signal = QtCore.Signal()
    thread = None
    time_key = "t"

    settings_key = "Seanopy_deformation"
    file_extension = ".saenopy"

    def add_parameters(self):
        pass

    def load_file(self, file):
        pass

    def get_label(self):
        return "", ""

    def get_copy_to_menu_item(self):
        result = self.list2.data[self.list2.currentRow()][2]
        return self.get_copy_to_menu(result.output, delete=True)

    def get_copy_to_menu(self, filename, delete=False):
        menu2 = QtWidgets.QMenu("Copy to Analysis" if delete is False else "Move to Group")
        menu2.setIcon(qta.icon("mdi.clipboard-arrow-right-outline"))
        self.copy_to_actions = []
        for index, folder in enumerate(self.data_folders):
            name, checked, files, color = folder
            act = QtWidgets.QAction(qta.icon("fa5s.circle", options=[dict(color=color)]), name, self)
            def clicked(*, index=index):
                if delete is True:
                    self.list2.delete_item()
                self.list.setCurrentRow(index)
                try:
                    self.add_files([filename])
                except Exception as e:
                    import traceback
                    traceback.print_exception(e)
                    print(e, file=sys.stderr)
                    QtWidgets.QMessageBox.critical(self, "Error", f"Measurement could not be added to Analysis.\n"
                                                                  f"Is it evaluated completely?")
                    return
                self.list2.setCurrentRow(self.list2.count()-2)
            act.triggered.connect(clicked)
            menu2.addAction(act)
            self.copy_to_actions.append(act)
        menu2.addSeparator()
        return menu2

    def __init__(self, parent=None, batch_evluate_instance=None):
        super().__init__(parent)

        self.batch_evluate_instance = batch_evluate_instance
        if batch_evluate_instance is not None:
            self.batch_evluate_instance.set_plot_window(self)

        # QSettings
        self.settings = QtCore.QSettings("Saenopy", self.settings_key)

        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setWindowTitle("Saenopy Evaluation")

        self.images = []
        self.data_folders = []
        self.current_plot_func = lambda: None

        with QtShortCuts.QVBoxLayout(self) as main_layout0:
         main_layout0.setContentsMargins(0, 0, 0, 0)
         with QtShortCuts.QHBoxLayout() as main_layout00:
             self.button_save = QtShortCuts.QPushButton(None, "save", lambda x: self.save())
             self.button_load = QtShortCuts.QPushButton(None, "load", lambda x: self.load())
             main_layout00.addStretch()
         with QtShortCuts.QHBoxLayout() as main_layout:
            with QtShortCuts.QVBoxLayout() as layout:
                with QtShortCuts.QGroupBox(None, "Groups") as (_, layout2):
                    layout2.setContentsMargins(0, 3, 0, 1)
                    self.list = ListWidget(layout2, True, add_item_button="add group", color_picker=True)
                    self.list.setStyleSheet("QListWidget{border: none}")
                    self.list.itemSelectionChanged.connect(self.listSelected)
                    self.list.itemChanged.connect(self.replot)
                    self.list.rows_moved_event.connect(self.replot)
                    self.list.itemChanged.connect(self.update_group_name)
                    self.list.addItemClicked.connect(self.addGroup)

                with QtShortCuts.QGroupBox(layout, "Group") as (self.box_group, layout2):
                    layout2.setContentsMargins(0, 3, 0, 1)
                    self.list2 = ListWidget(layout2, add_item_button="add files", copy_to_callback=self.get_copy_to_menu_item)
                    self.list2.setStyleSheet("QListWidget{border: none}")
                    self.list2.itemSelectionChanged.connect(self.run2)
                    self.list2.itemChanged.connect(self.replot)
                    self.list2.addItemClicked.connect(self.addFiles)

                    self.setAcceptDrops(True)

            with QtShortCuts.QGroupBox(main_layout, "Plot Forces") as (_, layout):
                self.add_parameters()

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

    def save(self, filename=None):
        if filename is None:
            filename = QtWidgets.QFileDialog.getSaveFileName(None, "Save Session", os.getcwd(), "JSON File (*.json)")[0]
        if filename:
            if not filename.endswith(".json"):
                filename += ".json"
            list_new = []
            for item in self.list.data:
                list_new.append({"name": item[0], "selected": item[1], "color": item[3], "paths": []})
                for item2 in item[2]:
                    list_new[-1]["paths"].append({"path": item2[0], "selected": item[1]})

            with open(filename, "w") as fp:
                json.dump(list_new, fp, indent=2)

    def load(self, filename=None):
        if filename is None:
            filename = QtWidgets.QFileDialog.getOpenFileName(None, "Load Session", os.getcwd(), "JSON File (*.json)")[0]
        if filename:
            with open(filename, "r") as fp:
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
            url = url.toLocalFile()
            if url[0] == "/" and url[2] == ":":
                url = url[1:]
            if url.endswith(self.file_extension):
                urls += [url]
            else:
                urls += glob.glob(url + "/**/*"+self.file_extension, recursive=True)
        self.add_files(urls)

    def add_files(self, urls):
        current_group = self.list2.data
        current_files = [d[0] for d in current_group]
        for file in urls:
            if file in current_files:
                print("File already in list", file)
                continue
            try:
                print("Add file", file)
                res = self.load_file(file)
                if res is None:
                    continue
                if self.list2.data is current_group:
                    self.list2.addData(file, True, res)
                    #self.replot()
                #app.processEvents()
            except FileNotFoundError:
                continue
        self.check_results_with_time()

    def check_results_with_time(self):
        time_values = False
        for name, checked, files, color in self.data_folders:
            if checked != 0:
                for name2, checked2, res, color in files:
                    data = res.get_data_structure()
                    if data["time_delta"] is not None:
                        time_values = True
        if time_values is False:
            self.barplot()
        if getattr(self, "agg", None) is not None:
            self.agg.setEnabled(time_values)
        self.button_run.setEnabled(time_values)
        self.button_run2.setEnabled(time_values)


    def update_group_name(self):
        if self.list.currentItem() is not None:
            self.box_group.setTitle(f"Files for '{self.list.currentItem().text()}'")
            self.box_group.setEnabled(True)
        else:
            self.box_group.setEnabled(False)

    def addGroup(self):
        text = f"Group{1+len(self.data_folders)}"
        item = self.list.addData(text, True, [], mpl.colors.to_hex(f"C{len(self.data_folders)}"))
        self.list.setCurrentItem(item)
        self.list.editItem(item)

    def addFiles(self):
        dialog = AddFilesDialog(self, self.settings, self.file_extension)
        if not dialog.exec():
            return

        text = os.path.normpath(dialog.inputText.value())
        files = glob.glob(text, recursive=True)

        self.add_files(files)

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
                        res.resulting_data["group"] = name
                        results.append(res.resulting_data)
        res = pd.concat(results)
        #res["t"] = res["index"] * self.dt.value() / 60
        res.to_csv("tmp_pandas.csv")
        return res

    def replot(self):
        self.check_results_with_time()
        if self.current_plot_func is not None:
            self.current_plot_func()

    def barplot(self):
        for button in self.plot_buttons:
            button.setChecked(False)
        self.button_run3.setChecked(True)
        self.current_plot_func = self.barplot
        self.canvas.setActive()
        plt.cla()
        mu_name, y_label = self.get_label()

        # get all the data as a pandas dataframe
        res = self.getAllCurrentPandasData()

        if getattr(self, "agg", None) is not None:
            # limit the dataframe to the comparison time
            res0 = res.groupby("filename", sort=False).agg("max")
            del res["group"]
            res = res.groupby("filename", sort=False).agg(self.agg.value())
            res["group"] = res0["group"]
        else:
            # limit the dataframe to the comparison time
            index = self.get_comparison_index()
            res = res[res.index == index]

        code_data = [res, ["group", mu_name]]

        color_dict = {d[0]: d[3] for d in self.data_folders}

        @export_as_string
        def plot(res, mu_name, y_label, color_dict2):
            # define the colors
            color_dict = color_dict2

            # iterate over the groups
            for index, (name, data) in enumerate(res.groupby("group", sort=False)[mu_name]):
                # add the bar with the mean value and the standard error as errorbar
                if np.isnan(data.sem()):
                    plt.bar(name, data.mean(), color=color_dict[name])
                else:
                    plt.bar(name, data.mean(), yerr=data.sem(), error_kw=dict(capsize=5), color=color_dict[name])
                # add the number of averaged points
                plt.text(index, float(data.mean() + np.nan_to_num(data.sem())), f"n={data.count()}", ha="center", va="bottom")

            # add ticks and labels
            plt.ylabel(y_label)
            # despine the axes
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            #plt.tight_layout()
            # show the plot
            self.canvas.draw()

        code = execute(plot, code_data[0][code_data[1]], mu_name=mu_name, y_label=y_label, color_dict2=color_dict)

        self.export_data = [code, code_data]
    
        
    def get_time_factor(self, maxtime): 
        factor = 1 
        x_label = "Time (s)"
        
        if maxtime > 2*60:
            factor = 60
            x_label = "Time (min)"
        if maxtime > 2*60*60:  
            factor = 60*60
            x_label = "Time (h)"
        if maxtime > 3*60*60*24:  
            factor = 60*60*24
            x_label = "Time (days)"
            
        return factor, x_label
        
   
    
    def plot_groups(self):
        for button in self.plot_buttons:
            button.setChecked(False)
        self.button_run2.setChecked(True)
        self.current_plot_func = self.plot_groups
        mu_name, y_label = self.get_label()

        self.canvas.setActive()
        plt.cla()
        res = self.getAllCurrentPandasData()

        code_data = [res, [self.time_key, "group", mu_name, "filename"]]
        
        # get best fitting time label
        factor, x_label = self.get_time_factor(np.max(res.t))
        
        
        # add a vertical line where the comparison time is
        if getattr(self, "input_tbar", None) and self.input_tbar.value() is not None:
            comp_h = self.get_comparison_index() * (res.iloc[1][self.time_key] - res.iloc[0][self.time_key])
            plt.axvline(comp_h/factor, color="k")

        color_dict = {d[0]: d[3] for d in self.data_folders}

        @export_as_string
        def plot(res, mu_name, y_label, color_dict2, x_label, factor):
            # define the colors
            color_dict = color_dict2

            # iterate over the groups
            for group_name, data in res.groupby("group", sort=False):
                # get the mean and sem
                x = data.groupby(self.time_key)[mu_name].agg(["mean", "sem", "count"])
                # plot the mean curve
                p, = plt.plot(x.index/factor, x["mean"], color=color_dict[group_name], lw=2, label=f"{group_name} (n={int(x['count'].mean())})")
                # add a shaded area for the standard error
                plt.fill_between(x.index/factor, x["mean"] - x["sem"], x["mean"] + x["sem"], facecolor=p.get_color(), lw=0, alpha=0.5)
                
            # add a grid
            plt.grid(True)
            # add labels
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.tight_layout()
            plt.legend()

            # show
            self.canvas.draw()

        code = execute(plot, code_data[0][code_data[1]], mu_name=mu_name, y_label=y_label, color_dict2=color_dict, factor=factor, x_label=x_label)

        self.export_data = [code, code_data]
        return

    def run2(self):
        if not self.button_run.isEnabled():
            return
        for button in self.plot_buttons:
            button.setChecked(False)
        self.button_run.setChecked(True)
        #return
        self.current_plot_func = self.run2
        mu_name, y_label = self.get_label()
        if 0:
            if self.type.value() == "Contractility":
                mu_name = 'Mean Contractility (µN)'
                std_name = 'St.dev. Contractility (µN)'
                y_label = 'Contractility (µN)'
            else:
                mu_name = 'Mean Pressure (Pa)'
                std_name = 'St.dev. Pressure (Pa)'
                y_label = 'Pressure (Pa)'

        try:
            res = self.data_folders[self.list.currentRow()][2][self.list2.currentRow()][2].resulting_data
        except IndexError:
            return

        #plt.figure(figsize=(6, 3))
        code_data = [res, [self.time_key, mu_name]]


        self.canvas.setActive()
        plt.cla()
        
                       
        # get best fitting time label
        factor2, x_label2 = self.get_time_factor(np.max(res.t))
        print (factor2)

        @export_as_string
        def plot(res, mu_name, y_label, plot_color, factor2, x_label2):
            mu = res[mu_name]

            # plot time course of mean values
            p, = plt.plot(res.t/factor2, mu, lw=2, color=plot_color)

            # add grid
            plt.grid(True)
            # add labels
            plt.xlabel(x_label2)
            plt.ylabel(y_label)
            plt.tight_layout()

            # show the plot
            self.canvas.draw()

        code = execute(plot, code_data[0][code_data[1]], mu_name=mu_name, y_label=y_label, plot_color=self.data_folders[self.list.currentRow()][3], factor2=factor2, x_label2=x_label2)

        self.export_data = [code, code_data]

    def export(self):
        dialog = ExportDialog(self, self.settings)
        if not dialog.exec():
            return

        filename = str(dialog.inputText.value())
        if not filename.endswith(".py"):
            filename += ".py"

        with open(filename, "wb") as fp:
            code = ""
            code += "import matplotlib.pyplot as plt\n"
            code += "import numpy as np\n"
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
