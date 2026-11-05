import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qtpy import QtWidgets, QtCore

from saenopy.gui.common import QtShortCuts

from saenopy.gui.common.PlottingWindowBase import PlottingWindowBase
from saenopy.gui.orientation.modules.result import ResultOrientation
from saenopy.gui.common.code_export import export_as_string
from saenopy.gui.common.gui_classes import execute


class PlottingWindow(PlottingWindowBase):
    settings_key = "saenopy_orientation"
    file_extension = ".saenopyOrientation"
    point_distance_metrics = {"point alignment over distance", "point angular deviation over distance",
                              "point coherency over distance"}
    # Keep saved/exported column names stable while using clearer GUI labels.
    metrics = {
        "global orientation": ("Cell-relative alignment", "Orientation", "cell"),
        "mean cell-relative deviation": ("Angular deviation to cell centre", "Mean Angle", "cell"),
        "normalized intensity (first shell)": ("Normalized intensity (first shell)", "normalized intensity (first shell)", "cell"),
        "orientation over distance": ("Cell-relative alignment over distance", "Orientation (individual)", "cell"),
        "normed intensity over distance": ("Normalized intensity over distance", "Intensity Norm (individual)", "cell"),
        "alignment x": ("Alignment to X axis", "Alignment x", "axis"),
        "alignment y": ("Alignment to Y axis", "Alignment y", "axis"),
        "mean absolute deviation x": ("Angular deviation to X axis", "Mean absolute deviation x", "axis"),
        "mean absolute deviation y": ("Angular deviation to Y axis", "Mean absolute deviation y", "axis"),
        "angle histogram x": ("Angle histogram: X axis", None, "axis"),
        "angle histogram y": ("Angle histogram: Y axis", None, "axis"),
        "point alignment": ("Alignment towards point", "Alignment point", "point"),
        "point angular deviation": ("Angular deviation to point", "Mean angle point", "point"),
        "point alignment over distance": ("Point alignment over distance", "Point alignment (shell)", "point"),
        "point angular deviation over distance": ("Angular deviation to point over distance", "Point angular deviation (shell)", "point"),
        "point coherency over distance": ("Coherency over distance from point", "Point coherency (shell)", "point"),
        "line alignment": ("Alignment to reference line", "Alignment line", "line"),
        "line angular deviation": ("Angular deviation to reference line", "Mean deviation line", "line"),
        "angle histogram line": ("Angle histogram: reference line", None, "line"),
        "coherency": ("Coherency", "Mean Coherency", "all"),
    }

    def add_parameters(self):
        self.type = QtShortCuts.QInputChoice(None, "Measurement", "global orientation",
            list(self.metrics), value_names=[spec[0] for spec in self.metrics.values()])
        self.type.valueChanged.connect(self.replot)
        self.agg = QtShortCuts.QInputChoice(None, "aggregate", "mean",
                                            ["mean", "max", "min", "median"])
        self.agg.valueChanged.connect(self.replot)
        self.distance_unit = QtShortCuts.QInputChoice(None, "Distance unit", "µm", ["µm", "pixel"])
        self.distance_unit.valueChanged.connect(self.replot)
        self.distance_unit.setVisible(False)
        self.metric_note = QtWidgets.QLabel("Select a metric for the cell or axis measurements in your groups.").addToLayout()
        self.metric_note.setWordWrap(True)

    def __init__(self, parent=None, batch_evluate_instance=None):
        super().__init__(parent, batch_evluate_instance)
        self.button_run.setText("Single Distance Course")
        self.button_run2.setText("Grouped Distance Courses")
        self.type.parentWidget().setTitle("Orientation statistics")
        self._make_compact_layout()

    def _make_compact_layout(self):
        """Let groups and plots resize without the wide base-layout button row."""
        self.setMinimumSize(0, 0)
        self.canvas.figure.patch.set_facecolor("white")
        content = self.layout().itemAt(1).layout()
        groups_layout = content.takeAt(0).layout()
        groups_layout.setParent(None)
        groups = QtWidgets.QWidget()
        groups.setLayout(groups_layout)
        plot = content.takeAt(0).widget()
        self.group_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        content.addWidget(self.group_splitter)
        self.group_splitter.addWidget(groups)
        self.group_splitter.addWidget(plot)
        self.group_splitter.setCollapsible(1, False)
        self.group_splitter.setStretchFactor(1, 1)
        self.group_splitter.setSizes([220, 650])
        plot_layout = plot.layout()
        for i in range(plot_layout.count()):
            row = plot_layout.itemAt(i).layout()
            if row is not None and row.indexOf(self.button_export) >= 0:
                plot_layout.takeAt(i)
                row.setParent(None)
                for button in (self.button_export, *self.plot_buttons):
                    row.removeWidget(button)
                row.deleteLater()
                break
        buttons = QtWidgets.QGridLayout()
        plot_layout.addLayout(buttons)
        for i, button in enumerate((self.button_run, self.button_run2, self.button_run3, self.button_export)):
            buttons.addWidget(button, i // 2, i % 2)

    def get_time_factor(self, maxtime):
        return 1, "Distance (µm)"

    def update_metric_availability(self):
        selected = [result for _, checked, files, _ in self.data_folders if checked
                    for _, included, result, _ in files if included]
        available = []
        for index, (key, (title, column, family)) in enumerate(self.metrics.items()):
            enabled = False
            for result in selected:
                if family in ("cell", "point", "line") and result.analysis_mode != {
                        "cell": "full", "point": "point_orientation", "line": "line_orientation"}[family]:
                    continue
                if key.startswith("angle histogram"):
                    histogram = self.histogram_for(result, key)
                    if histogram is not None:
                        count = histogram[:, 2].sum()
                        enabled = np.isfinite(count) and count > 0
                else:
                    data = getattr(result, "resulting_distance_data", pd.DataFrame()) if key in self.point_distance_metrics else result.resulting_data
                    if column in data:
                        enabled = np.isfinite(pd.to_numeric(data[column], errors="coerce")).any()
                if enabled:
                    break
            item = self.type.combobox.model().item(index)
            item.setEnabled(bool(enabled))
            scope = {"cell": "Cell-relative measurements", "axis": "X/Y statistics from any measurement mode", "point": "Point-relative measurements",
                     "line": "Line-relative measurements; each image uses its own selected line",
                     "all": "All measurement modes; mean local coherency, independent of reference direction"}[family]
            item.setToolTip(scope + ("." if enabled else ": no values in the selected measurements."))
            if enabled:
                available.append(key)
        self.type.combobox.setEnabled(bool(available))
        if available and self.type.value() not in available:
            self.type.setValue(available[0])
        return available

    def check_results_with_time(self):
        available = self.update_metric_availability()
        point_distance = self.type.value() in self.point_distance_metrics
        self.distance_unit.setVisible(point_distance)
        self.button_export.setEnabled(bool(available))
        self.button_run3.setEnabled(bool(available))
        if not available:
            self.button_run.setEnabled(False)
            self.button_run2.setEnabled(False)
            self.agg.setEnabled(False)
            self.metric_note.setText("Select evaluated measurements to enable their statistics.")
            self.canvas.setActive()
            plt.cla()
            self.canvas.draw()
            self.export_data = None
            self.current_plot_func = lambda: None
            return
        histogram = self.type.value().startswith("angle histogram")
        self.button_run3.setText("Grouped Histogram" if histogram else "Grouped Bar Plot")
        if point_distance:
            self.button_run.setEnabled(True)
            self.button_run2.setEnabled(True)
            self.button_run3.setEnabled(False)
            self.agg.setEnabled(False)
            if self.current_plot_func not in (self.plot_groups, self.run2):
                self.current_plot_func = self.plot_groups
            return
        if histogram:
            self.button_run.setEnabled(False)
            self.button_run2.setEnabled(False)
            self.agg.setEnabled(False)
            self.histogram()
            return
        time_values = False
        if self.type.value() == "global orientation":
            time_values = False
        elif self.type.value() == "normalized intensity (first shell)":
            time_values = False
        elif self.type.value() == "orientation over distance":
            time_values = True
        elif self.type.value() == "normed intensity over distance":
            time_values = True

        if time_values is False:
            self.barplot()
        elif self.current_plot_func not in (self.plot_groups, self.run2):
            self.current_plot_func = self.plot_groups
        if getattr(self, "agg", None) is not None:
            self.agg.setEnabled(time_values)
        self.button_run.setEnabled(time_values)
        self.button_run2.setEnabled(time_values)

    def load_file(self, file):
        res: ResultOrientation = ResultOrientation.load(file)
        res.resulting_data = []
        res.resulting_distance_data = pd.DataFrame()
        if res.analysis_mode == "point_orientation":
            from ..modules.point_distance import ensure_point_distance
            ensure_point_distance(res)
            res.resulting_distance_data = pd.DataFrame(res.results_distance or [])
            if len(res.resulting_distance_data):
                res.resulting_distance_data["filename"] = file

        if not res.results_total:
            raise ValueError("No orientation results; run the analysis first")
        if res.analysis_mode == "full":
            from ..modules.axis_statistics import axis_statistics
            computed = axis_statistics(res)
            if computed is not None:
                stats, res.axis_histogram = computed
                res.results_total[0].update(stats)
        if res.analysis_mode != "full":
            if self is not None and self.type.value() == "global orientation":
                self.type.setValue({"point_orientation": "point alignment", "line_orientation": "line alignment"}.get(res.analysis_mode, "alignment x"))
            # One row per image: group means weight images equally, not pixels.
            row = res.results_total[0].copy()
            # Existing axis files already store the coherency map. Derive the
            # image mean in memory so they work without rerunning or resaving.
            if "Mean Coherency" not in row:
                row["Mean Coherency"] = PlottingWindow.axis_mean_coherency(res)
            # With the package's linear alignment, this is exactly mean(abs(angle)),
            # including for old saved files; never average the signed axial angles.
            for axis in ("x", "y"):
                row[f"Mean absolute deviation {axis}"] = 45 * (1 - row[f"Alignment {axis}"])
            row.update(filename=file, t=0)
            row.update({name: float("nan") for name in ("Orientation", "normalized intensity (first shell)",
                                                        "Orientation (individual)", "Intensity Norm (individual)", "Mean Angle")})
            res.resulting_data = pd.DataFrame([row])
            return res
        global_data = res.results_total[0].copy()
        global_data.setdefault("Mean Coherency", float("nan"))
        for axis in ("x", "y"):
            global_data.setdefault(f"Alignment {axis}", float("nan"))
            global_data.setdefault(f"Mean absolute deviation {axis}", 45 * (1 - global_data[f"Alignment {axis}"]))
        global_data["normalized intensity (first shell)"] = res.results_distance[0]["Intensity Norm (individual)"] if res.results_distance else float("nan")

        for i in res.results_distance or [{"Shell_mid (µm)": 0}]:
            i = i.copy()
            i.update(global_data)
            i["filename"] = file
            i["t"] = i['Shell_mid (µm)']
            res.resulting_data.append(i)
        res.resulting_data = pd.DataFrame(res.resulting_data)
        return res

    @staticmethod
    def axis_mean_coherency(result):
        if result.coherence_map is None:
            return float("nan")
        values = np.asarray(result.coherence_map)
        valid = np.isfinite(values)
        if result.analysis_mode == "axis_orientation_masked":
            if result.validity_mask is None:
                return float("nan")
            valid &= np.asarray(result.validity_mask, dtype=bool)
        # Isotropic pixels with coherency 0 remain valid even when their
        # orientation angle is undefined; coherency needs no reference axis.
        return float(values[valid].mean()) if valid.any() else float("nan")

    def getAllCurrentPandasData(self):
        data = super().getAllCurrentPandasData()
        if data is None:
            self.metric_note.setText("No measurements selected for this metric.")
            return None
        # A mixed group must not turn missing axis/cell metrics into zero bars.
        metric, label = self.get_label()
        data = data.dropna(subset=[metric])
        family = self.metrics[self.type.value()][2]
        if family in ("all", "axis"):
            self.metric_note.setText(f"All measurement modes: {data.filename.nunique()} images with values. "
                                    "Group means give each image equal weight.")
        else:
            label = {"point": "Point-relative", "line": "Line-relative", "cell": "Cell-relative"}[family]
            self.metric_note.setText(f"{label} measurements: {data.filename.nunique()} images with values. "
                                    "Other measurement types are excluded. Group means give each image equal weight.")
        if self.type.value() == "coherency":
            self.metric_note.setText(self.metric_note.text() + " Mean local coherency: 0 isotropic, "
                                    "1 strongly directional; independent of the reference axis. "
                                    "Each image uses its analyzed region.")
        elif self.type.value() in ("alignment x", "alignment y"):
            self.metric_note.setText(self.metric_note.text() + " Alignment: +1 parallel, 0 at 45°, −1 perpendicular.")
        elif self.type.value().startswith("mean absolute deviation"):
            self.metric_note.setText(self.metric_note.text() + " Mean angular deviation: 0° parallel, 90° perpendicular.")
        return data if len(data) else None

    def barplot(self):
        if self.type.value() in self.point_distance_metrics:
            return self.point_distance_plot()
        if self.type.value().startswith("angle histogram"):
            return self.histogram()
        super().barplot()
        # The base plot clears its axes but leaves an obsolete export when empty.
        if self.getAllCurrentPandasData() is None:
            self.export_data = None
            self.canvas.draw()
        else:
            self.apply_metric_limits()

    def plot_groups(self):
        if self.type.value() in self.point_distance_metrics:
            return self.point_distance_plot()
        self.export_data = None
        super().plot_groups()
        self.apply_metric_limits()

    def run2(self):
        if self.type.value() in self.point_distance_metrics:
            return self.point_distance_plot(single=True)
        if not self.button_run.isEnabled():
            return
        self.export_data = None
        super().run2()
        self.apply_metric_limits()

    def point_distance_plot(self, single=False):
        """Keep shell courses separate from one-value-per-image statistics."""
        self.current_plot_func = self.run2 if single else self.plot_groups
        self.export_data = None
        self.canvas.setActive()
        plt.cla()
        for button in self.plot_buttons:
            button.setChecked(False)
        (self.button_run if single else self.button_run2).setChecked(True)
        metric, ylabel = self.get_label()
        unit = self.distance_unit.value()
        distance_column = "Shell_mid (px)" if unit == "pixel" else "Shell_mid (µm)"
        frames, widths, colors = [], [], {}
        for group_index, (group, checked, files, color) in enumerate(self.data_folders):
            if not checked or (single and group_index != self.list.currentRow()):
                continue
            for file_index, (filename, included, result, _) in enumerate(files):
                if (not included or result.analysis_mode != "point_orientation"
                        or (single and file_index != self.list2.currentRow())):
                    continue
                data = getattr(result, "resulting_distance_data", pd.DataFrame())
                if metric not in data or not np.isfinite(data[metric]).any():
                    continue
                data = data.copy()
                width = data["Shell_end (px)"].iloc[0]-data["Shell_start (px)"].iloc[0]
                widths.append(width if unit == "pixel" else width*result.pixel_size)
                data["distance"] = data[distance_column].round(9)
                data["group"] = group
                frames.append(data)
                colors[group] = color
        if not frames:
            self.metric_note.setText("Select an evaluated point measurement with distance-shell values.")
            self.canvas.draw()
            return
        if not np.allclose(widths, widths[0], rtol=1e-9, atol=1e-9):
            self.metric_note.setText("Grouped point courses need matching shell widths in the selected distance unit. Adjust shell_width and rerun find orientations.")
            self.canvas.draw()
            return
        data = pd.concat(frames, ignore_index=True)
        limits = (-1, 1) if self.type.value() == "point alignment over distance" else (
            (0, 90) if self.type.value() == "point angular deviation over distance" else (0, 1))
        xlabel = "Distance from reference point (px)" if unit == "pixel" else "Distance from reference point (µm)"
        self.metric_note.setText("Concentric rings around the selected point, within each image's mask. "
            "Each image contributes one mean per ring; groups show mean ± SEM across images. "
            "Empty rings remain gaps; n can vary between rings.")

        @export_as_string
        def plot(res, shell_metric, shell_ylabel, shell_xlabel, shell_colors, shell_limits):
            for group, rows in res.groupby("group", sort=False):
                stats = rows.groupby("distance")[shell_metric].agg(["mean", "sem", "count"])
                counts = stats.loc[stats["count"] > 0, "count"]
                n = str(int(counts.max())) if counts.min() == counts.max() else f"{int(counts.min())}–{int(counts.max())}"
                plt.plot(stats.index, stats["mean"], color=shell_colors[group], label=f"{group} (n={n} images/ring)")
                plt.fill_between(stats.index, stats["mean"]-stats["sem"], stats["mean"]+stats["sem"],
                                 color=shell_colors[group], alpha=.2)
            plt.xlabel(shell_xlabel)
            plt.ylabel(shell_ylabel)
            plt.ylim(*shell_limits)
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.legend()
            plt.tight_layout()
            self.canvas.draw()

        count_column = "Coherency pixels" if self.type.value() == "point coherency over distance" else "Valid pixels"
        columns = ["distance", "group", "filename", metric, count_column]
        code = execute(plot, data[columns], shell_metric=metric, shell_ylabel=ylabel,
                       shell_xlabel=xlabel, shell_colors=colors, shell_limits=limits)
        self.export_data = [code, [data, columns]]

    def apply_metric_limits(self):
        """Use the same meaningful scale in the GUI and the exported script."""
        key = self.type.value()
        if key in ("alignment x", "alignment y", "global orientation", "orientation over distance", "point alignment", "line alignment"):
            limits = (-1, 1)
        elif key.startswith("mean absolute deviation") or key in ("mean cell-relative deviation", "point angular deviation", "line angular deviation"):
            limits = (0, 90)
        elif key == "coherency":
            limits = (0, 1)
        else:
            return
        if self.export_data is not None:
            self.canvas.setActive()
            plt.ylim(*limits)
            plt.tight_layout()
            self.canvas.draw()
            self.export_data[0] = self.export_data[0].replace("plt.show()",
                f"plt.ylim{limits!r}\nplt.tight_layout()\nplt.show()")

    @staticmethod
    def histogram_for(result, key):
        if key == "angle histogram line":
            angles = result.get_map_data("angle_line")
            if angles is None:
                return None
            counts, bins = np.histogram(angles[np.isfinite(angles)], bins=np.linspace(-90, 90, 37))
            return np.column_stack([bins[:-1], bins[1:], counts])
        if result.axis_histogram is None:
            return None
        return np.asarray(result.axis_histogram)[:, [0, 1, 2 if key.endswith("x") else 3]]

    def get_histogram_data(self):
        """One normalized distribution per image; never pool raw pixel counts."""
        records = []
        for group, checked, files, color in self.data_folders:
            if not checked:
                continue
            for filename, selected, result, _ in files:
                if not selected:
                    continue
                hist = self.histogram_for(result, self.type.value())
                if hist is None:
                    continue
                counts = hist[:, 2]
                total = counts.sum()
                if not np.isfinite(total) or total <= 0:
                    continue
                for row, fraction in zip(hist, counts / total):
                    records.append(dict(group=group, filename=filename,
                        angle=(row[0]+row[1])/2, fraction=fraction))
        return pd.DataFrame(records, columns=["group", "filename", "angle", "fraction"])

    def histogram(self):
        self.current_plot_func = self.histogram
        for button in self.plot_buttons:
            button.setChecked(button is self.button_run3)
        self.canvas.setActive()
        plt.cla()
        res = self.get_histogram_data()
        scope = "Line-relative measurements" if self.type.value() == "angle histogram line" else "All measurement modes"
        self.metric_note.setText(f"{scope}: {res.filename.nunique()} nonempty images. "
            "Bin probabilities sum to 1 per image; groups show the mean and SEM across images. n is the number of images.")
        if res.empty:
            self.export_data = None
            self.canvas.draw()
            return
        xlabel = "Orientation angle to reference line (deg)" if self.type.value() == "angle histogram line" else f"Orientation angle towards {self.type.value()[-1]}-axis (deg)"
        colors = {d[0]: d[3] for d in self.data_folders}

        @export_as_string
        def plot(res, histogram_xlabel, histogram_colors):
            for group, data in res.groupby("group", sort=False):
                stats = data.groupby("angle")["fraction"].agg(["mean", "sem"])
                sem = stats["sem"].fillna(0)
                count = data.filename.nunique()
                plt.step(stats.index, stats["mean"], where="mid", color=histogram_colors[group],
                         label=f"{group} (n={count})")
                plt.fill_between(stats.index, stats["mean"]-sem, stats["mean"]+sem,
                                 step="mid", color=histogram_colors[group], alpha=.2)
            plt.xlim(-90,90)
            plt.ylim(bottom=0)
            plt.xlabel(histogram_xlabel)
            plt.ylabel("Probability")
            plt.legend()
            plt.tight_layout()
            self.canvas.draw()

        code = execute(plot, res, histogram_xlabel=xlabel, histogram_colors=colors)
        self.export_data = [code, [res, list(res.columns)]]

    def export(self):
        if getattr(self, "export_data", None) is None:
            QtWidgets.QMessageBox.information(self, "Export", "Select measurements with values for the current metric first.")
            return
        super().export()

    def get_label(self):
        key = self.type.value()
        title, column, _ = self.metrics[key]
        if key.startswith("mean absolute deviation") or key in ("mean cell-relative deviation", "point angular deviation", "point angular deviation over distance", "line angular deviation"):
            title += " (deg)"
        return column, title
