import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import jointforces as jf
from pathlib import Path

from saenopy.gui.orientation.modules.result import ResultOrientation, OrientationParametersDict
from qtpy import QtCore, QtWidgets, QtGui
from saenopy.gui.common import QtShortCuts

from saenopy.gui.common.gui_classes import CheckAbleGroup
from saenopy.gui.common.PipelineModule import PipelineModule, StateEnum
from saenopy.gui.common.code_export import get_code

class DeformationDetector(PipelineModule):
    pipeline_name = "find orientations"
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
                        self.sigma_tensor_type = QtShortCuts.QInputChoice(None, "", "µm", ["µm", "pixel"],
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
                        self.max_dist_unit = QtWidgets.QLabel("px").addToLayout()

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
                        self.angle_unit = QtWidgets.QLabel("deg").addToLayout()

                    with QtShortCuts.QHBoxLayout():
                        self.shell_width = QtShortCuts.QInputString(None, "shell_width", "5", type=float,
                                                                    settings=self.settings,
                                                                    settings_key="orientation/shell_width")
                        self.shell_width_type = QtShortCuts.QInputChoice(None, "", "µm", ["µm", "pixel"],
                                                                         settings=self.settings,
                                                                         settings_key="orientation/shell_width_type")

                    self.save_numpy = QtShortCuts.QInputBool(None, "save NumPy arrays + CSV", True)
                    self.save_plots = QtShortCuts.QInputBool(None, "save plots", True)
                    self.input_button = QtShortCuts.QPushButton(None, "find orientations", self.start_process)

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
            "SaveNumpy": self.save_numpy,
            "plotting": self.save_plots,
        })

        #self.progress_signal.connect(self.progress_callback)

    def check_available(self, result: ResultOrientation) -> bool:
        if result is None:
            return False
        if result.analysis_mode == "point_orientation":
            p = getattr(result, "orientation_parameters_tmp", None) or result.orientation_parameters or {}
            try:
                x, y = float(p.get("reference_x")), float(p.get("reference_y"))
                shape = result.shape or result.get_image(1).shape[:2]
                return bool(np.isfinite([x, y]).all() and 0 <= x < shape[1] and 0 <= y < shape[0])
            except (TypeError, ValueError, OSError):
                return False
        if result.analysis_mode == "line_orientation":
            from .ReferenceOverlay import parameter_line
            from CompactionAnalyzer.plotting import line_reference_angle
            p = getattr(result, "orientation_parameters_tmp", None) or result.orientation_parameters or {}
            try:
                line_reference_angle(parameter_line(p), result.shape)
            except ValueError:
                return False
        return True

    def state_changed(self, result):
        super().state_changed(result)
        if result is self.result and self.get_result_state(result) in (StateEnum.finished, StateEnum.failed):
            self.input_button.setText("find orientations")
            self.input_button.setEnabled(self.check_available(result))

    def setResult(self, result):
        super().setResult(result)
        full = result is None or result.analysis_mode == "full"
        for widget in (self.max_dist, self.max_dist_unit, self.ignore_cell, self.angle_sections,
                       self.angle_unit):
            widget.setVisible(full)
        point = result is not None and result.analysis_mode == "point_orientation"
        self.shell_width.setVisible(full or point)
        self.shell_width_type.setVisible(full or point)
        self.shell_width.setToolTip("Width of concentric rings around the selected point." if point
                                   else "Width of distance shells around the cell outline.")
        axis = result is not None and result.analysis_mode.startswith("axis_")
        self.save_numpy.setVisible(axis)
        self.save_plots.setVisible(axis)

    def process(self, result: ResultOrientation, orientation_parameters: OrientationParametersDict):
        from .analysis import analyze_orientation
        # Also initialize these when only "find orientations" was clicked.
        mapping = self.parent.sub_module_segmentation.parameter_mappings[0]
        mapping.ensure_tmp_params_initialized(result)
        # The pipeline worker saves once after this calculation succeeds.
        analyze_orientation(result, orientation_parameters, result.segmentation_parameters_tmp, save=False)

    def process_thread(self, result, params):
        # Preserve completed results if calculation, statistics or saving fails.
        fields = result.__save_parameters__ + ["mask_preview", "mask_preview_edge", "segmentation_preview"]
        previous = {name: getattr(result, name, None) for name in fields}
        super().process_thread(result, params)
        if self.get_result_state(result) == StateEnum.failed:
            for name, value in previous.items():
                setattr(result, name, value)
            self.parent.result_changed.emit(result)

    def get_code(self) -> Tuple[str, str]:
        if self.result is None:
            return "", ""
        from pprint import pformat
        for mapping in self.parameter_mappings + self.parent.sub_module_segmentation.parameter_mappings:
            mapping.ensure_tmp_params_initialized(self.result)
        result = self.result
        imports = ("from saenopy.gui.orientation.modules.result import ResultOrientation\n"
                   "from saenopy.gui.orientation.modules.analysis import analyze_orientation\n")
        # Use repr for Windows paths and pformat for the actual GUI parameters.
        code = (f"result = ResultOrientation(output={str(result.output)!r},\n"
                f"    image_fiber={str(result.get_absolute_path_fiber())!r},\n"
                f"    image_cell={None if result.image_cell is None else str(result.get_absolute_path_cell())!r},\n"
                f"    pixel_size={result.pixel_size!r}, analysis_mode={result.analysis_mode!r})\n"
                f"orientation_parameters = {pformat(result.orientation_parameters_tmp)}\n"
                f"segmentation_parameters = {pformat(result.segmentation_parameters_tmp)}\n"
                "analyze_orientation(result, orientation_parameters, segmentation_parameters)\n")
        return imports, code

    def sigma_tensor_button_clicked(self):
        parent = self

        class SigmaRange(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Determine Sigma Tensor")
                with QtShortCuts.QVBoxLayout(self) as layout:
                    self.output_folder = QtShortCuts.QInputFolder(None, "output folder", settings=settings,
                                                                  settings_key="orientation/sigma_tensor_range_output")
                    self.label_scale = QtWidgets.QLabel(f"Scale is {parent.scale.value()} px/µm").addToLayout(layout)
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
                        self.sigma_tensor_type = QtShortCuts.QInputChoice(None, "", "µm", ["µm", "pixel"],
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
                if self.sigma_tensor_type.value() == "µm":
                    sigma_tensor_min /= parent.scale.value()
                    sigma_tensor_max /= parent.scale.value()
                    sigma_tensor_step /= parent.scale.value()
                shell_width = parent.shell_width.value()
                if parent.shell_width_type.value() == "µm":
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
