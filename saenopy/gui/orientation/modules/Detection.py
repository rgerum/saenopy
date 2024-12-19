import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import jointforces as jf
from pathlib import Path

from saenopy.gui.orientation.modules.result import ResultOrientation, OrientationParametersDict
from qtpy import QtCore, QtWidgets, QtGui
from saenopy.gui.common import QtShortCuts

from saenopy.gui.common.gui_classes import CheckAbleGroup
from saenopy.gui.common.PipelineModule import PipelineModule
from saenopy.gui.common.code_export import get_code

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

    def check_available(self, result: ResultOrientation) -> bool:
        return True

    def process(self, result: ResultOrientation, orientation_parameters: OrientationParametersDict):
        segmentation_parameters = result.segmentation_parameters
        sigma_tensor = orientation_parameters["sigma_tensor"]
        if orientation_parameters["sigma_tensor_type"] == "um":
            sigma_tensor /= result.pixel_size
        shell_width = orientation_parameters["shell_width"]
        if orientation_parameters["shell_width_type"] == "um":
            shell_width /= result.pixel_size

        from CompactionAnalyzer.CompactionFunctions import StuctureAnalysisMain
        angle_no_reference, orientation_dev, ori, min_evex, excel_total, excel_angles, excel_distance = StuctureAnalysisMain(fiber_list=[result.get_absolute_path_fiber()],
                             cell_list=[result.get_absolute_path_cell()],
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
                             mode_saenopy=True,
                             )

        result.orientation_map = orientation_dev#np.load(Path(result.output).parent / Path(result.output).stem / "NumpyArrays" / "OrientationMap.npy")
        result.coherence_map = ori
        result.angle_to_x_map = angle_no_reference
        result.orientation_vectors = min_evex

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