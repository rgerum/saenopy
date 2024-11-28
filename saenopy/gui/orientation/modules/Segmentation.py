import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import jointforces as jf
from pathlib import Path

from saenopy.gui.orientation.modules.result import ResultOrientation, SegmentationParametersDict
from qtpy import QtCore, QtWidgets, QtGui
from qimage2ndarray import array2qimage
from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView

from saenopy.gui.common.gui_classes import QHLine, CheckAbleGroup
from saenopy.gui.common.PipelineModule import PipelineModule
from saenopy.gui.common.QTimeSlider import QTimeSlider
from saenopy.gui.common.resources import resource_icon
from saenopy.gui.common.code_export import get_code, export_as_string
from saenopy.gui.common.ModuleScaleBar import ModuleScaleBar
from saenopy.gui.common.ModuleColorBar import ModuleColorBar


class SegmentationDetector(PipelineModule):
    pipeline_name = "find deformations"
    use_thread = False
    progress_signal = QtCore.Signal(int, str)
    result: ResultOrientation = None

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__(parent, layout)

        with QtShortCuts.QVBoxLayout(self) as layout:
            layout.setContentsMargins(0, 0, 0, 0)
            with CheckAbleGroup(self, "segmentation").addToLayout() as self.group:
                with QtShortCuts.QVBoxLayout():
                    with QtShortCuts.QHBoxLayout():
                        self.seg_thresh = QtShortCuts.QInputString(None, "thresh", "1.0",
                                                                   type=float,
                                                                   settings=self.settings,
                                                                   settings_key="orientation/segmentation/thresh")
                        self.seg_invert = QtShortCuts.QInputBool(None, "invert", False, settings=self.settings,
                                                                 settings_key="orientation/segmentation/invert",
                                                                 tooltip="Tick if cell is dark on white background.")
                    # self.segmention_thres.valueChanged.connect(self.listSelected)
                    with QtShortCuts.QHBoxLayout():
                        self.seg_gauss1 = QtShortCuts.QInputString(None, "gauss1", "0.5", type=float,
                                                                   settings=self.settings,
                                                                   settings_key="orientation/segmentation/gauss1")
                        # self.seg_gaus1.valueChanged.connect(self.listSelected)
                        self.seg_gauss2 = QtShortCuts.QInputString(None, "gauss2", "100", type=float,
                                                                   settings=self.settings,
                                                                   settings_key="orientation/segmentation/gauss2")
                        # self.seg_gaus2.valueChanged.connect(self.listSelected)

                    self.input_button = QtShortCuts.QPushButton(None, "detect deformations", self.start_process)

        self.setParameterMapping("segmentation_parameters", {
            "thresh": self.seg_thresh,
            "gauss1": self.seg_gauss1,
            "gauss2": self.seg_gauss2,
            "invert": self.seg_invert,
        })

        #self.progress_signal.connect(self.progress_callback)

    def check_available(self, result: ResultOrientation) -> bool:
        return True

    def setResult(self, result: ResultOrientation):
        super().setResult(result)
        self.update_display()

    def process(self, result: ResultOrientation, segmentation_parameters: SegmentationParametersDict):
        from CompactionAnalyzer.CompactionFunctions import segment_cell, normalize
        from skimage import color

        im_cell = result.get_image(0)
        ## if 3 channels convert to grey
        if len(im_cell.shape) == 3:
            im_cell = color.rgb2gray(im_cell)
        im_cell_n = normalize(im_cell, 1, 99)

        segmentation = segment_cell(im_cell_n, thres=segmentation_parameters["thresh"],
                                  seg_gaus1=segmentation_parameters["gauss1"],
                                  seg_gaus2=segmentation_parameters["gauss2"],
                                  show_segmentation=False,
                                  seg_invert=segmentation_parameters["invert"],
                                  seg_iter=1,
                                  segmention_method="otsu",
                                  regional_max_correction=True,
                                  segmention_min_area=1000)

        result.segmentation = segmentation
        result.save()

    def get_code(self) -> Tuple[str, str]:
        import_code = "from saenopy.gui.spheroid.modules.result import get_stacks_spheroid\nimport jointforces as jf\n"

        @export_as_string
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
