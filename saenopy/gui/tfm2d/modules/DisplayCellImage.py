from pathlib import Path
from typing import Tuple

from saenopy.gui.common.code_export import get_code
from saenopy.gui.tfm2d.modules.result import get_stacks2D

from .PipelineModule import PipelineModule


class DisplayCellImage(PipelineModule):

    def __init__(self, parent=None, layout=None):
        super().__init__(parent, layout)
        self.parent = parent
        with self.parent.tabs.createTab("Cell Image") as self.tab:
            pass

    def check_evaluated(self, result):
        return True

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            im = self.result.get_image(-1)
            self.parent.draw.setImage(im, self.result.shape)

    def get_code(self) -> Tuple[str, str]:
        import_code = "from saenopy.gui.tfm2d.modules.result import Result2D, get_stacks2D\n"


        def code(output1, bf1, filename1, reference_stack1, pixel_size1, result_file, crop1):  # pragma: no cover
            # load the cell image, the relaxed and the contracted images
            # use * to load multiple experiments for batch processing
            # load_existing=True allows to load an existing file of these images if it already exists
            results = get_stacks2D(
                output1,
                bf_stack=bf1,
                active_stack=filename1,
                reference_stack=reference_stack1,
                pixel_size=pixel_size1,
                load_existing=True)
            # or if you want to explicitly load existing results files
            # use * to load multiple result files for batch processing
            # results = saenopy.load_results(result_file)

        data = dict(
            output1=str(Path(self.result.output).parent),
            bf1=self.result.get_absolute_path_bf(),
            filename1=self.result.get_absolute_path(),
            reference_stack1=self.result.get_absolute_path_reference(),
            pixel_size1=self.result.pixel_size,
            result_file=str(self.result.output),
        )

        code = get_code(code, data)
        return import_code, code
