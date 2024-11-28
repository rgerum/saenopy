from typing import Tuple
from pathlib import Path

import saenopy
import saenopy.multigrid_helper
import saenopy.get_deformations
import saenopy.materials

from saenopy.gui.common.PipelineModule import PipelineModule
from saenopy.gui.common.code_export import get_code, export_as_string


class Initializer(PipelineModule):

    def get_code(self) -> Tuple[str, str]:
        import_code = ""

        if self.result.time_delta is None:
            if self.result.stack_reference is not None:
                @export_as_string
                def code(filename, reference_stack1, output1, voxel_size1, result_file, crop1):  # pragma: no cover
                    # load the relaxed and the contracted stack
                    # {z} is the placeholder for the z stack
                    # {c} is the placeholder for the channels
                    # {t} is the placeholder for the time points
                    # use * to load multiple stacks for batch processing
                    # load_existing=True allows to load an existing file of these stacks if it already exists
                    results = saenopy.get_stacks(
                        filename,
                        reference_stack=reference_stack1,
                        output_path=output1,
                        voxel_size=voxel_size1,
                        crop=crop1,
                        load_existing=True)
                    # or if you want to explicitly load existing results files
                    # use * to load multiple result files for batch processing
                    # results = saenopy.load_results(result_file)

                data = dict(
                    filename=self.result.get_absolute_path(),
                    reference_stack1=self.result.get_absolute_path_reference(),
                    output1=str(Path(self.result.output).parent),
                    voxel_size1=self.result.stacks[0].voxel_size,
                    crop1=self.result.stacks[0].crop,
                    result_file=str(self.result.output),
                )
        else:
            if self.result.stack_reference is not None:
                @export_as_string
                def code(filename, reference_stack1, output1, voxel_size1, time_delta1, result_file, crop1):  # pragma: no cover
                    # load the relaxed and the contracted stack
                    # {z} is the placeholder for the z stack
                    # {c} is the placeholder for the channels
                    # {t} is the placeholder for the time points
                    # use * to load multiple stacks for batch processing
                    # load_existing=True allows to load an existing file of these stacks if it already exists
                    results = saenopy.get_stacks(
                        filename,
                        reference_stack=reference_stack1,
                        output_path=output1,
                        voxel_size=voxel_size1,
                        time_delta=time_delta1,
                        crop=crop1,
                        load_existing=True)
                    # or if you want to explicitly load existing results files
                    # use * to load multiple result files for batch processing
                    # results = saenopy.load_results(result_file)

                data = dict(
                    filename=self.result.get_absolute_path(),
                    reference_stack1=self.result.get_absolute_path_reference(),
                    output1=str(Path(self.result.output).parent),
                    result_file=str(self.result.output),
                    voxel_size1=self.result.stacks[0].voxel_size,
                    crop1=self.result.stacks[0].crop,
                    time_delta1=self.result.time_delta,
                )
            else:
                @export_as_string
                def code(filename, output1, voxel_size1, time_delta1, result_file, crop1):  # pragma: no cover
                    # load the relaxed and the contracted stack
                    # {z} is the placeholder for the z stack
                    # {c} is the placeholder for the channels
                    # {t} is the placeholder for the time points
                    # use * to load multiple stacks for batch processing
                    # load_existing=True allows to load an existing file of these stacks if it already exists
                    results = saenopy.get_stacks(
                        filename,
                        output_path=output1,
                        voxel_size=voxel_size1,
                        time_delta=time_delta1,
                        crop=crop1,
                        load_existing=True)
                    # or if you want to explicitly load existing results files
                    # use * to load multiple result files for batch processing
                    # results = saenopy.load_results(result_file)

                data = dict(
                    filename=self.result.get_absolute_path(),
                    output1=str(Path(self.result.output).parent),
                    voxel_size1=self.result.stacks[0].voxel_size,
                    time_delta1=self.result.time_delta,
                    crop1=self.result.stacks[0].crop,
                    result_file=str(self.result.output),
                )

        code = get_code(code, data)
        return import_code, code
