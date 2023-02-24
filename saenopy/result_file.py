import glob
import re
from pathlib import Path
import os
import natsort
from typing import List
import tifffile
from saenopy.getDeformations import Stack, format_glob
from saenopy.loadHelpers import Saveable
from saenopy.solver import Mesh, Solver


def get_channel_placeholder(filename):
    match = re.match(r".*{c:([^}]*)}.*", filename)
    if match:
        filename = re.sub(r"{c:[^}]*}", "{c}", filename)
        return filename, match.groups()[0]
    return filename, None


def get_iterator(values, name="", iter=None):
    if iter is None:
        for v in values:
            yield {name: v}
    else:
        for v in values:
            for t in iter:
                t[name] = v
                yield t

def process_line(filename, output_path):
    results = []

    if filename.endswith(".lif"):
        match = re.match(r"(.*)\{f\:(\d*)\}\{c\:(\d*)\}.lif", filename)
        filename, folder, channel = match.groups()
        files = glob.glob(filename+".lif", recursive=True)

        output_base = filename
        while "*" in str(output_base):
            output_base = Path(output_base).parent

        for file in files:
            relative = os.path.relpath(file, output_base)
            counts = {}
            from saenopy.gui.lif_reader import LifFile
            leica_file = LifFile(file).get_image(folder)
            counts["z"] = leica_file.dims.z
            counts["c"] = leica_file.channels
            if leica_file.dims.t > 1:
                counts["t"] = leica_file.dims.t
            results.append({"filename": f"{file[:-4]}{{f:{folder}}}{{c:{channel}}}.lif", "dimensions": counts, "output": Path(output_path) / relative})
            if leica_file.dims.t > 1:
                import numpy as np
                results[-1]["times"] = np.arange(counts["t"])
        return results
    filename, channel1 = get_channel_placeholder(filename)
    results1, output_base = format_glob(Path(filename))
    for (r1, d1) in results1.groupby("template"):
        counts = {"z": len(d1.z.unique())}
        iterator = get_iterator(d1.z.unique(), "z")
        if channel1 is not None:
            r1 = r1.replace("{c}", "{c:"+channel1+"}")
            counts["c"] = len(d1.c.unique())
            iterator = get_iterator(d1.c.unique(), "c", iterator)
        if "t" in d1.columns:
            counts["t"] = len(d1.t.unique())
            iterator = get_iterator(d1.t.unique(), "t", iterator)
        #print(r1, counts, np.prod([c for c in counts.values()]), len(d1))
        #assert np.prod([c for c in counts.values()]) == len(d1)
        # check if all files exist and have the same shape
        template = d1.iloc[0].template
        shape = None
        for props in iterator:
            filename = template.format(**props)
            if re.match("(.*.tif)\[(.*)\]", filename):
                filename, page = re.match("(.*.tif)\[(.*)\]", filename).groups()
            if not Path(filename).exists():
                raise FileNotFoundError(f"Could not find file {filename}")
            f = tifffile.TiffFile(filename)
            if shape is None:
                shape = f.pages[0].shape
            else:
                if f.pages[0].shape != shape:
                    raise ValueError(f"Shape of file {filename} ({f.pages[0].shape}) does not match previous shape ({shape})")

        # create the output path
        output = Path(output_path) / os.path.relpath(r1, output_base)
        output = output.parent / output.stem
        output = Path(str(output).replace("*", "").replace("{c}", "{c" + str(channel1) + "}").replace("{c:", "{c") + ".npz")

        results.append({"filename": r1, "dimensions": counts, "output": output})
        if "t" in d1.columns:
            results[-1]["times"] = natsort.natsorted(d1.t.unique())
    return results

def get_stacks(filename, output_path, voxel_size, time_delta=None, reference_stack=None,
               crop=None,
               exist_overwrite_callback=None,
               load_existing=False):
    filename = str(filename)
    output_path = str(output_path)
    if reference_stack is not None:
        reference_stack = str(reference_stack)
    results1 = process_line(filename, output_path)
    results = []
    if reference_stack is not None:
        results2 = process_line(reference_stack, output_path)
        if len(results1) != len(results2):
            raise ValueError(
                f"Number of active stacks ({len(results1)}) does not match the number of reference stacks ({len(results2)}).")
        for r1, r2 in zip(results1, results2):
            if r1["dimensions"]["z"] != r2["dimensions"]["z"]:
                raise ValueError("active and reference stack need the same number of z slices")
            if "t" in r2["dimensions"]:
                raise ValueError("the reference stack is not allowed to have different time points")
            if "c" in r1["dimensions"]:
                if "c" not in r2["dimensions"]:
                    raise ValueError("if the active stack has channels the reference stack also needs channels")
                if r1["dimensions"]["c"] != r2["dimensions"]["c"]:
                    raise ValueError("the active stack and the reference stack also need the same number of channels")

            if "t" in r1["dimensions"]:
                stacks = []
                times = r1["times"]
                if (crop is not None) and ("t" in crop):
                    times = times[slice(*crop["t"])]
                for t in times:
                    stacks.append(Stack(r1["filename"].replace("{t}", t), voxel_size, crop=crop))
            else:
                stacks = [Stack(r1["filename"], voxel_size, crop=crop)]

            output = r1["output"]
            if output.exists():
                if exist_overwrite_callback is not None:
                    mode = exist_overwrite_callback(output)
                    if mode == 0:
                        break
                    if mode == "read":
                        print('exists', output)
                        data = Result.load(output)
                        results.append(data)
                        continue
                elif load_existing is True:
                    data = Result.load(output)
                    results.append(data)
                    continue

            data = Result(
                output=output,
                template=r1["filename"],
                stack=stacks,
                stack_reference=Stack(r2["filename"], voxel_size, crop=crop),
                time_delta=time_delta,
            )
            data.save()
            results.append(data)
    else:
        for r1 in results1:
            if "t" in r1["dimensions"]:
                stacks = []
                times = r1["times"]
                if crop is not None and "t" in crop:
                    times = times[slice(*crop["t"])]
                for t in times:
                    if r1["filename"].endswith(".lif"):
                        stacks.append(Stack(r1["filename"].replace(".lif", f"{{t:{t}}}.lif"), voxel_size, crop=crop))
                    else:
                        stacks.append(Stack(r1["filename"].replace("{t}", t), voxel_size, crop=crop))
            else:
                stacks = [Stack(r1["filename"], voxel_size, crop=crop)]

            output = r1["output"]
            if output.exists():
                if exist_overwrite_callback is not None:
                    mode = exist_overwrite_callback(output)
                    if mode == 0:
                        break
                    if mode == "read":
                        print('exists', output)
                        data = Result.load(output)
                        results.append(data)
                        continue
                elif load_existing is True:
                    data = Result.load(output)
                    results.append(data)
                    continue

            if len(stacks) == 1:
                raise ValueError("when not using a time series, a reference stack is required.")
            data = Result(
                output=r1["output"],
                template=r1["filename"],
                stack=stacks,
                time_delta=time_delta,
            )
            data.save()
            results.append(data)
    return results


def common_start(values):
    if len(values) != 0:
        start = values[0]
        while start:
            if all(value.startswith(start) for value in values):
                return start
            start = start[:-1]
    return ""


def common_end(values):
    if len(values) != 0:
        end = values[0]
        while end:
            if all(value.endswith(end) for value in values):
                return end
            end = end[1:]
    return ""


class Result(Saveable):
    __save_parameters__ = ['template', 'stack', 'stack_reference', 'time_delta', 'piv_parameter', 'mesh_piv',
                           'interpolate_parameter', 'solve_parameter', 'solver',
                           '___save_name__', '___save_version__']
    ___save_name__ = "Result"
    ___save_version__ = "1.1"
    output: str = None
    state: False

    stack: List[Stack] = None
    stack_parameter: dict = None
    stack_reference: Stack = None
    template: str = None

    piv_parameter: dict = None
    mesh_piv: List[Mesh] = None

    interpolate_parameter: dict = None
    solve_parameter: dict = None
    solver: List[Solver] = None

    def __init__(self, output=None, template=None, stack=None, time_delta=None, **kwargs):
        self.output = str(output)
        if "___save_version__" in kwargs and kwargs["___save_version__"] == "1.0":
            if len(stack) == 2:
                kwargs["stack_reference"] = stack[0]
                stack = [stack[1]]

        self.stack = stack
        if stack is None:
            self.stack = []
        self.stack_parameter = dict(z_project_name=None, z_project_range=0)

        self.state = False
        self.time_delta = time_delta
        self.template = template

        if "stack_reference" in kwargs:
            self.mesh_piv = [None] * (len(self.stack))
        else:
            self.mesh_piv = [None] * (len(self.stack) - 1)
        self.solver = [None] * (len(self.mesh_piv))

        super().__init__(**kwargs)
        print(self)

    def save(self):
        Path(self.output).parent.mkdir(exist_ok=True, parents=True)
        super().save(self.output)

    def on_load(self, filename):
        self.output = str(Path(filename))

    def __repr__(self):
        def filename_to_string(filename):
            if isinstance(filename, list):
                return str(Path(common_start(filename) + "{z}" + common_end(filename)))
            return str(Path(filename))
        folders = [filename_to_string(stack.template) for stack in self.stack]
        base_folder = common_start(folders)
        base_folder = os.sep.join(base_folder.split(os.sep)[:-1])
        indent = "    "
        text = "Result(" + "\n"
        text += indent + "output = " + self.output + "\n"
        if self.template is not None:
            text += indent + "template = " + self.template + "\n"
        text += indent + "stacks = [" + "\n"
        text += indent + indent + "base_folder = " + base_folder + "\n"
        if self.stack_reference is not None:
            if self.stack_reference.template.startswith(base_folder):
                text += indent + indent + "reference = " + self.stack_reference.template[len(base_folder):] + "\n"
            else:
                text += indent + indent + "reference = " + self.stack_reference.template + "\n"
        if self.time_delta is not None:
            text += indent + indent + "time_delta = " + str(self.time_delta) + "\n"
        for stack, filename in zip(self.stack, folders):
            text += indent + indent + filename[len(base_folder):] + " " + str(stack.voxel_size) + " " + str(stack.channels) + "\n"
        text += indent + "]" + "\n"
        if self.piv_parameter:
            text += indent + "piv_parameter = " + str(self.piv_parameter) + "\n"
        if self.interpolate_parameter:
            text += indent + "interpolate_parameter = " + str(self.interpolate_parameter) + "\n"
        if self.solve_parameter:
            text += indent + "solve_parameter = " + str(self.solve_parameter) + "\n"
        text += ")" + "\n"
        return text
