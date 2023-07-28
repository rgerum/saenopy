import glob
import re
from pathlib import Path
import os
import natsort
from typing import List
import tifffile
from PIL import Image
from saenopy.stack import Stack, format_glob
from saenopy.saveable import Saveable
from saenopy.solver import Solver
from saenopy.get_deformations import PivMesh


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
        match = re.match(r"(.*)\{f\:(\d*)\}\{c\:(\d*)\}(\{t\:(\d*)\})?.lif", filename)
        filename, folder, channel, _, time = match.groups()
        files = glob.glob(filename+".lif", recursive=True)

        output_base = filename
        while "*" in str(output_base):
            output_base = Path(output_base).parent

        for file in files:
            relative = os.path.relpath(file, output_base)
            counts = {}
            from saenopy.gui.common.lif_reader import LifFile
            leica_file = LifFile(file).get_image(folder)
            counts["z"] = leica_file.dims.z
            counts["c"] = leica_file.channels
            if leica_file.dims.t > 1 and time is None:
                counts["t"] = leica_file.dims.t
            results.append({"filename": f"{file[:-4]}{{f:{folder}}}{{c:{channel}}}.lif", "dimensions": counts, "output": Path(output_path) / relative})
            if leica_file.dims.t > 1 and time is None:
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
            if re.match(r"(.*.tif)\[(.*)\]", filename):
                filename, page = re.match(r"(.*.tif)\[(.*)\]", filename).groups()
            # the file needs to exist
            #if not Path(filename).exists():
            #    raise FileNotFoundError(f"Could not find file {filename}")
            if Path(filename).suffix in [".tif", ".tiff"]:
                with tifffile.TiffFile(filename) as f:
                    file_shape = f.pages[0].shape
            else:
                with Image.open(filename) as im:
                    file_shape = (im.height, im.width)
            if shape is None:
                shape = file_shape
            else:
                if file_shape != shape:
                    raise ValueError(f"Shape of file {filename} ({file_shape}) does not match previous shape ({shape})")

        # create the output path
        output = Path(output_path) / os.path.relpath(r1, output_base)
        output = output.parent / output.stem
        output = Path(str(output).replace("*", "").replace("{c}", "{c" + str(channel1) + "}").replace("{c:", "{c") + ".saenopy")

        results.append({"filename": r1, "dimensions": counts, "output": output})
        if "t" in d1.columns:
            results[-1]["times"] = natsort.natsorted(d1.t.unique())
    return results


def normalize_path(template, output):
    template = str(Path(template).absolute())
    return template


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

            output = r1["output"]
            if "t" in r1["dimensions"]:
                stacks = []
                times = r1["times"]
                if (crop is not None) and ("t" in crop):
                    times = times[slice(*crop["t"])]
                for t in times:
                    stacks.append(Stack(normalize_path(r1["filename"].replace("{t}", t), output),
                                        voxel_size, crop=crop))
            else:
                stacks = [Stack(normalize_path(r1["filename"], output), voxel_size, crop=crop)]

            if output.exists():
                if exist_overwrite_callback is not None:
                    mode = exist_overwrite_callback(output)
                    if mode == 0:
                        break
                    if mode == "read":
                        data = Result.load(output)
                        data.is_read = True
                        results.append(data)
                        continue
                elif load_existing is True:
                    data = Result.load(output)
                    data.is_read = True
                    results.append(data)
                    continue

            data = Result(
                output=output,
                template=r1["filename"],
                stack=stacks,
                stack_reference=Stack(normalize_path(r2["filename"], output), voxel_size, crop=crop),
                time_delta=time_delta,
            )
            data.save()
            results.append(data)
    else:
        for r1 in results1:
            output = r1["output"]
            if "t" in r1["dimensions"]:
                stacks = []
                times = r1["times"]
                if crop is not None and "t" in crop:
                    times = times[slice(*crop["t"])]
                for t in times:
                    if r1["filename"].endswith(".lif"):
                        stacks.append(Stack(normalize_path(r1["filename"].replace(".lif", f"{{t:{t}}}.lif"), output), voxel_size, crop=crop))
                    else:
                        stacks.append(Stack(normalize_path(r1["filename"].replace("{t}", t), output), voxel_size, crop=crop))
            else:
                stacks = [Stack(normalize_path(r1["filename"], output), voxel_size, crop=crop)]

            if output.exists():
                if exist_overwrite_callback is not None:
                    mode = exist_overwrite_callback(output)
                    if mode == 0:
                        break
                    if mode == "read":
                        data = Result.load(output)
                        data.is_read = True
                        results.append(data)
                        continue
                elif load_existing is True:
                    data = Result.load(output)
                    data.is_read = True
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


def make_path_relative(template, output):
    template = str(Path(template).absolute())
    output = str(Path(output).absolute())
    # relative and optionally go up to two folders up
    try:
        template = Path(template).relative_to(output)
    except ValueError:
        try:
            template = Path("..") / Path(template).relative_to(Path(output).parent)
        except ValueError:
            try:
                template = Path("../..") / Path(template).relative_to(Path(output).parent.parent)
            except ValueError:
                pass
    return str(template)


def make_path_absolute(template, output):
    if not Path(template).is_absolute():
        return str(Path(output).absolute() / template)
    return str(Path(template).absolute())


class Result(Saveable):
    __save_parameters__ = ['stacks', 'stack_reference', 'template',
                           'time_delta', 'piv_parameters', 'mesh_piv',
                           'mesh_parameters', 'material_parameters',
                           'solve_parameters', 'solvers',
                           '___save_name__', '___save_version__']
    ___save_name__ = "Result"
    ___save_version__ = "1.4"
    output: str = None
    state: False

    stack_parameters: dict = None
    stacks: List[Stack] = None
    stack_reference: Stack = None
    template: str = None
    time_delta: float = None

    piv_parameters: dict = None
    mesh_piv: List[PivMesh] = None

    mesh_parameters: dict = None
    material_parameters: dict = None
    solve_parameters: dict = None
    solvers: List[Solver] = None

    @classmethod
    def from_dict(cls, data_dict):  # pragma: no cover
        def apply_rename(obj_data, rename):
            if isinstance(obj_data, list):
                return [apply_rename(o, rename) for o in obj_data]

            from typing import Callable
            for r in rename:
                if r["new"] is not None:
                    if isinstance(r["old"], Callable):
                        obj_data[r["new"]] = r["old"](obj_data)
                    elif r["old"] in obj_data:
                        obj_data[r["new"]] = obj_data[r["old"]]
                    elif "default" in r:
                        obj_data[r["new"]] = r["default"]
                    else:
                        raise ValueError(f"File does not contain parameter {r['old']} and {r['new']} does not have a "
                                         f"default value.")
                if r.get("renames_child", None) is not None:
                    apply_rename(obj_data[r["new"]], r.get("renames_child", None))

        def apply_delete(obj_data, rename):
            if isinstance(obj_data, list):
                return [apply_delete(o, rename) for o in obj_data]

            for r in rename:
                if r["old"] in obj_data and r["old"] != r["new"]:
                    del obj_data[r["old"]]
                if r.get("renames_child", None) is not None:
                    apply_delete(obj_data[r["new"]], r.get("renames_child", None))

        if data_dict["___save_version__"] < "1.1":
            if len(data_dict["stack"]) == 2:
                data_dict["stack_reference"] = data_dict["stack"][0]
                data_dict["stack"] = [data_dict["stack"][1]]

        if data_dict["___save_version__"] < "1.2":  # pragma: no cover
            print(f"convert old version {data_dict['___save_version__']} to 1.2")
            renames = [
                dict(old="stack", new="stack", renames_child=[
                    dict(old="shape", new="_shape"),
                    dict(old="leica_file", new=None),
                    dict(old="crop", new="crop", default=None),
                    dict(old="packed_files", new="packed_files", default=None),
                ]),
                dict(old="stack_reference", new="stack_reference", renames_child=[
                    dict(old="shape", new="_shape"),
                    dict(old="leica_file", new=None),
                    dict(old="crop", new="crop", default=None),
                    dict(old="packed_files", new="packed_files", default=None),
                ]),
                dict(old="piv_parameter", new="piv_parameters", renames_child=[
                    dict(old="win_um", new="window_size"),
                    dict(old="elementsize", new="element_size"),
                    dict(old="signoise_filter", new="signal_to_noise"),
                ]),
                dict(old="interpolate_parameter", new="mesh_parameters", renames_child=[
                    dict(old="inner_region", new=None),
                    dict(old="thinning_factor", new=None),
                    dict(old=lambda d: ("piv" if d["mesh_size_same"] else (d["mesh_size_x"], d["mesh_size_y"], d["mesh_size_z"])), new="mesh_size"),
                    dict(old="mesh_size_same", new=None),
                    dict(old="mesh_size_x", new=None),
                    dict(old="mesh_size_y", new=None),
                    dict(old="mesh_size_z", new=None),
                ]),
                dict(old="solve_parameter", new="material_parameters", renames_child=[
                    dict(old="d0", new="d_0"),
                    dict(old="ds", new="d_s"),
                    dict(old="alpha", new=None),
                    dict(old="stepper", new=None),
                    dict(old="i_max", new=None),
                    dict(old="rel_conv_crit", new=None),
                ], default=dict(k=1645, d0=0.0008, lambda_s=0.0075, ds=0.033)),
                dict(old="solve_parameter", new="solve_parameters", renames_child=[
                    dict(old="k", new=None),
                    dict(old="d0", new=None),
                    dict(old="lambda_s", new=None),
                    dict(old="ds", new=None),
                    dict(old="stepper", new="step_size"),
                    dict(old="i_max", new="max_iterations"),
                ], default=dict(alpha=1e10, stepper=0.33, i_max=100, rel_conv_crit=0.01)),

                dict(old="mesh_piv", new="mesh_piv", renames_child=[
                    dict(old="R", new="nodes"),
                    dict(old="T", new="tetrahedra"),
                    dict(old="node_vars", new=None),
                    dict(old=lambda d: d["node_vars"]["U_measured"], new="displacements_measured"),
                ]),

                dict(old="solver", new="solver", renames_child=[
                    dict(old=lambda d: dict(
                        nodes=d["R"],
                        tetrahedra=d["T"],
                        displacements=d["U"],
                        displacements_fixed=d.get("U_fixed", None),
                        displacements_target=d["U_target"],
                        displacements_target_mask=d["U_target_mask"],
                        regularisation_mask=d["reg_mask"],
                        movable=d["var"],
                        forces=d["f"],
                        forces_target=d["f_target"],
                        strain_energy=d["E_glo"],
                    ),
                         new="mesh"),
                    dict(old="R", new=None),
                    dict(old="T", new=None),
                    dict(old="U", new=None),
                    dict(old="U_fixed", new=None),
                    dict(old="U_target", new=None),
                    dict(old="U_target_mask", new=None),
                    dict(old="reg_mask", new=None),
                    dict(old="f", new=None),
                    dict(old="f_target", new=None),
                    dict(old="E_glo", new=None),
                    dict(old="var", new=None),
                    dict(old="regularisation_parameters", new="regularisation_parameters", default=None),
                    dict(old="relrec", new="regularisation_results", default=[]),
                    dict(old="material_model", new="material_model", renames_child=[
                        dict(old="d0", new="d_0"),
                        dict(old="ds", new="d_s"),
                    ], default=dict(k=1645, d0=0.0008, lambda_s=0.0075, ds=0.033)),
                ]),
                dict(old="time_delta", new="time_delta", default=None),
                dict(old="stack_parameters", new=None),
            ]
            apply_rename(data_dict, renames)
            apply_delete(data_dict, renames)

            data_dict["___save_version__"] = "1.2"
        if data_dict["___save_version__"] < "1.3":  # pragma: no cover
            print(f"convert old version {data_dict['___save_version__']} to 1.3")
            renames = [
                dict(old="stack", new="stacks"),
                dict(old="solver", new="solvers"),
            ]
            apply_rename(data_dict, renames)
            apply_delete(data_dict, renames)

            data_dict["___save_version__"] = "1.3"
        if data_dict["___save_version__"] < "1.4":  # pragma: no cover
            print(f"convert old version {data_dict['___save_version__']} to 1.4")
            renames = [
                dict(old="solvers", new="solvers", renames_child=[
                    dict(old="mesh", new="mesh", renames_child=[
                        dict(old="cell_boundary_mask", new="cell_boundary_mask", default=None)
                    ]),
                ]),
            ]
            apply_rename(data_dict, renames)
            apply_delete(data_dict, renames)

            data_dict["___save_version__"] = "1.4"
        return super().from_dict(data_dict)

    def __init__(self, output=None, template=None, stack=None, time_delta=None, **kwargs):
        if output is not None:
            self.output = str(Path(output).absolute())

        self.stacks = stack
        if stack is None:
            self.stacks = []
        self.stack_parameters = dict(z_project_name=None, z_project_range=0)

        self.state = False
        self.time_delta = time_delta
        self.template = template

        if "stack_reference" in kwargs:
            self.mesh_piv = [None] * (len(self.stacks))
        else:
            self.mesh_piv = [None] * (len(self.stacks) - 1)
        self.solvers = [None] * (len(self.mesh_piv))

        super().__init__(**kwargs)

        # add a reference to this instance to the stacks, so they know the path
        if output is not None:
            for stack in self.stacks:
                stack.paths_relative(self)
            if self.stack_reference is not None:
                self.stack_reference.paths_relative(self)
            self.template = make_path_relative(self.template, Path(self.output).parent)


        # if demo move parts to simulate empty result
        if os.environ.get("DEMO") == "true":  # pragma: no cover
            self.mesh_piv_demo = self.mesh_piv
            self.solver_demo = self.solvers
            if self.solvers[0] is not None and getattr(self.solvers[0], "regularisation_results", None):
                self.solver_relrec_demo = self.solvers[0].regularisation_results
                self.solvers[0].regularisation_results = None
            if "stack_reference" in kwargs:
                self.mesh_piv = [None] * (len(self.stacks))
            else:
                self.mesh_piv = [None] * (len(self.stacks) - 1)
            self.solvers = [None] * (len(self.mesh_piv))

    def get_absolute_path(self):
        return make_path_absolute(self.template, Path(self.output).parent)

    def get_absolute_path_reference(self):
        return make_path_absolute(self.stack_reference.template, Path(self.output).parent)

    def save(self, filename: str = None, file_format=".saenopy"):
        if filename is not None:
            for stack in self.stacks:
                stack.paths_absolute()
            if self.stack_reference is not None:
                self.stack_reference.paths_absolute()
            self.template = make_path_absolute(self.template, Path(self.output).parent)

            self.output = filename
            for stack in self.stacks:
                stack.paths_relative(self)
            if self.stack_reference is not None:
                self.stack_reference.paths_relative(self)
            self.template = make_path_relative(self.template, Path(self.output).parent)
        Path(self.output).parent.mkdir(exist_ok=True, parents=True)
        super().save(self.output, file_format=file_format)

    def clear_cache(self, solver_id: int):
        # only if there is a solver
        if self.solvers[solver_id] is None:
            return
        # get the solver object and convert the important parts to a dict
        data_dict = self.solvers[solver_id].to_dict()
        # delete the solver in the list so that the garbage collector can remove it
        self.solvers[solver_id] = None
        # create a new solver object from the data
        self.solvers[solver_id] = Solver.from_dict(data_dict)

    def on_load(self, filename: str):
        self.output = str(Path(filename))

        for stack in self.stacks:
            stack.parent = self
        if self.stack_reference is not None:
            self.stack_reference.parent = self

    def __repr__(self):
        folders = [str(Path(stack.template)) for stack in self.stacks]
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
        for stack, filename in zip(self.stacks, folders):
            text += indent + indent + filename[len(base_folder):] + " " + str(stack.voxel_size) + " " + str(stack.channels) + "\n"
        text += indent + "]" + "\n"
        if self.piv_parameters:
            text += indent + "piv_parameters = " + str(self.piv_parameters) + "\n"
        if self.mesh_parameters:
            text += indent + "mesh_parameters = " + str(self.mesh_parameters) + "\n"
        if self.material_parameters:
            text += indent + "material_parameters = " + str(self.material_parameters) + "\n"
        if self.solve_parameters:
            text += indent + "solve_parameters = " + str(self.solve_parameters) + "\n"
        text += ")" + "\n"
        return text
