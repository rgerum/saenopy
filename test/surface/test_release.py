"""Regression tests for file format 1.8 and the shared reconstruction API."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from saenopy import Result, Solver
from saenopy import physical_regularization as pr
from saenopy import surface_regularization as sr
from saenopy.reconstruction import fit_result, segmented_surface_area
from saenopy.materials import SemiAffineFiberMaterial
from test_physical_regularization import _one_tetra_solver


def example_result():
    result = Result()
    solver = _one_tetra_solver()
    solver.set_target_displacements(solver.mesh.nodes * -0.01, np.array([True, True, True, False]))
    result.solvers = [solver]
    result.mesh_parameters = dict(element_size=14, reference_stack="first", mesh_size="piv")
    result.solve_parameters = dict(alpha=1e8, physical_normalization=True, max_iterations=2)
    result.material_parameters = dict(k=1000, d_0=None, lambda_s=None, d_s=None)
    return result


def test_area_and_counts_exclude_boundary():
    result = example_result()
    solver = result.solvers[0]
    mask_before = solver.mesh.regularisation_mask.copy()
    info = pr.configure_solver(solver, 1e8, 14, np.ones(4, bool), surface_area_m2=3e-10)
    assert info["surface_node_count"] == 3
    assert info["bulk_node_count"] == 0
    assert info["regularized_node_count"] == 3
    assert info["reference_area_m2"] == pytest.approx(1e-10)
    assert info["surface_thickness_m"] == pytest.approx(
        info["regularized_volume_m3"] / 3 / (3e-10 / 3))
    np.testing.assert_array_equal(mask_before, solver.mesh.regularisation_mask)


def test_no_silent_area_estimate():
    with pytest.raises(ValueError, match="provide surface_area"):
        pr.configure_solver(_one_tetra_solver(), 1e8, 14, np.ones(4, bool))


def test_partial_nan_target_is_excluded():
    result = example_result()
    solver = result.solvers[0]
    solver.mesh.displacements_target[0, 1] = np.nan
    fit_result(result)
    assert not solver.mesh.displacements_target_mask[0]
    assert np.isfinite(solver.mesh.displacements).all()
    assert np.isfinite(solver.mesh.forces).all()


def test_normalized_objective_ignores_fixed_target():
    import scipy.sparse as sparse
    solver = _one_tetra_solver()
    solver.mesh.movable = np.array([True, True, True, False])
    solver.set_target_displacements(np.ones((4, 3)), np.ones(4, bool))
    pr.configure_solver(solver, 1e8, 14)
    solver.localweight = np.ones(4)
    solver.K_glo = sparse.eye(12, format="csr")
    solver.mesh.forces[:] = 0
    solver._compute_regularization_a_and_b(1)
    assert np.all(solver.b[-1] == 0)
    records = []
    solver._record_regularization_status(records, 1)
    assert records[-1][1] == pytest.approx(9)


def test_official_v12_names_and_future_version():
    result = example_result()
    data = copy.deepcopy(result.to_dict())
    data["___save_version__"] = "1.2"
    data["stack"] = data.pop("stacks")
    data["solver"] = data.pop("solvers")
    data["solve_parameters"].pop("physical_normalization")
    loaded = Result.from_dict(data)
    assert loaded.___save_version__ == "1.8"
    assert not loaded.solve_parameters["physical_normalization"]
    data = result.to_dict()
    data["___save_version__"] = "2.2"
    with pytest.raises(ValueError, match="newer Saenopy"):
        Result.from_dict(data)


@pytest.mark.parametrize("h,href,p", [(0, 14, 5), (14, np.nan, 5), (14, 14, np.inf)])
def test_invalid_scaling(h, href, p):
    with pytest.raises(ValueError):
        pr.mesh_scale_factor(h, href, p)


def test_reset_restores_methods():
    solver = _one_tetra_solver()
    for _ in range(3):
        pr.configure_solver(solver, 1e8, 14)
        assert "_compute_regularization_a_and_b" in solver.__dict__
        pr.reset_solver(solver)
        assert "_compute_regularization_a_and_b" not in solver.__dict__
        assert not hasattr(solver, "physical_data_weights")
    assert solver._compute_regularization_a_and_b.__func__ is Solver._compute_regularization_a_and_b


def test_surface_floor_and_switches(monkeypatch):
    result = example_result()
    solver = result.solvers[0]
    calls = []
    def fake(self, **kwargs):
        calls.append(kwargs)
        self.regularisation_parameters = dict(kwargs)
        return []
    monkeypatch.setattr(Solver, "solve_regularized", fake)
    geometry = dict(mask=np.array([True, True, False, False]),
                    inside=np.array([False, True, False, False]), area_m2=2e-10, threshold=5.)
    fit_result(result, parameters=dict(surface=True, max_iterations=100), prepared_surface=geometry)
    assert calls[-1]["i_min"] == 60
    assert not solver.mesh.displacements_target_mask[1]
    fit_result(result, parameters=dict(surface=True, max_iterations=2), prepared_surface=geometry)
    assert calls[-1]["i_min"] == 2
    fit_result(result, parameters=dict(surface=False, physical_normalization=False))
    assert solver.mesh.cell_boundary_mask is None
    assert solver.mesh.displacements_target_mask.all()
    assert not hasattr(solver, "physical_data_weights")
    assert calls[-1]["alpha"] == 1e8


def test_segmentation_area_scaling():
    body = np.zeros((12, 14, 16), bool)
    body[3:8, 4:10, 5:12] = True
    area = segmented_surface_area(body, [1, 2, 3])
    assert segmented_surface_area(body, [2, 4, 6]) == pytest.approx(4 * area)
    assert segmented_surface_area(body.transpose(1, 0, 2), [2, 1, 3]) == pytest.approx(area)
    body[0, 4, 5] = True
    with pytest.warns(RuntimeWarning, match="OPEN"):
        segmented_surface_area(body, [1, 2, 3])


def test_surface_fit_segments_current_parameters_without_preview(monkeypatch):
    import saenopy.reconstruction as reconstruction
    result = example_result()
    solver = result.solvers[0]
    result.solve_parameters.update(seg_threshold_factor=0.9, seg_dilate_layers=1)
    geometry = dict(mask=np.array([True, True, False, False]),
                    inside=np.zeros(4, bool), area_m2=2e-10, threshold=5.)
    calls = []
    def prepare(res, index, params):
        calls.append(dict(params))
        return geometry
    monkeypatch.setattr(reconstruction, "prepare_surface", prepare)
    def solve(self, **kwargs):
        self.regularisation_parameters = dict(kwargs)
        return []
    monkeypatch.setattr(Solver, "solve_regularized", solve)
    for previous_mask in (None, np.array([False, False, True, False])):
        solver.mesh.cell_boundary_mask = previous_mask
        fit_result(result, parameters=dict(surface=True, seg_threshold_factor=0.6,
                                           seg_dilate_layers=2))
        assert calls[-1]["seg_threshold_factor"] == 0.6
        assert calls[-1]["seg_dilate_layers"] == 2
        np.testing.assert_array_equal(solver.mesh.cell_boundary_mask, geometry["mask"])
    assert len(calls) == 2


@pytest.mark.parametrize("version", ["1.3", "1.4", "1.5", "1.6", "1.7"])
def test_old_format_migration(version):
    result = example_result()
    data = copy.deepcopy(result.to_dict())
    data["___save_version__"] = version
    # Official files did not contain the fork's normalization setting.
    data["solve_parameters"].pop("physical_normalization")
    data["solvers"].insert(0, None)
    loaded = Result.from_dict(data)
    assert loaded.___save_version__ == "1.8"
    assert "surface_area_method" not in loaded.solve_parameters
    assert loaded.solve_parameters["physical_normalization"] is False
    assert loaded.solvers[0] is None


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_v10_migration_keeps_material_and_solve_parameters_separate(version):
    """The official pre-1.2 layout reused one dict for both parameter groups."""
    result = example_result()
    data = copy.deepcopy(result.to_dict())
    data["___save_version__"] = version
    data["stack"] = []
    data.pop("stacks")
    data["piv_parameter"] = None
    data.pop("piv_parameters")
    data["interpolate_parameter"] = None
    data.pop("mesh_parameters")
    data["solve_parameter"] = dict(
        k=1000, d0=None, lambda_s=None, ds=None, alpha=1e8,
        stepper=0.33, i_max=2, rel_conv_crit=0.01)
    data.pop("material_parameters")
    data.pop("solve_parameters")
    data["mesh_piv"] = None
    data["solver"] = None
    data.pop("solvers")

    loaded = Result.from_dict(data)

    assert loaded.material_parameters["k"] == 1000
    assert loaded.material_parameters["d_0"] is None
    assert loaded.solve_parameters["alpha"] == 1e8
    assert loaded.solve_parameters["step_size"] == 0.33


def test_actual_solves_and_roundtrip(tmp_path):
    result = example_result()
    solver = fit_result(result)
    assert np.isfinite(solver.mesh.displacements).all()
    assert np.all(solver.mesh.forces[~solver.mesh.regularisation_mask] == 0)
    assert np.all(solver.mesh.forces_border[solver.mesh.regularisation_mask] == 0)
    result.solvers.insert(0, None)
    filename = tmp_path / "roundtrip.saenopy"
    result.save(filename)
    loaded = Result.load(filename)
    assert loaded.___save_version__ == "1.8"
    assert loaded.solvers[0] is None
    np.testing.assert_array_equal(loaded.solvers[1].mesh.forces, solver.mesh.forces)
    assert loaded.solvers[1].regularisation_parameters["alpha_visible"] == 1e8
    assert not hasattr(loaded.solvers[1], "physical_data_weights")
    before = filename.read_bytes()
    result.solve_parameters["bad_value"] = object()
    with pytest.raises(ValueError, match="object array"):
        result.save(filename)
    assert filename.read_bytes() == before


def test_gui_export_executes_shared_path(monkeypatch):
    from saenopy.gui.solver.modules.Regularizer import Regularizer
    import saenopy.reconstruction as reconstruction
    result = example_result()
    result.material_parameters_tmp = result.material_parameters.copy()
    result.solve_parameters_tmp = {**result.solve_parameters, "surface": True,
                                   "surface_area_method": "segmented",
                                   "alpha_reference_element_size_um": 16.0}
    imports, code = Regularizer.get_code(SimpleNamespace(result=result))
    calls = []
    monkeypatch.setattr(reconstruction, "fit_result", lambda *args, **kwargs: calls.append(kwargs))
    result.clear_cache = lambda index: calls.append("clear")
    result.save = lambda: calls.append("save")
    exec(compile(imports + code, "exported.py", "exec"), {"results": [result]})
    assert calls[0]["parameters"]["surface"] is True
    assert "surface_area_method" not in calls[0]["parameters"]
    assert calls[1:] == ["clear", "save"]


def test_gui_reset_preserves_boundary_mask():
    result = example_result()
    original = result.solvers[0].mesh.regularisation_mask.copy()
    result.solvers[0].mesh.displacements_target_mask[1] = False
    result.reset_regularisation_results()
    np.testing.assert_array_equal(result.solvers[0].mesh.regularisation_mask, original)
    assert not result.solvers[0].mesh.displacements_target_mask[1]


def test_cancelled_surface_fit_preserves_border_split():
    result = example_result()
    solver = result.solvers[0]
    solver.set_material_model(SemiAffineFiberMaterial(1000, None, None, None))
    pr.solve_regularized(solver, 1e8, 14, surface_mask=np.array([True, True, False, False]),
                         surface_area_m2=2e-10, max_iterations=10,
                         cancel_signal=SimpleNamespace(cancel=True))
    assert len(solver.regularisation_results) == 2
    assert np.all(solver.mesh.forces[~solver.mesh.regularisation_mask] == 0)
    assert np.all(solver.mesh.forces_border[solver.mesh.regularisation_mask] == 0)


def test_gui_mapping_keeps_non_widget_parameters():
    from saenopy.gui.common.PipelineModule import ParameterMapping
    result = example_result()
    result.solve_parameters["alpha_reference_element_size_um"] = 16.0
    result.solve_parameters["surface_min_iterations"] = 90
    widget = SimpleNamespace(valueChanged=SimpleNamespace(connect=lambda fn: None),
                             value=lambda: 1e10, setValue=lambda value: None)
    mapping = ParameterMapping("solve_parameters", {"alpha": widget})
    mapping.setResult(result)
    assert result.solve_parameters_tmp["alpha_reference_element_size_um"] == 16
    assert result.solve_parameters_tmp["surface_min_iterations"] == 90


def test_code_button_writes_valid_utf8_script(monkeypatch, tmp_path):
    from saenopy.gui.common.BatchEvaluateBase import BatchEvaluateBase
    from saenopy.gui.common.code_export import get_code
    from qtpy import QtWidgets
    destination = tmp_path / "export_ä.py"
    code = SimpleNamespace(_source_code="@decorator\ndef code(value):\n    path = value\n")
    filename = "C:\\O'Brien\\Messung_ä\\"
    text = get_code(code, {"value": filename})
    module = SimpleNamespace(get_code=lambda: ("", text))
    fake = SimpleNamespace(list=SimpleNamespace(data=[[None, None, object()]], currentRow=lambda: 0),
                           modules=[module])
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(destination), ""))
    BatchEvaluateBase.generate_code(fake)
    namespace = {}
    exec(compile(destination.read_text(encoding="utf-8"), str(destination), "exec"), namespace)
    assert namespace["path"] == filename


def test_new_surface_roundtrip_can_be_refitted(tmp_path):
    result = example_result()
    geometry = dict(mask=np.array([True, True, False, False]), inside=np.zeros(4, bool),
                    area_m2=2e-10, threshold=5.)
    original_mask = result.solvers[0].mesh.regularisation_mask.copy()
    fit_result(result, parameters={"surface": True}, prepared_surface=geometry)
    filename = tmp_path / "surface.saenopy"
    result.save(filename)
    loaded = Result.load(filename)
    np.testing.assert_array_equal(loaded.solvers[0].mesh.regularisation_mask, original_mask)
    assert loaded.solvers[0].regularisation_parameters["surface_area_m2"] == 2e-10
    fit_result(loaded, prepared_surface=geometry)
    assert loaded.solvers[0].regularisation_parameters["surface_node_count"] == 2
    loaded.clear_cache(0)
    assert loaded.solvers[0].regularisation_parameters["surface_area_m2"] == 2e-10


def test_original_classic_matches_supplied_upstream():
    import importlib.util
    from pathlib import Path
    import os
    upstream = Path(os.environ.get("SAENOPY_UPSTREAM_SOLVER", "__not_configured__"))
    if not upstream.is_file():
        pytest.skip("separate upstream checkout not installed")
    spec = importlib.util.spec_from_file_location("upstream_solver_check", upstream)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fork_result = example_result()
    original = module.Solver()
    mesh = fork_result.solvers[0].mesh
    original.set_nodes(mesh.nodes.copy())
    original.set_tetrahedra(mesh.tetrahedra.copy())
    original.set_target_displacements(mesh.displacements_target.copy(), mesh.regularisation_mask.copy())
    original.set_material_model(SemiAffineFiberMaterial(1000, None, None, None))
    original.solve_regularized(alpha=1e8, max_iterations=5)
    fit_result(fork_result, parameters=dict(surface=False, physical_normalization=False,
                                           max_iterations=5, cg_maxiter_factor=1))
    for name in ("displacements", "forces", "forces_border", "regularisation_mask"):
        np.testing.assert_allclose(getattr(fork_result.solvers[0].mesh, name), getattr(original.mesh, name), rtol=1e-12, atol=1e-25)


def test_surface_and_bulk_counts_use_same_reference_volume():
    solver = _one_tetra_solver()
    classic = pr.configure_solver(solver, 1e9, 7)
    surface = pr.configure_solver(solver, 1e9, 7, np.array([True, True, False, False]), surface_area_m2=2e-10)
    assert surface["surface_node_count"] == surface["bulk_node_count"] == 2
    assert surface["regularized_node_count"] == 4
    assert surface["reference_volume_m3"] == classic["reference_volume_m3"]
    assert surface["effective_bulk_alpha"] == pytest.approx(classic["internal_alpha"])
    solver.localweight = np.ones(4)
    solver._update_local_regularization_weigth("huber")
    np.testing.assert_allclose(solver.localweight[:2], 1)
    np.testing.assert_allclose(solver.localweight[2:], surface["bulk_factor_per_m"])
