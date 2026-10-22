"""Regression coverage for PR #94's segmentation, live display and cancel fixes."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from saenopy import Result, Solver, surface_regularization as sr
from saenopy.reconstruction import fit_result, segment_with_params
from test_release import example_result


@pytest.mark.parametrize("channels,params,expected", [(1, {}, 0), (2, {}, 0), (2, {"seg_channel": 1}, 1)])
def test_segmentation_channel_default_and_explicit_selection(monkeypatch, channels, params, expected):
    image = np.broadcast_to(np.arange(channels), (3, 4, 1, 5, channels)).copy()

    class Stack:
        voxel_size = (1., 1., 1.)

        def __getitem__(self, key):
            return image[key]

    monkeypatch.setattr(sr, "auto_threshold", lambda image, **kwargs: 0.5)
    monkeypatch.setattr(sr, "segment_cell", lambda image, *args, **kwargs: (image > 0, np.empty((0, 3))))
    _, selected, _, _, _ = segment_with_params(SimpleNamespace(stacks=[Stack()]), 0, params)
    np.testing.assert_array_equal(selected, image[:, :, 0, :, expected])


@pytest.mark.parametrize("explicit_channel", [None, 0, 1])
def test_legacy_channel_migration_preserves_explicit_selection(explicit_channel):
    data = copy.deepcopy(example_result().to_dict())
    data["___save_version__"] = "1.7"
    if explicit_channel is not None:
        data["solve_parameters"]["seg_channel"] = explicit_channel
    loaded = Result.from_dict(data)
    assert loaded.solve_parameters["seg_channel"] == (0 if explicit_channel is None else explicit_channel)


@pytest.mark.parametrize("stop_after,max_iterations", [(0, 3), (1, 3), (3, 3), (14, 20)])
def test_cancel_at_start_during_fit_or_at_plateau(monkeypatch, stop_after, max_iterations):
    from saenopy.gui.solver.modules.Regularizer import Regularizer
    result = example_result()
    signal = SimpleNamespace(cancel=stop_after == 0)

    def stationary_inner(self, *args):
        self.last_cg_info = dict(iterations=1, relative_residual=0., converged=True)
        self._cg_diagnostics.append(self.last_cg_info)
        return 0.

    def callback(solver, records, *args):
        if records and len(records) - 1 == stop_after:
            signal.cancel = True

    monkeypatch.setattr(Solver, "_solve_regularization_cg", stationary_inner)
    solver = fit_result(result, parameters=dict(max_iterations=max_iterations),
                        callback=callback, cancel_signal=signal)
    assert len(solver.regularisation_results) - 1 == stop_after
    assert solver.regularisation_parameters["cancelled"] is True
    assert Regularizer.check_status(None, result) == ("progress", 0, 1)
    assert np.all(solver.mesh.forces[~solver.mesh.regularisation_mask] == 0)
    assert np.all(solver.mesh.forces_border[solver.mesh.regularisation_mask] == 0)


def test_cancellation_inside_cg_does_not_apply_partial_update(monkeypatch):
    import saenopy.solver as module
    result = example_result()
    before = result.solvers[0].mesh.displacements.copy()
    signal = SimpleNamespace(cancel=False)

    def cancelled_cg(A, b, **kwargs):
        signal.cancel = True
        return np.ones_like(b), dict(iterations=1, relative_residual=.1, converged=False, reason="cancelled")

    monkeypatch.setattr(module, "cg", cancelled_cg)
    solver = fit_result(result, cancel_signal=signal)
    np.testing.assert_array_equal(solver.mesh.displacements, before)
    assert solver.regularisation_parameters["cancelled"] is True
    assert solver.regularisation_parameters["cg_converged"] == [False]


@pytest.mark.parametrize("frames", [1, 2])
def test_saved_cancelled_fit_resumes_from_partial_field(tmp_path, frames):
    from saenopy.gui.solver.modules.Regularizer import Regularizer
    result = example_result()
    if frames == 2:
        fit_result(result)
        result.solvers.append(example_result().solvers[0])
    index = frames - 1
    signal = SimpleNamespace(cancel=False)

    def cancel_after_one(solver, records, *args):
        if len(records) == 2:
            signal.cancel = True

    fit_result(result, index, parameters=dict(prev_t_as_start=True),
               callback=cancel_after_one, cancel_signal=signal)
    partial = result.solvers[index].mesh.displacements.copy()
    fresh_start = (result.solvers[0].mesh.displacements if index else
                   result.solvers[0].mesh.displacements_target).copy()
    assert not np.array_equal(partial, fresh_start)
    result.clear_cache(index)
    path = tmp_path / "cancelled.saenopy"
    result.save(path)
    loaded = Result.load(path)
    assert bool(loaded.solvers[index].regularisation_parameters["cancelled"])
    assert Regularizer.check_status(None, loaded) == ("progress", index, frames)
    starts = []
    fit_result(loaded, index, resume=True,
               callback=lambda solver, records, *args: starts.append(solver.mesh.displacements.copy()) if not records else None)
    np.testing.assert_array_equal(starts[0], partial)
    assert loaded.solvers[index].regularisation_parameters["cancelled"] is False
    assert Regularizer.check_status(None, loaded) == ("finished", frames, frames)
    # Explicit refitting retains the existing previous-frame/measurement start policy.
    starts.clear()
    fit_result(loaded, index,
               callback=lambda solver, records, *args: starts.append(solver.mesh.displacements.copy()) if not records else None)
    np.testing.assert_array_equal(starts[0], fresh_start)


def test_legacy_completed_solver_needs_no_cancel_field():
    from saenopy.gui.solver.modules.Regularizer import Regularizer
    result = example_result()
    fit_result(result)
    del result.solvers[0].regularisation_parameters["cancelled"]
    assert Regularizer.check_status(None, result) == ("finished", 1, 1)


def test_continue_keeps_completed_frames_and_publishes_final_snapshots(monkeypatch):
    import saenopy.reconstruction as reconstruction
    from saenopy.gui.solver.modules.Regularizer import Regularizer
    from saenopy.gui.solver.modules.live_fit import displayed_solver
    result = example_result()
    fit_result(result)
    first = result.solvers[0]
    original = first.mesh.forces.copy()
    result.solvers.extend([example_result().solvers[0], example_result().solvers[0]])
    result.solvers[1].regularisation_parameters = {"cancelled": True}
    result.solvers[1].regularisation_results = np.ones((2, 3))
    result._live_fit_active = True
    result._live_fit_solvers = {0: first}
    result.output = "unused.saenopy"
    saved, cleared, calls, snapshots = [], [], [], []
    result.save = lambda: saved.append(True)
    clear_cache = result.clear_cache
    result.clear_cache = lambda index: (cleared.append(index), clear_cache(index))
    no_op = SimpleNamespace(emit=lambda *args: None)

    def display(result, frame, snapshot):
        result._live_fit_solvers[frame] = snapshot
        snapshots.append((frame, snapshot))

    gui = SimpleNamespace(parent=SimpleNamespace(signal_process_status_update=no_op, result_changed=no_op),
                          iteration_finished=no_op, live_field_ready=SimpleNamespace(emit=display))

    def fit(result, index, callback, resume, **kwargs):
        calls.append((index, resume))
        if index == 2:
            # While frame 2 is running, the viewer must already show frame 1's
            # final result, despite completion inside the display throttle.
            np.testing.assert_array_equal(displayed_solver(result, 1).mesh.forces, np.full((4, 3), 11.))
        solver = result.solvers[index]
        solver.mesh.forces[:] = index
        callback(solver, [], 0, 300)
        solver.mesh.forces[:] = 10 + index
        callback(solver, np.ones((2, 3)), 0, 300)
        solver.regularisation_parameters = copy.deepcopy(first.regularisation_parameters)
        solver.regularisation_results = np.ones((2, 3))
        return solver

    monkeypatch.setattr(reconstruction, "fit_result", fit)
    monkeypatch.setattr("saenopy.gui.solver.modules.Regularizer.time.monotonic", lambda: 10.)
    Regularizer.process(gui, result, result.material_parameters, result.solve_parameters)
    assert calls == [(1, True), (2, False)]
    assert cleared == [1, 2] and len(saved) == 2
    assert result.solvers[0] is first
    np.testing.assert_array_equal(first.mesh.forces, original)
    assert [frame for frame, _ in snapshots] == [1, 1, 2, 2]
    result.solvers[1].mesh.forces[:] = 123
    np.testing.assert_array_equal(displayed_solver(result, 1).mesh.forces, np.full((4, 3), 11.))


def test_gui_cancels_final_frame_without_reporting_done(monkeypatch):
    from saenopy.gui.solver.modules.Regularizer import Regularizer
    from saenopy.gui.solver.modules.live_fit import fit_status_label
    result = example_result()
    result.output = "unused.saenopy"
    result.save = lambda: None
    snapshots = []
    no_op = SimpleNamespace(emit=lambda *args: None)
    gui = SimpleNamespace(parent=SimpleNamespace(signal_process_status_update=no_op, result_changed=no_op),
                          live_field_ready=SimpleNamespace(emit=lambda *args: snapshots.append(args[2])))

    def cancel(result, records, *args):
        if len(records) == 2:
            gui.cancel_p.cancel = True

    gui.iteration_finished = SimpleNamespace(emit=cancel)
    assert Regularizer.process(gui, result, result.material_parameters, result.solve_parameters) == "Terminated"
    assert Regularizer.check_status(None, result) == ("progress", 0, 1)
    assert "partial result" in fit_status_label(result, "Field.")
    np.testing.assert_array_equal(snapshots[-1].mesh.forces, result.solvers[0].mesh.forces)
