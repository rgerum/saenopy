"""Example variants and combined input/evaluated downloads remain portable."""
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pytest
import tifffile

from saenopy import examples
from saenopy.gui.orientation.modules.example_data import get_orientation_example_files
from saenopy.gui.orientation.modules.result import ResultOrientation


@pytest.mark.parametrize("legacy_shape", [False, True])
def test_rgb_image_shape_survives_saving_and_loading(tmp_path, legacy_shape):
    pixels = np.zeros((32, 48, 3), dtype=np.uint8)
    fiber = tmp_path / "fiber.tif"
    tifffile.imwrite(fiber, pixels)
    result = ResultOrientation(output=str(tmp_path / "fiber.saenopyOrientation"),
                               image_fiber=str(fiber), image_cell=None, pixel_size=.48)
    np.testing.assert_array_equal(result.get_image(1), pixels)
    assert result.shape == (32, 48)
    if legacy_shape:
        result.shape = pixels.shape
    result.save()
    loaded = ResultOrientation.load(result.output)
    assert loaded.shape == (32, 48)
    np.testing.assert_array_equal(loaded.get_image(1), pixels)


@pytest.fixture
def example(tmp_path, monkeypatch):
    monkeypatch.setattr(examples.appdirs, "user_data_dir", lambda *args: str(tmp_path))
    config = examples.get_examples_orientation()["FibroblastActin"]
    # Keep exercising mixed variants for packages/scripts that offer both.
    config["variants"].insert(0, {"name": "whole_image", "analysis_mode": "axis_orientation_only", "input_cell": None})
    for i in (1, 2):
        path = tmp_path / f"Orientation_FibroblastActin/data/field_{i:02d}/phalloidin.tif"
        path.parent.mkdir(parents=True)
        tifffile.imwrite(path, np.arange(32 * 48, dtype=np.uint16).reshape(32, 48))
    return config


def test_variants_keep_modes_and_parameters_when_reopened(example):
    results = get_orientation_example_files(example)
    assert len(results) == 4
    assert [r.analysis_mode for r in results] == ["axis_orientation_only"] * 2 + ["axis_orientation_masked"] * 2
    assert len({r.output for r in results}) == 4
    assert all(r.image_cell is None for r in results[:2])
    assert all(r.image_cell == r.image_fiber for r in results[2:])
    assert all(r.pixel_size == .48 and r.orientation_parameters["sigma_tensor"] == 3 for r in results)
    results[0].orientation_parameters["sigma_tensor"] = 9
    results[0].save()
    loaded = get_orientation_example_files(example, lambda path: "read")
    assert all(r.is_read for r in loaded)
    assert loaded[0].orientation_parameters["sigma_tensor"] == 9
    assert all(r.orientation_parameters["sigma_tensor"] == 3 for r in loaded[1:])
    assert example["orientation_parameters"]["sigma_tensor"] == 3


def test_cancel_stops_all_variants(example):
    results = get_orientation_example_files(example)
    for result in results[2:]:
        Path(result.output).unlink()
    assert get_orientation_example_files(example, lambda path: 0) == []
    assert not any(Path(result.output).exists() for result in results[2:])


def test_legacy_example_defaults_to_cell_analysis(example):
    config = dict(example)
    for name in ("variants", "orientation_parameters", "segmentation_parameters"):
        config.pop(name)
    results = get_orientation_example_files(config)
    assert len(results) == 2
    assert all(r.analysis_mode == "full" and r.image_cell is not None for r in results)
    assert all(r.orientation_parameters is None for r in results)


@pytest.mark.parametrize("name", ["FibroblastActin", "EmptyCollagen", "ActininMask", "SpheroidPoint", "CardiomyocyteLine"])
def test_combined_archive_downloaded_once_and_checks_all_results(tmp_path, monkeypatch, name):
    cache = tmp_path / "cache"
    monkeypatch.setattr(examples.appdirs, "user_data_dir", lambda *args: str(cache))
    config = examples.get_examples_orientation()[name]
    root = Path(config["url"]).stem
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for result_name in config["url_evaluated_file"]:
            z.writestr(root + "/example_output/" + result_name, b"fixture")
    calls = []

    def retrieve(url, destination, callback):
        calls.append(url)
        shutil.copyfile(archive, destination)

    monkeypatch.setattr(examples, "urlretrieve", retrieve)
    results = examples.load_example(name, evaluated=True)
    assert len(results) == len(config["url_evaluated_file"]) and all(path.is_file() for path in results)
    assert calls == [config["url"]]
    assert examples.load_example(name, evaluated=True) == results
    assert len(calls) == 1
    results[-1].unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        examples.load_example(name, evaluated=True)
