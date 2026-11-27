"""Import validation visits every selected time, channel and Z plane."""
import itertools

import numpy as np
import pytest
import tifffile

from saenopy.result_file import get_iterator, get_stacks


def test_iterator_covers_all_combinations_without_mutating_inputs():
    planes = [{"z": 0}, {"z": 1}]
    combinations = list(get_iterator([0, 1], "t", get_iterator([0, 1], "c", iter(planes))))
    assert [(item["t"], item["c"], item["z"]) for item in combinations] == list(
        itertools.product(range(2), repeat=3))
    assert planes == [{"z": 0}, {"z": 1}]
    assert len({id(item) for item in combinations}) == len(combinations)


@pytest.mark.parametrize("bad_frame", [None, (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("multipage", [False, True])
def test_all_time_channels_checked(tmp_path, bad_frame, multipage):
    for t, c in itertools.product(range(2), repeat=2):
        images = [np.full((20 if (t, c) == bad_frame and z == 1 else 21, 31),
                          100 * t + 10 * c + z, np.uint8) for z in range(2)]
        if multipage:
            with tifffile.TiffWriter(tmp_path / f"volume_t{t}_c{c}.tiff") as writer:
                for image in images:
                    writer.write(image, photometric="minisblack")
        else:
            for z, image in enumerate(images):
                tifffile.imwrite(tmp_path / f"volume_t{t}_c{c}_z{z}.tif", image)
    pattern = ("volume_t{t}_c{c:0}.tiff[z]" if multipage
               else "volume_t{t}_c{c:0}_z{z}.tif")
    output = tmp_path / "results"
    if bad_frame is not None:
        with pytest.raises(ValueError, match="does not match previous shape"):
            get_stacks(tmp_path / pattern, output, (1, 1, 1), time_delta=1)
        assert not list(output.rglob("*.saenopy"))
    else:
        result = get_stacks(tmp_path / pattern, output, (1, 1, 1), time_delta=1)[0]
        assert len(result.stacks) == 2
        for t, stack in enumerate(result.stacks):
            assert stack.shape == (21, 31, 2, 2)
            for c in range(2):
                np.testing.assert_array_equal(stack[0, 0, 0, :, c], [100 * t + 10 * c, 100 * t + 10 * c + 1])


@pytest.mark.parametrize("with_reference", [False, True])
def test_tiff_time_pages_import_and_crop(tmp_path, with_reference):
    for z in range(2):
        tifffile.imwrite(tmp_path / f"volume_z{z}.tiff",
                         np.stack([np.full((5, 7), 10 * t + z, np.uint8) for t in range(3)]),
                         photometric="minisblack")
        tifffile.imwrite(tmp_path / f"reference_z{z}.tiff", np.full((5, 7), 250 + z, np.uint8))
    reference = tmp_path / "reference_z{z}.tiff" if with_reference else None
    result = get_stacks(tmp_path / "volume_z{z}.tiff[t]", tmp_path / "results", (1, 1, 1),
                        time_delta=2, reference_stack=reference, crop={"t": (1, None)})[0]
    assert len(result.stacks) == 2
    for t, stack in enumerate(result.stacks, start=1):
        assert stack.shape == (5, 7, 2, 1)
        np.testing.assert_array_equal(stack[0, 0, 0, :, 0], [10 * t, 10 * t + 1])
    if with_reference:
        np.testing.assert_array_equal(result.stack_reference[0, 0, 0, :, 0], [250, 251])


@pytest.mark.parametrize("extension", ["tif", "tiff"])
@pytest.mark.parametrize("with_reference", [False, True])
def test_tiff_channel_pages_roundtrip(tmp_path, extension, with_reference):
    import saenopy
    for t, z in itertools.product(range(2), repeat=2):
        images = np.stack([np.full((5, 7), 100 * t + 10 * c + z, np.uint8) for c in range(3)])
        tifffile.imwrite(tmp_path / f"volume_t{t}_z{z}.{extension}", images, photometric="minisblack")
    filename = f"volume_t{{t}}_z{{z}}.{extension}"
    template = str(tmp_path / filename) + "[{c:1}]"
    reference = template.replace("{t}", "0").replace("{c:1}", "{c:2}") if with_reference else None
    result = get_stacks(template, tmp_path / "results", (1, 1, 1), time_delta=1,
                        reference_stack=reference)[0]
    surface = saenopy.add_image_source(result, template.replace("{c:1}", "{c:2}"), time_mode="series")
    mask = saenopy.add_image_source(result, template.replace("{z}", "0"), kind="mask", time_mode="series")
    result.save()
    result = saenopy.Result.load(result.output)
    for t, stack in enumerate(result.stacks):
        assert stack.channels == ["1", "0", "2"]
        assert stack.shape == (5, 7, 2, 3)
        for c, page in enumerate((1, 0, 2)):
            np.testing.assert_array_equal(stack[0, 0, 0, :, c], [100 * t + 10 * page, 100 * t + 10 * page + 1])
        source = saenopy.get_image_source(result, surface, t)
        assert source.channels == ["2"]
        np.testing.assert_array_equal(source[0, 0, 0, :, 0], [100 * t + 20, 100 * t + 21])
        source = saenopy.get_image_source(result, mask, t)
        assert source.channels == ["1"] and source.shape == (5, 7, 1, 1)
        np.testing.assert_array_equal(source[:, :, 0, 0, 0], 100 * t + 10)
    if with_reference:
        assert result.stack_reference.channels == ["2", "0", "1"]
        np.testing.assert_array_equal(result.stack_reference[0, 0, 0, :, 0], [20, 21])


def test_tiff_channel_pages_validate_later_channels(tmp_path):
    for t, z in itertools.product(range(2), repeat=2):
        with tifffile.TiffWriter(tmp_path / f"volume_t{t}_z{z}.tiff") as writer:
            for c in range(3):
                shape = (4, 7) if (t, z, c) == (1, 1, 2) else (5, 7)
                writer.write(np.zeros(shape, np.uint8), photometric="minisblack")
    with pytest.raises(ValueError, match=r"volume_t1_z1\.tiff\[2\].*does not match previous shape"):
        get_stacks(str(tmp_path / "volume_t{t}_z{z}.tiff") + "[{c:0}]",
                   tmp_path / "results", (1, 1, 1), time_delta=1)
    assert not list((tmp_path / "results").rglob("*.saenopy"))
