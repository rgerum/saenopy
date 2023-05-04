#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np
from mock_dir import mock_dir, create_tif, random_path
from saenopy.stack import Stack
from saenopy.result_file import get_stacks
import pytest
import tifffile


@pytest.fixture
def files_one_channel(random_path):
    file_structure = {
        "tmp": {
            "run-1": [f"Pos004_S001_z{z:03d}_ch00.tif" for z in range(2)],
            "run-1-reference": [f"Pos004_S001_z{z:03d}_ch00.tif" for z in range(2)],
        }
    }
    mock_dir(file_structure, callback=lambda file: create_tif(file, x=10, y=20))


@pytest.fixture
def files_z_pages(random_path):
    file_structure = {
        "tmp": {
            "run-1": [f"Pos004_S001_ch00.tif"],
            "run-1-reference": [f"Pos004_S001_ch00.tif"],
        }
    }
    mock_dir(file_structure, callback=lambda file: create_tif(file, x=50, y=50, z=10))


@pytest.fixture
def files_channels(random_path):
    file_structure = {
        "tmp": {
            "run-2": [f"Pos004_S001_z{z:03d}_ch{c:02d}.tif" for z in range(2) for c in range(3)],
            "run-2-reference": [f"Pos004_S001_z{z:03d}_ch{c:02d}.tif" for z in range(2) for c in range(3)],
            "run-2-reference_diff_channels": [f"Pos004_S001_z{z:03d}_ch{c:02d}.tif" for z in range(2) for c in
                                              range(2)],
        }
    }
    mock_dir(file_structure, callback=lambda file: create_tif(file, x=10, y=20))


@pytest.fixture
def files_time(random_path):
    file_structure = {
        "tmp": {
            "run-time": [f"Pos004_S001_z{z:03d}_ch{c:02d}_t{t:03d}.tif" for z in range(2) for c in range(3) for t in
                         range(2)],
        }
    }
    mock_dir(file_structure, callback=lambda file: create_tif(file, x=10, y=20))


def check_stack(stack, y0, x0, z0, c0, rgb0=1):
    y, x, z, c = stack.shape
    # y, x, rgb, z, c
    data = stack[:, :, :, :, :]
    y2, x2, rgb, c2, z2 = data.shape
    assert x == x2 == x0
    assert y == y2 == y0
    assert c2 == c == c0
    assert z == z2 == z0
    assert rgb == rgb0
    data = np.asarray(stack)
    y2, x2, rgb, z2 = data.shape
    assert x == x2 == x0
    assert y == y2 == y0
    assert z == z2 == z0
    assert rgb == rgb0


def test_stack_one_channel(files_one_channel):
    # get z of filename
    check_stack(Stack("tmp/run-1/Pos004_S001_z{z}_ch00.tif", (1, 1, 1)), 20, 10, 2, 1)

    # raise an error if no reference stack and no time is provided
    with pytest.raises(ValueError, match='when not using a time series, a reference stack is required.'):
        get_stacks("tmp/run-1/Pos004_S001_z{z}_ch00.tif", "tmp/run-1", [1, 1, 1])

    # check load stack with reference stack
    results = get_stacks("tmp/run-1/Pos004_S001_z{z}_ch00.tif", "tmp/run-1", [1, 1, 1],
                         reference_stack="tmp/run-1-reference/Pos004_S001_z{z}_ch00.tif")
    check_stack(results[0].stack[0], 20, 10, 2, 1)
    check_stack(results[0].stack_reference, 20, 10, 2, 1)

    with pytest.raises(ValueError, match='if the active stack has channels the reference stack also needs channels'):
        get_stacks("tmp/run-1/Pos004_S001_z{z}_ch{c:00}.tif", "tmp/run-1", [1, 1, 1],
                   reference_stack="tmp/run-1-reference/Pos004_S001_z{z}_ch00.tif")

    # check load stack with reference stack
    results = get_stacks("tmp/run-1/Pos004_S001_z{z}_ch{c:00}.tif", "tmp/run-1", [1, 1, 1],
                         reference_stack="tmp/run-1-reference/Pos004_S001_z{z}_ch{c:00}.tif")
    check_stack(results[0].stack[0], 20, 10, 2, 1)
    check_stack(results[0].stack_reference, 20, 10, 2, 1)

    # raise an error if stacks do not match
    with pytest.raises(ValueError, match='Shape of file .* does not match previous shape .*'):
        tifffile.imwrite("tmp/run-1/Pos004_S001_z001_ch00.tif", np.random.rand(22, 10))
        get_stacks("tmp/run-1/Pos004_S001_z{z}_ch00.tif", "tmp/run-1", [1, 1, 1],
                   reference_stack="tmp/run-1-reference/Pos004_S001_z{z}_ch00.tif")

    # raise an error if stacks do not match reference stack
    with pytest.raises(ValueError, match='active and reference stack need the same number of z slices'):
        tifffile.imwrite("tmp/run-1/Pos004_S001_z000_ch00.tif", np.random.rand(22, 10))
        tifffile.imwrite("tmp/run-1/Pos004_S001_z001_ch00.tif", np.random.rand(22, 10))
        tifffile.imwrite("tmp/run-1/Pos004_S001_z002_ch00.tif", np.random.rand(22, 10))
        get_stacks("tmp/run-1/Pos004_S001_z{z}_ch00.tif", "tmp/run-1", [1, 1, 1],
                   reference_stack="tmp/run-1-reference/Pos004_S001_z{z}_ch00.tif")


def test_stack_channels(files_channels):
    # ignore other channels
    check_stack(Stack("tmp/run-2/Pos004_S001_z{z}_ch00.tif", (1, 1, 1)), 20, 10, 2, 1)
    # incorporate other channels
    check_stack(Stack("tmp/run-2/Pos004_S001_z{z}_ch{c:00}.tif", (1, 1, 1)), 20, 10, 2, 3)

    # check load stack with reference stack
    results = get_stacks("tmp/run-2/Pos004_S001_z{z}_ch{c:00}.tif", "tmp/run-1", [1, 1, 1],
                         reference_stack="tmp/run-2-reference/Pos004_S001_z{z}_ch{c:00}.tif")
    check_stack(results[0].stack[0], 20, 10, 2, 3)
    check_stack(results[0].stack_reference, 20, 10, 2, 3)

    # check load stack with reference stack
    with pytest.raises(ValueError,
                       match='the active stack and the reference stack also need the same number of channels'):
        get_stacks("tmp/run-2/Pos004_S001_z{z}_ch{c:00}.tif", "tmp/run-1", [1, 1, 1],
                   reference_stack="tmp/run-2-reference_diff_channels/Pos004_S001_z{z}_ch{c:00}.tif")


def test_stack_time(files_time):
    # check load stack with reference stack
    results = get_stacks("tmp/run-time/Pos004_S001_z{z}_ch{c:00}_t{t}.tif", "tmp/run-1b", [1, 1, 1], time_delta=1)
    assert len(results[0].stack) == 2
    check_stack(results[0].stack[0], 20, 10, 2, 3)
    check_stack(results[0].stack[1], 20, 10, 2, 3)


def test_crop(files_one_channel):
    # crop y
    check_stack(Stack("tmp/run-1/Pos004_S001_z{z}_ch00.tif", (1, 1, 1), crop={"y": [5, 15]}), 10, 10, 2, 1)
    # crop x
    check_stack(Stack("tmp/run-1/Pos004_S001_z{z}_ch00.tif", (1, 1, 1), crop={"x": [2, 8]}), 20, 6, 2, 1)


def test_crop_z(files_z_pages):
    # use z of layers
    check_stack(Stack("tmp/run-1/Pos004_S001_ch00.tif[z]", (1, 1, 1)), 50, 50, 10, 1)

    # crop z
    check_stack(Stack("tmp/run-1/Pos004_S001_ch00.tif[z]", (1, 1, 1), crop={"z": [3, 6]}), 50, 50, 3, 1)

    # with 3 rgb layers
    #check_stack(Stack("tmp/run-1/Pos004_S001_z000_ch00.tif[z]", (1, 1, 1)), 13, 11, 10, 1, rgb0=3)
