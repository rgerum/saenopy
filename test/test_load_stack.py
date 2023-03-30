#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np
from mock_dir import MockDir
from saenopy.getDeformations import Stack
from saenopy.result_file import get_stacks
import pytest
import tifffile
from pathlib import Path


def create_tif(filename, y=20, x=10, z=1, rgb=None):
    with tifffile.TiffWriter(filename) as tif:
        for i in range(z):
            if rgb is None:
                tif.write(np.random.rand(y, x))
            else:
                tif.write(np.random.rand(y, x, rgb))


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


def test_stack():
    file_structure = {
        "tmp": {
            "run-1": [f"Pos004_S001_z{z:03d}_ch00.tif" for z in range(2)],
            "run-1-reference": [f"Pos004_S001_z{z:03d}_ch00.tif" for z in range(2)],
            "run-2": [f"Pos004_S001_z{z:03d}_ch{c:02d}.tif" for z in range(2) for c in range(3)],
            "run-2-reference": [f"Pos004_S001_z{z:03d}_ch{c:02d}.tif" for z in range(2) for c in range(3)],
            "run-2-reference_diff_channels": [f"Pos004_S001_z{z:03d}_ch{c:02d}.tif" for z in range(2) for c in range(2)],
            "run-time": [f"Pos004_S001_z{z:03d}_ch{c:02d}_t{t:03d}.tif" for z in range(2) for c in range(3) for t in range(2)],
        }
    }
    with MockDir(file_structure, create_tif):
        # get z of filename
        check_stack(Stack("tmp/run-1/Pos004_S001_z{z}_ch00.tif"), 20, 10, 2, 1)
        # ignore other channels
        check_stack(Stack("tmp/run-2/Pos004_S001_z{z}_ch00.tif"), 20, 10, 2, 1)
        # incorporate other channels
        check_stack(Stack("tmp/run-2/Pos004_S001_z{z}_ch{c:00}.tif"), 20, 10, 2, 3)

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

        # check load stack with reference stack
        results = get_stacks("tmp/run-2/Pos004_S001_z{z}_ch{c:00}.tif", "tmp/run-1", [1, 1, 1],
                             reference_stack="tmp/run-2-reference/Pos004_S001_z{z}_ch{c:00}.tif")
        check_stack(results[0].stack[0], 20, 10, 2, 3)
        check_stack(results[0].stack_reference, 20, 10, 2, 3)

        # check load stack with reference stack
        with pytest.raises(ValueError, match='the active stack and the reference stack also need the same number of channels'):
            get_stacks("tmp/run-2/Pos004_S001_z{z}_ch{c:00}.tif", "tmp/run-1", [1, 1, 1],
                       reference_stack="tmp/run-2-reference_diff_channels/Pos004_S001_z{z}_ch{c:00}.tif")

    with MockDir(file_structure, lambda file: create_tif(file, 13, 11, 10)):
        # use z of layers
        check_stack(Stack("tmp/run-1/Pos004_S001_z000_ch00.tif[z]"), 13, 11, 10, 1)

    with MockDir(file_structure, lambda file: create_tif(file, 13, 11, 10, 3)):
        # with 3 rgb layers
        check_stack(Stack("tmp/run-1/Pos004_S001_z000_ch00.tif[z]"), 13, 11, 10, 1, rgb0=3)

    with MockDir(file_structure, lambda file: create_tif(file, 20, 30, 1)):
        # crop y
        check_stack(Stack("tmp/run-1/Pos004_S001_z{z}_ch00.tif", crop={"y": [5, 15]}), 10, 30, 2, 1)
        # crop x
        check_stack(Stack("tmp/run-1/Pos004_S001_z{z}_ch00.tif", crop={"x": [5, 15]}), 20, 10, 2, 1)

    with MockDir(file_structure, lambda file: create_tif(file, 13, 11, 10)):
        # crop z
        check_stack(Stack("tmp/run-1/Pos004_S001_z000_ch00.tif[z]", crop={"z": [3, 6]}), 13, 11, 3, 1)

    # time
    with MockDir(file_structure, create_tif):
        # check load stack with reference stack
        results = get_stacks("tmp/run-time/Pos004_S001_z{z}_ch{c:00}_t{t}.tif", "tmp/run-1b", [1, 1, 1], time_delta=1)
        assert len(results[0].stack) == 2
        check_stack(results[0].stack[0], 20, 10, 2, 3)
        check_stack(results[0].stack[1], 20, 10, 2, 3)
