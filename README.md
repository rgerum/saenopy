SAENOPY: 3D Traction Force Microscopy with Python
=======

[![DOC](https://readthedocs.org/projects/saenopy/badge/)](https://saenopy.readthedocs.io)
[![Coverage Status](https://coveralls.io/repos/github/rgerum/saenopy/badge.svg?branch=master)](https://coveralls.io/github/rgerum/saenopy?branch=master)
[![PyTest](https://github.com/rgerum/saenopy/actions/workflows/test.yml/badge.svg)](https://github.com/rgerum/saenopy/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="saenopy/img/Logo.png" />
</p>


SAENOPY is a free open source 3D traction force microscopy (TFM) software. Its material model is especially well suited for
tissue-mimicking and typically highly non-linear biopolymer matrices such as collagen, fibrin, or Matrigel. 

It features a python package to use in scripts and an extensive graphical user interface.

Check out our [Documentation](https://saenopy.readthedocs.io) on how to install and use it.

## Installation

### Standalone
To use saenopy without a complicated installation you can use our standalone binaries to get started right away.

**Version 1.0.9** (downloads available after the release is published):

[Windows](https://github.com/rgerum/saenopy/releases/download/v1.0.9/saenopy.exe) · [Linux](https://github.com/rgerum/saenopy/releases/download/v1.0.9/saenopy) · [macOS](https://github.com/rgerum/saenopy/releases/download/v1.0.9/Saenopy.app.zip)

### Using Python

If you are experienced with python or even want to use our Python API, you need to install saenopy as a python package.
Saenopy can be installed directly using pip:

    ``pip install saenopy``

Now you can start the user interface with:

    ``saenopy``

Or by executing the script “gui_master.py” in your python interpreter.

## Getting started
To get started you can have a look at our collection of [example datasets](https://saenopy.readthedocs.io/en/latest/3d_tfm/auto_examples/index.html).

### Fiber orientation

Open **Orientation** to analyze fiber alignment with [CompactionAnalyzer](https://github.com/davidbhr/CompactionAnalyzer):

- **Relative to Cell**: matrix alignment around a cell as a marker of contractility.
- **Relative to X and Y Axes**: alignment to the image axes.
- **Relative to a Point or Line**: click one point for radial alignment, or two points to define a reference line.

For axis, point and line measurements, analyze the whole image or select an area with a mask from the same image or a second image. Preview and adjust the mask before analysis.

Load examples through **Add Measurements → Examples**. Drag saved `.saenopyOrientation` files or their folders into the GUI to inspect results. Use **Data Analysis** to compare alignment, angular deviation, coherency and angle histograms across conditions.

[Orientation guide](docs/source/2.5d_orientation/interface.rst) · [Examples and Python API](https://github.com/davidbhr/CompactionAnalyzer#examples-by-analysis-mode) · [Try it online](https://davidbhr.github.io/CompactionAnalyzer/)

## Preprint
If you want to cite saenopy you can reference our article:

*Dynamic traction force measurements of migrating immune cells in 3D biopolymer matrices*  
David Böhringer, Mar Cóndor, Lars Bischof, Tina Czerwinski, Niklas Gampl, Phuong Anh Ngo, Andreas Bauer, 
Caroline Voskens, Rocío López-Posadas, Kristian Franze, Silvia Budday, Christoph Mark, Ben Fabry, Richard Gerum  
**Nat. Phys. 20, 1816–1823 (2024)**; doi: [https://doi.org/10.1038/s41567-024-02632-8](https://doi.org/10.1038/s41567-024-02632-8)


