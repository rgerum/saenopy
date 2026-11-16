SAENOPY: 3D Traction Force Microscopy with Python
=======

[![DOC](https://app.readthedocs.org/projects/saenopy/badge/?version=latest)](https://saenopy.readthedocs.io/en/latest/)
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

**Latest stable release:** [Windows (EXE)](https://github.com/rgerum/saenopy/releases/latest/download/saenopy.exe) · [Linux](https://github.com/rgerum/saenopy/releases/latest/download/saenopy) · [macOS (ZIP)](https://github.com/rgerum/saenopy/releases/latest/download/Saenopy.app.zip)

These links follow GitHub's latest stable release automatically once its binaries are available. They do not select a development branch or a prerelease.

[All releases](https://github.com/rgerum/saenopy/releases)

**Surface development builds:** [Builds of the Surface branch](https://github.com/rgerum/saenopy/actions/workflows/installer.yml?query=branch%3Aadd_surfaceregularization_and_meshnormalization).
Choose the newest successful run with an available `saenopy_windows` artifact and extract its ZIP to run `saenopy.exe`.
These are experimental builds. Artifact downloads require GitHub sign-in and expire; this build list is separate from the stable download links above.

### Using Python

If you are experienced with python or even want to use our Python API, you need to install saenopy as a python package.
Saenopy can be installed directly using pip:

    ``pip install saenopy``

Now you can start the user interface with:

    ``saenopy``

Or by executing the script “gui_master.py” in your python interpreter.

## Getting started
To get started you can have a look at our collection of [example datasets](https://saenopy.readthedocs.io/en/latest/3d_tfm/auto_examples/index.html).

Standard bulk regularisation does not require a cell stain. Optional surface regularisation uses a segmented cell stain to distinguish cell-surface and bulk forces; see the [regularisation parameters](https://saenopy.readthedocs.io/en/latest/3d_tfm/parameters_regularisation.html).

## Preprint
If you want to cite saenopy you can reference our article:

*Dynamic traction force measurements of migrating immune cells in 3D biopolymer matrices*  
David Böhringer, Mar Cóndor, Lars Bischof, Tina Czerwinski, Niklas Gampl, Phuong Anh Ngo, Andreas Bauer, 
Caroline Voskens, Rocío López-Posadas, Kristian Franze, Silvia Budday, Christoph Mark, Ben Fabry, Richard Gerum  
**Nat. Phys. 20, 1816–1823 (2024)**; doi: [https://doi.org/10.1038/s41567-024-02632-8](https://doi.org/10.1038/s41567-024-02632-8)


