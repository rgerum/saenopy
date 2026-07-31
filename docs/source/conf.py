# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.metadata
from sphinx_gallery.sorting import FileNameSortKey

project = 'saenopy'
copyright = '2019-2023, Richard Gerum, David Böhringer'
author = 'Richard Gerum'
release = importlib.metadata.metadata('saenopy')['version']

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'sphinx_gallery.gen_gallery',
]

sphinx_gallery_conf = {
     'examples_dirs': '3d_tfm/examples',   # path to your example scripts
     'gallery_dirs': '3d_tfm/auto_examples',  # path to where to save gallery generated output
     'line_numbers': True,
     'download_all_examples': False,
     'min_reported_time': 10,
     'plot_gallery': False,
     'remove_config_comments': True,
     'within_subsection_order': FileNameSortKey,
}

templates_path = ['_templates']
# sphinx-gallery writes a .rst and a matching .ipynb for every example, and
# nbsphinx registers .ipynb as a source suffix, so without this Sphinx sees two
# candidate sources for each gallery page. The notebooks stay on disk and remain
# downloadable from the gallery; they are just not built as pages themselves.
exclude_patterns = [
    '**.ipynb_checkpoints',
    '**/auto_examples/*.ipynb',
    # sphinx-gallery reads this as the gallery header and copies it into
    # auto_examples; it is not a page of its own
    '**/examples/README.rst',
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
