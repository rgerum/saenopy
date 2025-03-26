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
    'sphinxcontrib.jquery',  # to fix read the docs jQuery bug (https://github.com/readthedocs/sphinx_rtd_theme/issues/1452)
]

sphinx_gallery_conf = {
     'examples_dirs': 'examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     'line_numbers': True,
     'download_all_examples': False,
     'min_reported_time': 10,
     'plot_gallery': False,
     'remove_config_comments': True,
     'within_subsection_order': FileNameSortKey,
}

templates_path = ['_templates']
exclude_patterns = ['**.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme

html_theme = 'furo'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']
