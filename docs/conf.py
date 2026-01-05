# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# -- Project information -----------------------------------------------------

project = "species"
copyright = "2026, Tomas Stolker"
author = "Tomas Stolker"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.

extensions = [
    "sphinx_json",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx"
]

# Disable notebook timeout
nbsphinx_timeout = -1

# Allow errors from notebooks
nbsphinx_allow_errors = True

autoclass_content = "both"

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

exclude_patterns = ["_build",
                    "Thumbs.db",
                    ".DS_Store",
                    "tutorials/.ipynb_checkpoints/*"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinxawesome_theme"

html_theme_options = {
    "show_breadcrumbs": True,
    "show_scrolltop": True,
    "show_prev_next": True,
    "main_nav_links": {
        "GitHub": "https://github.com/tomasstolker/species",
        "PyPI": "https://pypi.org/project/species/",
    },
}

html_context = {
    "github_user": "tomasstolker",
    "github_repo": "species",
    "github_version": "main",
    "doc_path": "docs",
}

html_static_path = []

html_logo = "_static/species_logo.png"
html_search_language = "en"
