# Ensure package is in Python path
import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TorchNaut"
copyright = "2025, Domokos M. Kelen"
author = "Domokos M. Kelen"
release = "2025-02-19"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Enables extracting docstrings
    "sphinx.ext.napoleon",  # Supports Google- & NumPy-style docstrings
    "sphinx.ext.viewcode",  # Adds source code links
    "sphinx.ext.autosummary",  # Generates function/class summaries
    "myst_parser",  # If you are using Markdown
    "sphinx_autodoc_typehints",  # Optional: Improves type hints in docs
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))  # Adjust if necessary

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,  # Include functions without docstrings
    "show-inheritance": True,
}
autodoc_member_order = "bysource"  # Keeps function order same as in code
autodoc_inherit_docstrings = True  # Inherit docstrings from parent classes
