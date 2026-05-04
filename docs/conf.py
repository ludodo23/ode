## docs/conf.py — Sphinx configuration for ode

import subprocess, os

from pathlib import Path

version = Path("../VERSION").read_text().strip()

# -- Project info -------------------------------------------------------------
project   = "ode"
copyright = "2026, Ludovic Andrieux"
author    = "Ludovic Andrieux"
release   = version

# -- Extensions ---------------------------------------------------------------
extensions = [
    "breathe",          # bridge Doxygen XML → Sphinx
    "myst_parser",      # parse .md files
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]

# -- MyST (Markdown) ----------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- Breathe ------------------------------------------------------------------
breathe_projects        = {"ode": "../doxygen/xml"}
breathe_default_project = "ode"

# -- HTML output --------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "titles_only": False,
}
html_static_path = ["_static"]

# -- General ------------------------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
