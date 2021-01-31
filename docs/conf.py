import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../deltaflow/"))

# -- Project information -----------------------------------------------------
project = "DeltaFlow"
copyright = "2021, Aidan Swope"
author = "Aidan Swope"
release = "1.0"

# -- General configuration ---------------------------------------------------
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc configuration ---------------------------------------------------
autodoc_mock_imports = ["jax", "jaxlib", "numpy", "tqdm", "ffmpeg", "matplotlib"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
