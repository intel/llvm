# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import datetime

# -- Project information -----------------------------------------------------

now = datetime.datetime.now()

project = 'oneAPI DPC++ Compiler'
copyright = str(now.year) + ', Intel Corporation'
author = 'Intel Corporation'

# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser'
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'friendly'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'haiku'

# The suffix of source filenames.
source_suffix = ['.rst', '.md']

# Extensions are mostly in asciidoc which has poor support in Sphinx
exclude_patterns = ['extensions/*']

suppress_warnings = [ 'misc.highlighting_failure' ]

def on_missing_reference(app, env, node, contnode):
    if node['reftype'] == 'any':
        contnode['refuri'] = "https://github.com/intel/llvm/tree/sycl/sycl/doc/" + node['reftarget']
        return contnode
    else:
        return None

def setup(app):
    app.connect('missing-reference', on_missing_reference)
