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
from docutils import nodes
import re


# -- Project information -----------------------------------------------------

now = datetime.datetime.now()

project = "oneAPI DPC++ Compiler"
copyright = str(now.year) + ", Intel Corporation"
author = "Intel Corporation"

# -- General configuration ---------------------------------------------------

master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["myst_parser"]

# Implicit targets for cross reference
myst_heading_anchors = 5

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "haiku"

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

exclude_patterns = [
    # We are migrating the SYCL extensions to reStructuredText, but some of the
    # older *.rst files in the extensions directories have errors, so ignore
    # them.
    "extensions/supported/C-CXX-StandardLibrary.rst",
    "extensions/supported/sycl_ext_intel_esimd/*",
    # These files are instructions to developers about how to write extensions,
    # so do not build them into the extensions documentation.
    "extensions/template.rst",
    "extensions/*.md",
    "extensions/*/*.md",
    # These OpenCL and SPIR-V extensions are asciidoc which has poor support in
    # Sphinx.
    "design/opencl-extensions/*",
    "design/spirv-extensions/*",
    # Sphinx complains about syntax errors in these files.
    "design/DeviceLibExtensions.rst",
    "design/SYCLPipesLoweringToSPIRV.rst",
    "design/fpga_io_pipes_design.rst",
    "design/Reduction_status.md",
]

suppress_warnings = ["misc.highlighting_failure"]


def on_missing_reference(app, env, node, contnode):
    # Get the directory that contains the *source* file of the link.  These
    # files are always relative to the directory containing "conf.py"
    # (<top>/sycl/doc).  For example, the file "sycl/doc/design/foo.md" will
    # have a directory "design".
    refdoc_components = node["refdoc"].split("/")
    dirs = "/".join(refdoc_components[:-1])
    if dirs:
        dirs += "/"

    # A missing reference usually occurs when the target file of the link is
    # not processed by Sphinx.  Compensate by creating a link that goes to the
    # file's location in the GitHub repo.
    new_target = (
        "https://github.com/intel/llvm/tree/sycl/sycl/doc/" + dirs + node["reftarget"]
    )

    newnode = nodes.reference("", "", internal=False, refuri=new_target)
    newnode.append(contnode)
    return newnode


# These match only relative URLs because ":" is missing from the
# match set, which means they won't match a URL starting with
# "http:", etc.
reRelativeRstUri = re.compile("([a-z0-9_/.-]*)\.rst")
reRelativeAsciidocUri = re.compile("[a-z0-9_/.-]*\.asciidoc")


# We want the extension specification documents to be readable in either of two
# ways:
#
#   * From the HTML that is generated from Sphinx, or
#   * By using a web browser to navigate to the .rst file in the repo.
#
# The second method works because the GitHub server renders an .rst file into
# HTML when serving its contents to the browser.
#
# One challenge with this are cross-files references.  The GitHub server
# approach works well when a reference uses a standard relative URL like:
#
# > This is a `reference`_ to another file.
# >
# > .. _`reference`: relative/path/to/file.rst
#
# However this style of cross-file reference does *not* work well with the
# Sphinx tools.  In Sphinx, you would instead normally use the syntax
# ":doc:`relative/path/to/file` for such a cross-file link.  However, that
# syntax is not understood by the GitHub server.
#
# In order to have cross-file links that work in both scenarios, we use the
# standard relative URL approach (understood by GitHub) and then establish a
# "doctree-resolved" callback to fix up the links in the Sphinx generated HTML.
def on_doctree_resolved(app, doctree, docname):

    # Get the directory that contains the *source* file of the link.  These
    # files are always relative to the directory containing "conf.py"
    # (<top>/sycl/doc).  For example, the file "sycl/doc/extension/supported/foo.rst"
    # will have a directory "extension/supported/".
    refdoc_components = docname.split("/")
    dirs = "/".join(refdoc_components[:-1])
    if dirs:
        dirs += "/"

    # Look for references from this file to other files.
    for ref in doctree.traverse(nodes.reference):
        if "refuri" in ref:
            uri = ref["refuri"]

            # This is a relative link to another .rst file.  We assume the
            # target .rst file was also processed by Sphinx, so we just need to
            # replace the ".rst" suffix with ".html".
            m = reRelativeRstUri.fullmatch(uri)
            if m:
                ref["refuri"] = m[1] + ".html"
                continue

            # This is a relative link to an .asciidoc file.  These files are not
            # processed by Sphinx, so there is no generated HTML.  Instead,
            # change the link to point to the file in the main branch of the
            # GitHub repo.  Although this is not ideal, it's better than a
            # broken link.  We are in the process of migrating the .asciidoc
            # specifications to .rst.  When that is complete, this
            # transformation won't be needed anymore.
            m = reRelativeAsciidocUri.fullmatch(uri)
            if m:
                ref["refuri"] = (
                    "https://github.com/intel/llvm/tree/sycl/sycl/doc/" + dirs + uri
                )
                continue


def setup(app):
    app.connect("missing-reference", on_missing_reference)
    app.connect("doctree-resolved", on_doctree_resolved)
