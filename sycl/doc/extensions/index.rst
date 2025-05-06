=====================================
DPC++ Extensions to the SYCL Language
=====================================

The DPC++ compiler supports several extensions to the SYCL language, which are
divided into several categories.

Supported Extensions
====================

The "supported" extensions are considered stable.
Future releases of the compiler may add APIs to these extensions, but existing
APIs will generally be compatible with new compiler releases.
If it becomes necessary to drop support for an API, it will first go through a
deprecation period before being removed.

.. toctree::
   :maxdepth: 1

   supported/sycl_ext_oneapi_free_function_queries

Experimental Extensions
=======================

The "experimental" extensions are still in development and are not stable.
They may be changed or even removed in subsequent releases of the compiler
without prior notice.
As a result, they are not recommended for use in production code.

.. toctree::
   :maxdepth: 1

   experimental/sycl_ext_oneapi_work_group_memory

Proposed Extensions
===================

The "proposed" extensions are not implemented, so they cannot be used.
These extension documents are published here to gather community feedback.

.. toctree::
   :maxdepth: 1

   proposed/sycl_ext_oneapi_num_compute_units

Other Extensions
================

The extension documentation is currently being transitioned to Sphinx.
Some extension specifications are still in an older format (generally
AsciiDoc), and those specifications are not listed above.
See `the GitHub repo`_ for those specifications.

.. _`the GitHub repo`: https://github.com/intel/llvm/tree/sycl/sycl/doc/extensions
