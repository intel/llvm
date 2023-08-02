<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-ESIMD:

================================================================================
ESIMD
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.

Motivation
--------------------------------------------------------------------------------
The `DPC++ ESIMD extension <https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md>`_
provides the user the ability to write explicitly vectorized code. In order
for DPC++ PI to query if a backend supports ESIMD, the Unified Runtime
will need a new device info enumeration.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_ESIMD_SUPPORT_EXP

Changelog
--------------------------------------------------------------------------------

+-----------+------------------------+
| Revision  | Changes                |
+===========+========================+
| 1.0       | Initial Draft           |
+-----------+------------------------+


Contributors
--------------------------------------------------------------------------------

* Sarnie, Nick `nick.sarnie@intel.com <nick.sarnie@intel.com>`_
