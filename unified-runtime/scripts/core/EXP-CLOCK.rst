<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-clock:

================================================================================
Clock
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------
This experimental extension enables the sycl_ext_oneapi_clock feature:
https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_clock.asciidoc
It introduces descriptors to query sub-group/work-group/device clock support.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_CLOCK_SUB_GROUP_SUPPORT_EXP
    * ${X}_DEVICE_INFO_CLOCK_WORK_GROUP_SUPPORT_EXP
    * ${X}_DEVICE_INFO_CLOCK_DEVICE_SUPPORT_EXP

Changelog
--------------------------------------------------------------------------------

+-----------+------------------------+
| Revision  | Changes                |
+===========+========================+
| 1.0       | Initial Draft           |
+-----------+------------------------+


Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return ${X}_RESULT_SUCCESS from
the ${x}DeviceGetInfo call with new ${X}_DEVICE_INFO_CLOCK_SUB_GROUP_SUPPORT_EXP,
${X}_DEVICE_INFO_CLOCK_WORK_GROUP_SUPPORT_EXP or ${X}_DEVICE_INFO_CLOCK_DEVICE_SUPPORT_EXP
device descriptors.

Contributors
--------------------------------------------------------------------------------

* Kornev, Nikita `nikita.kornev@intel.com <nikita.kornev@intel.com>`_
