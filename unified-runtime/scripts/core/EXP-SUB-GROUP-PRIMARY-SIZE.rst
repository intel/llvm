<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-sub-group-primary-size:

================================================================================
Sub-group primary size
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------
Some devices expose a "primary" sub-group size, which is a device-specific named
size that is independent of the kernels run on it. Usually, this sub-group size
can be specified by name in kernel code, but in order for the host code to know
this size, the corresponding device info query is introduced.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_SUB_GROUP_PRIMARY_SIZE_EXP

Changelog
--------------------------------------------------------------------------------

+-----------+------------------------+
| Revision  | Changes                |
+===========+========================+
| 1.0       | Initial Draft          |
+-----------+------------------------+


Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return ${X}_RESULT_SUCCESS from
the ${x}DeviceGetInfo call with the new ${X}_DEVICE_INFO_SUB_GROUP_PRIMARY_SIZE_EXP
device descriptor.


Contributors
--------------------------------------------------------------------------------

* Steffen Larsen `steffen.larsen@intel.com <steffen.larsen@intel.com>`_
