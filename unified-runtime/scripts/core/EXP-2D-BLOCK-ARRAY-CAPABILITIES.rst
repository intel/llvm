<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-2D-block-array-capabilities:

================================================================================
2D Block Array Capabilities
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------
Some Intel GPU devices support 2D block array operations which may be used to optimize applications on Intel GPUs.
This extension provides a device descriptor which allows to query the 2D block array capabilities of a device.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_2D_BLOCK_ARRAY_CAPABILITIES_EXP

* ${x}_exp_device_2d_block_array_capability_flags_t
    * ${X}_EXP_DEVICE_2D_BLOCK_ARRAY_CAPABILITY_FLAG_LOAD
    * ${X}_EXP_DEVICE_2D_BLOCK_ARRAY_CAPABILITY_FLAG_STORE

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
the ${x}DeviceGetInfo call with the new ${X}_DEVICE_INFO_2D_BLOCK_ARRAY_CAPABILITIES_EXP
device descriptor.


Contributors
--------------------------------------------------------------------------------

* Artur Gainullin `artur.gainullin@intel.com <artur.gainullin@intel.com>`_
