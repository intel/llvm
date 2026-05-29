<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-usm-host-alloc-register:

================================================================================
USM Host Alloc Register
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------

This extension enables applications to register externally-allocated host
memory (such as memory-mapped files or standard malloc allocations) with the
runtime, allowing the device to access it efficiently as USM host memory
without requiring data copies.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_USM_HOST_ALLOC_REGISTER_SUPPORT_EXP
* ${x}_structure_type_t
    * ${X}_STRUCTURE_TYPE_EXP_USM_HOST_ALLOC_REGISTER_PROPERTIES

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}USMHostAllocRegisterExp
* ${x}USMHostAllocUnregisterExp

Changelog
--------------------------------------------------------------------------------

+-----------+---------------------------+
| Revision  | Changes                   |
+===========+===========================+
| 1.0       | Initial Draft             |
+-----------+---------------------------+


Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return true for the
new ``${X}_DEVICE_INFO_USM_HOST_ALLOC_REGISTER_SUPPORT_EXP`` device info query.


Contributors
--------------------------------------------------------------------------------

* Krzysztof Swiecicki `krzysztof.swiecicki@intel.com <krzysztof.swiecicki@intel.com>`_
