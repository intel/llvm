<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-inter-process-communication:

================================================================================
Inter Process Communication
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------
This extension introduces functionality for allowing processes to share common
objects, such as device USM memory allocations. Doing so lets processes actively
communicate with each other through the devices, by explicitly managing handles
that represent shareable objects for inter-process communication.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_IPC_MEMORY_SUPPORT_EXP

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Inter-Process Communication
   * ${x}IPCGetMemHandleExp
   * ${x}IPCPutMemHandleExp
   * ${x}IPCOpenMemHandleExp
   * ${x}IPCCloseMemHandleExp

Changelog
--------------------------------------------------------------------------------

+-----------+------------------------+
| Revision  | Changes                |
+===========+========================+
| 1.0       | Initial Draft          |
+-----------+------------------------+

Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return true for the new
``${X}_DEVICE_INFO_IPC_MEMORY_SUPPORT_EXP`` device info query.

Contributors
--------------------------------------------------------------------------------

* Larsen, Steffen `steffen.larsen@intel.com <steffen.larsen@intel.com>`_
