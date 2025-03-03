<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-usm-context-memcpy:

================================================================================
USM Context Memcpy
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------

In order to support device globals there's a need for a blocking USM write
operation that doesn't need a queue. This is to facilitate fast initialization
of the device global memory via native APIs that enable this kind of operation.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_USM_CONTEXT_MEMCPY_SUPPORT_EXP

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}USMContextMemcpyExp

Changelog
--------------------------------------------------------------------------------

+-----------+---------------------------+
| Revision  | Changes                   |
+===========+===========================+
| 1.0       | Initial Draft             |
+-----------+---------------------------+


Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return true for the new
``${X}_DEVICE_INFO_USM_CONTEXT_MEMCPY_SUPPORT_EXP`` device info query.


Contributors
--------------------------------------------------------------------------------

* Aaron Greig `aaron.greig@codeplay.com <aaron.greig@codeplay.com>`
