<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-per-event-profiling:

================================================================================
Per-Event Profiling
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------
Some devices can produce profiling information for individual events without the
owning queue being created with ${X}_QUEUE_FLAG_PROFILING_ENABLE. This extension
adds a device query to report that capability.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_PER_EVENT_PROFILING_SUPPORT_EXP

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
``${X}_DEVICE_INFO_PER_EVENT_PROFILING_SUPPORT_EXP`` device info query.


Contributors
--------------------------------------------------------------------------------

* Krzysztof Weronski `krzysztofx.weronski@intel.com <krzysztofx.weronski@intel.com>`_
