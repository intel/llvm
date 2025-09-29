<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-enqueue-timestamp-recording:

================================================================================
Enqueue Timestamp Recording
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------
Currently, the only way to get timestamp information is through enabling
profiling on a queue and retrieving the information from events coming from
commands submitted to it. However, not all systems give full control of the
queue construction to the programmer wanting the profiling information. To amend
this, this extension adds the ability to enqueue a timestamp recording on any
queue, with or without profiling enabled. This event can in turn be queried for
the usual profiling information.


API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_TIMESTAMP_RECORDING_SUPPORT_EXP

* ${x}_command_t
    * ${X}_COMMAND_TIMESTAMP_RECORDING_EXP

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}EnqueueTimestampRecordingExp

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
`${X}_DEVICE_INFO_TIMESTAMP_RECORDING_SUPPORT_EXP` device info query.


Contributors
--------------------------------------------------------------------------------

* Steffen Larsen `steffen.larsen@intel.com <steffen.larsen@intel.com>`_
