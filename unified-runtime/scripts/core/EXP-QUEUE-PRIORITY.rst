<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-queue-priority:

================================================================================
Queue Priority
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------

The ${X}_QUEUE_FLAG_PRIORITY_LOW and ${X}_QUEUE_FLAG_PRIORITY_HIGH flags only
allow two priorities in addition to the default one. Some devices support more
priority levels. This extension allows querying the range of priorities
supported by a device and creating queues with a priority from that range.

Lower values indicate higher priority and 0 is the default priority.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_QUEUE_PRIORITY_SUPPORT_EXP
    * ${X}_DEVICE_INFO_QUEUE_PRIORITY_HIGHEST_EXP
    * ${X}_DEVICE_INFO_QUEUE_PRIORITY_LOWEST_EXP
* ${x}_structure_type_t
    * ${X}_STRUCTURE_TYPE_EXP_QUEUE_PRIORITY_PROPERTIES

Structs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_exp_queue_priority_properties_t

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
new ``${X}_DEVICE_INFO_QUEUE_PRIORITY_SUPPORT_EXP`` device info query and
return the supported range for ``${X}_DEVICE_INFO_QUEUE_PRIORITY_HIGHEST_EXP``
and ``${X}_DEVICE_INFO_QUEUE_PRIORITY_LOWEST_EXP``.

${x}QueueCreate *must* return ${X}_RESULT_ERROR_INVALID_QUEUE_PROPERTIES if
${x}_exp_queue_priority_properties_t is combined with
${X}_QUEUE_FLAG_PRIORITY_LOW or ${X}_QUEUE_FLAG_PRIORITY_HIGH.


Contributors
--------------------------------------------------------------------------------

* Gainullin, Artur `artur.gainullin@intel.com <artur.gainullin@intel.com>`_
