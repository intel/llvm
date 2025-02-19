<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-low-power-events:

================================================================================
Low Power Events
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------

By default, level-zero uses busy polling for waiting on event completion when
performing host-based synchronization through APIs such as `${x}QueueFinish`.
This provides the lowest possible latency for the calling thread, but
it may lead to increased CPU utilization.

This extension introduces a new hint flag for `${x}QueueCreate`, allowing users to
indicate to the runtime that they are willing to sacrifice event completion
latency in order to reduce CPU utilization. This may be implemented using
interrupt-driven event completion, where the calling thread yields until
woken up by the driver.

For applications that want to selectively choose which events should utilize
the low-power mode, this extension also adds a new `${x}EnqueueEventsWaitWithBarrierExt` function.
This enqueue method can be used with an analogous property flag that may cause
its output event to be low-power. This barrier is meant to be used on a regular event
just before calling synchronization APIs (such as `${x}QueueFinish`) to introduce a low-power event.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_LOW_POWER_EVENTS_EXP
* ${x}_queue_flags_t
    * ${X}_QUEUE_FLAG_LOW_POWER_EVENTS_EXP
* ${x}_exp_enqueue_ext_flags_t
    * ${X}_EXP_ENQUEUE_EXT_FLAG_LOW_POWER_EVENTS
* ${x}_structure_type_t
    * {X}_STRUCTURE_TYPE_EXP_ENQUEUE_EXT_PROPERTIES

Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

${x}_exp_enqueue_ext_properties_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}EnqueueEventsWaitWithBarrierExt

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
``${X}_DEVICE_INFO_LOW_POWER_EVENTS_EXP`` device info query.


Contributors
--------------------------------------------------------------------------------

* Piotr Balcer `piotr.balcer@intel.com <piotr.balcer@intel.com>`_
