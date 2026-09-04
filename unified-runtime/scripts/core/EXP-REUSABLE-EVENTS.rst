<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-reusable-events:

================================================================================
Reusable Events
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------

Some applications need to create event objects independently of queue submission
and then reuse those events for synchronization and profiling across multiple
submissions. This extension introduces explicit event creation for that use case
and allows `${x}EnqueueEventsWaitWithBarrierExt` to signal a caller-provided
event instead of always creating a new one.


API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_REUSABLE_EVENTS_SUPPORT_EXP
* ${x}_structure_type_t
    * ${X}_STRUCTURE_TYPE_EXP_EVENT_DESC

Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_exp_event_flags_t
* ${x}_exp_event_desc_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}EventCreateExp


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
``${X}_DEVICE_INFO_REUSABLE_EVENTS_SUPPORT_EXP`` device info query. A reusable
event may only be passed to `${x}EnqueueEventsWaitWithBarrierExt` for a queue
whose device reports this query as true.

For this extension, signaling reusable events is supported through
`${x}EnqueueEventsWaitWithBarrierExt` by passing a non-``NULL`` `phEvent`
that points to a reusable event created by `${x}EventCreateExp`.

Waiting on reusable events is supported by passing those events as
dependencies in wait lists to enqueue commands.

Reusable events follow the standard event lifetime rules. An event created
through `${x}EventCreateExp` must eventually be released with
`${x}EventRelease`, and applications may use `${x}EventRetain` and
`${x}EventGetInfo` (with `${X}_EVENT_INFO_REFERENCE_COUNT`) for explicit
reference-count management.


Contributors
--------------------------------------------------------------------------------

* Krzysztof Weronski `krzysztofx.weronski@intel.com <krzysztofx.weronski@intel.com>`_
