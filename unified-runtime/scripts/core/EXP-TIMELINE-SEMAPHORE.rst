<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-timeline-semaphore:

================================================================================
Timeline Semaphore
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------

A timeline semaphore holds a 64-bit value that only increases. A signal sets a
new, larger value, and a wait completes once the value is greater than or equal
to a target value. Waits do not consume the value, so the same semaphore can be
used for many rounds without a reset, and a wait can be enqueued before the
signal that satisfies it.


API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_TIMELINE_SEMAPHORE_SUPPORT_EXP
    * ${X}_DEVICE_INFO_TIMELINE_SEMAPHORE_MAX_VALUE_EXP
* ${x}_command_t
    * ${X}_COMMAND_TIMELINE_SEMAPHORE_SIGNAL_EXP
    * ${X}_COMMAND_TIMELINE_SEMAPHORE_WAIT_EXP

Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_exp_timeline_semaphore_handle_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}TimelineSemaphoreCreateExp
* ${x}TimelineSemaphoreRetainExp
* ${x}TimelineSemaphoreReleaseExp
* ${x}TimelineSemaphoreHostWaitExp
* ${x}TimelineSemaphoreHostSignalExp
* ${x}EnqueueTimelineSemaphoreSignalExp
* ${x}EnqueueTimelineSemaphoreWaitExp


Changelog
--------------------------------------------------------------------------------

+-----------+-----------------------------------------------+
| Revision  | Changes                                       |
+===========+===============================================+
| 1.0       | Initial Draft                                 |
+-----------+-----------------------------------------------+


Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return true for the new
``${X}_DEVICE_INFO_TIMELINE_SEMAPHORE_SUPPORT_EXP`` device info query, and
return the largest value a semaphore may hold for
``${X}_DEVICE_INFO_TIMELINE_SEMAPHORE_MAX_VALUE_EXP``. The maximum value of a
semaphore is the minimum of that value over the devices of its context, and
initial, signal and wait values above it are rejected with
``${X}_RESULT_ERROR_INVALID_VALUE``.

The enqueue functions honor only their event wait list and, on an in-order
queue, the queue order. They do not act as barriers on out-of-order queues;
callers that need barrier semantics enqueue a barrier first.


Contributors
--------------------------------------------------------------------------------

* Slawomir Ptak
