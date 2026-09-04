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
objects, such as device USM memory allocations and event objects. Doing so lets
processes actively communicate with each other through the devices, by
explicitly managing handles that represent shareable objects for inter-process
communication.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_IPC_MEMORY_SUPPORT_EXP
    * ${X}_DEVICE_INFO_IPC_EVENT_SUPPORT_EXP
* ${x}_exp_event_flags_t
    * ${X}_EXP_EVENT_FLAG_IPC_EXP

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Inter-Process Communication (Memory)
   * ${x}IPCGetMemHandleExp
   * ${x}IPCPutMemHandleExp
   * ${x}IPCOpenMemHandleExp
   * ${x}IPCCloseMemHandleExp
* Inter-Process Communication (Event)
   * ${x}IPCGetEventHandleExp
   * ${x}IPCPutEventHandleExp
   * ${x}IPCOpenEventHandleExp

Event sharing semantics
--------------------------------------------------------------------------------
An event must be created by ${x}EventCreateExp with the
``${X}_EXP_EVENT_FLAG_IPC_EXP`` flag set in ``${x}_exp_event_desc_t::flags`` to
be eligible for inter-process sharing. The flag cannot be enabled after the
event is created.

An event opened with ``${x}IPCOpenEventHandleExp`` shares state with the source
event that produced the IPC handle. Either event can be signaled, waited on, or
queried, and a state change made through one event is observable through the
other.

An event opened with ``${x}IPCOpenEventHandleExp`` is a normal
``${x}_event_handle_t`` with an initial reference count of 1. It may be passed
to entry points that accept events, retained with ``${x}EventRetain``, and must
be released with ``${x}EventRelease``. On the final release, the adapter
performs the native cleanup required for an opened IPC event.

Events created with ``${X}_EXP_EVENT_FLAG_ENABLE_PROFILING`` cannot be shared
via IPC.

Changelog
--------------------------------------------------------------------------------

+-----------+----------------------------------------------------------+
| Revision  | Changes                                                  |
+===========+==========================================================+
| 1.0       | Initial Draft                                            |
+-----------+----------------------------------------------------------+
| 1.1       | Added IPC event sharing APIs, the                        |
|           | ``IPC_EVENT_SUPPORT_EXP`` device info query, and the     |
|           | ``${X}_EXP_EVENT_FLAG_IPC_EXP`` event creation flag.     |
+-----------+----------------------------------------------------------+

Support
--------------------------------------------------------------------------------

Adapters which support inter-process sharing of memory allocations *must*
return true for the ``${X}_DEVICE_INFO_IPC_MEMORY_SUPPORT_EXP`` device info
query.

Adapters which support inter-process sharing of event objects *must* return
true for the ``${X}_DEVICE_INFO_IPC_EVENT_SUPPORT_EXP`` device info query.

Contributors
--------------------------------------------------------------------------------

* Larsen, Steffen `steffen.larsen@intel.com <steffen.larsen@intel.com>`_
