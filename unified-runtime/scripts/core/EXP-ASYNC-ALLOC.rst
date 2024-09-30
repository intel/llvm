<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-async-allocations:

================================================================================
Async Allocation Functions
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------

Asynchronous allocations can allow queues to allocate and free memory between
UR command enqueues without forcing synchronization points in the asynchronous
command DAG associated with a queue. This can allow applications to compose
memory allocation and command execution asynchronously, which can improve
performance.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_ENQUEUE_USM_ALLOCATIONS_EXP
* ${x}_command_t
    * ${X}_COMMAND_ENQUEUE_USM_DEVICE_ALLOC_EXP
    * ${X}_COMMAND_ENQUEUE_USM_SHARED_ALLOC_EXP
    * ${X}_COMMAND_ENQUEUE_USM_HOST_ALLOC_EXP
    * ${X}_COMMAND_ENQUEUE_USM_FREE_EXP
* ${x}_exp_enqueue_usm_alloc_flags_t

Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

${x}_exp_enqueue_usm_alloc_properties_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}EnqueueUSMDeviceAllocExp
* ${x}EnqueueUSMSharedAllocExp
* ${x}EnqueueUSMHostAllocExp
* ${x}EnqueueUSMFreeExp

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
``${X}_DEVICE_INFO_ENQUEUE_USM_ALLOCATIONS_EXP`` device info query.


Contributors
--------------------------------------------------------------------------------

* Hugh Delaney `hugh.delaney@codeplay.com <hugh.delaney@codeplay.com>`_
