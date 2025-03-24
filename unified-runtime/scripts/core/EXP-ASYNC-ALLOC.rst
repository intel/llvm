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
command DAG associated with a queue. Through the enqueue-ordering semantics,
memory allocated within a pool can be reused so as to avoid expensive and 
redundant calls into the OS, which can improve performance.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_ASYNC_USM_ALLOCATIONS_SUPPORT_EXP
* ${x}_usm_pool_flags_t
    * ${X}_USM_POOL_FLAG_USE_NATIVE_MEMORY_POOL_EXP
    * ${X}_USM_POOL_FLAG_READ_ONLY_EXP
* ${x}_usm_pool_info_t
    * ${X}_USM_POOL_INFO_RELEASE_THRESHOLD_EXP
    * ${X}_USM_POOL_INFO_MAXIMUM_SIZE_EXP
    * ${X}_USM_POOL_INFO_RESERVED_CURRENT_EXP
    * ${X}_USM_POOL_INFO_RESERVED_HIGH_EXP
    * ${X}_USM_POOL_INFO_USED_CURRENT_EXP
    * ${X}_USM_POOL_INFO_USED_HIGH_EXP
* ${x}_command_t
    * ${X}_COMMAND_ENQUEUE_USM_DEVICE_ALLOC_EXP
    * ${X}_COMMAND_ENQUEUE_USM_SHARED_ALLOC_EXP
    * ${X}_COMMAND_ENQUEUE_USM_HOST_ALLOC_EXP
    * ${X}_COMMAND_ENQUEUE_USM_FREE_EXP
* ${x}_structure_type_t
    * ${X}_STRUCTURE_TYPE_EXP_ASYNC_USM_ALLOC_PROPERTIES
* ${x}_exp_async_usm_alloc_flags_t

Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_exp_async_usm_alloc_properties_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}EnqueueUSMDeviceAllocExp
* ${x}EnqueueUSMSharedAllocExp
* ${x}EnqueueUSMHostAllocExp
* ${x}EnqueueUSMFreeExp
* ${x}USMPoolCreateExp
* ${x}USMPoolDestroyExp
* ${x}USMPoolGetDefaultDevicePoolExp
* ${x}USMPoolGetInfoExp
* ${x}USMPoolSetInfoExp
* ${x}USMPoolSetDevicePoolExp
* ${x}USMPoolGetDevicePoolExp
* ${x}USMPoolTrimToExp


Changelog
--------------------------------------------------------------------------------

+----------+----------------------------------------------------------+
| Revision | Changes                                                  |
+==========+==========================================================+
| 1.0      | Initial Draft                                            |
+----------+----------------------------------------------------------+
| 1.1      | Fix typos/warnings/descriptions                          |
|          | Change enum values                                       |
|          | Add missing properties/enums/funcs to API list           |
+----------+----------------------------------------------------------+
| 1.2      | Rename DEVICE_INFO_ASYNC_USM_ALLOCATIONS_EXP to          |
|          | DEVICE_INFO_ASYNC_USM_ALLOCATIONS_SUPPORT_EXP for        |
|          | better consistency with other UR enums                   |
+----------+----------------------------------------------------------+
| 1.3      | Remove USMPoolSetThresholdExp                            |
|          | Replace with USMPoolSetInfoExp                           |
+----------+----------------------------------------------------------+

Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return true for the new
``${X}_DEVICE_INFO_ASYNC_USM_ALLOCATIONS_SUPPORT_EXP`` device info query.


Contributors
--------------------------------------------------------------------------------

* Hugh Delaney `hugh.delaney@codeplay.com <hugh.delaney@codeplay.com>`_
* Sean Stirling `sean.stirling@codeplay.com <sean.stirling@codeplay.com>`_
