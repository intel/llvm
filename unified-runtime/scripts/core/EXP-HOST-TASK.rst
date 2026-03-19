<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-host-task:

================================================================================
Host Task
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------
This API enables launching a host function on a queue. Unlike native commands,
the host function executes asynchronously within the queue context.

CUDA (via cudaLaunchHostFunc) and HIP support this natively.
L0v2 adapter calls native Level Zero API `zeCommandListAppendHostFunction`.

Limitations:
 - Host functions must not call into UR.
 - Access to USM shared memory is not supported on GPUs lacking page fault support.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_ENQUEUE_HOST_TASK_SUPPORT_EXP
* ${x}_command_t
    * ${X}_COMMAND_HOST_TASK_EXP
* ${x}_exp_host_task_flags_t

Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

${x}_exp_host_task_properties_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}EnqueueHostTaskExp

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
``${X}_DEVICE_INFO_ENQUEUE_HOST_TASK_SUPPORT_EXP`` device info query.


Contributors
--------------------------------------------------------------------------------

* Piotr Balcer `piotr.balcer@intel.com <piotr.balcer@intel.com>`_
