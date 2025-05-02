<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-cooperative-kernels:

================================================================================
Cooperative Kernels
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------
Cooperative kernels are kernels that use cross-workgroup synchronization
features. All enqueued workgroups must run concurrently for cooperative kernels
to execute without hanging. This experimental feature provides an API for
querying the maximum number of workgroups and launching cooperative kernels.

Any device can support cooperative kernels by restricting the maximum number of
workgroups to 1. Devices that support cross-workgroup synchronization can
specify a larger maximum for a given cooperative kernel.

The functions defined here align with those specified in Level Zero.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_COOPERATIVE_KERNEL_SUPPORT_EXP

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}EnqueueCooperativeKernelLaunchExp
* ${x}KernelSuggestMaxCooperativeGroupCountExp

Changelog
--------------------------------------------------------------------------------
+-----------+---------------------------------------------+
| Revision  | Changes                                     |
+===========+=============================================+
| 1.0       | Initial Draft                               |
+-----------+---------------------------------------------+
| 1.1       | Switch from extension string macro to       |
|           | device info enum for reporting support.     |
+-----------+---------------------------------------------+

Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return ``true`` when
queried for ${X}_DEVICE_INFO_COOPERATIVE_KERNEL_SUPPORT_EXP via
${x}DeviceGetInfo. Conversely, before using any of the functionality defined in
this experimental feature the user *must* use the device query to determine if
the adapter supports this feature.

Contributors
--------------------------------------------------------------------------------
* Michael Aziz `michael.aziz@intel.com <michael.aziz@intel.com>`_
* Aaron Greig `aaron.greig@codeplay.com <aaron.greig@codeplay.com>`_
