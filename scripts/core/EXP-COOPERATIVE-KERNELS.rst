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

Macros
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${X}_COOPERATIVE_KERNELS_EXTENSION_STRING_EXP

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}EnqueueCooperativeKernelLaunchExp
* ${x}KernelSuggestMaxCooperativeGroupCountExp

Changelog
--------------------------------------------------------------------------------
+-----------+------------------------+
| Revision  | Changes                |
+===========+========================+
| 1.0       | Initial Draft           |
+-----------+------------------------+

Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return the valid string 
defined in ``${X}_COOPERATIVE_KERNELS_EXTENSION_STRING_EXP`` 
as one of the options from ${x}DeviceGetInfo when querying for 
${X}_DEVICE_INFO_EXTENSIONS. Conversely, before using any of the 
functionality defined in this experimental feature the user *must* use the 
device query to determine if the adapter supports this feature.

Contributors
--------------------------------------------------------------------------------
* Michael Aziz `michael.aziz@intel.com <michael.aziz@intel.com>`_
