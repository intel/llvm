<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-launch-properties:

================================================================================
LAUNCH Properties
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Terminology
--------------------------------------------------------------------------------
"Launch Properties" is used to indicate optional kernel launch properties that
can be specified at the time of a kernel launch. Such properties can be used to
enable hardware specific kernel launch features.

Motivation
--------------------------------------------------------------------------------
Advances in hardware sometimes require new kernel properties. One example is
distributed shared memory as used by Nvidia Hopper GPUs. Launching a kernel
that supports distributed shared memory requires specifying a set of "cluster"
dimensions, in units of work-groups, over which the shared memory is
"distributed". Additionally some applications require specification of kernel
properties at launch-time.

This extension is a future-proof and portable solution that supports these two
requirements. Instead of using a fixed set of kernel enqueue arguments, the
approach is to introduce the ${x}_exp_launch_property_t type that enables a
more extendable API.

Each ${x}_exp_launch_property_t instance corresponds to a specific kernel
launch property.
Only one new function is introduced: ${x}EnqueueKernelLaunchCustomExp.
${x}EnqueueKernelLaunchCustomExp takes an array of ${x}_exp_launch_property_t
as an argument, and launches a kernel using these properties.
${x}EnqueueKernelLaunchCustomExp corresponds closely to the CUDA Driver API
``cuLaunchKernelEx``.

Many kernel lauch properties can be supported, such as cooperative kernel
launches. As such, eventually this extension should be able to replace the
cooperative kernels Unified-Runtime extension.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_LAUNCH_PROPERTIES_SUPPORT_EXP

* ${x}_exp_launch_property_id_t

Unions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_exp_launch_property_value_t

Structs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_exp_launch_property_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}EnqueueKernelLaunchCustomExp

Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return ``true`` when
queried for ${X}_DEVICE_INFO_LAUNCH_PROPERTIES_SUPPORT_EXP via
${x}DeviceGetInfo. Conversely, before using any of the functionality defined
in this experimental feature the user *must* use the device query to determine
if the adapter supports this feature.


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

Contributors
--------------------------------------------------------------------------------

* JackAKirk `jack.kirk@codeplay.com <jack.kirk@codeplay.com>`_
* Aaron Greig `aaron.greig@codeplay.com <aaron.greig@codeplay.com>`_
