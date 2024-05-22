<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-launch-attributes:

================================================================================
LAUNCH Attributes
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Terminology
--------------------------------------------------------------------------------
"Launch Attribute" is used to indicate an optional kernel attribute that can
be specified at the time of a kernel launch. Such attributes can be used to
enable hardware specific kernel features.

Motivation
--------------------------------------------------------------------------------
Advances in hardware sometimes require new kernel attributes. One example is
distributed shared memory as used by Nvidia Hopper GPUs. Launching a kernel
that supports distributed shared memory requires specifying a thread cluster
dimension over which the shared memory is accessible. Additionally some
applications require specification of kernel attributes at runtime.

This extension is a future-proof and portable solution that supports these two requirements.
Instead of using a fixed set of kernel arguments, the approach is to introduce the
`exp_launch_attribute_t` type that enables a more flexible approach.
Each `exp_launch_attribute_t` instance corresponds to a specific kernel launch attribute.
One new function is introduced; `urEnqueueKernelLaunchCustomExp` takes an
array of `exp_launch_attribute_t` as an argument, and launches a kernel using these
attributes. `urEnqueueKernelLaunchCustomExp` corresponds closely to the CUDA Driver API
`cuLaunchKernelEx`.

Many kernel properties can be supported, such as cooperative kernels. As such,
eventually this extension should be able to replace the cooperative kernels
UR extension.

API
--------------------------------------------------------------------------------

Macros
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${X}_LAUNCH_ATTRIBUTES_EXTENSION_STRING_EXP

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_exp_launch_attribute_id_t

Unions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_exp_launch_attribute_value_t

Structs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}_exp_launch_attribute_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ${x}EnqueueKernelLaunchCustomExp

Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return the valid string
defined in ``${X}_LAUNCH_ATTRIBUTES_EXTENSION_STRING_EXP`` as one of the options from
${x}DeviceGetInfo when querying for ${X}_DEVICE_INFO_EXTENSIONS.

Changelog
--------------------------------------------------------------------------------

+-----------+---------------------------------------------+
| Revision  | Changes                                     |
+===========+=============================================+
| 1.0       | Initial Draft                               |
+-----------+---------------------------------------------+

Contributors
--------------------------------------------------------------------------------

* JackAKirk `jack.kirk@codeplay.com <jack.kirk@codeplay.com>`_
