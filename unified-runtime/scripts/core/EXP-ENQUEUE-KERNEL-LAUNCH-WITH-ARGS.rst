<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-enqueue-kernel-launch-with-args:

================================================================================
Enqueue Kernel Launch With Args
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.



Motivation
--------------------------------------------------------------------------------

If an application is setting a kernel's args and launching that kernel in the
same place, we can eliminate some overhead by allowing this to be accomplished
with one API call, rather than requiring one call for each argument and one to
launch. This also aligns with developments in the Level Zero backend, as well
as how CUDA and HIP handle kernel args.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_structure_type_t
    ${X}_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES

* ${x}_exp_kernel_arg_type_t
    ${X}_EXP_KERNEL_ARG_TYPE_VALUE
    ${X}_EXP_KERNEL_ARG_TYPE_POINTER
    ${X}_EXP_KERNEL_ARG_TYPE_MEM_OBJ
    ${X}_EXP_KERNEL_ARG_TYPE_LOCAL
    ${X}_EXP_KERNEL_ARG_TYPE_SAMPLER

Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_exp_kernel_arg_mem_obj_tuple_t
* ${x}_exp_kernel_arg_value_t
* ${x}_exp_kernel_arg_properties_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}EnqueueKernelLaunchWithArgsExp

Changelog
--------------------------------------------------------------------------------

+-----------+---------------------------------------------+
| Revision  | Changes                                     |
+===========+=============================================+
| 1.0       | Initial Draft                               |
+-----------+---------------------------------------------+

Support
--------------------------------------------------------------------------------

Adapters must support this feature. A naive implementation can easily be
constructed as a wrapper around the existing APIs for setting kernel args and
launching.

Contributors
--------------------------------------------------------------------------------

* Aaron Greig `aaron.greig@codeplay.com <aaron.greig@codeplay.com>`_
