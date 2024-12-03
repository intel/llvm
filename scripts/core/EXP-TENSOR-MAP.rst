<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-enqueue-native-command:

================================================================================
Tensor Mapping APIs
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------

Used to target the CUDA entry points ``cuTensorMapEncodeIm2col`` and
``cuTensorMapEncodeTiled``.

For some tensor core operations on ``sm_90+`` Nvidia devices, a tensor
descriptor must be built on the host and passed to the kernel. The interfaces
mentioned above, and mapped to UR in this extension, provide the APIs necessary
to create these tensor descriptor objects, that can then be passed to the
kernels.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

${x}_exp_tensor_map_data_type_flags_t
${x}_exp_tensor_map_interleave_flags_t
${x}_exp_tensor_map_l2_promotion_flags_t
${x}_exp_tensor_map_swizzle_flags_t
${x}_exp_tensor_map_oob_fill_flags_t

Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

${x}_exp_tensor_map_handle_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}TensorMapEncodeIm2ColExp
* ${x}TensorMapEncodeTiledExp

Changelog
--------------------------------------------------------------------------------

+-----------+------------------------+
| Revision  | Changes                |
+===========+========================+
| 1.0       | Initial Draft          |
+-----------+------------------------+


Support
--------------------------------------------------------------------------------

This extension is only supported on the ``UR_PLATFORM_BACKEND_CUDA`` backend.

Contributors
--------------------------------------------------------------------------------

* Hugh Delaney `hugh.delaney@codeplay.com <hugh.delaney@codeplay.com>`_
