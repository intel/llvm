<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-device-is-integrated-gpu:

================================================================================
Device is integrated GPU
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------
This experimental extension enables the sycl_ext_oneapi_device_is_integrated_gpu
feature:
http://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_device_is_integrated_gpu.asciidoc.
It introduces descriptor to query if device is integrated GPU.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_IS_INTEGRATED_GPU

Changelog
--------------------------------------------------------------------------------

+-----------+------------------------+
| Revision  | Changes                |
+===========+========================+
| 1.0       | Initial Draft           |
+-----------+------------------------+


Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return ${X}_RESULT_SUCCESS
from the ${x}DeviceGetInfo call with this new ${X}_DEVICE_INFO_IS_INTEGRATED_GPU
device descriptor.

Contributors
--------------------------------------------------------------------------------

* Vodopyanov, Dmitry `dmitry.vodopyanov@intel.com <dmitry.vodopyanov@intel.com>`_
