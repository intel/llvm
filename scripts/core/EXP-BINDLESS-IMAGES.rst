
<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _exp-bindless-images:

================================================================================
Bindless Images
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.

================================================================================
Terminology
================================================================================
For the purposes of this document, a bindless image is one which provides
access to the underlying data via image reference handles. At the application
level, this allows the user to implement programs where the number of images
is not known at compile-time, and store all handles to images -- irrespective
of varying formats and layouts -- in some container, e.g. a dynamic array.

================================================================================
Motivation
================================================================================
The `DPC++ bindless images extension <https://github.com/intel/llvm/pull/8307>`_
has sought to provide the flexibility of bindless images at the SYCL
application level. This extension has been implemented using the CUDA backend of
the DPC++ PI. With the movement to migrate from PI to the Unified Runtime in
DPC++, as seen in `Port CUDA plugin to Unified Runtime
<https://github.com/intel/llvm/pull/9512/>`_, the Unified Runtime's support for
this experimental feature would enable the DPC++ bindless images extension to be
migrated to UR without issue.

================================================================================
Overview
================================================================================

In this document, we propose the following experimental additions to the Unified
Runtime:

* Bindless images support

  * Sampled images
  * Unsampled images
  * Mipmaps
  * USM backed images

* Interoperability support

  * External memory
  * Semaphores

================================================================================
API
================================================================================

--------------------------------------------------------------------------------
Definitions
--------------------------------------------------------------------------------

* ${x}_exp_sampler_mip_properties_t

The following definitions will be implementation-dependent

* ${x}_exp_image_handle_t
* ${x}_exp_image_mem_handle_t
* ${x}_exp_interop_mem_handle_t
* ${x}_exp_interop_semaphore_handle_t

--------------------------------------------------------------------------------
Enums
--------------------------------------------------------------------------------

* ${x}_device_info_t
* ${x}_command_t
* ${x}_exp_image_copy_flags_t

--------------------------------------------------------------------------------
Interface
--------------------------------------------------------------------------------

* USM
   * ${x}USMPitchedAllocExp

* Bindless Images
   * ${x}BindlessImagesUnsampledImageHandleDestroyExp
   * ${x}BindlessImagesSampledImageHandleDestroyExp
   * ${x}BindlessImagesImageAllocateExp
   * ${x}BindlessImagesImageFreeExp
   * ${x}BindlessImagesUnsampledImageCreateExp
   * ${x}BindlessImagesSampledImageCreateExp
   * ${x}BindlessImagesImageCopyExp
   * ${x}BindlessImagesImageGetInfoExp
   * ${x}BindlessImagesMipmapGetLevelExp
   * ${x}BindlessImagesMipmapFreeExp

* Interop
   * ${x}BindlessImagesImportOpaqueFDExp
   * ${x}BindlessImagesMapExternalArrayExp
   * ${x}BindlessImagesReleaseInteropExp
   * ${x}BindlessImagesImportExternalSemaphoreOpaqueFDExp
   * ${x}BindlessImagesDestroyExternalSemaphoreExp
   * ${x}BindlessImagesWaitExternalSemaphoreExp
   * ${x}BindlessImagesSignalExternalSemaphoreExp


================================================================================
Changelog
================================================================================

+-----------+------------------------+
| Revision  | Changes                |
+===========+========================+
| 1         | Intial Draft           |
+-----------+------------------------+

================================================================================
Contributors
================================================================================

* Isaac Ault `isaac.ault@codeplay.com <isaac.ault@codeplay.com>`_
* Duncan Brawley `duncan.brawley@codeplay.com <duncan.brawley@codeplay.com>`_
* Przemek Malon `przemek.malon@codeplay.com <przemek.malon@codeplay.com>`_
* Chedy Najjar `chedy.najjar@codeplay.com <chedy.najjar@codeplay.com>`_
* Sean Stirling `sean.stirling@codeplay.com <sean.stirling@codeplay.com>`_
* Peter Zuzek `peter@codeplay.com peter@codeplay.com <peter@codeplay.com>`_
