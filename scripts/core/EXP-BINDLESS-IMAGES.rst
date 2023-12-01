<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-bindless-images:

================================================================================
Bindless Images
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Terminology
--------------------------------------------------------------------------------
For the purposes of this document, a bindless image is one which provides
access to the underlying data via image reference handles. At the application
level, this allows the user to implement programs where the number of images
is not known at compile-time, and store all handles to images -- irrespective
of varying formats and layouts -- in some container, e.g. a dynamic array.


Motivation
--------------------------------------------------------------------------------
The `DPC++ bindless images extension <https://github.com/intel/llvm/pull/8307>`_
has sought to provide the flexibility of bindless images at the SYCL
application level. This extension has been implemented using the CUDA backend of
the DPC++ PI. With the movement to migrate from PI to the Unified Runtime in
DPC++, as seen in `Port CUDA plugin to Unified Runtime
<https://github.com/intel/llvm/pull/9512/>`_, the Unified Runtime's support for
this experimental feature would enable the DPC++ bindless images extension to be
migrated to UR without issue.

Overview
--------------------------------------------------------------------------------
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

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_structure_type_t
    ${X}_STRUCTURE_TYPE_EXP_SAMPLER_MIP_PROPERTIES
    ${X}_STRUCTURE_TYPE_EXP_INTEROP_MEM_DESC
    ${X}_STRUCTURE_TYPE_EXP_INTEROP_SEMAPHORE_DESC
    ${X}_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR
    ${X}_STRUCTURE_TYPE_EXP_WIN32_HANDLE
    ${X}_STRUCTURE_TYPE_EXP_LAYERED_IMAGE_PROPERTIES 
    ${X}_STRUCTURE_TYPE_EXP_SAMPLER_ADDR_MODES

* ${x}_device_info_t
    * ${X}_DEVICE_INFO_BINDLESS_IMAGES_SUPPORT_EXP
    * ${X}_DEVICE_INFO_BINDLESS_IMAGES_SHARED_USM_SUPPORT_EXP
    * ${X}_DEVICE_INFO_BINDLESS_IMAGES_1D_USM_SUPPORT_EXP
    * ${X}_DEVICE_INFO_BINDLESS_IMAGES_2D_USM_SUPPORT_EXP
    * ${X}_DEVICE_INFO_IMAGE_PITCH_ALIGN_EXP
    * ${X}_DEVICE_INFO_MAX_IMAGE_LINEAR_WIDTH_EXP
    * ${X}_DEVICE_INFO_MAX_IMAGE_LINEAR_HEIGHT_EXP
    * ${X}_DEVICE_INFO_MAX_IMAGE_LINEAR_PITCH_EXP
    * ${X}_DEVICE_INFO_MIPMAP_SUPPORT_EXP
    * ${X}_DEVICE_INFO_MIPMAP_ANISOTROPY_SUPPORT_EXP
    * ${X}_DEVICE_INFO_MIPMAP_MAX_ANISOTROPY_EXP
    * ${X}_DEVICE_INFO_MIPMAP_LEVEL_REFERENCE_SUPPORT_EXP
    * ${X}_DEVICE_INFO_INTEROP_MEMORY_IMPORT_SUPPORT_EXP
    * ${X}_DEVICE_INFO_INTEROP_MEMORY_EXPORT_SUPPORT_EXP
    * ${X}_DEVICE_INFO_INTEROP_SEMAPHORE_IMPORT_SUPPORT_EXP
    * ${X}_DEVICE_INFO_INTEROP_SEMAPHORE_EXPORT_SUPPORT_EXP

* ${x}_command_t
    * ${X}_COMMAND_INTEROP_SEMAPHORE_WAIT_EXP
    * ${X}_COMMAND_INTEROP_SEMAPHORE_SIGNAL_EXP

* ${x}_exp_image_copy_flags_t
    * ${X}_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE
    * ${X}_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST
    * ${X}_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE

* ${x}_function_t
    * ${X}_FUNCTION_USM_PITCHED_ALLOC_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_UNSAMPLED_IMAGE_HANDLE_DESTROY_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_SAMPLED_IMAGE_HANDLE_DESTROY_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_IMAGE_ALLOCATE_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_IMAGE_FREE_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_UNSAMPLED_IMAGE_CREATE_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_SAMPLED_IMAGE_CREATE_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_IMAGE_COPY_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_IMAGE_GET_INFO_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_MIPMAP_GET_LEVEL_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_MIPMAP_FREE_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_IMPORT_OPAQUE_FD_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_MAP_EXTERNAL_ARRAY_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_RELEASE_INTEROP_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_IMPORT_EXTERNAL_SEMAPHORE_OPAQUE_FD_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_DESTROY_EXTERNAL_SEMAPHORE_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_WAIT_EXTERNAL_SEMAPHORE_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_SIGNAL_EXTERNAL_SEMAPHORE_EXP

Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_exp_sampler_mip_properties_t
* ${x}_exp_image_handle_t
* ${x}_exp_image_mem_handle_t
* ${x}_exp_interop_mem_handle_t
* ${x}_exp_interop_semaphore_handle_t
* ${x}_exp_interop_mem_desc_t
* ${x}_exp_interop_semaphore_desc_t
* ${x}_exp_file_descriptor_t
* ${x}_exp_win32_handle_t
* ${x}_exp_layered_image_properties_t
* ${x}_exp_sampler_addr_modes_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

Changelog
--------------------------------------------------------------------------------

+----------+----------------------------------------------------------+
| Revision | Changes                                                  |
+==========+==========================================================+
| 1.0      | Initial Draft                                            |
+----------+----------------------------------------------------------+
| 2.0      || Added device parameters to UR functions.                |
|          || Added sub-region copy parameters to image copy function.|
|          || Removed 3D USM capabilities.                            |
|          || Added mip filter mode.                                  |
+----------+----------------------------------------------------------+
| 3.0      | Added device query for bindless images on shared USM     |
+----------+-------------------------------------------------------------+
| 4.0      || Added platform specific interop resource handles.          |
|          || Added and updated to use new interop resource descriptors. |
+----------+-------------------------------------------------------------+
| 5.0      | Update interop struct and func param names to adhere to convention. |
+----------+-------------------------------------------------------------+
| 6.0      | Fix semaphore import function parameter name.               |
+----------+-------------------------------------------------------------+
| 7.0      | Add layered image properties struct.                        |
+----------+-------------------------------------------------------------+
| 8.0      | Added structure for sampler addressing modes per dimension. |
+------------------------------------------------------------------------+

Contributors
--------------------------------------------------------------------------------

* Isaac Ault `isaac.ault@codeplay.com <isaac.ault@codeplay.com>`_
* Duncan Brawley `duncan.brawley@codeplay.com <duncan.brawley@codeplay.com>`_
* Przemek Malon `przemek.malon@codeplay.com <przemek.malon@codeplay.com>`_
* Chedy Najjar `chedy.najjar@codeplay.com <chedy.najjar@codeplay.com>`_
* Sean Stirling `sean.stirling@codeplay.com <sean.stirling@codeplay.com>`_
* Peter Zuzek `peter@codeplay.com peter@codeplay.com <peter@codeplay.com>`_
