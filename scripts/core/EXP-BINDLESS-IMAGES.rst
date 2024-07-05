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
  * Image arrays
  * Cubemaps
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
    ${X}_STRUCTURE_TYPE_EXP_SAMPLER_ADDR_MODES
    ${X}_STRUCTURE_TYPE_EXP_SAMPLER_CUBEMAP_PROPERTIES

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
    * ${X}_DEVICE_INFO_CUBEMAP_SUPPORT_EXP
    * ${X}_DEVICE_INFO_CUBEMAP_SEAMLESS_FILTERING_SUPPORT_EXP
    * ${X}_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_USM_EXP
    * ${X}_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_EXP
    * ${X}_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_USM_EXP
    * ${X}_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_EXP
    * ${X}_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_3D_USM_EXP
    * ${X}_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_3D_EXP

* ${x}_command_t
    * ${X}_COMMAND_INTEROP_SEMAPHORE_WAIT_EXP
    * ${X}_COMMAND_INTEROP_SEMAPHORE_SIGNAL_EXP

* ${x}_exp_image_copy_flags_t
    * ${X}_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE
    * ${X}_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST
    * ${X}_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE

* ${x}_exp_sampler_cubemap_filter_mode_t
    * ${X}_EXP_SAMPLER_CUBEMAP_FILTER_MODE_SEAMLESS
    * ${X}_EXP_SAMPLER_CUBEMAP_FILTER_MODE_DISJOINTED

* ${x}_exp_external_mem_type_t
    * ${X}_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD
    * ${X}_EXP_EXTERNAL_MEM_TYPE_WIN32_NT
    * ${X}_EXP_EXTERNAL_MEM_TYPE_WIN32_NT_DX12_RESOURCE

* ${x}_exp_external_semaphore_type_t
    * ${X}_EXP_EXTERNAL_SEMAPHORE_TYPE_OPAQUE_FD
    * ${X}_EXP_EXTERNAL_SEMAPHORE_TYPE_WIN32_NT
    * ${X}_EXP_EXTERNAL_SEMAPHORE_TYPE_WIN32_NT_DX12_FENCE

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
    * ${X}_FUNCTION_BINDLESS_IMAGES_IMPORT_EXTERNAL_MEMORY_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_MAP_EXTERNAL_ARRAY_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_RELEASE_INTEROP_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_IMPORT_EXTERNAL_SEMAPHORE_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_DESTROY_EXTERNAL_SEMAPHORE_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_WAIT_EXTERNAL_SEMAPHORE_EXP
    * ${X}_FUNCTION_BINDLESS_IMAGES_SIGNAL_EXTERNAL_SEMAPHORE_EXP

* ${x}_mem_type_t
    * ${X}_MEM_TYPE_IMAGE_CUBEMAP_EXP

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
* ${x}_exp_sampler_addr_modes_t
* ${x}_exp_sampler_cubemap_properties_t

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
   * ${x}BindlessImagesImportExternalMemoryExp
   * ${x}BindlessImagesMapExternalArrayExp
   * ${x}BindlessImagesReleaseInteropExp
   * ${x}BindlessImagesImportExternalSemaphoreExp
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
| 9.0      | Remove layered image properties struct.                     |
+------------------------------------------------------------------------+
| 10.0     | Added cubemap image type, sampling properties, and device   |
|          | queries.                                                    |
+------------------------------------------------------------------------+
| 11.0     | Added device queries for sampled image fetch capabilities.  |
+----------+-------------------------------------------------------------+
| 12.0     | Added image arrays to list of supported bindless images     |
+----------+-------------------------------------------------------------+
| 13.0     || Interop import API has been adapted to cater to multiple   |
|          ||  external memory and semaphore handle types                |
|          || Removed the following APIs:                                |
|          ||  - ImportExternalOpaqueFDExp                               |
|          ||  - ImportExternalSemaphoreOpaqueFDExp                      |
|          || Added the following APIs:                                  |
|          ||  - ImportExternalMemoryExp                                 |
|          ||  - ImportExternalSemaphoreExp                              |
|          || Added the following enums:                                 |
|          ||  - exp_external_mem_type_t                                 |
|          ||  - exp_external_semaphore_type_t                           |
|          || Semaphore oparations now take value parameters which set   |
|          || the state the semaphore should wait on or signal.          |
|          || Introduced resource enums for DX12 interop:                |
|          ||  - ${X}_EXP_EXTERNAL_MEM_TYPE_WIN32_NT_DX12_RESOURCE       |
|          ||  - ${X}_EXP_EXTERNAL_SEMAPHORE_TYPE_WIN32_NT_DX12_FENCE    |
+------------------------------------------------------------------------+

Contributors
--------------------------------------------------------------------------------

* Isaac Ault `isaac.ault@codeplay.com <isaac.ault@codeplay.com>`_
* Duncan Brawley `duncan.brawley@codeplay.com <duncan.brawley@codeplay.com>`_
* Przemek Malon `przemek.malon@codeplay.com <przemek.malon@codeplay.com>`_
* Chedy Najjar `chedy.najjar@codeplay.com <chedy.najjar@codeplay.com>`_
* Sean Stirling `sean.stirling@codeplay.com <sean.stirling@codeplay.com>`_
* Peter Zuzek `peter@codeplay.com peter@codeplay.com <peter@codeplay.com>`_
