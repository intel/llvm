<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-memory-export:

================================================================================
Memory Export
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.

Motivation
--------------------------------------------------------------------------------


The `DPC++ Memory Export extension` provides new APIs for
allocating and deallocating exportable device memory, and obtaining a handle to
that memory which can be used in external APIs. This is useful when applications
want to share device memory between different APIs.

Without the ability to allocate exportable memory and obtain an interoperable
handle, applications have to copy device memory allocated by one API to
the host, then copy that host memory back to the device in a memory region
allocated by a second API. If the second API modifies that memory, then this
process would have to be repeated in the opposite direction in order for the
first API to see the changes made to that memory.

This extension enables copy-free sharing of SYCL allocated device memory with
external APIs.

Overview
--------------------------------------------------------------------------------
In this document, we propose the following experimental additions to the Unified
Runtime:

* Exporting Memory

  * Allocating an exportable device resident memory region.
  * Obtaining a handle to the allocated exportable memory region, for the
    purpose of using this handle to import the memory in external APIs such as
    Vulkan or DirectX.
  * Freeing the exportable device resident memory region.


Please note that the following enums and types used by this extension are
dependent on the Bindless Images extension. Their definitions are provided in
`exp-bindless-images.yml`.

* ${x}_exp_external_mem_handle_t

* ${x}_exp_external_mem_type_t
    * ${X}_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD
    * ${X}_EXP_EXTERNAL_MEM_TYPE_WIN32_NT

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_MEMORY_EXPORT_EXPORTABLE_DEVICE_MEM_EXP

* ${x}_exp_external_mem_type_t
    * ${X}_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD
    * ${X}_EXP_EXTERNAL_MEM_TYPE_WIN32_NT

* ${x}_function_t
    * ${X}_FUNCTION_MEMORY_EXPORT_ALLOC_EXPORTABLE_MEMORY_EXP
    * ${X}_FUNCTION_MEMORY_EXPORT_EXPORT_MEMORY_HANDLE_EXP
    * ${X}_FUNCTION_MEMORY_EXPORT_FREE_EXPORTABLE_MEMORY_EXP

Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_exp_external_mem_handle_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}MemoryExportAllocExportableMemoryExp
* ${x}MemoryExportFreeExportableMemoryExp
* ${x}MemoryExportExportMemoryHandleExp

Changelog
--------------------------------------------------------------------------------

+----------+----------------------------------------------------------+
| Revision | Changes                                                  |
+==========+==========================================================+
| 1.0      | Initial draft with Level Zero adapter implementation     |
+----------+----------------------------------------------------------+

Contributors
--------------------------------------------------------------------------------

* Przemek Malon `przemek.malon@codeplay.com <przemek.malon@codeplay.com>`_
