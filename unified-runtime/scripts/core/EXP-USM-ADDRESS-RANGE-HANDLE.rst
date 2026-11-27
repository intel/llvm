<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-usm-address-range-handle:

================================================================================
USM Address Range Handle
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Terminology
--------------------------------------------------------------------------------

Address range handle
    An OS handle, such as a Linux DMA-BUF file descriptor, that names the
    memory backing a range of device addresses.

Location
    The position of the requested address range within the object named by an
    address range handle, described by ``offset``, ``base`` and ``size``. The
    handle names device addresses [``base``, ``base`` + ``size``) and byte
    ``offset`` of the handle corresponds to the requested pointer.

Motivation
--------------------------------------------------------------------------------

Devices such as RDMA network adapters can access device memory directly when
given a DMA-BUF file descriptor for it, e.g. through ``ibv_reg_dmabuf_mr``.
The existing memory export extension only produces such a handle for memory
allocated with ${x}MemoryExportAllocExportableMemoryExp, so device memory
allocated by other means, e.g. ${x}USMDeviceAlloc, cannot be shared this way.

This extension returns an OS handle for the memory backing an address range of
any device USM allocation, or of a mapped virtual memory range.

Device memory is commonly sub-allocated from larger objects, either by the
adapter's USM pools or by the native driver. An OS handle names a whole
object, so this extension always reports where the requested range lies
within it.

The extension maps onto:

* Level Zero: ``zeIntelMemGetHandleForAddressRange``
  (``ZE_intel_experimental_mem_get_handle_for_address_range``).
* CUDA: ``cuMemGetHandleForAddressRange`` with
  ``CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD``. A CUDA implementation would align
  the requested range to the host page size, clamp it to the allocation
  reported by ``cuMemGetAddressRange``, and report the aligned range as the
  location. The CUDA adapter does not implement this yet.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_ADDRESS_RANGE_EXPORT_TYPES_EXP
* ${x}_structure_type_t
    * ${X}_STRUCTURE_TYPE_EXP_ADDRESS_RANGE_HANDLE_PROPERTIES
* ${x}_exp_address_range_export_type_t
    * ${X}_EXP_ADDRESS_RANGE_EXPORT_TYPE_DMA_BUF_FD
* ${x}_exp_address_range_export_type_flags_t
    * ${X}_EXP_ADDRESS_RANGE_EXPORT_TYPE_FLAG_DMA_BUF_FD
* ${x}_exp_address_range_handle_flags_t
* ${x}_function_t
    * ${X}_FUNCTION_USM_GET_HANDLE_FOR_ADDRESS_RANGE_EXP

Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_exp_address_range_handle_properties_t
* ${x}_exp_address_range_handle_value_t
* ${x}_exp_address_range_export_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}USMGetHandleForAddressRangeExp

Usage
--------------------------------------------------------------------------------

.. parsed-literal::

    ${x}_exp_address_range_export_type_flags_t types = 0;
    ${x}DeviceGetInfo(hDevice, ${X}_DEVICE_INFO_ADDRESS_RANGE_EXPORT_TYPES_EXP,
                    sizeof(types), &types, nullptr);
    if (!(types & ${X}_EXP_ADDRESS_RANGE_EXPORT_TYPE_FLAG_DMA_BUF_FD)) {
        return; // not supported
    }

    void *ptr = nullptr;
    ${x}USMDeviceAlloc(hContext, hDevice, nullptr, nullptr, bytes, &ptr);

    ${x}_exp_address_range_export_t exported{};
    ${x}_result_t result = ${x}USMGetHandleForAddressRangeExp(
        hContext, ptr, bytes, ${X}_EXP_ADDRESS_RANGE_EXPORT_TYPE_DMA_BUF_FD,
        nullptr, &exported);
    if (result != ${X}_RESULT_SUCCESS) {
        return; // e.g. the memory cannot be exported
    }

    // Register only the requested bytes. The I/O virtual address is the
    // device address of ptr.
    ibv_mr *mr = ibv_reg_dmabuf_mr(pd, exported.offset, bytes,
                                   reinterpret_cast<uint64_t>(ptr),
                                   exported.handle.fd, access);
    close(exported.handle.fd); // the registration holds its own reference

    // ... RDMA traffic ...

    ibv_dereg_mr(mr); // before the memory is freed
    ${x}USMFree(hContext, ptr);

A virtual memory range may be backed by several objects. In that case, the
handle may cover only a prefix of the requested range, and the remainder is
obtained by calling ${x}USMGetHandleForAddressRangeExp again with
``exported.base + exported.size``:

.. parsed-literal::

    for (size_t done = 0; done < len;) {
        auto *p = static_cast<std::byte *>(va) + done;
        ${x}_exp_address_range_export_t exported{};
        ${x}_result_t result = ${x}USMGetHandleForAddressRangeExp(
            hContext, p, len - done,
            ${X}_EXP_ADDRESS_RANGE_EXPORT_TYPE_DMA_BUF_FD, nullptr, &exported);
        if (result != ${X}_RESULT_SUCCESS) {
            break; // e.g. part of the range is not mapped
        }
        size_t chunk = std::min(
            len - done, static_cast<size_t>(exported.size - exported.offset));
        register_range(exported.handle.fd, exported.offset, chunk, p);
        close(exported.handle.fd);
        done += chunk;
    }

Semantics
--------------------------------------------------------------------------------

* **Location.** The handle names device addresses [``pExport->base``,
  ``pExport->base + pExport->size``), which may extend beyond the requested
  range. ``base`` is aligned to the host page size, so ``offset`` has the same
  remainder modulo the host page size as ``pMem``. On success, if the
  requested range lies within one device USM allocation, it is covered in
  full.

* **Isolation.** The handle is not an isolation boundary. It may give access
  to unrelated allocations that share the backing object. Consumers should
  restrict their own mappings or registrations to the requested range.

* **Ownership.** The caller owns the handle. A DMA-BUF file descriptor is
  released with ``close()``, or by passing it to an API that takes ownership
  of it. There is no ${x} function that releases the handle.

* **Lifetime.** The handle may be closed before or after the memory is freed,
  but must not be used to access the memory after the memory has been freed or
  unmapped. The application must not free or unmap the memory while the
  handle, or anything created from it, is in use. After the memory is freed,
  it may be reused by subsequent allocations, including allocations that share
  the backing object.

* **Visibility.** Device writes are visible through the handle once the host
  has synchronized with the commands that performed them. Writes made through
  the handle are visible to commands enqueued after those writes have
  completed.

* **No concurrent device work.** No commands that access device memory of the
  device owning ``pMem`` may be executing during the call. The implementation
  may need to transform the whole object backing the range, e.g. decompress
  it, which affects unrelated allocations that share that object.

Limitations
--------------------------------------------------------------------------------

* Only Linux DMA-BUF file descriptors are supported.
* Only device memory is supported. Host and shared USM allocations are
  rejected.
* Virtual memory ranges are supported only if the backend can export their
  physical memory. Level Zero requires each range to map a whole physical
  memory object from its start, and CUDA requires the range to be fully
  mapped.
* On Level Zero, compressed memory that cannot be decompressed is rejected.
  Decompression of the backing object is permanent.

Future work
--------------------------------------------------------------------------------

* ${x}_exp_address_range_handle_properties_t is reserved for future controls. One
  candidate is selecting the DMA-BUF mapping path on platforms that can reach
  device memory through more than one hardware path, as in CUDA's
  ``CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE``. Such paths form separate
  ordering domains, which a specification of that control would need to
  address.

Changelog
--------------------------------------------------------------------------------

+-----------+------------------------+
| Revision  | Changes                |
+===========+========================+
| 1.0       | Initial Draft          |
+-----------+------------------------+

Support
--------------------------------------------------------------------------------

Adapters which support this experimental feature *must* return a non-zero set
of flags when ${x}DeviceGetInfo is queried with
${X}_DEVICE_INFO_ADDRESS_RANGE_EXPORT_TYPES_EXP. Before using
${x}USMGetHandleForAddressRangeExp, the user *must* use this query to determine
whether the device supports the required handle type.

Contributors
--------------------------------------------------------------------------------

* Piotr Balcer `piotr.balcer@intel.com <piotr.balcer@intel.com>`_
