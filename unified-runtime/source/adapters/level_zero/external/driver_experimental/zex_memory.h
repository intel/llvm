/*
 * Copyright (C) 2020-2026 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef _ZEX_MEMORY_H
#define _ZEX_MEMORY_H
#if defined(__cplusplus)
#pragma once
#endif

#include <level_zero/ze_api.h>

#include "zex_common.h"

///////////////////////////////////////////////////////////////////////////////
// It indicates that the application wants the L0 driver implementation to use
// memory referenced by **ptr passed in `zeMemAllocHost` or `zeMemAllocShared`.
// Can be set in `ze_host_mem_alloc_flags_t`.
constexpr uint32_t ZEX_HOST_MEM_ALLOC_FLAG_USE_HOST_PTR = ZE_BIT(30);

///////////////////////////////////////////////////////////////////////////////
#ifndef ZEX_MEM_IPC_HANDLES_NAME
/// @brief Multiple IPC handles driver extension name
#define ZEX_MEM_IPC_HANDLES_NAME "ZEX_mem_ipc_handles"
#endif // ZEX_MEM_IPC_HANDLES_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Multiple IPC handles driver extension Version(s)
typedef enum _zex_mem_ipc_handles_version_t {
  ZEX_MEM_IPC_HANDLES_VERSION_1_0 = ZE_MAKE_VERSION(1, 0), ///< version 1.0
  ZEX_MEM_IPC_HANDLES_VERSION_CURRENT =
      ZE_MAKE_VERSION(1, 0), ///< latest known version
  ZEX_MEM_IPC_HANDLES_VERSION_FORCE_UINT32 = 0x7fffffff

} zex_mem_ipc_handles_version_t;

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns an array IPC memory handles for the specified allocation
///
/// @details
///     - Takes a pointer to a device memory allocation and returns an array of
//        IPC memory handle for exporting it for use in another process.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + `ptr` not known
ze_result_t ZE_APICALL zexMemGetIpcHandles(
    ze_context_handle_t hContext, ///< [in] handle of the context object
    const void *ptr, ///< [in] pointer to the device memory allocation
    uint32_t
        *numIpcHandles, ///< [in,out] number of IPC handles associated with the
                        ///< allocation if numIpcHandles is zero, then the
                        ///< driver shall update the value with the total number
                        ///< of IPC handles associated with the allocation.
    ze_ipc_mem_handle_t
        *pIpcHandles ///< [in,out][optional][range(0, *numIpcHandles)] returned
                     ///< array of IPC memory handles
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates an allocation associated with an array of IPC memory handles
///        imported from another process.
///
/// @details
///     - Takes an array of IPC memory handles from a remote process and
///     associates it
///       with a device pointer usable in this process.
///     - The device pointer in this process should not be freed with
///       ::zeMemFree, but rather with ::zeMemCloseIpcHandle.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + handles not known
ze_result_t ZE_APICALL zexMemOpenIpcHandles(
    ze_context_handle_t hContext, ///< [in] handle of the context object
    ze_device_handle_t hDevice, ///< [in] handle of the device to associate with
                                ///< the IPC memory handle
    uint32_t numIpcHandles, ///< [in] number of IPC handles associated with the
                            ///< allocation
    ze_ipc_mem_handle_t *pIpcHandles, ///< [in][range(0, *numIpcHandles)] array
                                      ///< of IPC memory handles
    ze_ipc_memory_flags_t flags,      ///< [in] flags controlling the operation.
                                 ///< must be 0 (default) or a valid combination
                                 ///< of ::ze_ipc_memory_flag_t.
    void **pptr ///< [out] pointer to device allocation in this process
);

#define ZE_RESULT_ERROR_INCOMPATIBLE_RESOURCE                                  \
  EXTENDED_ENUM(ze_result_t, 0x7fff0003)

/// @brief Map device memory allocation to host address space
///
/// @details
///    - The application may call this function from simultaneous threads.
///    - The implementation of this function should be lock-free.
///    - Host address is only valid on the host, using it on device is invalid.
///    - Freeing the memory will unmap the host memory mapping.
///    - The application must only use the memory allocation for the context
///       which was provided during allocation.
///    - The application is responsible for properly flushing/sequencing host
///       writes towards the device memory.
///    - To potentially solve ZE_RESULT_ERROR_INCOMPATIBLE_RESOURCE the user may
///       need to allocate using sharing flags by using
///       $x_external_memory_export_desc_t.
///    - pNext points to extensible list of extensions that may be added in
///    future (none defined yet).
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///     - ::ZE_RESULT_ERROR_INCOMPATIBLE_RESOURCE
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
ze_result_t ZE_APICALL zeIntelMemMapDeviceMemToHost(
    ze_context_handle_t hContext, ///< [in] handle of the context
    const void *ptr, ///< [in] pointer to the device memory to be mapped
    void **pptr,     ///< [out] pointer to the host mapped memory
    void *pNext      ///< [in][optional] must be null or a pointer to an
                     ///< extension-specific structure
);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZEX_MEMORY_H
