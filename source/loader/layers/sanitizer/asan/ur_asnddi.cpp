/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_trcddi.cpp
 *
 */

#include "common.h"
#include "sanitizer_interceptor.hpp"
#include "ur_asan_layer.hpp"

#include <iostream>
#include <stdio.h>

namespace ur_asan_layer {

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMHostAlloc
__urdlllocal ur_result_t UR_APICALL urUSMHostAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    const ur_usm_desc_t
        *pUSMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t
        pool, ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t
        size, ///< [in] size in bytes of the USM memory object to be allocated
    void **ppMem ///< [out] pointer to USM host memory object
) {
    auto pfnHostAlloc = context.urDdiTable.USM.pfnHostAlloc;

    if (nullptr == pfnHostAlloc) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    context.logger.debug("=== urUSMHostAlloc");

    return context.interceptor->allocateMemory(
        hContext, nullptr, pUSMDesc, pool, size, ppMem, USMMemoryType::HOST);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMDeviceAlloc
__urdlllocal ur_result_t UR_APICALL urUSMDeviceAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_usm_desc_t
        *pUSMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t
        pool, ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t
        size, ///< [in] size in bytes of the USM memory object to be allocated
    void **ppMem ///< [out] pointer to USM device memory object
) {
    auto pfnDeviceAlloc = context.urDdiTable.USM.pfnDeviceAlloc;

    if (nullptr == pfnDeviceAlloc) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    context.logger.debug("=== urUSMDeviceAlloc");

    return context.interceptor->allocateMemory(
        hContext, hDevice, pUSMDesc, pool, size, ppMem, USMMemoryType::DEVICE);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMSharedAlloc
__urdlllocal ur_result_t UR_APICALL urUSMSharedAlloc(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_usm_desc_t *
        pUSMDesc, ///< [in][optional] Pointer to USM memory allocation descriptor.
    ur_usm_pool_handle_t
        pool, ///< [in][optional] Pointer to a pool created using urUSMPoolCreate
    size_t
        size, ///< [in] size in bytes of the USM memory object to be allocated
    void **ppMem ///< [out] pointer to USM shared memory object
) {
    auto pfnSharedAlloc = context.urDdiTable.USM.pfnSharedAlloc;

    if (nullptr == pfnSharedAlloc) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    context.logger.debug("=== urUSMSharedAlloc");

    return context.interceptor->allocateMemory(
        hContext, hDevice, pUSMDesc, pool, size, ppMem, USMMemoryType::SHARE);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMFree
__urdlllocal ur_result_t UR_APICALL urUSMFree(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem                    ///< [in] pointer to USM memory object
) {
    auto pfnFree = context.urDdiTable.USM.pfnFree;

    if (nullptr == pfnFree) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    context.logger.debug("=== urUSMFree");

    return context.interceptor->releaseMemory(hContext, pMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelCreate
__urdlllocal ur_result_t UR_APICALL urKernelCreate(
    ur_program_handle_t hProgram, ///< [in] handle of the program instance
    const char *pKernelName,      ///< [in] pointer to null-terminated string.
    ur_kernel_handle_t
        *phKernel ///< [out] pointer to handle of kernel object created.
) {
    auto pfnCreate = context.urDdiTable.Kernel.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }
    // context.logger.debug("=== urKernelCreate");

    ur_result_t result = pfnCreate(hProgram, pKernelName, phKernel);
    // if (result == UR_RESULT_SUCCESS) {
    //     context.interceptor->addKernel(hProgram, *phKernel);
    // }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urQueueCreate
__urdlllocal ur_result_t UR_APICALL urQueueCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_device_handle_t hDevice,   ///< [in] handle of the device object
    const ur_queue_properties_t
        *pProperties, ///< [in][optional] pointer to queue creation properties.
    ur_queue_handle_t
        *phQueue ///< [out] pointer to handle of queue object created
) {
    auto pfnCreate = context.urDdiTable.Queue.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }
    // context.logger.debug("=== urQueueCreate");

    ur_result_t result = pfnCreate(hContext, hDevice, pProperties, phQueue);
    if (result == UR_RESULT_SUCCESS) {
        result = context.interceptor->addQueue(hContext, *phQueue);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueKernelLaunch
__urdlllocal ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue,   ///< [in] handle of the queue object
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t
        workDim, ///< [in] number of dimensions, from 1 to 3, to specify the global and
                 ///< work-group work-items
    const size_t *
        pGlobalWorkOffset, ///< [in] pointer to an array of workDim unsigned values that specify the
    ///< offset used to calculate the global ID of a work-item
    const size_t *
        pGlobalWorkSize, ///< [in] pointer to an array of workDim unsigned values that specify the
    ///< number of global work-items in workDim that will execute the kernel
    ///< function
    const size_t *
        pLocalWorkSize, ///< [in][optional] pointer to an array of workDim unsigned values that
    ///< specify the number of local work-items forming a work-group that will
    ///< execute the kernel function.
    ///< If nullptr, the runtime implementation will choose the work-group
    ///< size.
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before the kernel execution.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that no wait
    ///< event.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< kernel execution instance.
) {
    auto pfnKernelLaunch = context.urDdiTable.Enqueue.pfnKernelLaunch;

    if (nullptr == pfnKernelLaunch) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    context.logger.debug("=== urEnqueueKernelLaunch");
    ur_event_handle_t lk_event{};
    std::vector<ur_event_handle_t> events(numEventsInWaitList + 1);
    for (unsigned i = 0; i < numEventsInWaitList; ++i) {
        events.push_back(phEventWaitList[i]);
    }

    // launchKernel must append to num_events_in_wait_list, not prepend
    context.interceptor->launchKernel(hKernel, hQueue, lk_event);
    if (lk_event) {
        events.push_back(lk_event);
    }

    ur_result_t result = pfnKernelLaunch(
        hQueue, hKernel, workDim, pGlobalWorkOffset, pGlobalWorkSize,
        pLocalWorkSize, numEventsInWaitList, phEventWaitList, phEvent);

    if (result == UR_RESULT_SUCCESS) {
        context.interceptor->postLaunchKernel(hKernel, hQueue, phEvent, false);
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemBufferCreate
__urdlllocal ur_result_t UR_APICALL urMemBufferCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_mem_flags_t flags, ///< [in] allocation and usage information flags
    size_t size, ///< [in] size in bytes of the memory object to be allocated
    const ur_buffer_properties_t
        *pProperties, ///< [in][optional] pointer to buffer creation properties
    ur_mem_handle_t
        *phBuffer ///< [out] pointer to handle of the memory buffer created
) {
    return context.interceptor->createMemoryBuffer(hContext, flags, size,
                                                   pProperties, phBuffer);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemBufferPartition
__urdlllocal ur_result_t UR_APICALL urMemBufferPartition(
    ur_mem_handle_t
        hBuffer,          ///< [in] handle of the buffer object to allocate from
    ur_mem_flags_t flags, ///< [in] allocation and usage information flags
    ur_buffer_create_type_t bufferCreateType, ///< [in] buffer creation type
    const ur_buffer_region_t
        *pRegion, ///< [in] pointer to buffer create region information
    ur_mem_handle_t
        *phMem ///< [out] pointer to the handle of sub buffer created
) {
    auto pfnBufferPartition = context.urDdiTable.Mem.pfnBufferPartition;

    if (nullptr == pfnBufferPartition) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    return context.interceptor->partitionMemoryBuffer(
        hBuffer, flags, bufferCreateType, pRegion, phMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgLocal
__urdlllocal ur_result_t UR_APICALL urKernelSetArgLocal(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    size_t
        argSize, ///< [in] size of the local buffer to be allocated by the runtime
    const ur_kernel_arg_local_properties_t
        *pProperties ///< [in][optional] pointer to local buffer properties.
) {
    auto pfnSetArgLocal = context.urDdiTable.Kernel.pfnSetArgLocal;

    if (nullptr == pfnSetArgLocal) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    context.logger.debug("=== urKernelSetArgLocal");

    uptr rzLog = ComputeRZLog(argSize);
    uptr rzSize = RZLog2Size(rzLog);
    uptr roundedSize = RoundUpTo(argSize, ASAN_SHADOW_GRANULARITY);
    uptr neededSize = roundedSize + rzSize;

    // TODO: How to update its shadow memory?
    ur_result_t result =
        pfnSetArgLocal(hKernel, argIndex, neededSize, pProperties);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgMemObj
__urdlllocal ur_result_t UR_APICALL urKernelSetArgMemObj(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_mem_obj_properties_t
        *pProperties, ///< [in][optional] pointer to Memory object properties.
    ur_mem_handle_t hArgValue ///< [in][optional] handle of Memory object.
) {
    auto pfnSetArgMemObj = context.urDdiTable.Kernel.pfnSetArgMemObj;

    if (nullptr == pfnSetArgMemObj) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    context.logger.debug("=== urKernelSetArgMemObj");

    ur_result_t result =
        pfnSetArgMemObj(hKernel, argIndex, pProperties, hArgValue);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextCreate
__urdlllocal ur_result_t UR_APICALL urContextCreate(
    uint32_t numDevices, ///< [in] the number of devices given in phDevices
    const ur_device_handle_t
        *phDevices, ///< [in][range(0, numDevices)] array of handle of devices.
    const ur_context_properties_t *
        pProperties, ///< [in][optional] pointer to context creation properties.
    ur_context_handle_t
        *phContext ///< [out] pointer to handle of context object created
) {
    auto pfnCreate = context.urDdiTable.Context.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_result_t result =
        pfnCreate(numDevices, phDevices, pProperties, phContext);

    if (result == UR_RESULT_SUCCESS) {
        auto Context = *phContext;
        result = context.interceptor->addContext(Context);
        if (result != UR_RESULT_SUCCESS) {
            return result;
        }
        for (uint32_t i = 0; i < numDevices; ++i) {
            result = context.interceptor->addDevice(Context, phDevices[i]);
            if (result != UR_RESULT_SUCCESS) {
                return result;
            }
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urContextCreateWithNativeHandle(
    ur_native_handle_t
        hNativeContext,  ///< [in][nocheck] the native handle of the context.
    uint32_t numDevices, ///< [in] number of devices associated with the context
    const ur_device_handle_t *
        phDevices, ///< [in][range(0, numDevices)] list of devices associated with the context
    const ur_context_native_properties_t *
        pProperties, ///< [in][optional] pointer to native context properties struct
    ur_context_handle_t *
        phContext ///< [out] pointer to the handle of the context object created.
) {
    auto pfnCreateWithNativeHandle =
        context.urDdiTable.Context.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_result_t result = pfnCreateWithNativeHandle(
        hNativeContext, numDevices, phDevices, pProperties, phContext);

    if (result == UR_RESULT_SUCCESS) {
        auto Context = *phContext;
        result = context.interceptor->addContext(Context);
        if (result != UR_RESULT_SUCCESS) {
            return result;
        }
        for (uint32_t i = 0; i < numDevices; ++i) {
            result = context.interceptor->addDevice(Context, phDevices[i]);
            if (result != UR_RESULT_SUCCESS) {
                return result;
            }
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Context table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetContextProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_context_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    auto &dditable = ur_asan_layer::context.urDdiTable.Context;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_asan_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_asan_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    dditable.pfnCreate = pDdiTable->pfnCreate;
    pDdiTable->pfnCreate = ur_asan_layer::urContextCreate;

    // dditable.pfnRetain = pDdiTable->pfnRetain;
    // pDdiTable->pfnRetain = ur_asan_layer::urContextRetain;

    // dditable.pfnRelease = pDdiTable->pfnRelease;
    // pDdiTable->pfnRelease = ur_asan_layer::urContextRelease;

    // dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    // pDdiTable->pfnGetInfo = ur_asan_layer::urContextGetInfo;

    // dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    // pDdiTable->pfnGetNativeHandle = ur_asan_layer::urContextGetNativeHandle;

    dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    pDdiTable->pfnCreateWithNativeHandle =
        ur_asan_layer::urContextCreateWithNativeHandle;

    // dditable.pfnSetExtendedDeleter = pDdiTable->pfnSetExtendedDeleter;
    // pDdiTable->pfnSetExtendedDeleter =
    //     ur_asan_layer::urContextSetExtendedDeleter;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Mem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetMemProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_mem_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    // auto &dditable = ur_asan_layer::context.urDdiTable.Mem;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_asan_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_asan_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // dditable.pfnImageCreate = pDdiTable->pfnImageCreate;
    // pDdiTable->pfnImageCreate = ur_asan_layer::urMemImageCreate;

    // dditable.pfnBufferCreate = pDdiTable->pfnBufferCreate;
    pDdiTable->pfnBufferCreate = ur_asan_layer::urMemBufferCreate;

    // dditable.pfnRetain = pDdiTable->pfnRetain;
    // pDdiTable->pfnRetain = ur_asan_layer::urMemRetain;

    // dditable.pfnRelease = pDdiTable->pfnRelease;
    // pDdiTable->pfnRelease = ur_asan_layer::urMemRelease;

    // dditable.pfnBufferPartition = pDdiTable->pfnBufferPartition;
    pDdiTable->pfnBufferPartition = ur_asan_layer::urMemBufferPartition;

    // dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    // pDdiTable->pfnGetNativeHandle = ur_asan_layer::urMemGetNativeHandle;

    // dditable.pfnBufferCreateWithNativeHandle =
    //     pDdiTable->pfnBufferCreateWithNativeHandle;
    // pDdiTable->pfnBufferCreateWithNativeHandle =
    //     ur_asan_layer::urMemBufferCreateWithNativeHandle;

    // dditable.pfnImageCreateWithNativeHandle =
    //     pDdiTable->pfnImageCreateWithNativeHandle;
    // pDdiTable->pfnImageCreateWithNativeHandle =
    //     ur_asan_layer::urMemImageCreateWithNativeHandle;

    // dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    // pDdiTable->pfnGetInfo = ur_asan_layer::urMemGetInfo;

    // dditable.pfnImageGetInfo = pDdiTable->pfnImageGetInfo;
    // pDdiTable->pfnImageGetInfo = ur_asan_layer::urMemImageGetInfo;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Enqueue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetEnqueueProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_enqueue_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    // auto &dditable = ur_asan_layer::context.urDdiTable.Enqueue;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_asan_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_asan_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // dditable.pfnKernelLaunch = pDdiTable->pfnKernelLaunch;
    pDdiTable->pfnKernelLaunch = ur_asan_layer::urEnqueueKernelLaunch;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Kernel table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetKernelProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_kernel_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    // auto &dditable = ur_asan_layer::context.urDdiTable.Kernel;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_asan_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_asan_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // dditable.pfnCreate = pDdiTable->pfnCreate;
    pDdiTable->pfnCreate = ur_asan_layer::urKernelCreate;
    // pDdiTable->pfnSetArgLocal = ur_asan_layer::urKernelSetArgLocal;
    pDdiTable->pfnSetArgMemObj = ur_asan_layer::urKernelSetArgMemObj;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Queue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetQueueProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_queue_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    // auto &dditable = ur_asan_layer::context.urDdiTable.Queue;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_asan_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_asan_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // dditable.pfnCreate = pDdiTable->pfnCreate;
    pDdiTable->pfnCreate = ur_asan_layer::urQueueCreate;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's USM table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetUSMProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_usm_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    // auto &dditable = ur_asan_layer::context.urDdiTable.USM;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_asan_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_asan_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // dditable.pfnHostAlloc = pDdiTable->pfnHostAlloc;
    // pDdiTable->pfnHostAlloc = ur_asan_layer::urUSMHostAlloc;

    // dditable.pfnDeviceAlloc = pDdiTable->pfnDeviceAlloc;
    pDdiTable->pfnDeviceAlloc = ur_asan_layer::urUSMDeviceAlloc;

    // dditable.pfnSharedAlloc = pDdiTable->pfnSharedAlloc;
    // pDdiTable->pfnSharedAlloc = ur_asan_layer::urUSMSharedAlloc;

    // dditable.pfnFree = pDdiTable->pfnFree;
    // pDdiTable->pfnFree = ur_asan_layer::urUSMFree;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Device table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetDeviceProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_device_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    // auto &dditable = ur_asan_layer::context.urDdiTable.Device;

    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_asan_layer::context.version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_asan_layer::context.version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    // dditable.pfnGet = pDdiTable->pfnGet;
    // pDdiTable->pfnGet = ur_asan_layer::urDeviceGet;

    // dditable.pfnGetInfo = pDdiTable->pfnGetInfo;
    // pDdiTable->pfnGetInfo = ur_asan_layer::urDeviceGetInfo;

    // dditable.pfnRetain = pDdiTable->pfnRetain;
    // pDdiTable->pfnRetain = ur_asan_layer::urDeviceRetain;

    // dditable.pfnRelease = pDdiTable->pfnRelease;
    // pDdiTable->pfnRelease = ur_asan_layer::urDeviceRelease;

    // dditable.pfnPartition = pDdiTable->pfnPartition;
    // pDdiTable->pfnPartition = ur_asan_layer::urDevicePartition;

    // dditable.pfnSelectBinary = pDdiTable->pfnSelectBinary;
    // pDdiTable->pfnSelectBinary = ur_asan_layer::urDeviceSelectBinary;

    // dditable.pfnGetNativeHandle = pDdiTable->pfnGetNativeHandle;
    // pDdiTable->pfnGetNativeHandle = ur_asan_layer::urDeviceGetNativeHandle;

    // dditable.pfnCreateWithNativeHandle = pDdiTable->pfnCreateWithNativeHandle;
    // pDdiTable->pfnCreateWithNativeHandle =
    //     ur_asan_layer::urDeviceCreateWithNativeHandle;

    // dditable.pfnGetGlobalTimestamps = pDdiTable->pfnGetGlobalTimestamps;
    // pDdiTable->pfnGetGlobalTimestamps =
    //     ur_asan_layer::urDeviceGetGlobalTimestamps;

    return result;
}

ur_result_t context_t::init(ur_dditable_t *dditable,
                            const std::set<std::string> &enabledLayerNames) {
    ur_result_t result = UR_RESULT_SUCCESS;

    context.logger.info("ur_asan_layer init");

    // if (!enabledLayerNames.count(name)) {
    //     return result;
    // }

    if (UR_RESULT_SUCCESS == result) {
        // FIXME: Just copy needed APIs?
        urDdiTable = *dditable;

        result = ur_asan_layer::urGetContextProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Context);

        // result = ur_asan_layer::urGetDeviceProcAddrTable(
        //     UR_API_VERSION_CURRENT, &dditable->Device);

        result = ur_asan_layer::urGetEnqueueProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Enqueue);

        result = ur_asan_layer::urGetKernelProcAddrTable(UR_API_VERSION_CURRENT,
                                                         &dditable->Kernel);

        result = ur_asan_layer::urGetQueueProcAddrTable(UR_API_VERSION_CURRENT,
                                                        &dditable->Queue);

        result = ur_asan_layer::urGetUSMProcAddrTable(UR_API_VERSION_CURRENT,
                                                      &dditable->USM);

        result = ur_asan_layer::urGetMemProcAddrTable(UR_API_VERSION_CURRENT,
                                                      &dditable->Mem);
    }

    return result;
}
} // namespace ur_asan_layer
