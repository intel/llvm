/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_ddi.cpp
 *
 */

#include "asan_ddi.hpp"
#include "asan_interceptor.hpp"
#include "asan_options.hpp"
#include "sanitizer_common/sanitizer_stacktrace.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"
#include "ur_sanitizer_layer.hpp"

#include <memory>

namespace ur_sanitizer_layer {
namespace asan {

namespace {

ur_result_t setupContext(ur_context_handle_t Context, uint32_t numDevices,
                         const ur_device_handle_t *phDevices) {
    std::shared_ptr<ContextInfo> CI;
    UR_CALL(getAsanInterceptor()->insertContext(Context, CI));
    for (uint32_t i = 0; i < numDevices; ++i) {
        auto hDevice = phDevices[i];
        std::shared_ptr<DeviceInfo> DI;
        UR_CALL(getAsanInterceptor()->insertDevice(hDevice, DI));
        DI->Type = GetDeviceType(Context, hDevice);
        if (DI->Type == DeviceType::UNKNOWN) {
            getContext()->logger.error("Unsupport device");
            return UR_RESULT_ERROR_INVALID_DEVICE;
        }
        getContext()->logger.info(
            "DeviceInfo {} (Type={}, IsSupportSharedSystemUSM={})",
            (void *)DI->Handle, ToString(DI->Type),
            DI->IsSupportSharedSystemUSM);
        getContext()->logger.info("Add {} into context {}", (void *)DI->Handle,
                                  (void *)Context);
        if (!DI->Shadow) {
            UR_CALL(DI->allocShadowMemory(Context));
        }
        CI->DeviceList.emplace_back(hDevice);
        CI->AllocInfosMap[hDevice];
    }
    return UR_RESULT_SUCCESS;
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterGet
__urdlllocal ur_result_t UR_APICALL urAdapterGet(
    uint32_t
        NumEntries, ///< [in] the number of adapters to be added to phAdapters.
    ///< If phAdapters is not NULL, then NumEntries should be greater than
    ///< zero, otherwise ::UR_RESULT_ERROR_INVALID_SIZE,
    ///< will be returned.
    ur_adapter_handle_t *
        phAdapters, ///< [out][optional][range(0, NumEntries)] array of handle of adapters.
    ///< If NumEntries is less than the number of adapters available, then
    ///< ::urAdapterGet shall only retrieve that number of platforms.
    uint32_t *
        pNumAdapters ///< [out][optional] returns the total number of adapters available.
) {
    auto pfnAdapterGet = getContext()->urDdiTable.Global.pfnAdapterGet;

    if (nullptr == pfnAdapterGet) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    ur_result_t result = pfnAdapterGet(NumEntries, phAdapters, pNumAdapters);
    if (result == UR_RESULT_SUCCESS && phAdapters) {
        const uint32_t NumAdapters = pNumAdapters ? *pNumAdapters : NumEntries;
        for (uint32_t i = 0; i < NumAdapters; ++i) {
            UR_CALL(getAsanInterceptor()->holdAdapter(phAdapters[i]));
        }
    }

    return result;
}

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
    auto pfnHostAlloc = getContext()->urDdiTable.USM.pfnHostAlloc;

    if (nullptr == pfnHostAlloc) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urUSMHostAlloc");

    return getAsanInterceptor()->allocateMemory(
        hContext, nullptr, pUSMDesc, pool, size, AllocType::HOST_USM, ppMem);
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
    auto pfnDeviceAlloc = getContext()->urDdiTable.USM.pfnDeviceAlloc;

    if (nullptr == pfnDeviceAlloc) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urUSMDeviceAlloc");

    return getAsanInterceptor()->allocateMemory(
        hContext, hDevice, pUSMDesc, pool, size, AllocType::DEVICE_USM, ppMem);
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
    auto pfnSharedAlloc = getContext()->urDdiTable.USM.pfnSharedAlloc;

    if (nullptr == pfnSharedAlloc) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urUSMSharedAlloc");

    return getAsanInterceptor()->allocateMemory(
        hContext, hDevice, pUSMDesc, pool, size, AllocType::SHARED_USM, ppMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMFree
__urdlllocal ur_result_t UR_APICALL urUSMFree(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    void *pMem                    ///< [in] pointer to USM memory object
) {
    auto pfnFree = getContext()->urDdiTable.USM.pfnFree;

    if (nullptr == pfnFree) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urUSMFree");

    return getAsanInterceptor()->releaseMemory(hContext, pMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreateWithIL
__urdlllocal ur_result_t UR_APICALL urProgramCreateWithIL(
    ur_context_handle_t hContext, ///< [in] handle of the context instance
    const void *pIL,              ///< [in] pointer to IL binary.
    size_t length,                ///< [in] length of `pIL` in bytes.
    const ur_program_properties_t *
        pProperties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t
        *phProgram ///< [out] pointer to handle of program object created.
) {
    auto pfnProgramCreateWithIL =
        getContext()->urDdiTable.Program.pfnCreateWithIL;

    if (nullptr == pfnProgramCreateWithIL) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urProgramCreateWithIL");

    UR_CALL(
        pfnProgramCreateWithIL(hContext, pIL, length, pProperties, phProgram));
    UR_CALL(getAsanInterceptor()->insertProgram(*phProgram));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreateWithBinary
__urdlllocal ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t hContext, ///< [in] handle of the context instance
    uint32_t numDevices,          ///< [in] number of devices
    ur_device_handle_t *
        phDevices, ///< [in][range(0, numDevices)] a pointer to a list of device handles. The
                   ///< binaries are loaded for devices specified in this list.
    size_t *
        pLengths, ///< [in][range(0, numDevices)] array of sizes of program binaries
                  ///< specified by `pBinaries` (in bytes).
    const uint8_t **
        ppBinaries, ///< [in][range(0, numDevices)] pointer to program binaries to be loaded
                    ///< for devices specified by `phDevices`.
    const ur_program_properties_t *
        pProperties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t
        *phProgram ///< [out] pointer to handle of Program object created.
) {
    auto pfnProgramCreateWithBinary =
        getContext()->urDdiTable.Program.pfnCreateWithBinary;

    if (nullptr == pfnProgramCreateWithBinary) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urProgramCreateWithBinary");

    UR_CALL(pfnProgramCreateWithBinary(hContext, numDevices, phDevices,
                                       pLengths, ppBinaries, pProperties,
                                       phProgram));
    UR_CALL(getAsanInterceptor()->insertProgram(*phProgram));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    ur_native_handle_t
        hNativeProgram, ///< [in][nocheck] the native handle of the program.
    ur_context_handle_t hContext, ///< [in] handle of the context instance
    const ur_program_native_properties_t *
        pProperties, ///< [in][optional] pointer to native program properties struct.
    ur_program_handle_t *
        phProgram ///< [out] pointer to the handle of the program object created.
) {
    auto pfnProgramCreateWithNativeHandle =
        getContext()->urDdiTable.Program.pfnCreateWithNativeHandle;

    if (nullptr == pfnProgramCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urProgramCreateWithNativeHandle");

    UR_CALL(pfnProgramCreateWithNativeHandle(hNativeProgram, hContext,
                                             pProperties, phProgram));
    UR_CALL(getAsanInterceptor()->insertProgram(*phProgram));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRetain
__urdlllocal ur_result_t UR_APICALL urProgramRetain(
    ur_program_handle_t
        hProgram ///< [in][retain] handle for the Program to retain
) {
    auto pfnRetain = getContext()->urDdiTable.Program.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urProgramRetain");

    UR_CALL(pfnRetain(hProgram));

    auto ProgramInfo = getAsanInterceptor()->getProgramInfo(hProgram);
    UR_ASSERT(ProgramInfo != nullptr, UR_RESULT_ERROR_INVALID_VALUE);
    ProgramInfo->RefCount++;

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramBuild
__urdlllocal ur_result_t UR_APICALL urProgramBuild(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_program_handle_t hProgram, ///< [in] handle of the program object
    const char *pOptions          ///< [in] string of build options
) {
    auto pfnProgramBuild = getContext()->urDdiTable.Program.pfnBuild;

    if (nullptr == pfnProgramBuild) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urProgramBuild");

    UR_CALL(pfnProgramBuild(hContext, hProgram, pOptions));

    UR_CALL(getAsanInterceptor()->registerProgram(hProgram));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramBuildExp
__urdlllocal ur_result_t UR_APICALL urProgramBuildExp(
    ur_program_handle_t hProgram, ///< [in] Handle of the program to build.
    uint32_t numDevices,          ///< [in] number of devices
    ur_device_handle_t *
        phDevices, ///< [in][range(0, numDevices)] pointer to array of device handles
    const char *
        pOptions ///< [in][optional] pointer to build options null-terminated string.
) {
    auto pfnBuildExp = getContext()->urDdiTable.ProgramExp.pfnBuildExp;

    if (nullptr == pfnBuildExp) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urProgramBuildExp");

    UR_CALL(pfnBuildExp(hProgram, numDevices, phDevices, pOptions));
    UR_CALL(getAsanInterceptor()->registerProgram(hProgram));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramLink
__urdlllocal ur_result_t UR_APICALL urProgramLink(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    uint32_t count, ///< [in] number of program handles in `phPrograms`.
    const ur_program_handle_t *
        phPrograms, ///< [in][range(0, count)] pointer to array of program handles.
    const char *
        pOptions, ///< [in][optional] pointer to linker options null-terminated string.
    ur_program_handle_t
        *phProgram ///< [out] pointer to handle of program object created.
) {
    auto pfnProgramLink = getContext()->urDdiTable.Program.pfnLink;

    if (nullptr == pfnProgramLink) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urProgramLink");

    UR_CALL(pfnProgramLink(hContext, count, phPrograms, pOptions, phProgram));

    UR_CALL(getAsanInterceptor()->registerProgram(*phProgram));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramLinkExp
ur_result_t UR_APICALL urProgramLinkExp(
    ur_context_handle_t hContext, ///< [in] handle of the context instance.
    uint32_t numDevices,          ///< [in] number of devices
    ur_device_handle_t *
        phDevices, ///< [in][range(0, numDevices)] pointer to array of device handles
    uint32_t count, ///< [in] number of program handles in `phPrograms`.
    const ur_program_handle_t *
        phPrograms, ///< [in][range(0, count)] pointer to array of program handles.
    const char *
        pOptions, ///< [in][optional] pointer to linker options null-terminated string.
    ur_program_handle_t
        *phProgram ///< [out] pointer to handle of program object created.
) {
    auto pfnProgramLinkExp = getContext()->urDdiTable.ProgramExp.pfnLinkExp;

    if (nullptr == pfnProgramLinkExp) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urProgramLinkExp");

    UR_CALL(pfnProgramLinkExp(hContext, numDevices, phDevices, count,
                              phPrograms, pOptions, phProgram));

    UR_CALL(getAsanInterceptor()->registerProgram(*phProgram));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRelease
ur_result_t UR_APICALL urProgramRelease(
    ur_program_handle_t
        hProgram ///< [in][release] handle for the Program to release
) {
    auto pfnProgramRelease = getContext()->urDdiTable.Program.pfnRelease;

    if (nullptr == pfnProgramRelease) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urProgramRelease");

    UR_CALL(pfnProgramRelease(hProgram));

    auto ProgramInfo = getAsanInterceptor()->getProgramInfo(hProgram);
    UR_ASSERT(ProgramInfo != nullptr, UR_RESULT_ERROR_INVALID_VALUE);
    if (--ProgramInfo->RefCount == 0) {
        UR_CALL(getAsanInterceptor()->unregisterProgram(hProgram));
        UR_CALL(getAsanInterceptor()->eraseProgram(hProgram));
    }

    return UR_RESULT_SUCCESS;
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
    auto pfnKernelLaunch = getContext()->urDdiTable.Enqueue.pfnKernelLaunch;

    if (nullptr == pfnKernelLaunch) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urEnqueueKernelLaunch");

    USMLaunchInfo LaunchInfo(GetContext(hKernel), GetDevice(hQueue),
                             pGlobalWorkSize, pLocalWorkSize, pGlobalWorkOffset,
                             workDim);
    UR_CALL(LaunchInfo.initialize());

    UR_CALL(getAsanInterceptor()->preLaunchKernel(hKernel, hQueue, LaunchInfo));

    ur_event_handle_t hEvent{};
    ur_result_t result =
        pfnKernelLaunch(hQueue, hKernel, workDim, pGlobalWorkOffset,
                        pGlobalWorkSize, LaunchInfo.LocalWorkSize.data(),
                        numEventsInWaitList, phEventWaitList, &hEvent);

    if (result == UR_RESULT_SUCCESS) {
        UR_CALL(getAsanInterceptor()->postLaunchKernel(hKernel, hQueue,
                                                       LaunchInfo));
    }

    if (phEvent) {
        *phEvent = hEvent;
    }

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
    auto pfnCreate = getContext()->urDdiTable.Context.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urContextCreate");

    ur_result_t result =
        pfnCreate(numDevices, phDevices, pProperties, phContext);

    if (result == UR_RESULT_SUCCESS) {
        UR_CALL(setupContext(*phContext, numDevices, phDevices));
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextCreateWithNativeHandle
__urdlllocal ur_result_t UR_APICALL urContextCreateWithNativeHandle(
    ur_native_handle_t
        hNativeContext, ///< [in][nocheck] the native handle of the getContext()->
    ur_adapter_handle_t hAdapter,
    uint32_t numDevices, ///< [in] number of devices associated with the context
    const ur_device_handle_t *
        phDevices, ///< [in][range(0, numDevices)] list of devices associated with the context
    const ur_context_native_properties_t *
        pProperties, ///< [in][optional] pointer to native context properties struct
    ur_context_handle_t *
        phContext ///< [out] pointer to the handle of the context object created.
) {
    auto pfnCreateWithNativeHandle =
        getContext()->urDdiTable.Context.pfnCreateWithNativeHandle;

    if (nullptr == pfnCreateWithNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urContextCreateWithNativeHandle");

    ur_result_t result =
        pfnCreateWithNativeHandle(hNativeContext, hAdapter, numDevices,
                                  phDevices, pProperties, phContext);

    if (result == UR_RESULT_SUCCESS) {
        UR_CALL(setupContext(*phContext, numDevices, phDevices));
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRetain
__urdlllocal ur_result_t UR_APICALL urContextRetain(
    ur_context_handle_t
        hContext ///< [in] handle of the context to get a reference of.
) {
    auto pfnRetain = getContext()->urDdiTable.Context.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urContextRetain");

    UR_CALL(pfnRetain(hContext));

    auto ContextInfo = getAsanInterceptor()->getContextInfo(hContext);
    UR_ASSERT(ContextInfo != nullptr, UR_RESULT_ERROR_INVALID_VALUE);
    ContextInfo->RefCount++;

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRelease
__urdlllocal ur_result_t UR_APICALL urContextRelease(
    ur_context_handle_t hContext ///< [in] handle of the context to release.
) {
    auto pfnRelease = getContext()->urDdiTable.Context.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urContextRelease");

    UR_CALL(pfnRelease(hContext));

    auto ContextInfo = getAsanInterceptor()->getContextInfo(hContext);
    UR_ASSERT(ContextInfo != nullptr, UR_RESULT_ERROR_INVALID_VALUE);
    if (--ContextInfo->RefCount == 0) {
        UR_CALL(getAsanInterceptor()->eraseContext(hContext));
    }

    return UR_RESULT_SUCCESS;
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
    auto pfnBufferCreate = getContext()->urDdiTable.Mem.pfnBufferCreate;

    if (nullptr == pfnBufferCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    if (nullptr == phBuffer) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    getContext()->logger.debug("==== urMemBufferCreate");

    void *Host = nullptr;
    if (pProperties) {
        Host = pProperties->pHost;
    }

    char *hostPtrOrNull = (flags & UR_MEM_FLAG_USE_HOST_POINTER)
                              ? ur_cast<char *>(Host)
                              : nullptr;

    std::shared_ptr<MemBuffer> pMemBuffer =
        std::make_shared<MemBuffer>(hContext, size, hostPtrOrNull);

    if (Host && (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) {
        std::shared_ptr<ContextInfo> CtxInfo =
            getAsanInterceptor()->getContextInfo(hContext);
        for (const auto &hDevice : CtxInfo->DeviceList) {
            ManagedQueue InternalQueue(hContext, hDevice);
            char *Handle = nullptr;
            UR_CALL(pMemBuffer->getHandle(hDevice, Handle));
            UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
                InternalQueue, true, Handle, Host, size, 0, nullptr, nullptr));
        }
    }

    ur_result_t result = getAsanInterceptor()->insertMemBuffer(pMemBuffer);
    *phBuffer = ur_cast<ur_mem_handle_t>(pMemBuffer.get());

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemGetInfo
__urdlllocal ur_result_t UR_APICALL urMemGetInfo(
    ur_mem_handle_t
        hMemory,            ///< [in] handle to the memory object being queried.
    ur_mem_info_t propName, ///< [in] type of the info to retrieve.
    size_t
        propSize, ///< [in] the number of bytes of memory pointed to by pPropValue.
    void *
        pPropValue, ///< [out][optional][typename(propName, propSize)] array of bytes holding
                    ///< the info.
    ///< If propSize is less than the real number of bytes needed to return
    ///< the info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    ///< pPropValue is not used.
    size_t *
        pPropSizeRet ///< [out][optional] pointer to the actual size in bytes of the queried propName.
) {
    auto pfnGetInfo = getContext()->urDdiTable.Mem.pfnGetInfo;

    if (nullptr == pfnGetInfo) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urMemGetInfo");

    if (auto MemBuffer = getAsanInterceptor()->getMemBuffer(hMemory)) {
        UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
        switch (propName) {
        case UR_MEM_INFO_CONTEXT: {
            return ReturnValue(MemBuffer->Context);
        }
        case UR_MEM_INFO_SIZE: {
            return ReturnValue(size_t{MemBuffer->Size});
        }
        default: {
            return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
        }
        }
    } else {
        UR_CALL(
            pfnGetInfo(hMemory, propName, propSize, pPropValue, pPropSizeRet));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRetain
__urdlllocal ur_result_t UR_APICALL urMemRetain(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to get access
) {
    auto pfnRetain = getContext()->urDdiTable.Mem.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urMemRetain");

    if (auto MemBuffer = getAsanInterceptor()->getMemBuffer(hMem)) {
        MemBuffer->RefCount++;
    } else {
        UR_CALL(pfnRetain(hMem));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRelease
__urdlllocal ur_result_t UR_APICALL urMemRelease(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to release
) {
    auto pfnRelease = getContext()->urDdiTable.Mem.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urMemRelease");

    if (auto MemBuffer = getAsanInterceptor()->getMemBuffer(hMem)) {
        if (--MemBuffer->RefCount != 0) {
            return UR_RESULT_SUCCESS;
        }
        UR_CALL(MemBuffer->free());
        UR_CALL(getAsanInterceptor()->eraseMemBuffer(hMem));
    } else {
        UR_CALL(pfnRelease(hMem));
    }

    return UR_RESULT_SUCCESS;
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
    auto pfnBufferPartition = getContext()->urDdiTable.Mem.pfnBufferPartition;

    if (nullptr == pfnBufferPartition) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urMemBufferPartition");

    if (auto ParentBuffer = getAsanInterceptor()->getMemBuffer(hBuffer)) {
        if (ParentBuffer->Size < (pRegion->origin + pRegion->size)) {
            return UR_RESULT_ERROR_INVALID_BUFFER_SIZE;
        }
        std::shared_ptr<MemBuffer> SubBuffer = std::make_shared<MemBuffer>(
            ParentBuffer, pRegion->origin, pRegion->size);
        UR_CALL(getAsanInterceptor()->insertMemBuffer(SubBuffer));
        *phMem = reinterpret_cast<ur_mem_handle_t>(SubBuffer.get());
    } else {
        UR_CALL(pfnBufferPartition(hBuffer, flags, bufferCreateType, pRegion,
                                   phMem));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemGetNativeHandle
__urdlllocal ur_result_t UR_APICALL urMemGetNativeHandle(
    ur_mem_handle_t hMem, ///< [in] handle of the mem.
    ur_device_handle_t hDevice,
    ur_native_handle_t
        *phNativeMem ///< [out] a pointer to the native handle of the mem.
) {
    auto pfnGetNativeHandle = getContext()->urDdiTable.Mem.pfnGetNativeHandle;

    if (nullptr == pfnGetNativeHandle) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urMemGetNativeHandle");

    if (auto MemBuffer = getAsanInterceptor()->getMemBuffer(hMem)) {
        char *Handle = nullptr;
        UR_CALL(MemBuffer->getHandle(hDevice, Handle));
        *phNativeMem = ur_cast<ur_native_handle_t>(Handle);
    } else {
        UR_CALL(pfnGetNativeHandle(hMem, hDevice, phNativeMem));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferRead
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(offset, size)] handle of the buffer object
    bool blockingRead, ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,     ///< [in] offset in bytes in the buffer object
    size_t size,       ///< [in] size in bytes of data being read
    void *pDst, ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemBufferRead = getContext()->urDdiTable.Enqueue.pfnMemBufferRead;

    if (nullptr == pfnMemBufferRead) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urEnqueueMemBufferRead");

    if (auto MemBuffer = getAsanInterceptor()->getMemBuffer(hBuffer)) {
        ur_device_handle_t Device = GetDevice(hQueue);
        char *pSrc = nullptr;
        UR_CALL(MemBuffer->getHandle(Device, pSrc));
        UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
            hQueue, blockingRead, pDst, pSrc + offset, size,
            numEventsInWaitList, phEventWaitList, phEvent));
    } else {
        UR_CALL(pfnMemBufferRead(hQueue, hBuffer, blockingRead, offset, size,
                                 pDst, numEventsInWaitList, phEventWaitList,
                                 phEvent));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferWrite
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(offset, size)] handle of the buffer object
    bool
        blockingWrite, ///< [in] indicates blocking (true), non-blocking (false)
    size_t offset,     ///< [in] offset in bytes in the buffer object
    size_t size,       ///< [in] size in bytes of data being written
    const void
        *pSrc, ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemBufferWrite = getContext()->urDdiTable.Enqueue.pfnMemBufferWrite;

    if (nullptr == pfnMemBufferWrite) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urEnqueueMemBufferWrite");

    if (auto MemBuffer = getAsanInterceptor()->getMemBuffer(hBuffer)) {
        ur_device_handle_t Device = GetDevice(hQueue);
        char *pDst = nullptr;
        UR_CALL(MemBuffer->getHandle(Device, pDst));
        UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
            hQueue, blockingWrite, pDst + offset, pSrc, size,
            numEventsInWaitList, phEventWaitList, phEvent));
    } else {
        UR_CALL(pfnMemBufferWrite(hQueue, hBuffer, blockingWrite, offset, size,
                                  pSrc, numEventsInWaitList, phEventWaitList,
                                  phEvent));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferReadRect
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferReadRect(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(bufferOrigin, region)] handle of the buffer object
    bool blockingRead, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOrigin, ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOrigin,   ///< [in] 3D offset in the host region
    ur_rect_region_t
        region, ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t
        bufferRowPitch, ///< [in] length of each row in bytes in the buffer object
    size_t
        bufferSlicePitch, ///< [in] length of each 2D slice in bytes in the buffer object being read
    size_t
        hostRowPitch, ///< [in] length of each row in bytes in the host memory region pointed by
                      ///< dst
    size_t
        hostSlicePitch, ///< [in] length of each 2D slice in bytes in the host memory region
                        ///< pointed by dst
    void *pDst, ///< [in] pointer to host memory where data is to be read into
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemBufferReadRect =
        getContext()->urDdiTable.Enqueue.pfnMemBufferReadRect;

    if (nullptr == pfnMemBufferReadRect) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urEnqueueMemBufferReadRect");

    if (auto MemBuffer = getAsanInterceptor()->getMemBuffer(hBuffer)) {
        char *SrcHandle = nullptr;
        ur_device_handle_t Device = GetDevice(hQueue);
        UR_CALL(MemBuffer->getHandle(Device, SrcHandle));

        UR_CALL(EnqueueMemCopyRectHelper(
            hQueue, SrcHandle, ur_cast<char *>(pDst), bufferOrigin, hostOrigin,
            region, bufferRowPitch, bufferSlicePitch, hostRowPitch,
            hostSlicePitch, blockingRead, numEventsInWaitList, phEventWaitList,
            phEvent));
    } else {
        UR_CALL(pfnMemBufferReadRect(
            hQueue, hBuffer, blockingRead, bufferOrigin, hostOrigin, region,
            bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch,
            pDst, numEventsInWaitList, phEventWaitList, phEvent));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferWriteRect
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferWriteRect(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(bufferOrigin, region)] handle of the buffer object
    bool
        blockingWrite, ///< [in] indicates blocking (true), non-blocking (false)
    ur_rect_offset_t bufferOrigin, ///< [in] 3D offset in the buffer
    ur_rect_offset_t hostOrigin,   ///< [in] 3D offset in the host region
    ur_rect_region_t
        region, ///< [in] 3D rectangular region descriptor: width, height, depth
    size_t
        bufferRowPitch, ///< [in] length of each row in bytes in the buffer object
    size_t
        bufferSlicePitch, ///< [in] length of each 2D slice in bytes in the buffer object being
                          ///< written
    size_t
        hostRowPitch, ///< [in] length of each row in bytes in the host memory region pointed by
                      ///< src
    size_t
        hostSlicePitch, ///< [in] length of each 2D slice in bytes in the host memory region
                        ///< pointed by src
    void
        *pSrc, ///< [in] pointer to host memory where data is to be written from
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] points to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemBufferWriteRect =
        getContext()->urDdiTable.Enqueue.pfnMemBufferWriteRect;

    if (nullptr == pfnMemBufferWriteRect) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urEnqueueMemBufferWriteRect");

    if (auto MemBuffer = getAsanInterceptor()->getMemBuffer(hBuffer)) {
        char *DstHandle = nullptr;
        ur_device_handle_t Device = GetDevice(hQueue);
        UR_CALL(MemBuffer->getHandle(Device, DstHandle));

        UR_CALL(EnqueueMemCopyRectHelper(
            hQueue, ur_cast<char *>(pSrc), DstHandle, hostOrigin, bufferOrigin,
            region, hostRowPitch, hostSlicePitch, bufferRowPitch,
            bufferSlicePitch, blockingWrite, numEventsInWaitList,
            phEventWaitList, phEvent));
    } else {
        UR_CALL(pfnMemBufferWriteRect(
            hQueue, hBuffer, blockingWrite, bufferOrigin, hostOrigin, region,
            bufferRowPitch, bufferSlicePitch, hostRowPitch, hostSlicePitch,
            pSrc, numEventsInWaitList, phEventWaitList, phEvent));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferCopy
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferCopy(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBufferSrc, ///< [in][bounds(srcOffset, size)] handle of the src buffer object
    ur_mem_handle_t
        hBufferDst, ///< [in][bounds(dstOffset, size)] handle of the dest buffer object
    size_t srcOffset, ///< [in] offset into hBufferSrc to begin copying from
    size_t dstOffset, ///< [in] offset info hBufferDst to begin copying into
    size_t size,      ///< [in] size in bytes of data being copied
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemBufferCopy = getContext()->urDdiTable.Enqueue.pfnMemBufferCopy;

    if (nullptr == pfnMemBufferCopy) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urEnqueueMemBufferCopy");

    auto SrcBuffer = getAsanInterceptor()->getMemBuffer(hBufferSrc);
    auto DstBuffer = getAsanInterceptor()->getMemBuffer(hBufferDst);

    UR_ASSERT((SrcBuffer && DstBuffer) || (!SrcBuffer && !DstBuffer),
              UR_RESULT_ERROR_INVALID_MEM_OBJECT);

    if (SrcBuffer && DstBuffer) {
        ur_device_handle_t Device = GetDevice(hQueue);
        char *SrcHandle = nullptr;
        UR_CALL(SrcBuffer->getHandle(Device, SrcHandle));

        char *DstHandle = nullptr;
        UR_CALL(DstBuffer->getHandle(Device, DstHandle));

        UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
            hQueue, false, DstHandle + dstOffset, SrcHandle + srcOffset, size,
            numEventsInWaitList, phEventWaitList, phEvent));
    } else {
        UR_CALL(pfnMemBufferCopy(hQueue, hBufferSrc, hBufferDst, srcOffset,
                                 dstOffset, size, numEventsInWaitList,
                                 phEventWaitList, phEvent));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferCopyRect
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferCopyRect(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBufferSrc, ///< [in][bounds(srcOrigin, region)] handle of the source buffer object
    ur_mem_handle_t
        hBufferDst, ///< [in][bounds(dstOrigin, region)] handle of the dest buffer object
    ur_rect_offset_t srcOrigin, ///< [in] 3D offset in the source buffer
    ur_rect_offset_t dstOrigin, ///< [in] 3D offset in the destination buffer
    ur_rect_region_t
        region, ///< [in] source 3D rectangular region descriptor: width, height, depth
    size_t
        srcRowPitch, ///< [in] length of each row in bytes in the source buffer object
    size_t
        srcSlicePitch, ///< [in] length of each 2D slice in bytes in the source buffer object
    size_t
        dstRowPitch, ///< [in] length of each row in bytes in the destination buffer object
    size_t
        dstSlicePitch, ///< [in] length of each 2D slice in bytes in the destination buffer object
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemBufferCopyRect =
        getContext()->urDdiTable.Enqueue.pfnMemBufferCopyRect;

    if (nullptr == pfnMemBufferCopyRect) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urEnqueueMemBufferCopyRect");

    auto SrcBuffer = getAsanInterceptor()->getMemBuffer(hBufferSrc);
    auto DstBuffer = getAsanInterceptor()->getMemBuffer(hBufferDst);

    UR_ASSERT((SrcBuffer && DstBuffer) || (!SrcBuffer && !DstBuffer),
              UR_RESULT_ERROR_INVALID_MEM_OBJECT);

    if (SrcBuffer && DstBuffer) {
        ur_device_handle_t Device = GetDevice(hQueue);
        char *SrcHandle = nullptr;
        UR_CALL(SrcBuffer->getHandle(Device, SrcHandle));

        char *DstHandle = nullptr;
        UR_CALL(DstBuffer->getHandle(Device, DstHandle));

        UR_CALL(EnqueueMemCopyRectHelper(
            hQueue, SrcHandle, DstHandle, srcOrigin, dstOrigin, region,
            srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch, false,
            numEventsInWaitList, phEventWaitList, phEvent));
    } else {
        UR_CALL(pfnMemBufferCopyRect(
            hQueue, hBufferSrc, hBufferDst, srcOrigin, dstOrigin, region,
            srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch,
            numEventsInWaitList, phEventWaitList, phEvent));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferFill
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferFill(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(offset, size)] handle of the buffer object
    const void *pPattern, ///< [in] pointer to the fill pattern
    size_t patternSize,   ///< [in] size in bytes of the pattern
    size_t offset,        ///< [in] offset into the buffer
    size_t size, ///< [in] fill size in bytes, must be a multiple of patternSize
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemBufferFill = getContext()->urDdiTable.Enqueue.pfnMemBufferFill;

    if (nullptr == pfnMemBufferFill) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urEnqueueMemBufferFill");

    if (auto MemBuffer = getAsanInterceptor()->getMemBuffer(hBuffer)) {
        char *Handle = nullptr;
        ur_device_handle_t Device = GetDevice(hQueue);
        UR_CALL(MemBuffer->getHandle(Device, Handle));
        UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMFill(
            hQueue, Handle + offset, patternSize, pPattern, size,
            numEventsInWaitList, phEventWaitList, phEvent));
    } else {
        UR_CALL(pfnMemBufferFill(hQueue, hBuffer, pPattern, patternSize, offset,
                                 size, numEventsInWaitList, phEventWaitList,
                                 phEvent));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemBufferMap
__urdlllocal ur_result_t UR_APICALL urEnqueueMemBufferMap(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hBuffer, ///< [in][bounds(offset, size)] handle of the buffer object
    bool blockingMap, ///< [in] indicates blocking (true), non-blocking (false)
    ur_map_flags_t mapFlags, ///< [in] flags for read, write, readwrite mapping
    size_t offset, ///< [in] offset in bytes of the buffer region being mapped
    size_t size,   ///< [in] size in bytes of the buffer region being mapped
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent, ///< [out][optional] return an event object that identifies this particular
                 ///< command instance.
    void **ppRetMap ///< [out] return mapped pointer.  TODO: move it before
                    ///< numEventsInWaitList?
) {
    auto pfnMemBufferMap = getContext()->urDdiTable.Enqueue.pfnMemBufferMap;

    if (nullptr == pfnMemBufferMap) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urEnqueueMemBufferMap");

    if (auto MemBuffer = getAsanInterceptor()->getMemBuffer(hBuffer)) {

        // Translate the host access mode info.
        MemBuffer::AccessMode AccessMode = MemBuffer::UNKNOWN;
        if (mapFlags & UR_MAP_FLAG_WRITE_INVALIDATE_REGION) {
            AccessMode = MemBuffer::WRITE_ONLY;
        } else {
            if (mapFlags & UR_MAP_FLAG_READ) {
                AccessMode = MemBuffer::READ_ONLY;
                if (mapFlags & UR_MAP_FLAG_WRITE) {
                    AccessMode = MemBuffer::READ_WRITE;
                }
            } else if (mapFlags & UR_MAP_FLAG_WRITE) {
                AccessMode = MemBuffer::WRITE_ONLY;
            }
        }

        UR_ASSERT(AccessMode != MemBuffer::UNKNOWN,
                  UR_RESULT_ERROR_INVALID_ARGUMENT);

        ur_device_handle_t Device = GetDevice(hQueue);
        // If the buffer used host pointer, then we just reuse it. If not, we
        // need to manually allocate a new host USM.
        if (MemBuffer->HostPtr) {
            *ppRetMap = MemBuffer->HostPtr + offset;
        } else {
            ur_context_handle_t Context = GetContext(hQueue);
            ur_usm_desc_t USMDesc{};
            USMDesc.align = MemBuffer->getAlignment();
            ur_usm_pool_handle_t Pool{};
            UR_CALL(getAsanInterceptor()->allocateMemory(
                Context, nullptr, &USMDesc, Pool, size, AllocType::HOST_USM,
                ppRetMap));
        }

        // Actually, if the access mode is write only, we don't need to do this
        // copy. However, in that way, we cannot generate a event to user. So,
        // we'll aways do copy here.
        char *SrcHandle = nullptr;
        UR_CALL(MemBuffer->getHandle(Device, SrcHandle));
        UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
            hQueue, blockingMap, *ppRetMap, SrcHandle + offset, size,
            numEventsInWaitList, phEventWaitList, phEvent));

        {
            std::scoped_lock<ur_shared_mutex> Guard(MemBuffer->Mutex);
            UR_ASSERT(MemBuffer->Mappings.find(*ppRetMap) ==
                          MemBuffer->Mappings.end(),
                      UR_RESULT_ERROR_INVALID_VALUE);
            MemBuffer->Mappings[*ppRetMap] = {offset, size};
        }
    } else {
        UR_CALL(pfnMemBufferMap(hQueue, hBuffer, blockingMap, mapFlags, offset,
                                size, numEventsInWaitList, phEventWaitList,
                                phEvent, ppRetMap));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueMemUnmap
__urdlllocal ur_result_t UR_APICALL urEnqueueMemUnmap(
    ur_queue_handle_t hQueue, ///< [in] handle of the queue object
    ur_mem_handle_t
        hMem,         ///< [in] handle of the memory (buffer or image) object
    void *pMappedPtr, ///< [in] mapped host address
    uint32_t numEventsInWaitList, ///< [in] size of the event wait list
    const ur_event_handle_t *
        phEventWaitList, ///< [in][optional][range(0, numEventsInWaitList)] pointer to a list of
    ///< events that must be complete before this command can be executed.
    ///< If nullptr, the numEventsInWaitList must be 0, indicating that this
    ///< command does not wait on any event to complete.
    ur_event_handle_t *
        phEvent ///< [out][optional] return an event object that identifies this particular
                ///< command instance.
) {
    auto pfnMemUnmap = getContext()->urDdiTable.Enqueue.pfnMemUnmap;

    if (nullptr == pfnMemUnmap) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urEnqueueMemUnmap");

    if (auto MemBuffer = getAsanInterceptor()->getMemBuffer(hMem)) {
        MemBuffer::Mapping Mapping{};
        {
            std::scoped_lock<ur_shared_mutex> Guard(MemBuffer->Mutex);
            auto It = MemBuffer->Mappings.find(pMappedPtr);
            UR_ASSERT(It != MemBuffer->Mappings.end(),
                      UR_RESULT_ERROR_INVALID_VALUE);
            Mapping = It->second;
            MemBuffer->Mappings.erase(It);
        }

        // Write back mapping memory data to device and release mapping memory
        // if we allocated a host USM. But for now, UR doesn't support event
        // call back, we can only do blocking copy here.
        char *DstHandle = nullptr;
        ur_context_handle_t Context = GetContext(hQueue);
        ur_device_handle_t Device = GetDevice(hQueue);
        UR_CALL(MemBuffer->getHandle(Device, DstHandle));
        UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
            hQueue, true, DstHandle + Mapping.Offset, pMappedPtr, Mapping.Size,
            numEventsInWaitList, phEventWaitList, phEvent));

        if (!MemBuffer->HostPtr) {
            UR_CALL(getAsanInterceptor()->releaseMemory(Context, pMappedPtr));
        }
    } else {
        UR_CALL(pfnMemUnmap(hQueue, hMem, pMappedPtr, numEventsInWaitList,
                            phEventWaitList, phEvent));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelCreate
__urdlllocal ur_result_t UR_APICALL urKernelCreate(
    ur_program_handle_t hProgram, ///< [in] handle of the program instance
    const char *pKernelName,      ///< [in] pointer to null-terminated string.
    ur_kernel_handle_t
        *phKernel ///< [out] pointer to handle of kernel object created.
) {
    auto pfnCreate = getContext()->urDdiTable.Kernel.pfnCreate;

    if (nullptr == pfnCreate) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urKernelCreate");

    UR_CALL(pfnCreate(hProgram, pKernelName, phKernel));
    UR_CALL(getAsanInterceptor()->insertKernel(*phKernel));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRetain
__urdlllocal ur_result_t UR_APICALL urKernelRetain(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to retain
) {
    auto pfnRetain = getContext()->urDdiTable.Kernel.pfnRetain;

    if (nullptr == pfnRetain) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urKernelRetain");

    UR_CALL(pfnRetain(hKernel));

    auto KernelInfo = getAsanInterceptor()->getKernelInfo(hKernel);
    KernelInfo->RefCount++;

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRelease
__urdlllocal ur_result_t urKernelRelease(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to release
) {
    auto pfnRelease = getContext()->urDdiTable.Kernel.pfnRelease;

    if (nullptr == pfnRelease) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urKernelRelease");
    UR_CALL(pfnRelease(hKernel));

    auto KernelInfo = getAsanInterceptor()->getKernelInfo(hKernel);
    if (--KernelInfo->RefCount == 0) {
        UR_CALL(getAsanInterceptor()->eraseKernel(hKernel));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgValue
__urdlllocal ur_result_t UR_APICALL urKernelSetArgValue(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    size_t argSize,    ///< [in] size of argument type
    const ur_kernel_arg_value_properties_t
        *pProperties, ///< [in][optional] pointer to value properties.
    const void
        *pArgValue ///< [in] argument value represented as matching arg type.
) {
    auto pfnSetArgValue = getContext()->urDdiTable.Kernel.pfnSetArgValue;

    if (nullptr == pfnSetArgValue) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urKernelSetArgValue");

    std::shared_ptr<MemBuffer> MemBuffer;
    std::shared_ptr<KernelInfo> KernelInfo;
    if (argSize == sizeof(ur_mem_handle_t) &&
        (MemBuffer = getAsanInterceptor()->getMemBuffer(
             *ur_cast<const ur_mem_handle_t *>(pArgValue)))) {
        auto KernelInfo = getAsanInterceptor()->getKernelInfo(hKernel);
        std::scoped_lock<ur_shared_mutex> Guard(KernelInfo->Mutex);
        KernelInfo->BufferArgs[argIndex] = std::move(MemBuffer);
    } else {
        UR_CALL(
            pfnSetArgValue(hKernel, argIndex, argSize, pProperties, pArgValue));
    }

    return UR_RESULT_SUCCESS;
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
    auto pfnSetArgMemObj = getContext()->urDdiTable.Kernel.pfnSetArgMemObj;

    if (nullptr == pfnSetArgMemObj) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug("==== urKernelSetArgMemObj");

    std::shared_ptr<MemBuffer> MemBuffer;
    std::shared_ptr<KernelInfo> KernelInfo;
    if ((MemBuffer = getAsanInterceptor()->getMemBuffer(hArgValue))) {
        auto KernelInfo = getAsanInterceptor()->getKernelInfo(hKernel);
        std::scoped_lock<ur_shared_mutex> Guard(KernelInfo->Mutex);
        KernelInfo->BufferArgs[argIndex] = std::move(MemBuffer);
    } else {
        UR_CALL(pfnSetArgMemObj(hKernel, argIndex, pProperties, hArgValue));
    }

    return UR_RESULT_SUCCESS;
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
    auto pfnSetArgLocal = getContext()->urDdiTable.Kernel.pfnSetArgLocal;

    if (nullptr == pfnSetArgLocal) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug(
        "==== urKernelSetArgLocal (argIndex={}, argSize={})", argIndex,
        argSize);

    {
        auto KI = getAsanInterceptor()->getKernelInfo(hKernel);
        std::scoped_lock<ur_shared_mutex> Guard(KI->Mutex);
        // TODO: get local variable alignment
        auto argSizeWithRZ = GetSizeAndRedzoneSizeForLocal(
            argSize, ASAN_SHADOW_GRANULARITY, ASAN_SHADOW_GRANULARITY);
        KI->LocalArgs[argIndex] = LocalArgsInfo{argSize, argSizeWithRZ};
        argSize = argSizeWithRZ;
    }

    ur_result_t result =
        pfnSetArgLocal(hKernel, argIndex, argSize, pProperties);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgPointer
__urdlllocal ur_result_t UR_APICALL urKernelSetArgPointer(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_pointer_properties_t
        *pProperties, ///< [in][optional] pointer to USM pointer properties.
    const void *
        pArgValue ///< [in][optional] Pointer obtained by USM allocation or virtual memory
    ///< mapping operation. If null then argument value is considered null.
) {
    auto pfnSetArgPointer = getContext()->urDdiTable.Kernel.pfnSetArgPointer;

    if (nullptr == pfnSetArgPointer) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    getContext()->logger.debug(
        "==== urKernelSetArgPointer (argIndex={}, pArgValue={})", argIndex,
        pArgValue);

    std::shared_ptr<KernelInfo> KI;
    if (getAsanInterceptor()->getOptions().DetectKernelArguments) {
        auto KI = getAsanInterceptor()->getKernelInfo(hKernel);
        std::scoped_lock<ur_shared_mutex> Guard(KI->Mutex);
        KI->PointerArgs[argIndex] = {pArgValue, GetCurrentBacktrace()};
    }

    ur_result_t result =
        pfnSetArgPointer(hKernel, argIndex, pProperties, pArgValue);

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Global table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetGlobalProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_global_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_sanitizer_layer::getContext()->version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_sanitizer_layer::getContext()->version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnAdapterGet = ur_sanitizer_layer::asan::urAdapterGet;

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
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_sanitizer_layer::getContext()->version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_sanitizer_layer::getContext()->version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreate = ur_sanitizer_layer::asan::urContextCreate;
    pDdiTable->pfnRetain = ur_sanitizer_layer::asan::urContextRetain;
    pDdiTable->pfnRelease = ur_sanitizer_layer::asan::urContextRelease;

    pDdiTable->pfnCreateWithNativeHandle =
        ur_sanitizer_layer::asan::urContextCreateWithNativeHandle;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Program table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetProgramProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_program_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_sanitizer_layer::getContext()->version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_sanitizer_layer::getContext()->version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    pDdiTable->pfnCreateWithIL =
        ur_sanitizer_layer::asan::urProgramCreateWithIL;
    pDdiTable->pfnCreateWithBinary =
        ur_sanitizer_layer::asan::urProgramCreateWithBinary;
    pDdiTable->pfnCreateWithNativeHandle =
        ur_sanitizer_layer::asan::urProgramCreateWithNativeHandle;
    pDdiTable->pfnBuild = ur_sanitizer_layer::asan::urProgramBuild;
    pDdiTable->pfnLink = ur_sanitizer_layer::asan::urProgramLink;
    pDdiTable->pfnRetain = ur_sanitizer_layer::asan::urProgramRetain;
    pDdiTable->pfnRelease = ur_sanitizer_layer::asan::urProgramRelease;

    return UR_RESULT_SUCCESS;
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
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_sanitizer_layer::getContext()->version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_sanitizer_layer::getContext()->version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreate = ur_sanitizer_layer::asan::urKernelCreate;
    pDdiTable->pfnRetain = ur_sanitizer_layer::asan::urKernelRetain;
    pDdiTable->pfnRelease = ur_sanitizer_layer::asan::urKernelRelease;
    pDdiTable->pfnSetArgValue = ur_sanitizer_layer::asan::urKernelSetArgValue;
    pDdiTable->pfnSetArgMemObj = ur_sanitizer_layer::asan::urKernelSetArgMemObj;
    pDdiTable->pfnSetArgLocal = ur_sanitizer_layer::asan::urKernelSetArgLocal;
    pDdiTable->pfnSetArgPointer =
        ur_sanitizer_layer::asan::urKernelSetArgPointer;

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
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_sanitizer_layer::getContext()->version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_sanitizer_layer::getContext()->version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnBufferCreate = ur_sanitizer_layer::asan::urMemBufferCreate;
    pDdiTable->pfnRetain = ur_sanitizer_layer::asan::urMemRetain;
    pDdiTable->pfnRelease = ur_sanitizer_layer::asan::urMemRelease;
    pDdiTable->pfnBufferPartition =
        ur_sanitizer_layer::asan::urMemBufferPartition;
    pDdiTable->pfnGetNativeHandle =
        ur_sanitizer_layer::asan::urMemGetNativeHandle;
    pDdiTable->pfnGetInfo = ur_sanitizer_layer::asan::urMemGetInfo;

    return result;
}
/// @brief Exported function for filling application's ProgramExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::UR_RESULT_ERROR_UNSUPPORTED_VERSION
__urdlllocal ur_result_t UR_APICALL urGetProgramExpProcAddrTable(
    ur_api_version_t version, ///< [in] API version requested
    ur_program_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_sanitizer_layer::getContext()->version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_sanitizer_layer::getContext()->version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnBuildExp = ur_sanitizer_layer::asan::urProgramBuildExp;
    pDdiTable->pfnLinkExp = ur_sanitizer_layer::asan::urProgramLinkExp;

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
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_sanitizer_layer::getContext()->version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_sanitizer_layer::getContext()->version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnMemBufferRead =
        ur_sanitizer_layer::asan::urEnqueueMemBufferRead;
    pDdiTable->pfnMemBufferWrite =
        ur_sanitizer_layer::asan::urEnqueueMemBufferWrite;
    pDdiTable->pfnMemBufferReadRect =
        ur_sanitizer_layer::asan::urEnqueueMemBufferReadRect;
    pDdiTable->pfnMemBufferWriteRect =
        ur_sanitizer_layer::asan::urEnqueueMemBufferWriteRect;
    pDdiTable->pfnMemBufferCopy =
        ur_sanitizer_layer::asan::urEnqueueMemBufferCopy;
    pDdiTable->pfnMemBufferCopyRect =
        ur_sanitizer_layer::asan::urEnqueueMemBufferCopyRect;
    pDdiTable->pfnMemBufferFill =
        ur_sanitizer_layer::asan::urEnqueueMemBufferFill;
    pDdiTable->pfnMemBufferMap =
        ur_sanitizer_layer::asan::urEnqueueMemBufferMap;
    pDdiTable->pfnMemUnmap = ur_sanitizer_layer::asan::urEnqueueMemUnmap;
    pDdiTable->pfnKernelLaunch =
        ur_sanitizer_layer::asan::urEnqueueKernelLaunch;

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
    if (nullptr == pDdiTable) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    if (UR_MAJOR_VERSION(ur_sanitizer_layer::getContext()->version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_sanitizer_layer::getContext()->version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }

    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnDeviceAlloc = ur_sanitizer_layer::asan::urUSMDeviceAlloc;
    pDdiTable->pfnHostAlloc = ur_sanitizer_layer::asan::urUSMHostAlloc;
    pDdiTable->pfnSharedAlloc = ur_sanitizer_layer::asan::urUSMSharedAlloc;
    pDdiTable->pfnFree = ur_sanitizer_layer::asan::urUSMFree;

    return result;
}

} // namespace asan

ur_result_t context_t::init(ur_dditable_t *dditable,
                            const std::set<std::string> &enabledLayerNames,
                            [[maybe_unused]] codeloc_data codelocData) {
    ur_result_t result = UR_RESULT_SUCCESS;

    if (enabledLayerNames.count("UR_LAYER_ASAN")) {
        enabledType = SanitizerType::AddressSanitizer;
        initAsanInterceptor();
    } else if (enabledLayerNames.count("UR_LAYER_MSAN")) {
        enabledType = SanitizerType::MemorySanitizer;
    } else if (enabledLayerNames.count("UR_LAYER_TSAN")) {
        enabledType = SanitizerType::ThreadSanitizer;
    }

    // Only support AddressSanitizer now
    if (enabledType != SanitizerType::AddressSanitizer) {
        return result;
    }

    urDdiTable = *dditable;

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::asan::urGetGlobalProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Global);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::asan::urGetContextProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Context);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::asan::urGetKernelProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Kernel);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::asan::urGetProgramProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Program);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::asan::urGetKernelProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Kernel);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::asan::urGetMemProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Mem);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::asan::urGetProgramExpProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->ProgramExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::asan::urGetEnqueueProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->Enqueue);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::asan::urGetUSMProcAddrTable(
            UR_API_VERSION_CURRENT, &dditable->USM);
    }

    return result;
}

} // namespace ur_sanitizer_layer
