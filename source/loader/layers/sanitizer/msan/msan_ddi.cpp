/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_ddi.cpp
 *
 */

#include "msan_ddi.hpp"
#include "msan_interceptor.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"
#include "ur_sanitizer_layer.hpp"

#include <memory>

namespace ur_sanitizer_layer {
namespace msan {

namespace {

ur_result_t setupContext(ur_context_handle_t Context, uint32_t numDevices,
                         const ur_device_handle_t *phDevices) {
    std::shared_ptr<ContextInfo> CI;
    UR_CALL(getMsanInterceptor()->insertContext(Context, CI));
    for (uint32_t i = 0; i < numDevices; ++i) {
        auto hDevice = phDevices[i];
        std::shared_ptr<DeviceInfo> DI;
        UR_CALL(getMsanInterceptor()->insertDevice(hDevice, DI));
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

bool isInstrumentedKernel(ur_kernel_handle_t hKernel) {
    auto hProgram = GetProgram(hKernel);
    auto PI = getMsanInterceptor()->getProgramInfo(hProgram);
    return PI->isKernelInstrumented(hKernel);
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urAdapterGet
ur_result_t urAdapterGet(
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

    // FIXME: This is a W/A to disable heap extended for MSAN so that we can reserve large VA of GPU.
    setenv("NEOReadDebugKeys", "1", 1);
    setenv("AllocateHostAllocationsInHeapExtendedHost", "0", 1);
    setenv("UseHighAlignmentForHeapExtended", "0", 1);

    ur_result_t result = pfnAdapterGet(NumEntries, phAdapters, pNumAdapters);
    if (result == UR_RESULT_SUCCESS && phAdapters) {
        const uint32_t NumAdapters = pNumAdapters ? *pNumAdapters : NumEntries;
        for (uint32_t i = 0; i < NumAdapters; ++i) {
            UR_CALL(getMsanInterceptor()->holdAdapter(phAdapters[i]));
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urUSMDeviceAlloc
ur_result_t urUSMDeviceAlloc(
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
    getContext()->logger.debug("==== urUSMDeviceAlloc");

    return getMsanInterceptor()->allocateMemory(hContext, hDevice, pUSMDesc,
                                                pool, size, ppMem);
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreateWithIL
ur_result_t urProgramCreateWithIL(
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

    getContext()->logger.debug("==== urProgramCreateWithIL");

    UR_CALL(
        pfnProgramCreateWithIL(hContext, pIL, length, pProperties, phProgram));
    UR_CALL(getMsanInterceptor()->insertProgram(*phProgram));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreateWithBinary
ur_result_t urProgramCreateWithBinary(
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

    getContext()->logger.debug("==== urProgramCreateWithBinary");

    UR_CALL(pfnProgramCreateWithBinary(hContext, numDevices, phDevices,
                                       pLengths, ppBinaries, pProperties,
                                       phProgram));
    UR_CALL(getMsanInterceptor()->insertProgram(*phProgram));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramCreateWithNativeHandle
ur_result_t urProgramCreateWithNativeHandle(
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

    getContext()->logger.debug("==== urProgramCreateWithNativeHandle");

    UR_CALL(pfnProgramCreateWithNativeHandle(hNativeProgram, hContext,
                                             pProperties, phProgram));
    UR_CALL(getMsanInterceptor()->insertProgram(*phProgram));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRetain
ur_result_t
urProgramRetain(ur_program_handle_t
                    hProgram ///< [in][retain] handle for the Program to retain
) {
    auto pfnRetain = getContext()->urDdiTable.Program.pfnRetain;

    getContext()->logger.debug("==== urProgramRetain");

    UR_CALL(pfnRetain(hProgram));

    auto ProgramInfo = getMsanInterceptor()->getProgramInfo(hProgram);
    UR_ASSERT(ProgramInfo != nullptr, UR_RESULT_ERROR_INVALID_VALUE);
    ProgramInfo->RefCount++;

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramBuild
ur_result_t urProgramBuild(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_program_handle_t hProgram, ///< [in] handle of the program object
    const char *pOptions          ///< [in] string of build options
) {
    auto pfnProgramBuild = getContext()->urDdiTable.Program.pfnBuild;

    getContext()->logger.debug("==== urProgramBuild");

    UR_CALL(pfnProgramBuild(hContext, hProgram, pOptions));

    UR_CALL(getMsanInterceptor()->registerProgram(hProgram));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramBuildExp
ur_result_t urProgramBuildExp(
    ur_program_handle_t hProgram, ///< [in] Handle of the program to build.
    uint32_t numDevices,          ///< [in] number of devices
    ur_device_handle_t *
        phDevices, ///< [in][range(0, numDevices)] pointer to array of device handles
    const char *
        pOptions ///< [in][optional] pointer to build options null-terminated string.
) {
    auto pfnBuildExp = getContext()->urDdiTable.ProgramExp.pfnBuildExp;

    getContext()->logger.debug("==== urProgramBuildExp");

    UR_CALL(pfnBuildExp(hProgram, numDevices, phDevices, pOptions));
    UR_CALL(getMsanInterceptor()->registerProgram(hProgram));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramLink
ur_result_t urProgramLink(
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

    getContext()->logger.debug("==== urProgramLink");

    UR_CALL(pfnProgramLink(hContext, count, phPrograms, pOptions, phProgram));

    UR_CALL(getMsanInterceptor()->insertProgram(*phProgram));
    UR_CALL(getMsanInterceptor()->registerProgram(*phProgram));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramLinkExp
ur_result_t urProgramLinkExp(
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

    getContext()->logger.debug("==== urProgramLinkExp");

    UR_CALL(pfnProgramLinkExp(hContext, numDevices, phDevices, count,
                              phPrograms, pOptions, phProgram));

    UR_CALL(getMsanInterceptor()->insertProgram(*phProgram));
    UR_CALL(getMsanInterceptor()->registerProgram(*phProgram));

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urProgramRelease
ur_result_t urProgramRelease(
    ur_program_handle_t
        hProgram ///< [in][release] handle for the Program to release
) {
    auto pfnProgramRelease = getContext()->urDdiTable.Program.pfnRelease;

    getContext()->logger.debug("==== urProgramRelease");

    UR_CALL(pfnProgramRelease(hProgram));

    auto ProgramInfo = getMsanInterceptor()->getProgramInfo(hProgram);
    UR_ASSERT(ProgramInfo != nullptr, UR_RESULT_ERROR_INVALID_VALUE);
    if (--ProgramInfo->RefCount == 0) {
        UR_CALL(getMsanInterceptor()->unregisterProgram(hProgram));
        UR_CALL(getMsanInterceptor()->eraseProgram(hProgram));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urEnqueueKernelLaunch
ur_result_t urEnqueueKernelLaunch(
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

    getContext()->logger.debug("==== urEnqueueKernelLaunch");

    if (!isInstrumentedKernel(hKernel)) {
        return pfnKernelLaunch(hQueue, hKernel, workDim, pGlobalWorkOffset,
                               pGlobalWorkSize, pLocalWorkSize,
                               numEventsInWaitList, phEventWaitList, phEvent);
    }

    USMLaunchInfo LaunchInfo(GetContext(hQueue), GetDevice(hQueue),
                             pGlobalWorkSize, pLocalWorkSize, pGlobalWorkOffset,
                             workDim);
    UR_CALL(LaunchInfo.initialize());

    UR_CALL(getMsanInterceptor()->preLaunchKernel(hKernel, hQueue, LaunchInfo));

    ur_event_handle_t hEvent{};
    ur_result_t result =
        pfnKernelLaunch(hQueue, hKernel, workDim, pGlobalWorkOffset,
                        pGlobalWorkSize, LaunchInfo.LocalWorkSize.data(),
                        numEventsInWaitList, phEventWaitList, &hEvent);

    if (result == UR_RESULT_SUCCESS) {
        UR_CALL(getMsanInterceptor()->postLaunchKernel(hKernel, hQueue,
                                                       LaunchInfo));
    }

    if (phEvent) {
        *phEvent = hEvent;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextCreate
ur_result_t urContextCreate(
    uint32_t numDevices, ///< [in] the number of devices given in phDevices
    const ur_device_handle_t
        *phDevices, ///< [in][range(0, numDevices)] array of handle of devices.
    const ur_context_properties_t *
        pProperties, ///< [in][optional] pointer to context creation properties.
    ur_context_handle_t
        *phContext ///< [out] pointer to handle of context object created
) {
    auto pfnCreate = getContext()->urDdiTable.Context.pfnCreate;

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
ur_result_t urContextCreateWithNativeHandle(
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
ur_result_t urContextRetain(
    ur_context_handle_t
        hContext ///< [in] handle of the context to get a reference of.
) {
    auto pfnRetain = getContext()->urDdiTable.Context.pfnRetain;

    getContext()->logger.debug("==== urContextRetain");

    UR_CALL(pfnRetain(hContext));

    auto ContextInfo = getMsanInterceptor()->getContextInfo(hContext);
    UR_ASSERT(ContextInfo != nullptr, UR_RESULT_ERROR_INVALID_VALUE);
    ContextInfo->RefCount++;

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urContextRelease
ur_result_t urContextRelease(
    ur_context_handle_t hContext ///< [in] handle of the context to release.
) {
    auto pfnRelease = getContext()->urDdiTable.Context.pfnRelease;

    getContext()->logger.debug("==== urContextRelease");

    UR_CALL(pfnRelease(hContext));

    auto ContextInfo = getMsanInterceptor()->getContextInfo(hContext);
    UR_ASSERT(ContextInfo != nullptr, UR_RESULT_ERROR_INVALID_VALUE);
    if (--ContextInfo->RefCount == 0) {
        UR_CALL(getMsanInterceptor()->eraseContext(hContext));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemBufferCreate
ur_result_t urMemBufferCreate(
    ur_context_handle_t hContext, ///< [in] handle of the context object
    ur_mem_flags_t flags, ///< [in] allocation and usage information flags
    size_t size, ///< [in] size in bytes of the memory object to be allocated
    const ur_buffer_properties_t
        *pProperties, ///< [in][optional] pointer to buffer creation properties
    ur_mem_handle_t
        *phBuffer ///< [out] pointer to handle of the memory buffer created
) {
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
            getMsanInterceptor()->getContextInfo(hContext);
        for (const auto &hDevice : CtxInfo->DeviceList) {
            ManagedQueue InternalQueue(hContext, hDevice);
            char *Handle = nullptr;
            UR_CALL(pMemBuffer->getHandle(hDevice, Handle));
            UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
                InternalQueue, true, Handle, Host, size, 0, nullptr, nullptr));
        }
    }

    ur_result_t result = getMsanInterceptor()->insertMemBuffer(pMemBuffer);
    *phBuffer = ur_cast<ur_mem_handle_t>(pMemBuffer.get());

    return result;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemGetInfo
ur_result_t urMemGetInfo(
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

    getContext()->logger.debug("==== urMemGetInfo");

    if (auto MemBuffer = getMsanInterceptor()->getMemBuffer(hMemory)) {
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
ur_result_t urMemRetain(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to get access
) {
    auto pfnRetain = getContext()->urDdiTable.Mem.pfnRetain;

    getContext()->logger.debug("==== urMemRetain");

    if (auto MemBuffer = getMsanInterceptor()->getMemBuffer(hMem)) {
        MemBuffer->RefCount++;
    } else {
        UR_CALL(pfnRetain(hMem));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemRelease
ur_result_t urMemRelease(
    ur_mem_handle_t hMem ///< [in] handle of the memory object to release
) {
    auto pfnRelease = getContext()->urDdiTable.Mem.pfnRelease;

    getContext()->logger.debug("==== urMemRelease");

    if (auto MemBuffer = getMsanInterceptor()->getMemBuffer(hMem)) {
        if (--MemBuffer->RefCount != 0) {
            return UR_RESULT_SUCCESS;
        }
        UR_CALL(MemBuffer->free());
        UR_CALL(getMsanInterceptor()->eraseMemBuffer(hMem));
    } else {
        UR_CALL(pfnRelease(hMem));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemBufferPartition
ur_result_t urMemBufferPartition(
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

    getContext()->logger.debug("==== urMemBufferPartition");

    if (auto ParentBuffer = getMsanInterceptor()->getMemBuffer(hBuffer)) {
        if (ParentBuffer->Size < (pRegion->origin + pRegion->size)) {
            return UR_RESULT_ERROR_INVALID_BUFFER_SIZE;
        }
        std::shared_ptr<MemBuffer> SubBuffer = std::make_shared<MemBuffer>(
            ParentBuffer, pRegion->origin, pRegion->size);
        UR_CALL(getMsanInterceptor()->insertMemBuffer(SubBuffer));
        *phMem = reinterpret_cast<ur_mem_handle_t>(SubBuffer.get());
    } else {
        UR_CALL(pfnBufferPartition(hBuffer, flags, bufferCreateType, pRegion,
                                   phMem));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urMemGetNativeHandle
ur_result_t urMemGetNativeHandle(
    ur_mem_handle_t hMem, ///< [in] handle of the mem.
    ur_device_handle_t hDevice,
    ur_native_handle_t
        *phNativeMem ///< [out] a pointer to the native handle of the mem.
) {
    auto pfnGetNativeHandle = getContext()->urDdiTable.Mem.pfnGetNativeHandle;

    getContext()->logger.debug("==== urMemGetNativeHandle");

    if (auto MemBuffer = getMsanInterceptor()->getMemBuffer(hMem)) {
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
ur_result_t urEnqueueMemBufferRead(
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

    getContext()->logger.debug("==== urEnqueueMemBufferRead");

    if (auto MemBuffer = getMsanInterceptor()->getMemBuffer(hBuffer)) {
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
ur_result_t urEnqueueMemBufferWrite(
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

    getContext()->logger.debug("==== urEnqueueMemBufferWrite");

    if (auto MemBuffer = getMsanInterceptor()->getMemBuffer(hBuffer)) {
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
ur_result_t urEnqueueMemBufferReadRect(
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

    getContext()->logger.debug("==== urEnqueueMemBufferReadRect");

    if (auto MemBuffer = getMsanInterceptor()->getMemBuffer(hBuffer)) {
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
ur_result_t urEnqueueMemBufferWriteRect(
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

    getContext()->logger.debug("==== urEnqueueMemBufferWriteRect");

    if (auto MemBuffer = getMsanInterceptor()->getMemBuffer(hBuffer)) {
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
ur_result_t urEnqueueMemBufferCopy(
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

    getContext()->logger.debug("==== urEnqueueMemBufferCopy");

    auto SrcBuffer = getMsanInterceptor()->getMemBuffer(hBufferSrc);
    auto DstBuffer = getMsanInterceptor()->getMemBuffer(hBufferDst);

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
ur_result_t urEnqueueMemBufferCopyRect(
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

    getContext()->logger.debug("==== urEnqueueMemBufferCopyRect");

    auto SrcBuffer = getMsanInterceptor()->getMemBuffer(hBufferSrc);
    auto DstBuffer = getMsanInterceptor()->getMemBuffer(hBufferDst);

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
ur_result_t urEnqueueMemBufferFill(
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

    getContext()->logger.debug("==== urEnqueueMemBufferFill");

    if (auto MemBuffer = getMsanInterceptor()->getMemBuffer(hBuffer)) {
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
ur_result_t urEnqueueMemBufferMap(
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

    getContext()->logger.debug("==== urEnqueueMemBufferMap");

    if (auto MemBuffer = getMsanInterceptor()->getMemBuffer(hBuffer)) {

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
            UR_CALL(getContext()->urDdiTable.USM.pfnHostAlloc(
                Context, &USMDesc, Pool, size, ppRetMap));
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
ur_result_t urEnqueueMemUnmap(
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

    getContext()->logger.debug("==== urEnqueueMemUnmap");

    if (auto MemBuffer = getMsanInterceptor()->getMemBuffer(hMem)) {
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
            UR_CALL(getContext()->urDdiTable.USM.pfnFree(Context, pMappedPtr));
        }
    } else {
        UR_CALL(pfnMemUnmap(hQueue, hMem, pMappedPtr, numEventsInWaitList,
                            phEventWaitList, phEvent));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelCreate
ur_result_t urKernelCreate(
    ur_program_handle_t hProgram, ///< [in] handle of the program instance
    const char *pKernelName,      ///< [in] pointer to null-terminated string.
    ur_kernel_handle_t
        *phKernel ///< [out] pointer to handle of kernel object created.
) {
    auto pfnCreate = getContext()->urDdiTable.Kernel.pfnCreate;

    getContext()->logger.debug("==== urKernelCreate");

    UR_CALL(pfnCreate(hProgram, pKernelName, phKernel));
    if (isInstrumentedKernel(*phKernel)) {
        UR_CALL(getMsanInterceptor()->insertKernel(*phKernel));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRetain
ur_result_t urKernelRetain(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to retain
) {
    auto pfnRetain = getContext()->urDdiTable.Kernel.pfnRetain;

    getContext()->logger.debug("==== urKernelRetain");

    UR_CALL(pfnRetain(hKernel));

    auto KernelInfo = getMsanInterceptor()->getKernelInfo(hKernel);
    if (KernelInfo) {
        KernelInfo->RefCount++;
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelRelease
ur_result_t urKernelRelease(
    ur_kernel_handle_t hKernel ///< [in] handle for the Kernel to release
) {
    auto pfnRelease = getContext()->urDdiTable.Kernel.pfnRelease;

    getContext()->logger.debug("==== urKernelRelease");
    UR_CALL(pfnRelease(hKernel));

    auto KernelInfo = getMsanInterceptor()->getKernelInfo(hKernel);
    if (KernelInfo) {
        if (--KernelInfo->RefCount == 0) {
            UR_CALL(getMsanInterceptor()->eraseKernel(hKernel));
        }
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Intercept function for urKernelSetArgValue
ur_result_t urKernelSetArgValue(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    size_t argSize,    ///< [in] size of argument type
    const ur_kernel_arg_value_properties_t
        *pProperties, ///< [in][optional] pointer to value properties.
    const void
        *pArgValue ///< [in] argument value represented as matching arg type.
) {
    auto pfnSetArgValue = getContext()->urDdiTable.Kernel.pfnSetArgValue;

    getContext()->logger.debug("==== urKernelSetArgValue");

    std::shared_ptr<MemBuffer> MemBuffer;
    std::shared_ptr<KernelInfo> KernelInfo;
    if (argSize == sizeof(ur_mem_handle_t) &&
        (MemBuffer = getMsanInterceptor()->getMemBuffer(
             *ur_cast<const ur_mem_handle_t *>(pArgValue))) &&
        (KernelInfo = getMsanInterceptor()->getKernelInfo(hKernel))) {
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
ur_result_t urKernelSetArgMemObj(
    ur_kernel_handle_t hKernel, ///< [in] handle of the kernel object
    uint32_t argIndex, ///< [in] argument index in range [0, num args - 1]
    const ur_kernel_arg_mem_obj_properties_t
        *pProperties, ///< [in][optional] pointer to Memory object properties.
    ur_mem_handle_t hArgValue ///< [in][optional] handle of Memory object.
) {
    auto pfnSetArgMemObj = getContext()->urDdiTable.Kernel.pfnSetArgMemObj;

    getContext()->logger.debug("==== urKernelSetArgMemObj");

    std::shared_ptr<MemBuffer> MemBuffer;
    std::shared_ptr<KernelInfo> KernelInfo;
    if ((MemBuffer = getMsanInterceptor()->getMemBuffer(hArgValue)) &&
        (KernelInfo = getMsanInterceptor()->getKernelInfo(hKernel))) {
        std::scoped_lock<ur_shared_mutex> Guard(KernelInfo->Mutex);
        KernelInfo->BufferArgs[argIndex] = std::move(MemBuffer);
    } else {
        UR_CALL(pfnSetArgMemObj(hKernel, argIndex, pProperties, hArgValue));
    }

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Global table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
ur_result_t urGetGlobalProcAddrTable(
    ur_global_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnAdapterGet = ur_sanitizer_layer::msan::urAdapterGet;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Context table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
ur_result_t urGetContextProcAddrTable(
    ur_context_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreate = ur_sanitizer_layer::msan::urContextCreate;
    pDdiTable->pfnRetain = ur_sanitizer_layer::msan::urContextRetain;
    pDdiTable->pfnRelease = ur_sanitizer_layer::msan::urContextRelease;

    pDdiTable->pfnCreateWithNativeHandle =
        ur_sanitizer_layer::msan::urContextCreateWithNativeHandle;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Program table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
ur_result_t urGetProgramProcAddrTable(
    ur_program_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    pDdiTable->pfnCreateWithIL =
        ur_sanitizer_layer::msan::urProgramCreateWithIL;
    pDdiTable->pfnCreateWithBinary =
        ur_sanitizer_layer::msan::urProgramCreateWithBinary;
    pDdiTable->pfnCreateWithNativeHandle =
        ur_sanitizer_layer::msan::urProgramCreateWithNativeHandle;
    pDdiTable->pfnBuild = ur_sanitizer_layer::msan::urProgramBuild;
    pDdiTable->pfnLink = ur_sanitizer_layer::msan::urProgramLink;
    pDdiTable->pfnRetain = ur_sanitizer_layer::msan::urProgramRetain;
    pDdiTable->pfnRelease = ur_sanitizer_layer::msan::urProgramRelease;

    return UR_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Kernel table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
ur_result_t urGetKernelProcAddrTable(
    ur_kernel_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnCreate = ur_sanitizer_layer::msan::urKernelCreate;
    pDdiTable->pfnRetain = ur_sanitizer_layer::msan::urKernelRetain;
    pDdiTable->pfnRelease = ur_sanitizer_layer::msan::urKernelRelease;
    pDdiTable->pfnSetArgValue = ur_sanitizer_layer::msan::urKernelSetArgValue;
    pDdiTable->pfnSetArgMemObj = ur_sanitizer_layer::msan::urKernelSetArgMemObj;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Mem table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
ur_result_t urGetMemProcAddrTable(
    ur_mem_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnBufferCreate = ur_sanitizer_layer::msan::urMemBufferCreate;
    pDdiTable->pfnRetain = ur_sanitizer_layer::msan::urMemRetain;
    pDdiTable->pfnRelease = ur_sanitizer_layer::msan::urMemRelease;
    pDdiTable->pfnBufferPartition =
        ur_sanitizer_layer::msan::urMemBufferPartition;
    pDdiTable->pfnGetNativeHandle =
        ur_sanitizer_layer::msan::urMemGetNativeHandle;
    pDdiTable->pfnGetInfo = ur_sanitizer_layer::msan::urMemGetInfo;

    return result;
}
/// @brief Exported function for filling application's ProgramExp table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
ur_result_t urGetProgramExpProcAddrTable(
    ur_program_exp_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnBuildExp = ur_sanitizer_layer::msan::urProgramBuildExp;
    pDdiTable->pfnLinkExp = ur_sanitizer_layer::msan::urProgramLinkExp;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Enqueue table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
ur_result_t urGetEnqueueProcAddrTable(
    ur_enqueue_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnMemBufferRead =
        ur_sanitizer_layer::msan::urEnqueueMemBufferRead;
    pDdiTable->pfnMemBufferWrite =
        ur_sanitizer_layer::msan::urEnqueueMemBufferWrite;
    pDdiTable->pfnMemBufferReadRect =
        ur_sanitizer_layer::msan::urEnqueueMemBufferReadRect;
    pDdiTable->pfnMemBufferWriteRect =
        ur_sanitizer_layer::msan::urEnqueueMemBufferWriteRect;
    pDdiTable->pfnMemBufferCopy =
        ur_sanitizer_layer::msan::urEnqueueMemBufferCopy;
    pDdiTable->pfnMemBufferCopyRect =
        ur_sanitizer_layer::msan::urEnqueueMemBufferCopyRect;
    pDdiTable->pfnMemBufferFill =
        ur_sanitizer_layer::msan::urEnqueueMemBufferFill;
    pDdiTable->pfnMemBufferMap =
        ur_sanitizer_layer::msan::urEnqueueMemBufferMap;
    pDdiTable->pfnMemUnmap = ur_sanitizer_layer::msan::urEnqueueMemUnmap;
    pDdiTable->pfnKernelLaunch =
        ur_sanitizer_layer::msan::urEnqueueKernelLaunch;

    return result;
}
///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's USM table
///        with current process' addresses
///
/// @returns
///     - ::UR_RESULT_SUCCESS
///     - ::UR_RESULT_ERROR_INVALID_NULL_POINTER
ur_result_t urGetUSMProcAddrTable(
    ur_usm_dditable_t
        *pDdiTable ///< [in,out] pointer to table of DDI function pointers
) {
    ur_result_t result = UR_RESULT_SUCCESS;

    pDdiTable->pfnDeviceAlloc = ur_sanitizer_layer::msan::urUSMDeviceAlloc;

    return result;
}

ur_result_t urCheckVersion(ur_api_version_t version) {
    if (UR_MAJOR_VERSION(ur_sanitizer_layer::getContext()->version) !=
            UR_MAJOR_VERSION(version) ||
        UR_MINOR_VERSION(ur_sanitizer_layer::getContext()->version) >
            UR_MINOR_VERSION(version)) {
        return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
    }
    return UR_RESULT_SUCCESS;
}

} // namespace msan

ur_result_t initMsanDDITable(ur_dditable_t *dditable) {
    ur_result_t result = UR_RESULT_SUCCESS;

    getContext()->logger.always("==== DeviceSanitizer: MSAN");

    if (UR_RESULT_SUCCESS == result) {
        result =
            ur_sanitizer_layer::msan::urCheckVersion(UR_API_VERSION_CURRENT);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::msan::urGetGlobalProcAddrTable(
            &dditable->Global);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::msan::urGetContextProcAddrTable(
            &dditable->Context);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::msan::urGetKernelProcAddrTable(
            &dditable->Kernel);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::msan::urGetProgramProcAddrTable(
            &dditable->Program);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::msan::urGetKernelProcAddrTable(
            &dditable->Kernel);
    }

    if (UR_RESULT_SUCCESS == result) {
        result =
            ur_sanitizer_layer::msan::urGetMemProcAddrTable(&dditable->Mem);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::msan::urGetProgramExpProcAddrTable(
            &dditable->ProgramExp);
    }

    if (UR_RESULT_SUCCESS == result) {
        result = ur_sanitizer_layer::msan::urGetEnqueueProcAddrTable(
            &dditable->Enqueue);
    }

    if (UR_RESULT_SUCCESS == result) {
        result =
            ur_sanitizer_layer::msan::urGetUSMProcAddrTable(&dditable->USM);
    }

    if (result != UR_RESULT_SUCCESS) {
        getContext()->logger.error("Initialize MSAN DDI table failed: {}",
                                   result);
    }

    return result;
}

} // namespace ur_sanitizer_layer
