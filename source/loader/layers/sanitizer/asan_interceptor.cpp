//===----------------------------------------------------------------------===//
/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_interceptor.cpp
 *
 */

#include "asan_interceptor.hpp"
#include "asan_options.hpp"
#include "asan_quarantine.hpp"
#include "asan_report.hpp"
#include "asan_shadow_setup.hpp"
#include "stacktrace.hpp"
#include "ur_sanitizer_utils.hpp"

namespace ur_sanitizer_layer {

namespace {

uptr MemToShadow_CPU(uptr USM_SHADOW_BASE, uptr UPtr) {
    return USM_SHADOW_BASE + (UPtr >> ASAN_SHADOW_SCALE);
}

uptr MemToShadow_PVC(uptr USM_SHADOW_BASE, uptr UPtr) {
    if (UPtr & 0xFF00000000000000ULL) { // Device USM
        return USM_SHADOW_BASE + 0x80000000000ULL +
               ((UPtr & 0xFFFFFFFFFFFFULL) >> ASAN_SHADOW_SCALE);
    } else { // Only consider 47bit VA
        return USM_SHADOW_BASE +
               ((UPtr & 0x7FFFFFFFFFFFULL) >> ASAN_SHADOW_SCALE);
    }
}

ur_result_t urEnqueueUSMSet(ur_queue_handle_t Queue, void *Ptr, char Value,
                            size_t Size, uint32_t NumEvents = 0,
                            const ur_event_handle_t *EventWaitList = nullptr,
                            ur_event_handle_t *OutEvent = nullptr) {
    if (Size == 0) {
        return UR_RESULT_SUCCESS;
    }
    return context.urDdiTable.Enqueue.pfnUSMFill(
        Queue, Ptr, 1, &Value, Size, NumEvents, EventWaitList, OutEvent);
}

ur_result_t enqueueMemSetShadow(ur_context_handle_t Context,
                                std::shared_ptr<DeviceInfo> &DeviceInfo,
                                ur_queue_handle_t Queue, uptr Ptr, uptr Size,
                                u8 Value) {
    if (Size == 0) {
        return UR_RESULT_SUCCESS;
    }
    if (DeviceInfo->Type == DeviceType::CPU) {
        uptr ShadowBegin = MemToShadow_CPU(DeviceInfo->ShadowOffset, Ptr);
        uptr ShadowEnd =
            MemToShadow_CPU(DeviceInfo->ShadowOffset, Ptr + Size - 1);

        // Poison shadow memory outside of asan runtime is not allowed, so we
        // need to avoid memset's call from being intercepted.
        static auto MemSet =
            (void *(*)(void *, int, size_t))GetMemFunctionPointer("memset");
        if (!MemSet) {
            return UR_RESULT_ERROR_UNKNOWN;
        }
        context.logger.debug("enqueueMemSetShadow(addr={}, count={}, value={})",
                             (void *)ShadowBegin, ShadowEnd - ShadowBegin + 1,
                             (void *)(size_t)Value);
        MemSet((void *)ShadowBegin, Value, ShadowEnd - ShadowBegin + 1);
    } else if (DeviceInfo->Type == DeviceType::GPU_PVC) {
        uptr ShadowBegin = MemToShadow_PVC(DeviceInfo->ShadowOffset, Ptr);
        uptr ShadowEnd =
            MemToShadow_PVC(DeviceInfo->ShadowOffset, Ptr + Size - 1);
        assert(ShadowBegin <= ShadowEnd);
        {
            static const size_t PageSize =
                GetVirtualMemGranularity(Context, DeviceInfo->Handle);

            ur_physical_mem_properties_t Desc{
                UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES, nullptr, 0};
            static ur_physical_mem_handle_t PhysicalMem{};

            // Make sure [Ptr, Ptr + Size] is mapped to physical memory
            for (auto MappedPtr = RoundDownTo(ShadowBegin, PageSize);
                 MappedPtr <= ShadowEnd; MappedPtr += PageSize) {
                if (!PhysicalMem) {
                    auto URes = context.urDdiTable.PhysicalMem.pfnCreate(
                        Context, DeviceInfo->Handle, PageSize, &Desc,
                        &PhysicalMem);
                    if (URes != UR_RESULT_SUCCESS) {
                        context.logger.error("urPhysicalMemCreate(): {}", URes);
                        return URes;
                    }
                }

                context.logger.debug("urVirtualMemMap: {} ~ {}",
                                     (void *)MappedPtr,
                                     (void *)(MappedPtr + PageSize - 1));

                // FIXME: No flag to check the failed reason is VA is already mapped
                auto URes = context.urDdiTable.VirtualMem.pfnMap(
                    Context, (void *)MappedPtr, PageSize, PhysicalMem, 0,
                    UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE);
                if (URes != UR_RESULT_SUCCESS) {
                    context.logger.debug("urVirtualMemMap(): {}", URes);
                }

                // Initialize to zero
                if (URes == UR_RESULT_SUCCESS) {
                    // Reset PhysicalMem to null since it's been mapped
                    PhysicalMem = nullptr;

                    auto URes =
                        urEnqueueUSMSet(Queue, (void *)MappedPtr, 0, PageSize);
                    if (URes != UR_RESULT_SUCCESS) {
                        context.logger.error("urEnqueueUSMFill(): {}", URes);
                        return URes;
                    }
                }
            }
        }

        auto URes = urEnqueueUSMSet(Queue, (void *)ShadowBegin, Value,
                                    ShadowEnd - ShadowBegin + 1);
        context.logger.debug(
            "enqueueMemSetShadow (addr={}, count={}, value={}): {}",
            (void *)ShadowBegin, ShadowEnd - ShadowBegin + 1,
            (void *)(size_t)Value, URes);
        if (URes != UR_RESULT_SUCCESS) {
            context.logger.error("urEnqueueUSMFill(): {}", URes);
            return URes;
        }
    } else {
        context.logger.error("Unsupport device type");
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return UR_RESULT_SUCCESS;
}

} // namespace

SanitizerInterceptor::SanitizerInterceptor() {
    if (Options().MaxQuarantineSizeMB) {
        m_Quarantine = std::make_unique<Quarantine>(
            static_cast<uint64_t>(Options().MaxQuarantineSizeMB) * 1024 * 1024);
    }
}

SanitizerInterceptor::~SanitizerInterceptor() {
    DestroyShadowMemoryOnCPU();
    DestroyShadowMemoryOnPVC();
}

/// The memory chunk allocated from the underlying allocator looks like this:
/// L L L L L L U U U U U U R R
///   L -- left redzone words (0 or more bytes)
///   U -- user memory.
///   R -- right redzone (0 or more bytes)
///
/// ref: "compiler-rt/lib/asan/asan_allocator.cpp" Allocator::Allocate
ur_result_t SanitizerInterceptor::allocateMemory(
    ur_context_handle_t Context, ur_device_handle_t Device,
    const ur_usm_desc_t *Properties, ur_usm_pool_handle_t Pool, size_t Size,
    AllocType Type, void **ResultPtr) {

    auto ContextInfo = getContextInfo(Context);
    std::shared_ptr<DeviceInfo> DeviceInfo =
        Device ? getDeviceInfo(Device) : nullptr;

    /// Modified from llvm/compiler-rt/lib/asan/asan_allocator.cpp
    uint32_t Alignment = Properties ? Properties->align : 0;
    // Alignment must be zero or a power-of-two
    if (0 != (Alignment & (Alignment - 1))) {
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    const uint32_t MinAlignment = ASAN_SHADOW_GRANULARITY;
    if (Alignment == 0) {
        Alignment = DeviceInfo ? DeviceInfo->Alignment : MinAlignment;
    }
    if (Alignment < MinAlignment) {
        Alignment = MinAlignment;
    }

    uptr RZLog = ComputeRZLog(Size, Options().MinRZSize, Options().MaxRZSize);
    uptr RZSize = RZLog2Size(RZLog);
    uptr RoundedSize = RoundUpTo(Size, Alignment);
    uptr NeededSize = RoundedSize + RZSize * 2;
    if (Alignment > MinAlignment) {
        NeededSize += Alignment;
    }

    void *Allocated = nullptr;

    if (Type == AllocType::DEVICE_USM) {
        UR_CALL(context.urDdiTable.USM.pfnDeviceAlloc(
            Context, Device, Properties, Pool, NeededSize, &Allocated));
    } else if (Type == AllocType::HOST_USM) {
        UR_CALL(context.urDdiTable.USM.pfnHostAlloc(Context, Properties, Pool,
                                                    NeededSize, &Allocated));
    } else if (Type == AllocType::SHARED_USM) {
        UR_CALL(context.urDdiTable.USM.pfnSharedAlloc(
            Context, Device, Properties, Pool, NeededSize, &Allocated));
    } else if (Type == AllocType::MEM_BUFFER) {
        UR_CALL(context.urDdiTable.USM.pfnDeviceAlloc(
            Context, Device, Properties, Pool, NeededSize, &Allocated));
    } else {
        context.logger.error("Unsupport memory type");
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    uptr AllocBegin = reinterpret_cast<uptr>(Allocated);
    [[maybe_unused]] uptr AllocEnd = AllocBegin + NeededSize;
    uptr UserBegin = AllocBegin + RZSize;
    if (!IsAligned(UserBegin, Alignment)) {
        UserBegin = RoundUpTo(UserBegin, Alignment);
    }
    uptr UserEnd = UserBegin + Size;
    assert(UserEnd <= AllocEnd);

    *ResultPtr = reinterpret_cast<void *>(UserBegin);

    auto AI = std::make_shared<AllocInfo>(AllocInfo{AllocBegin,
                                                    UserBegin,
                                                    UserEnd,
                                                    NeededSize,
                                                    Type,
                                                    false,
                                                    Context,
                                                    Device,
                                                    GetCurrentBacktrace(),
                                                    {}});

    AI->print();

    // For updating shadow memory
    if (Device) { // Device/Shared USM
        ContextInfo->insertAllocInfo({Device}, AI);
    } else { // Host USM
        ContextInfo->insertAllocInfo(ContextInfo->DeviceList, AI);
    }

    // For memory release
    {
        std::scoped_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
        m_AllocationMap.emplace(AI->AllocBegin, std::move(AI));
    }

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::releaseMemory(ur_context_handle_t Context,
                                                void *Ptr) {
    auto ContextInfo = getContextInfo(Context);

    auto Addr = reinterpret_cast<uptr>(Ptr);
    auto AllocInfoItOp = findAllocInfoByAddress(Addr);

    if (!AllocInfoItOp) {
        // "Addr" might be a host pointer
        ReportBadFree(Addr, GetCurrentBacktrace(), nullptr);
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    auto AllocInfoIt = *AllocInfoItOp;
    auto &AllocInfo = AllocInfoIt->second;

    if (AllocInfo->Context != Context) {
        if (AllocInfo->UserBegin == Addr) {
            ReportBadContext(Addr, GetCurrentBacktrace(), AllocInfo);
        } else {
            // "Addr" might be a host pointer
            ReportBadFree(Addr, GetCurrentBacktrace(), nullptr);
        }
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (Addr != AllocInfo->UserBegin) {
        ReportBadFree(Addr, GetCurrentBacktrace(), AllocInfo);
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (AllocInfo->IsReleased) {
        ReportDoubleFree(Addr, GetCurrentBacktrace(), AllocInfo);
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    AllocInfo->IsReleased = true;
    AllocInfo->ReleaseStack = GetCurrentBacktrace();

    if (AllocInfo->Type == AllocType::HOST_USM) {
        ContextInfo->insertAllocInfo(ContextInfo->DeviceList, AllocInfo);
    } else {
        ContextInfo->insertAllocInfo({AllocInfo->Device}, AllocInfo);
    }

    // If quarantine is disabled, USM is freed immediately
    if (!m_Quarantine) {
        context.logger.debug("Free: {}", (void *)AllocInfo->AllocBegin);
        std::scoped_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
        m_AllocationMap.erase(AllocInfoIt);
        return context.urDdiTable.USM.pfnFree(Context,
                                              (void *)(AllocInfo->AllocBegin));
    }

    auto ReleaseList = m_Quarantine->put(AllocInfo->Device, AllocInfoIt);
    if (ReleaseList.size()) {
        std::scoped_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
        for (auto &It : ReleaseList) {
            context.logger.info("Quarantine Free: {}",
                                (void *)It->second->AllocBegin);
            m_AllocationMap.erase(It);
            UR_CALL(context.urDdiTable.USM.pfnFree(
                Context, (void *)(It->second->AllocBegin)));
        }
    }

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::preLaunchKernel(ur_kernel_handle_t Kernel,
                                                  ur_queue_handle_t Queue,
                                                  USMLaunchInfo &LaunchInfo) {
    auto Context = GetContext(Queue);
    auto Device = GetDevice(Queue);
    auto ContextInfo = getContextInfo(Context);
    auto DeviceInfo = getDeviceInfo(Device);
    auto KernelInfo = getKernelInfo(Kernel);

    UR_CALL(LaunchInfo.updateKernelInfo(*KernelInfo.get()));

    ManagedQueue InternalQueue(Context, Device);
    if (!InternalQueue) {
        context.logger.error("Failed to create internal queue");
        return UR_RESULT_ERROR_INVALID_QUEUE;
    }

    UR_CALL(
        prepareLaunch(Context, DeviceInfo, InternalQueue, Kernel, LaunchInfo));

    UR_CALL(updateShadowMemory(ContextInfo, DeviceInfo, InternalQueue));

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::postLaunchKernel(ur_kernel_handle_t Kernel,
                                                   ur_queue_handle_t Queue,
                                                   USMLaunchInfo &LaunchInfo) {
    // FIXME: We must use block operation here, until we support urEventSetCallback
    auto Result = context.urDdiTable.Queue.pfnFinish(Queue);

    if (Result == UR_RESULT_SUCCESS) {
        const auto &AH = LaunchInfo.Data->SanitizerReport;
        if (!AH.Flag) {
            return UR_RESULT_SUCCESS;
        }
        if (AH.ErrorType == DeviceSanitizerErrorType::USE_AFTER_FREE) {
            ReportUseAfterFree(AH, Kernel, GetContext(Queue));
        } else if (AH.ErrorType == DeviceSanitizerErrorType::OUT_OF_BOUNDS) {
            ReportOutOfBoundsError(AH, Kernel);
        } else {
            ReportGenericError(AH);
        }
    }

    return Result;
}

ur_result_t DeviceInfo::allocShadowMemory(ur_context_handle_t Context) {
    if (Type == DeviceType::CPU) {
        UR_CALL(SetupShadowMemoryOnCPU(ShadowOffset, ShadowOffsetEnd));
    } else if (Type == DeviceType::GPU_PVC) {
        UR_CALL(SetupShadowMemoryOnPVC(Context, ShadowOffset, ShadowOffsetEnd));
    } else {
        context.logger.error("Unsupport device type");
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }
    context.logger.info("ShadowMemory(Global): {} - {}", (void *)ShadowOffset,
                        (void *)ShadowOffsetEnd);
    return UR_RESULT_SUCCESS;
}

/// Each 8 bytes of application memory are mapped into one byte of shadow memory
/// The meaning of that byte:
///  - Negative: All bytes are not accessible (poisoned)
///  - 0: All bytes are accessible
///  - 1 <= k <= 7: Only the first k bytes is accessible
///
/// ref: https://github.com/google/sanitizers/wiki/AddressSanitizerAlgorithm#mapping
ur_result_t SanitizerInterceptor::enqueueAllocInfo(
    ur_context_handle_t Context, std::shared_ptr<DeviceInfo> &DeviceInfo,
    ur_queue_handle_t Queue, std::shared_ptr<AllocInfo> &AI) {
    if (AI->IsReleased) {
        int ShadowByte;
        switch (AI->Type) {
        case AllocType::HOST_USM:
            ShadowByte = kUsmHostDeallocatedMagic;
            break;
        case AllocType::DEVICE_USM:
            ShadowByte = kUsmDeviceDeallocatedMagic;
            break;
        case AllocType::SHARED_USM:
            ShadowByte = kUsmSharedDeallocatedMagic;
            break;
        case AllocType::MEM_BUFFER:
            ShadowByte = kMemBufferDeallocatedMagic;
            break;
        default:
            ShadowByte = 0xff;
            assert(false && "Unknow AllocInfo Type");
        }
        UR_CALL(enqueueMemSetShadow(Context, DeviceInfo, Queue, AI->AllocBegin,
                                    AI->AllocSize, ShadowByte));
        return UR_RESULT_SUCCESS;
    }

    // Init zero
    UR_CALL(enqueueMemSetShadow(Context, DeviceInfo, Queue, AI->AllocBegin,
                                AI->AllocSize, 0));

    uptr TailBegin = RoundUpTo(AI->UserEnd, ASAN_SHADOW_GRANULARITY);
    uptr TailEnd = AI->AllocBegin + AI->AllocSize;

    // User tail
    if (TailBegin != AI->UserEnd) {
        auto Value =
            AI->UserEnd - RoundDownTo(AI->UserEnd, ASAN_SHADOW_GRANULARITY);
        UR_CALL(enqueueMemSetShadow(Context, DeviceInfo, Queue, AI->UserEnd, 1,
                                    static_cast<u8>(Value)));
    }

    int ShadowByte;
    switch (AI->Type) {
    case AllocType::HOST_USM:
        ShadowByte = kUsmHostRedzoneMagic;
        break;
    case AllocType::DEVICE_USM:
        ShadowByte = kUsmDeviceRedzoneMagic;
        break;
    case AllocType::SHARED_USM:
        ShadowByte = kUsmSharedRedzoneMagic;
        break;
    case AllocType::MEM_BUFFER:
        ShadowByte = kMemBufferRedzoneMagic;
        break;
    case AllocType::DEVICE_GLOBAL:
        ShadowByte = kDeviceGlobalRedzoneMagic;
        break;
    default:
        ShadowByte = 0xff;
        assert(false && "Unknow AllocInfo Type");
    }

    // Left red zone
    UR_CALL(enqueueMemSetShadow(Context, DeviceInfo, Queue, AI->AllocBegin,
                                AI->UserBegin - AI->AllocBegin, ShadowByte));

    // Right red zone
    UR_CALL(enqueueMemSetShadow(Context, DeviceInfo, Queue, TailBegin,
                                TailEnd - TailBegin, ShadowByte));

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::updateShadowMemory(
    std::shared_ptr<ContextInfo> &ContextInfo,
    std::shared_ptr<DeviceInfo> &DeviceInfo, ur_queue_handle_t Queue) {
    auto &AllocInfos = ContextInfo->AllocInfosMap[DeviceInfo->Handle];
    std::scoped_lock<ur_shared_mutex> Guard(AllocInfos.Mutex);

    for (auto &AI : AllocInfos.List) {
        UR_CALL(enqueueAllocInfo(ContextInfo->Handle, DeviceInfo, Queue, AI));
    }
    AllocInfos.List.clear();

    return UR_RESULT_SUCCESS;
}

ur_result_t
SanitizerInterceptor::registerDeviceGlobals(ur_context_handle_t Context,
                                            ur_program_handle_t Program) {
    std::vector<ur_device_handle_t> Devices = GetProgramDevices(Program);

    auto ContextInfo = getContextInfo(Context);

    for (auto Device : Devices) {
        ManagedQueue Queue(Context, Device);

        uint64_t NumOfDeviceGlobal;
        auto Result = context.urDdiTable.Enqueue.pfnDeviceGlobalVariableRead(
            Queue, Program, kSPIR_AsanDeviceGlobalCount, true,
            sizeof(NumOfDeviceGlobal), 0, &NumOfDeviceGlobal, 0, nullptr,
            nullptr);
        if (Result != UR_RESULT_SUCCESS) {
            context.logger.info("No device globals");
            continue;
        }

        std::vector<DeviceGlobalInfo> GVInfos(NumOfDeviceGlobal);
        Result = context.urDdiTable.Enqueue.pfnDeviceGlobalVariableRead(
            Queue, Program, kSPIR_AsanDeviceGlobalMetadata, true,
            sizeof(DeviceGlobalInfo) * NumOfDeviceGlobal, 0, &GVInfos[0], 0,
            nullptr, nullptr);
        if (Result != UR_RESULT_SUCCESS) {
            context.logger.error("Device Global[{}] Read Failed: {}",
                                 kSPIR_AsanDeviceGlobalMetadata, Result);
            return Result;
        }

        auto DeviceInfo = getDeviceInfo(Device);
        for (size_t i = 0; i < NumOfDeviceGlobal; i++) {
            auto AI = std::make_shared<AllocInfo>(
                AllocInfo{GVInfos[i].Addr,
                          GVInfos[i].Addr,
                          GVInfos[i].Addr + GVInfos[i].Size,
                          GVInfos[i].SizeWithRedZone,
                          AllocType::DEVICE_GLOBAL,
                          false,
                          Context,
                          Device,
                          GetCurrentBacktrace(),
                          {}});

            ContextInfo->insertAllocInfo({Device}, AI);
        }
    }

    return UR_RESULT_SUCCESS;
}

ur_result_t
SanitizerInterceptor::insertContext(ur_context_handle_t Context,
                                    std::shared_ptr<ContextInfo> &CI) {
    std::scoped_lock<ur_shared_mutex> Guard(m_ContextMapMutex);

    if (m_ContextMap.find(Context) != m_ContextMap.end()) {
        CI = m_ContextMap.at(Context);
        return UR_RESULT_SUCCESS;
    }

    CI = std::make_shared<ContextInfo>(Context);

    // Don't move CI, since it's a return value as well
    m_ContextMap.emplace(Context, CI);

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::eraseContext(ur_context_handle_t Context) {
    std::scoped_lock<ur_shared_mutex> Guard(m_ContextMapMutex);
    assert(m_ContextMap.find(Context) != m_ContextMap.end());
    m_ContextMap.erase(Context);
    // TODO: Remove devices in each context
    return UR_RESULT_SUCCESS;
}

ur_result_t
SanitizerInterceptor::insertDevice(ur_device_handle_t Device,
                                   std::shared_ptr<DeviceInfo> &DI) {
    std::scoped_lock<ur_shared_mutex> Guard(m_DeviceMapMutex);

    if (m_DeviceMap.find(Device) != m_DeviceMap.end()) {
        DI = m_DeviceMap.at(Device);
        return UR_RESULT_SUCCESS;
    }

    DI = std::make_shared<ur_sanitizer_layer::DeviceInfo>(Device);

    // Query device type
    DI->Type = GetDeviceType(Device);
    if (DI->Type == DeviceType::UNKNOWN) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    // Query alignment
    UR_CALL(context.urDdiTable.Device.pfnGetInfo(
        Device, UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN, sizeof(DI->Alignment),
        &DI->Alignment, nullptr));

    // Don't move DI, since it's a return value as well
    m_DeviceMap.emplace(Device, DI);

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::eraseDevice(ur_device_handle_t Device) {
    std::scoped_lock<ur_shared_mutex> Guard(m_DeviceMapMutex);
    assert(m_DeviceMap.find(Device) != m_DeviceMap.end());
    m_DeviceMap.erase(Device);
    // TODO: Remove devices in each context
    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::insertKernel(ur_kernel_handle_t Kernel) {
    std::scoped_lock<ur_shared_mutex> Guard(m_KernelMapMutex);
    if (m_KernelMap.find(Kernel) != m_KernelMap.end()) {
        return UR_RESULT_SUCCESS;
    }
    m_KernelMap.emplace(Kernel, std::make_shared<KernelInfo>(Kernel));
    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::eraseKernel(ur_kernel_handle_t Kernel) {
    std::scoped_lock<ur_shared_mutex> Guard(m_KernelMapMutex);
    assert(m_KernelMap.find(Kernel) != m_KernelMap.end());
    m_KernelMap.erase(Kernel);
    return UR_RESULT_SUCCESS;
}

ur_result_t
SanitizerInterceptor::insertMemBuffer(std::shared_ptr<MemBuffer> MemBuffer) {
    std::scoped_lock<ur_shared_mutex> Guard(m_MemBufferMapMutex);
    assert(m_MemBufferMap.find(ur_cast<ur_mem_handle_t>(MemBuffer.get())) ==
           m_MemBufferMap.end());
    m_MemBufferMap.emplace(reinterpret_cast<ur_mem_handle_t>(MemBuffer.get()),
                           MemBuffer);
    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::eraseMemBuffer(ur_mem_handle_t MemHandle) {
    std::scoped_lock<ur_shared_mutex> Guard(m_MemBufferMapMutex);
    assert(m_MemBufferMap.find(MemHandle) != m_MemBufferMap.end());
    m_MemBufferMap.erase(MemHandle);
    return UR_RESULT_SUCCESS;
}

std::shared_ptr<MemBuffer>
SanitizerInterceptor::getMemBuffer(ur_mem_handle_t MemHandle) {
    std::shared_lock<ur_shared_mutex> Guard(m_MemBufferMapMutex);
    if (m_MemBufferMap.find(MemHandle) != m_MemBufferMap.end()) {
        return m_MemBufferMap[MemHandle];
    }
    return nullptr;
}

ur_result_t SanitizerInterceptor::prepareLaunch(
    ur_context_handle_t Context, std::shared_ptr<DeviceInfo> &DeviceInfo,
    ur_queue_handle_t Queue, ur_kernel_handle_t Kernel,
    USMLaunchInfo &LaunchInfo) {
    auto Program = GetProgram(Kernel);

    do {
        // Set membuffer arguments
        auto KernelInfo = getKernelInfo(Kernel);
        for (const auto &[ArgIndex, MemBuffer] : KernelInfo->BufferArgs) {
            char *ArgPointer = nullptr;
            UR_CALL(MemBuffer->getHandle(DeviceInfo->Handle, ArgPointer));
            ur_result_t URes = context.urDdiTable.Kernel.pfnSetArgPointer(
                Kernel, ArgIndex, nullptr, ArgPointer);
            if (URes != UR_RESULT_SUCCESS) {
                context.logger.error(
                    "Failed to set buffer {} as the {} arg to kernel {}: {}",
                    ur_cast<ur_mem_handle_t>(MemBuffer.get()), ArgIndex, Kernel,
                    URes);
            }
        }

        // Set launch info argument
        auto ArgNums = GetKernelNumArgs(Kernel);
        if (ArgNums) {
            context.logger.debug(
                "launch_info {} (numLocalArgs={}, localArgs={})",
                (void *)LaunchInfo.Data, LaunchInfo.Data->NumLocalArgs,
                (void *)LaunchInfo.Data->LocalArgs);
            ur_result_t URes = context.urDdiTable.Kernel.pfnSetArgPointer(
                Kernel, ArgNums - 1, nullptr, LaunchInfo.Data);
            if (URes != UR_RESULT_SUCCESS) {
                context.logger.error("Failed to set launch info: {}", URes);
                return URes;
            }
        }

        // Write global variable to program
        auto EnqueueWriteGlobal = [Queue, Program](const char *Name,
                                                   const void *Value,
                                                   size_t Size) {
            auto Result =
                context.urDdiTable.Enqueue.pfnDeviceGlobalVariableWrite(
                    Queue, Program, Name, false, Size, 0, Value, 0, nullptr,
                    nullptr);
            if (Result != UR_RESULT_SUCCESS) {
                context.logger.warning(
                    "Failed to write device global \"{}\": {}", Name, Result);
                return false;
            }
            return true;
        };

        // Write debug
        // We use "uint64_t" here because EnqueueWriteGlobal will fail when it's "uint32_t"
        uint64_t Debug = Options().Debug ? 1 : 0;
        EnqueueWriteGlobal(kSPIR_AsanDebug, &Debug, sizeof(Debug));

        // Write shadow memory offset for global memory
        EnqueueWriteGlobal(kSPIR_AsanShadowMemoryGlobalStart,
                           &DeviceInfo->ShadowOffset,
                           sizeof(DeviceInfo->ShadowOffset));
        EnqueueWriteGlobal(kSPIR_AsanShadowMemoryGlobalEnd,
                           &DeviceInfo->ShadowOffsetEnd,
                           sizeof(DeviceInfo->ShadowOffsetEnd));

        // Write device type
        EnqueueWriteGlobal(kSPIR_DeviceType, &DeviceInfo->Type,
                           sizeof(DeviceInfo->Type));

        if (DeviceInfo->Type == DeviceType::CPU) {
            break;
        }

        if (LaunchInfo.LocalWorkSize.empty()) {
            LaunchInfo.LocalWorkSize.reserve(3);
            // FIXME: This is W/A until urKernelSuggestGroupSize is added
            LaunchInfo.LocalWorkSize[0] = 1;
            LaunchInfo.LocalWorkSize[1] = 1;
            LaunchInfo.LocalWorkSize[2] = 1;
        }

        const size_t *LocalWorkSize = LaunchInfo.LocalWorkSize.data();
        uint32_t NumWG = 1;
        for (uint32_t Dim = 0; Dim < LaunchInfo.WorkDim; ++Dim) {
            NumWG *= (LaunchInfo.GlobalWorkSize[Dim] + LocalWorkSize[Dim] - 1) /
                     LocalWorkSize[Dim];
        }

        auto EnqueueAllocateDevice = [Context, &DeviceInfo, Queue,
                                      NumWG](size_t Size, uptr &Ptr) {
            auto URes = context.urDdiTable.USM.pfnDeviceAlloc(
                Context, DeviceInfo->Handle, nullptr, nullptr, Size,
                (void **)&Ptr);
            if (URes != UR_RESULT_SUCCESS) {
                context.logger.error(
                    "Failed to allocate shadow memory for local memory: {}",
                    URes);
                context.logger.error(
                    "Maybe the number of workgroup ({}) too large", NumWG);
                return URes;
            }
            // Initialize shadow memory of local memory
            URes = urEnqueueUSMSet(Queue, (void *)Ptr, 0, Size);
            if (URes == UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) {
                context.logger.error(
                    "Failed to allocate shadow memory for local memory: {}",
                    URes);
                context.logger.error(
                    "Maybe the number of workgroup ({}) too large", NumWG);
                return URes;
            }
            return URes;
        };

        // Write shadow memory offset for local memory
        if (Options().DetectLocals) {
            // CPU needn't this
            if (DeviceInfo->Type == DeviceType::GPU_PVC) {
                size_t LocalMemorySize = GetLocalMemorySize(DeviceInfo->Handle);
                size_t LocalShadowMemorySize =
                    (NumWG * LocalMemorySize) >> ASAN_SHADOW_SCALE;

                context.logger.debug(
                    "LocalMemoryInfo(WorkGroup={}, LocalMemorySize={}, "
                    "LocalShadowMemorySize={})",
                    NumWG, LocalMemorySize, LocalShadowMemorySize);

                UR_CALL(EnqueueAllocateDevice(
                    LocalShadowMemorySize, LaunchInfo.Data->LocalShadowOffset));

                LaunchInfo.Data->LocalShadowOffsetEnd =
                    LaunchInfo.Data->LocalShadowOffset + LocalShadowMemorySize -
                    1;

                context.logger.info(
                    "ShadowMemory(Local, {} - {})",
                    (void *)LaunchInfo.Data->LocalShadowOffset,
                    (void *)LaunchInfo.Data->LocalShadowOffsetEnd);
            }
        }
    } while (false);

    return UR_RESULT_SUCCESS;
}

std::optional<AllocationIterator>
SanitizerInterceptor::findAllocInfoByAddress(uptr Address) {
    std::shared_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
    auto It = m_AllocationMap.upper_bound(Address);
    if (It == m_AllocationMap.begin()) {
        return std::optional<AllocationIterator>{};
    }
    --It;
    // Make sure we got the right AllocInfo
    assert(Address >= It->second->AllocBegin &&
           Address < It->second->AllocBegin + It->second->AllocSize &&
           "Wrong AllocInfo for the address");
    return It;
}

ur_result_t USMLaunchInfo::initialize() {
    UR_CALL(context.urDdiTable.Context.pfnRetain(Context));
    UR_CALL(context.urDdiTable.Device.pfnRetain(Device));
    UR_CALL(context.urDdiTable.USM.pfnSharedAlloc(
        Context, Device, nullptr, nullptr, sizeof(LaunchInfo), (void **)&Data));
    *Data = LaunchInfo{};
    return UR_RESULT_SUCCESS;
}

ur_result_t USMLaunchInfo::updateKernelInfo(const KernelInfo &KI) {
    auto NumArgs = KI.LocalArgs.size();
    if (NumArgs) {
        Data->NumLocalArgs = NumArgs;
        UR_CALL(context.urDdiTable.USM.pfnSharedAlloc(
            Context, Device, nullptr, nullptr, sizeof(LocalArgsInfo) * NumArgs,
            (void **)&Data->LocalArgs));
        uint32_t i = 0;
        for (auto [ArgIndex, ArgInfo] : KI.LocalArgs) {
            Data->LocalArgs[i++] = ArgInfo;
            context.logger.debug(
                "local_args (argIndex={}, size={}, sizeWithRZ={})", ArgIndex,
                ArgInfo.Size, ArgInfo.SizeWithRedZone);
        }
    }
    return UR_RESULT_SUCCESS;
}

USMLaunchInfo::~USMLaunchInfo() {
    [[maybe_unused]] ur_result_t Result;
    if (Data) {
        auto Type = GetDeviceType(Device);
        if (Type == DeviceType::GPU_PVC) {
            if (Data->PrivateShadowOffset) {
                Result = context.urDdiTable.USM.pfnFree(
                    Context, (void *)Data->PrivateShadowOffset);
                assert(Result == UR_RESULT_SUCCESS);
            }
            if (Data->LocalShadowOffset) {
                Result = context.urDdiTable.USM.pfnFree(
                    Context, (void *)Data->LocalShadowOffset);
                assert(Result == UR_RESULT_SUCCESS);
            }
        }
        if (Data->LocalArgs) {
            Result = context.urDdiTable.USM.pfnFree(Context,
                                                    (void *)Data->LocalArgs);
            assert(Result == UR_RESULT_SUCCESS);
        }
        Result = context.urDdiTable.USM.pfnFree(Context, (void *)Data);
        assert(Result == UR_RESULT_SUCCESS);
    }
    Result = context.urDdiTable.Context.pfnRelease(Context);
    assert(Result == UR_RESULT_SUCCESS);
    Result = context.urDdiTable.Device.pfnRelease(Device);
    assert(Result == UR_RESULT_SUCCESS);
}

} // namespace ur_sanitizer_layer
