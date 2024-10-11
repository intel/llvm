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
#include "asan_validator.hpp"
#include "stacktrace.hpp"
#include "ur_sanitizer_utils.hpp"

namespace ur_sanitizer_layer {

namespace {

uptr MemToShadow_CPU(uptr USM_SHADOW_BASE, uptr UPtr) {
    return USM_SHADOW_BASE + (UPtr >> ASAN_SHADOW_SCALE);
}

uptr MemToShadow_DG2(uptr USM_SHADOW_BASE, uptr UPtr) {
    if (UPtr & 0xFFFF000000000000ULL) { // Device USM
        return USM_SHADOW_BASE + 0x80000000000ULL +
               ((UPtr & 0x7FFFFFFFFFFFULL) >> ASAN_SHADOW_SCALE);
    } else { // Host/Shared USM
        return USM_SHADOW_BASE + (UPtr >> ASAN_SHADOW_SCALE);
    }
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
    return getContext()->urDdiTable.Enqueue.pfnUSMFill(
        Queue, Ptr, 1, &Value, Size, NumEvents, EventWaitList, OutEvent);
}

ur_result_t enqueueMemSetShadow(std::shared_ptr<ContextInfo> &ContextInfo,
                                std::shared_ptr<DeviceInfo> &DeviceInfo,
                                ur_queue_handle_t Queue, uptr Ptr, uptr Size,
                                u8 Value) {
    if (Size == 0) {
        return UR_RESULT_SUCCESS;
    }
    if (DeviceInfo->Type == DeviceType::CPU) {
        ///
        /// CPU Device: CPU needs to use a special memset function
        ///
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
        getContext()->logger.debug(
            "enqueueMemSetShadow(addr={}, count={}, value={})",
            (void *)ShadowBegin, ShadowEnd - ShadowBegin + 1,
            (void *)(size_t)Value);
        MemSet((void *)ShadowBegin, Value, ShadowEnd - ShadowBegin + 1);
    } else {
        ///
        /// GPU Device: GPU needs to manually map physical memory before memset
        ///
        uptr ShadowBegin = 0, ShadowEnd = 0;

        if (DeviceInfo->Type == DeviceType::GPU_PVC) {
            ShadowBegin = MemToShadow_PVC(DeviceInfo->ShadowOffset, Ptr);
            ShadowEnd =
                MemToShadow_PVC(DeviceInfo->ShadowOffset, Ptr + Size - 1);
        } else if (DeviceInfo->Type == DeviceType::GPU_DG2) {
            ShadowBegin = MemToShadow_DG2(DeviceInfo->ShadowOffset, Ptr);
            ShadowEnd =
                MemToShadow_DG2(DeviceInfo->ShadowOffset, Ptr + Size - 1);
        } else {
            getContext()->logger.error("Unsupport device type");
            return UR_RESULT_ERROR_INVALID_ARGUMENT;
        }

        assert(ShadowBegin <= ShadowEnd);

        {
            static const size_t PageSize = GetVirtualMemGranularity(
                ContextInfo->Handle, DeviceInfo->Handle);

            ur_physical_mem_properties_t Desc{
                UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES, nullptr, 0};
            static ur_physical_mem_handle_t PhysicalMem{};

            // Make sure [Ptr, Ptr + Size] is mapped to physical memory
            for (auto MappedPtr = RoundDownTo(ShadowBegin, PageSize);
                 MappedPtr <= ShadowEnd; MappedPtr += PageSize) {
                if (!PhysicalMem) {
                    auto URes = getContext()->urDdiTable.PhysicalMem.pfnCreate(
                        ContextInfo->Handle, DeviceInfo->Handle, PageSize,
                        &Desc, &PhysicalMem);
                    if (URes != UR_RESULT_SUCCESS) {
                        getContext()->logger.error("urPhysicalMemCreate(): {}",
                                                   URes);
                        return URes;
                    }
                }

                getContext()->logger.debug("urVirtualMemMap: {} ~ {}",
                                           (void *)MappedPtr,
                                           (void *)(MappedPtr + PageSize - 1));

                // FIXME: No flag to check the failed reason is VA is already mapped
                auto URes = getContext()->urDdiTable.VirtualMem.pfnMap(
                    ContextInfo->Handle, (void *)MappedPtr, PageSize,
                    PhysicalMem, 0, UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE);
                if (URes != UR_RESULT_SUCCESS) {
                    getContext()->logger.debug("urVirtualMemMap({}, {}): {}",
                                               (void *)MappedPtr, PageSize,
                                               URes);
                }

                // Initialize to zero
                if (URes == UR_RESULT_SUCCESS) {
                    // Reset PhysicalMem to null since it's been mapped
                    PhysicalMem = nullptr;
                    ContextInfo->Stats.UpdateShadowMmaped(PageSize);

                    auto URes =
                        urEnqueueUSMSet(Queue, (void *)MappedPtr, 0, PageSize);
                    if (URes != UR_RESULT_SUCCESS) {
                        getContext()->logger.error("urEnqueueUSMFill(): {}",
                                                   URes);
                        return URes;
                    }
                }
            }
        }

        auto URes = urEnqueueUSMSet(Queue, (void *)ShadowBegin, Value,
                                    ShadowEnd - ShadowBegin + 1);
        getContext()->logger.debug(
            "enqueueMemSetShadow (addr={}, count={}, value={}): {}",
            (void *)ShadowBegin, ShadowEnd - ShadowBegin + 1,
            (void *)(size_t)Value, URes);
        if (URes != UR_RESULT_SUCCESS) {
            getContext()->logger.error("urEnqueueUSMFill(): {}", URes);
            return URes;
        }
    }
    return UR_RESULT_SUCCESS;
}

} // namespace

SanitizerInterceptor::SanitizerInterceptor() {
    if (getOptions().MaxQuarantineSizeMB) {
        m_Quarantine = std::make_unique<Quarantine>(
            static_cast<uint64_t>(getOptions().MaxQuarantineSizeMB) * 1024 *
            1024);
    }
}

SanitizerInterceptor::~SanitizerInterceptor() {
    DestroyShadowMemoryOnCPU();
    DestroyShadowMemoryOnPVC();
    DestroyShadowMemoryOnDG2();

    // We must release these objects before releasing adapters, since
    // they may use the adapter in their destructor
    m_Quarantine = nullptr;
    m_MemBufferMap.clear();
    m_AllocationMap.clear();
    m_KernelMap.clear();
    m_ContextMap.clear();

    for (auto Adapter : m_Adapters) {
        getContext()->urDdiTable.Global.pfnAdapterRelease(Adapter);
    }
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

    uptr RZLog =
        ComputeRZLog(Size, getOptions().MinRZSize, getOptions().MaxRZSize);
    uptr RZSize = RZLog2Size(RZLog);
    uptr RoundedSize = RoundUpTo(Size, Alignment);
    uptr NeededSize = RoundedSize + RZSize * 2;
    if (Alignment > MinAlignment) {
        NeededSize += Alignment;
    }

    void *Allocated = nullptr;

    if (Type == AllocType::DEVICE_USM) {
        UR_CALL(getContext()->urDdiTable.USM.pfnDeviceAlloc(
            Context, Device, Properties, Pool, NeededSize, &Allocated));
    } else if (Type == AllocType::HOST_USM) {
        UR_CALL(getContext()->urDdiTable.USM.pfnHostAlloc(
            Context, Properties, Pool, NeededSize, &Allocated));
    } else if (Type == AllocType::SHARED_USM) {
        UR_CALL(getContext()->urDdiTable.USM.pfnSharedAlloc(
            Context, Device, Properties, Pool, NeededSize, &Allocated));
    } else if (Type == AllocType::MEM_BUFFER) {
        UR_CALL(getContext()->urDdiTable.USM.pfnDeviceAlloc(
            Context, Device, Properties, Pool, NeededSize, &Allocated));
    } else {
        getContext()->logger.error("Unsupport memory type");
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    // Udpate statistics
    ContextInfo->Stats.UpdateUSMMalloced(NeededSize, NeededSize - Size);

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
    // NOTE: AllocInfoIt will be erased later, so "AllocInfo" must be a new reference here
    auto AllocInfo = AllocInfoIt->second;

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
        getContext()->logger.debug("Free: {}", (void *)AllocInfo->AllocBegin);

        ContextInfo->Stats.UpdateUSMRealFreed(AllocInfo->AllocSize,
                                              AllocInfo->getRedzoneSize());

        std::scoped_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
        m_AllocationMap.erase(AllocInfoIt);

        return getContext()->urDdiTable.USM.pfnFree(
            Context, (void *)(AllocInfo->AllocBegin));
    }

    // If quarantine is enabled, cache it
    auto ReleaseList = m_Quarantine->put(AllocInfo->Device, AllocInfoIt);
    if (ReleaseList.size()) {
        std::scoped_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
        for (auto &It : ReleaseList) {
            getContext()->logger.info("Quarantine Free: {}",
                                      (void *)It->second->AllocBegin);

            ContextInfo->Stats.UpdateUSMRealFreed(AllocInfo->AllocSize,
                                                  AllocInfo->getRedzoneSize());

            m_AllocationMap.erase(It);

            UR_CALL(getContext()->urDdiTable.USM.pfnFree(
                Context, (void *)(It->second->AllocBegin)));
        }
    }
    ContextInfo->Stats.UpdateUSMFreed(AllocInfo->AllocSize);

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
        getContext()->logger.error("Failed to create internal queue");
        return UR_RESULT_ERROR_INVALID_QUEUE;
    }

    UR_CALL(prepareLaunch(ContextInfo, DeviceInfo, InternalQueue, Kernel,
                          LaunchInfo));

    UR_CALL(updateShadowMemory(ContextInfo, DeviceInfo, InternalQueue));

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::postLaunchKernel(ur_kernel_handle_t Kernel,
                                                   ur_queue_handle_t Queue,
                                                   USMLaunchInfo &LaunchInfo) {
    // FIXME: We must use block operation here, until we support urEventSetCallback
    auto Result = getContext()->urDdiTable.Queue.pfnFinish(Queue);

    if (Result == UR_RESULT_SUCCESS) {
        for (const auto &AH : LaunchInfo.Data->SanitizerReport) {
            if (!AH.Flag) {
                continue;
            }
            switch (AH.ErrorType) {
            case DeviceSanitizerErrorType::USE_AFTER_FREE:
                ReportUseAfterFree(AH, Kernel, GetContext(Queue));
                break;
            case DeviceSanitizerErrorType::OUT_OF_BOUNDS:
            case DeviceSanitizerErrorType::MISALIGNED:
            case DeviceSanitizerErrorType::NULL_POINTER:
                ReportGenericError(AH, Kernel);
                break;
            default:
                ReportFatalError(AH);
            }
            if (!AH.IsRecover) {
                exit(1);
            }
        }
    }

    return Result;
}

ur_result_t DeviceInfo::allocShadowMemory(ur_context_handle_t Context) {
    if (Type == DeviceType::CPU) {
        UR_CALL(SetupShadowMemoryOnCPU(ShadowOffset, ShadowOffsetEnd));
    } else if (Type == DeviceType::GPU_PVC) {
        UR_CALL(SetupShadowMemoryOnPVC(Context, ShadowOffset, ShadowOffsetEnd));
    } else if (Type == DeviceType::GPU_DG2) {
        UR_CALL(SetupShadowMemoryOnDG2(Context, ShadowOffset, ShadowOffsetEnd));
    } else {
        getContext()->logger.error("Unsupport device type");
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }
    getContext()->logger.info("ShadowMemory(Global): {} - {}",
                              (void *)ShadowOffset, (void *)ShadowOffsetEnd);

    // Set shadow memory for null pointer
    ManagedQueue Queue(Context, Handle);

    auto CI = getContext()->interceptor->getContextInfo(Context);
    auto DI = getContext()->interceptor->getDeviceInfo(Handle);
    auto URes =
        enqueueMemSetShadow(CI, DI, Queue, 0, 1, kNullPointerRedzoneMagic);
    if (URes != UR_RESULT_SUCCESS) {
        getContext()->logger.error("enqueueMemSetShadow(NullPointerRZ): {}",
                                   URes);
        return URes;
    }
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
    std::shared_ptr<ContextInfo> &ContextInfo,
    std::shared_ptr<DeviceInfo> &DeviceInfo, ur_queue_handle_t Queue,
    std::shared_ptr<AllocInfo> &AI) {
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
        UR_CALL(enqueueMemSetShadow(ContextInfo, DeviceInfo, Queue,
                                    AI->AllocBegin, AI->AllocSize, ShadowByte));
        return UR_RESULT_SUCCESS;
    }

    // Init zero
    UR_CALL(enqueueMemSetShadow(ContextInfo, DeviceInfo, Queue, AI->AllocBegin,
                                AI->AllocSize, 0));

    uptr TailBegin = RoundUpTo(AI->UserEnd, ASAN_SHADOW_GRANULARITY);
    uptr TailEnd = AI->AllocBegin + AI->AllocSize;

    // User tail
    if (TailBegin != AI->UserEnd) {
        auto Value =
            AI->UserEnd - RoundDownTo(AI->UserEnd, ASAN_SHADOW_GRANULARITY);
        UR_CALL(enqueueMemSetShadow(ContextInfo, DeviceInfo, Queue, AI->UserEnd,
                                    1, static_cast<u8>(Value)));
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
    UR_CALL(enqueueMemSetShadow(ContextInfo, DeviceInfo, Queue, AI->AllocBegin,
                                AI->UserBegin - AI->AllocBegin, ShadowByte));

    // Right red zone
    UR_CALL(enqueueMemSetShadow(ContextInfo, DeviceInfo, Queue, TailBegin,
                                TailEnd - TailBegin, ShadowByte));

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::updateShadowMemory(
    std::shared_ptr<ContextInfo> &ContextInfo,
    std::shared_ptr<DeviceInfo> &DeviceInfo, ur_queue_handle_t Queue) {
    auto &AllocInfos = ContextInfo->AllocInfosMap[DeviceInfo->Handle];
    std::scoped_lock<ur_shared_mutex> Guard(AllocInfos.Mutex);

    for (auto &AI : AllocInfos.List) {
        UR_CALL(enqueueAllocInfo(ContextInfo, DeviceInfo, Queue, AI));
    }
    AllocInfos.List.clear();

    return UR_RESULT_SUCCESS;
}

ur_result_t
SanitizerInterceptor::registerDeviceGlobals(ur_context_handle_t Context,
                                            ur_program_handle_t Program) {
    std::vector<ur_device_handle_t> Devices = GetDevices(Program);

    auto ContextInfo = getContextInfo(Context);

    for (auto Device : Devices) {
        ManagedQueue Queue(Context, Device);

        uint64_t NumOfDeviceGlobal;
        auto Result =
            getContext()->urDdiTable.Enqueue.pfnDeviceGlobalVariableRead(
                Queue, Program, kSPIR_AsanDeviceGlobalCount, true,
                sizeof(NumOfDeviceGlobal), 0, &NumOfDeviceGlobal, 0, nullptr,
                nullptr);
        if (Result != UR_RESULT_SUCCESS) {
            getContext()->logger.info("No device globals");
            continue;
        }

        std::vector<DeviceGlobalInfo> GVInfos(NumOfDeviceGlobal);
        Result = getContext()->urDdiTable.Enqueue.pfnDeviceGlobalVariableRead(
            Queue, Program, kSPIR_AsanDeviceGlobalMetadata, true,
            sizeof(DeviceGlobalInfo) * NumOfDeviceGlobal, 0, &GVInfos[0], 0,
            nullptr, nullptr);
        if (Result != UR_RESULT_SUCCESS) {
            getContext()->logger.error("Device Global[{}] Read Failed: {}",
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

    DI->IsSupportSharedSystemUSM = GetDeviceUSMCapability(
        Device, UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT);

    // Query alignment
    UR_CALL(getContext()->urDdiTable.Device.pfnGetInfo(
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
    std::shared_ptr<ContextInfo> &ContextInfo,
    std::shared_ptr<DeviceInfo> &DeviceInfo, ur_queue_handle_t Queue,
    ur_kernel_handle_t Kernel, USMLaunchInfo &LaunchInfo) {
    auto Program = GetProgram(Kernel);

    do {
        auto KernelInfo = getKernelInfo(Kernel);

        // Validate pointer arguments
        if (getOptions().DetectKernelArguments) {
            for (const auto &[ArgIndex, PtrPair] : KernelInfo->PointerArgs) {
                auto Ptr = PtrPair.first;
                if (Ptr == nullptr) {
                    continue;
                }
                if (auto ValidateResult = ValidateUSMPointer(
                        ContextInfo->Handle, DeviceInfo->Handle, (uptr)Ptr)) {
                    ReportInvalidKernelArgument(Kernel, ArgIndex, (uptr)Ptr,
                                                ValidateResult, PtrPair.second);
                    exit(1);
                }
            }
        }

        // Set membuffer arguments
        for (const auto &[ArgIndex, MemBuffer] : KernelInfo->BufferArgs) {
            char *ArgPointer = nullptr;
            UR_CALL(MemBuffer->getHandle(DeviceInfo->Handle, ArgPointer));
            ur_result_t URes = getContext()->urDdiTable.Kernel.pfnSetArgPointer(
                Kernel, ArgIndex, nullptr, ArgPointer);
            if (URes != UR_RESULT_SUCCESS) {
                getContext()->logger.error(
                    "Failed to set buffer {} as the {} arg to kernel {}: {}",
                    ur_cast<ur_mem_handle_t>(MemBuffer.get()), ArgIndex, Kernel,
                    URes);
            }
        }

        // Set launch info argument
        auto ArgNums = GetKernelNumArgs(Kernel);
        if (ArgNums) {
            getContext()->logger.debug(
                "launch_info {} (numLocalArgs={}, localArgs={})",
                (void *)LaunchInfo.Data, LaunchInfo.Data->NumLocalArgs,
                (void *)LaunchInfo.Data->LocalArgs);
            ur_result_t URes = getContext()->urDdiTable.Kernel.pfnSetArgPointer(
                Kernel, ArgNums - 1, nullptr, LaunchInfo.Data);
            if (URes != UR_RESULT_SUCCESS) {
                getContext()->logger.error("Failed to set launch info: {}",
                                           URes);
                return URes;
            }
        }

        // Write global variable to program
        auto EnqueueWriteGlobal = [Queue, Program](
                                      const char *Name, const void *Value,
                                      size_t Size, bool ReportWarning = true) {
            auto Result =
                getContext()->urDdiTable.Enqueue.pfnDeviceGlobalVariableWrite(
                    Queue, Program, Name, false, Size, 0, Value, 0, nullptr,
                    nullptr);
            if (ReportWarning && Result != UR_RESULT_SUCCESS) {
                getContext()->logger.warning(
                    "Failed to write device global \"{}\": {}", Name, Result);
                return false;
            }
            return true;
        };

        // Write debug
        // We use "uint64_t" here because EnqueueWriteGlobal will fail when it's "uint32_t"
        // Because EnqueueWriteGlobal is a async write, so
        // we need to extend its lifetime
        static uint64_t Debug = getOptions().Debug ? 1 : 0;
        EnqueueWriteGlobal(kSPIR_AsanDebug, &Debug, sizeof(Debug), false);

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

        if (LaunchInfo.LocalWorkSize.empty()) {
            LaunchInfo.LocalWorkSize.resize(LaunchInfo.WorkDim);
            auto URes =
                getContext()->urDdiTable.Kernel.pfnGetSuggestedLocalWorkSize(
                    Kernel, Queue, LaunchInfo.WorkDim,
                    LaunchInfo.GlobalWorkOffset, LaunchInfo.GlobalWorkSize,
                    LaunchInfo.LocalWorkSize.data());
            if (URes != UR_RESULT_SUCCESS) {
                if (URes != UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
                    return URes;
                }
                // If urKernelGetSuggestedLocalWorkSize is not supported by driver, we fallback
                // to inefficient implementation
                for (size_t Dim = 0; Dim < LaunchInfo.WorkDim; ++Dim) {
                    LaunchInfo.LocalWorkSize[Dim] = 1;
                }
            }
        }

        const size_t *LocalWorkSize = LaunchInfo.LocalWorkSize.data();
        uint32_t NumWG = 1;
        for (uint32_t Dim = 0; Dim < LaunchInfo.WorkDim; ++Dim) {
            NumWG *= (LaunchInfo.GlobalWorkSize[Dim] + LocalWorkSize[Dim] - 1) /
                     LocalWorkSize[Dim];
        }

        auto EnqueueAllocateShadowMemory = [Context = ContextInfo->Handle,
                                            Device = DeviceInfo->Handle,
                                            Queue](size_t Size, uptr &Ptr) {
            void *Allocated = nullptr;
            auto URes = getContext()->urDdiTable.USM.pfnDeviceAlloc(
                Context, Device, nullptr, nullptr, Size, &Allocated);
            if (URes != UR_RESULT_SUCCESS) {
                return URes;
            }
            // Initialize shadow memory
            URes = urEnqueueUSMSet(Queue, Allocated, 0, Size);
            if (URes != UR_RESULT_SUCCESS) {
                [[maybe_unused]] auto URes =
                    getContext()->urDdiTable.USM.pfnFree(Context, Allocated);
                assert(URes == UR_RESULT_SUCCESS &&
                       "urUSMFree failed at allocating shadow memory");
                Allocated = nullptr;
            }
            Ptr = (uptr)Allocated;
            return URes;
        };

        auto LocalMemoryUsage =
            GetKernelLocalMemorySize(Kernel, DeviceInfo->Handle);
        auto PrivateMemoryUsage =
            GetKernelPrivateMemorySize(Kernel, DeviceInfo->Handle);

        getContext()->logger.info(
            "KernelInfo {} (LocalMemory={}, PrivateMemory={})", (void *)Kernel,
            LocalMemoryUsage, PrivateMemoryUsage);

        // Write shadow memory offset for local memory
        if (getOptions().DetectLocals) {
            // CPU needn't this
            if (DeviceInfo->Type == DeviceType::GPU_PVC ||
                DeviceInfo->Type == DeviceType::GPU_DG2) {
                const size_t LocalMemorySize =
                    GetDeviceLocalMemorySize(DeviceInfo->Handle);
                const size_t LocalShadowMemorySize =
                    (NumWG * LocalMemorySize) >> ASAN_SHADOW_SCALE;

                getContext()->logger.debug(
                    "LocalMemory(WorkGroup={}, LocalMemorySize={}, "
                    "LocalShadowMemorySize={})",
                    NumWG, LocalMemorySize, LocalShadowMemorySize);

                if (EnqueueAllocateShadowMemory(
                        LocalShadowMemorySize,
                        LaunchInfo.Data->LocalShadowOffset) !=
                    UR_RESULT_SUCCESS) {
                    getContext()->logger.warning(
                        "Failed to allocate shadow memory for local "
                        "memory, maybe the number of workgroup ({}) is too "
                        "large",
                        NumWG);
                    getContext()->logger.warning(
                        "Skip checking local memory of kernel <{}>",
                        GetKernelName(Kernel));
                } else {
                    LaunchInfo.Data->LocalShadowOffsetEnd =
                        LaunchInfo.Data->LocalShadowOffset +
                        LocalShadowMemorySize - 1;

                    ContextInfo->Stats.UpdateShadowMalloced(
                        LocalShadowMemorySize);

                    getContext()->logger.info(
                        "ShadowMemory(Local, {} - {})",
                        (void *)LaunchInfo.Data->LocalShadowOffset,
                        (void *)LaunchInfo.Data->LocalShadowOffsetEnd);
                }
            }
        }

        // Write shadow memory offset for private memory
        if (getOptions().DetectPrivates) {
            if (DeviceInfo->Type == DeviceType::CPU) {
                LaunchInfo.Data->PrivateShadowOffset = DeviceInfo->ShadowOffset;
            } else if (DeviceInfo->Type == DeviceType::GPU_PVC ||
                       DeviceInfo->Type == DeviceType::GPU_DG2) {
                const size_t PrivateShadowMemorySize =
                    (NumWG * ASAN_PRIVATE_SIZE) >> ASAN_SHADOW_SCALE;

                getContext()->logger.debug("PrivateMemory(WorkGroup={}, "
                                           "PrivateShadowMemorySize={})",
                                           NumWG, PrivateShadowMemorySize);

                if (EnqueueAllocateShadowMemory(
                        PrivateShadowMemorySize,
                        LaunchInfo.Data->PrivateShadowOffset) !=
                    UR_RESULT_SUCCESS) {
                    getContext()->logger.warning(
                        "Failed to allocate shadow memory for private "
                        "memory, maybe the number of workgroup ({}) is too "
                        "large",
                        NumWG);
                    getContext()->logger.warning(
                        "Skip checking private memory of kernel <{}>",
                        GetKernelName(Kernel));
                } else {
                    LaunchInfo.Data->PrivateShadowOffsetEnd =
                        LaunchInfo.Data->PrivateShadowOffset +
                        PrivateShadowMemorySize - 1;

                    ContextInfo->Stats.UpdateShadowMalloced(
                        PrivateShadowMemorySize);

                    getContext()->logger.info(
                        "ShadowMemory(Private, {} - {})",
                        (void *)LaunchInfo.Data->PrivateShadowOffset,
                        (void *)LaunchInfo.Data->PrivateShadowOffsetEnd);
                }
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

std::vector<AllocationIterator>
SanitizerInterceptor::findAllocInfoByContext(ur_context_handle_t Context) {
    std::shared_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
    std::vector<AllocationIterator> AllocInfos;
    for (auto It = m_AllocationMap.begin(); It != m_AllocationMap.end(); It++) {
        const auto &[_, AI] = *It;
        if (AI->Context == Context) {
            AllocInfos.emplace_back(It);
        }
    }
    return AllocInfos;
}

ContextInfo::~ContextInfo() {
    Stats.Print(Handle);

    [[maybe_unused]] auto Result =
        getContext()->urDdiTable.Context.pfnRelease(Handle);
    assert(Result == UR_RESULT_SUCCESS);

    // check memory leaks
    std::vector<AllocationIterator> AllocInfos =
        getContext()->interceptor->findAllocInfoByContext(Handle);
    for (const auto &It : AllocInfos) {
        const auto &[_, AI] = *It;
        if (!AI->IsReleased) {
            ReportMemoryLeak(AI);
        }
    }
}

ur_result_t USMLaunchInfo::initialize() {
    UR_CALL(getContext()->urDdiTable.Context.pfnRetain(Context));
    UR_CALL(getContext()->urDdiTable.Device.pfnRetain(Device));
    UR_CALL(getContext()->urDdiTable.USM.pfnSharedAlloc(
        Context, Device, nullptr, nullptr, sizeof(LaunchInfo), (void **)&Data));
    *Data = LaunchInfo{};
    return UR_RESULT_SUCCESS;
}

ur_result_t USMLaunchInfo::updateKernelInfo(const KernelInfo &KI) {
    auto NumArgs = KI.LocalArgs.size();
    if (NumArgs) {
        Data->NumLocalArgs = NumArgs;
        UR_CALL(getContext()->urDdiTable.USM.pfnSharedAlloc(
            Context, Device, nullptr, nullptr, sizeof(LocalArgsInfo) * NumArgs,
            (void **)&Data->LocalArgs));
        uint32_t i = 0;
        for (auto [ArgIndex, ArgInfo] : KI.LocalArgs) {
            Data->LocalArgs[i++] = ArgInfo;
            getContext()->logger.debug(
                "local_args (argIndex={}, size={}, sizeWithRZ={})", ArgIndex,
                ArgInfo.Size, ArgInfo.SizeWithRedZone);
        }
    }
    return UR_RESULT_SUCCESS;
}

USMLaunchInfo::~USMLaunchInfo() {
    [[maybe_unused]] ur_result_t Result;
    if (Data) {
        auto Type = GetDeviceType(Context, Device);
        auto ContextInfo = getContext()->interceptor->getContextInfo(Context);
        if (Type == DeviceType::GPU_PVC || Type == DeviceType::GPU_DG2) {
            if (Data->PrivateShadowOffset) {
                ContextInfo->Stats.UpdateShadowFreed(
                    Data->PrivateShadowOffsetEnd - Data->PrivateShadowOffset +
                    1);
                Result = getContext()->urDdiTable.USM.pfnFree(
                    Context, (void *)Data->PrivateShadowOffset);
                assert(Result == UR_RESULT_SUCCESS);
            }
            if (Data->LocalShadowOffset) {
                ContextInfo->Stats.UpdateShadowFreed(
                    Data->LocalShadowOffsetEnd - Data->LocalShadowOffset + 1);
                Result = getContext()->urDdiTable.USM.pfnFree(
                    Context, (void *)Data->LocalShadowOffset);
                assert(Result == UR_RESULT_SUCCESS);
            }
        }
        if (Data->LocalArgs) {
            Result = getContext()->urDdiTable.USM.pfnFree(
                Context, (void *)Data->LocalArgs);
            assert(Result == UR_RESULT_SUCCESS);
        }
        Result = getContext()->urDdiTable.USM.pfnFree(Context, (void *)Data);
        assert(Result == UR_RESULT_SUCCESS);
    }
    Result = getContext()->urDdiTable.Context.pfnRelease(Context);
    assert(Result == UR_RESULT_SUCCESS);
    Result = getContext()->urDdiTable.Device.pfnRelease(Device);
    assert(Result == UR_RESULT_SUCCESS);
}

} // namespace ur_sanitizer_layer
