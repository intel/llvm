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
#include "asan_quarantine.hpp"
#include "asan_report.hpp"
#include "asan_shadow_setup.hpp"
#include "device_sanitizer_report.hpp"
#include "stacktrace.hpp"
#include "ur_sanitizer_utils.hpp"

namespace ur_sanitizer_layer {

namespace {

// These magic values are written to shadow for better error
// reporting.
constexpr int kUsmDeviceRedzoneMagic = (char)0x81;
constexpr int kUsmHostRedzoneMagic = (char)0x82;
constexpr int kUsmSharedRedzoneMagic = (char)0x83;
constexpr int kMemBufferRedzoneMagic = (char)0x84;
constexpr int kDeviceGlobalRedZoneMagic = (char)0x85;

const int kUsmDeviceDeallocatedMagic = (char)0x91;
const int kUsmHostDeallocatedMagic = (char)0x92;
const int kUsmSharedDeallocatedMagic = (char)0x93;
const int kMemBufferDeallocatedMagic = (char)0x93;

constexpr auto kSPIR_AsanShadowMemoryGlobalStart =
    "__AsanShadowMemoryGlobalStart";
constexpr auto kSPIR_AsanShadowMemoryGlobalEnd = "__AsanShadowMemoryGlobalEnd";
constexpr auto kSPIR_AsanShadowMemoryLocalStart =
    "__AsanShadowMemoryLocalStart";
constexpr auto kSPIR_AsanShadowMemoryLocalEnd = "__AsanShadowMemoryLocalEnd";

constexpr auto kSPIR_DeviceType = "__DeviceType";
constexpr auto kSPIR_AsanDebug = "__AsanDebug";

constexpr auto kSPIR_DeviceSanitizerReportMem = "__DeviceSanitizerReportMem";

constexpr auto kSPIR_AsanDeviceGlobalCount = "__AsanDeviceGlobalCount";
constexpr auto kSPIR_AsanDeviceGlobalMetadata = "__AsanDeviceGlobalMetadata";

struct ManagedQueue {
    ManagedQueue(ur_context_handle_t Context, ur_device_handle_t Device) {
        [[maybe_unused]] auto Result = context.urDdiTable.Queue.pfnCreate(
            Context, Device, nullptr, &Handle);
        assert(Result == UR_RESULT_SUCCESS);
    }

    ~ManagedQueue() {
        [[maybe_unused]] auto Result =
            context.urDdiTable.Queue.pfnRelease(Handle);
        assert(Result == UR_RESULT_SUCCESS);
    }

    operator ur_queue_handle_t() { return Handle; }

  private:
    ur_queue_handle_t Handle = nullptr;
};

uptr MemToShadow_CPU(uptr USM_SHADOW_BASE, uptr UPtr) {
    return USM_SHADOW_BASE + (UPtr >> 3);
}

uptr MemToShadow_PVC(uptr USM_SHADOW_BASE, uptr UPtr) {
    if (UPtr & 0xFF00000000000000ULL) { // Device USM
        return USM_SHADOW_BASE + 0x200000000000ULL +
               ((UPtr & 0xFFFFFFFFFFFFULL) >> 3);
    } else { // Only consider 47bit VA
        return USM_SHADOW_BASE + ((UPtr & 0x7FFFFFFFFFFFULL) >> 3);
    }
}

} // namespace

SanitizerInterceptor::SanitizerInterceptor() {
    auto Options = getenv_to_map("UR_LAYER_ASAN_OPTIONS");
    if (!Options.has_value()) {
        return;
    }
    auto KV = Options->find("debug");
    if (KV != Options->end()) {
        auto Value = KV->second.front();
        if (Value == "1" || Value == "true") {
            cl_Debug = 1;
        }
    }
    KV = Options->find("quarantine_size_mb");
    if (KV != Options->end()) {
        auto Value = KV->second.front();
        cl_MaxQuarantineSizeMB = std::stoul(Value);
    }
    if (cl_MaxQuarantineSizeMB) {
        m_Quarantine = std::make_unique<Quarantine>(cl_MaxQuarantineSizeMB);
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
    void **ResultPtr, AllocType Type) {
    auto Alignment = Properties->align;
    assert(Alignment == 0 || IsPowerOfTwo(Alignment));

    auto ContextInfo = getContextInfo(Context);
    std::shared_ptr<DeviceInfo> DeviceInfo =
        Device ? getDeviceInfo(Device) : nullptr;

    if (Alignment == 0) {
        Alignment =
            DeviceInfo ? DeviceInfo->Alignment : ASAN_SHADOW_GRANULARITY;
    }

    // Copy from LLVM compiler-rt/lib/asan
    uptr RZLog = ComputeRZLog(Size);
    uptr RZSize = RZLog2Size(RZLog);
    uptr RoundedSize = RoundUpTo(Size, Alignment);
    uptr NeededSize = RoundedSize + RZSize * 2;

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
    } else {
        context.logger.error("Unsupport memory type");
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    // Copy from LLVM compiler-rt/lib/asan
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

    // For updating shadow memory
    if (Device) { // Device/Shared USM
        ContextInfo->insertAllocInfo({Device}, AI);
    } else { // Host USM
        ContextInfo->insertAllocInfo(ContextInfo->DeviceList, AI);
    }

    // For memory release
    {
        std::scoped_lock<ur_shared_mutex> Guard(m_AllocationsMapMutex);
        m_AllocationsMap.emplace(std::move(AI));
    }

    context.logger.info(
        "AllocInfos(AllocBegin={},  User={}-{}, NeededSize={}, Type={})",
        (void *)AllocBegin, (void *)UserBegin, (void *)UserEnd, NeededSize,
        ToString(Type));

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::releaseMemory(ur_context_handle_t Context,
                                                void *Ptr) {
    auto ContextInfo = getContextInfo(Context);

    auto Addr = reinterpret_cast<uptr>(Ptr);
    auto AllocInfos = findAllocInfoByAddress(Addr);

    if (AllocInfos.empty()) {
        ReportBadFree(Addr, GetCurrentBacktrace(), nullptr);
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    std::vector<std::shared_ptr<AllocInfo>> CurrentContext;

    for (auto It = AllocInfos.begin(); It != AllocInfos.end();) {
        if (It->get()->Context == Context) {
            CurrentContext.emplace_back(*It);
            It = AllocInfos.erase(It);
            continue;
        }
        ++It;
    }

    if (CurrentContext.empty()) {
        // bad context
        ReportBadFreeContext(Addr, GetCurrentBacktrace(), AllocInfos);
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    for (auto &AllocInfo : CurrentContext) {
        context.logger.debug("AllocInfo(AllocBegin={}, UserBegin={})",
                             (void *)AllocInfo->AllocBegin,
                             (void *)AllocInfo->UserBegin);

        if (AllocInfo->IsReleased) {
            ReportDoubleFree(Addr, GetCurrentBacktrace(), AllocInfo);
            return UR_RESULT_ERROR_INVALID_ARGUMENT;
        }

        if (Addr != AllocInfo->UserBegin) {
            ReportBadFree(Addr, GetCurrentBacktrace(), AllocInfo);
            return UR_RESULT_ERROR_INVALID_ARGUMENT;
        }

        AllocInfo->IsReleased = true;
        AllocInfo->ReleaseStack = GetCurrentBacktrace();

        // auto Device =
        //     getUSMAllocDevice(Context, (const void *)AllocInfo->AllocBegin);
        auto Device = AllocInfo->Device;
        // TODO: Check Device

        // TODO: Quarantine Cache
        if (m_Quarantine) {
            auto ReleaseList = m_Quarantine->put(Device, AllocInfo);
            for (auto &AI : ReleaseList) {
            }
        }

        // auto Res =
        //     context.urDdiTable.USM.pfnFree(Context, (void *)AllocInfo->AllocBegin);
        // if (Res != UR_RESULT_SUCCESS) {
        //     return Res;
        // }

        if (AllocInfo->Type == AllocType::HOST_USM) {
            ContextInfo->insertAllocInfo(ContextInfo->DeviceList, AllocInfo);
        } else {
            ContextInfo->insertAllocInfo({Device}, AllocInfo);
        }
    }

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::preLaunchKernel(ur_kernel_handle_t Kernel,
                                                  ur_queue_handle_t Queue,
                                                  LaunchInfo &LaunchInfo,
                                                  uint32_t numWorkgroup) {
    auto Context = getContext(Queue);
    auto Device = getDevice(Queue);
    auto ContextInfo = getContextInfo(Context);
    auto DeviceInfo = getDeviceInfo(Device);

    ManagedQueue InternalQueue(Context, Device);
    if (!InternalQueue) {
        return UR_RESULT_ERROR_INVALID_QUEUE;
    }

    UR_CALL(prepareLaunch(Context, DeviceInfo, InternalQueue, Kernel,
                          LaunchInfo, numWorkgroup));

    UR_CALL(updateShadowMemory(ContextInfo, DeviceInfo, InternalQueue));

    UR_CALL(context.urDdiTable.Queue.pfnFinish(InternalQueue));

    return UR_RESULT_SUCCESS;
}

void SanitizerInterceptor::postLaunchKernel(ur_kernel_handle_t Kernel,
                                            ur_queue_handle_t Queue,
                                            ur_event_handle_t &Event,
                                            LaunchInfo &LaunchInfo) {
    auto Program = getProgram(Kernel);
    ur_event_handle_t ReadEvent{};

    // If kernel has defined SPIR_DeviceSanitizerReportMem, then we try to read it
    // to host, but it's okay that it isn't defined
    // FIXME: We must use block operation here, until we support urEventSetCallback
    auto Result = context.urDdiTable.Enqueue.pfnDeviceGlobalVariableRead(
        Queue, Program, kSPIR_DeviceSanitizerReportMem, true,
        sizeof(LaunchInfo.SPIR_DeviceSanitizerReportMem), 0,
        &LaunchInfo.SPIR_DeviceSanitizerReportMem, 1, &Event, &ReadEvent);

    if (Result == UR_RESULT_SUCCESS) {
        Event = ReadEvent;

        auto AH = &LaunchInfo.SPIR_DeviceSanitizerReportMem;
        if (!AH->Flag) {
            return;
        }
        ReportGenericError(*AH, Kernel, getContext(Queue), getDevice(Queue));
    }
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

ur_result_t SanitizerInterceptor::enqueueMemSetShadow(
    ur_context_handle_t Context, std::shared_ptr<DeviceInfo> &DeviceInfo,
    ur_queue_handle_t Queue, uptr Ptr, uptr Size, u8 Value) {

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

        MemSet((void *)ShadowBegin, Value, ShadowEnd - ShadowBegin + 1);
        context.logger.debug(
            "enqueueMemSetShadow (addr={}, count={}, value={})",
            (void *)ShadowBegin, ShadowEnd - ShadowBegin + 1,
            (void *)(size_t)Value);
    } else if (DeviceInfo->Type == DeviceType::GPU_PVC) {
        uptr ShadowBegin = MemToShadow_PVC(DeviceInfo->ShadowOffset, Ptr);
        uptr ShadowEnd =
            MemToShadow_PVC(DeviceInfo->ShadowOffset, Ptr + Size - 1);

        {
            static const size_t PageSize = [Context, &DeviceInfo]() {
                size_t Size;
                [[maybe_unused]] auto Result =
                    context.urDdiTable.VirtualMem.pfnGranularityGetInfo(
                        Context, DeviceInfo->Handle,
                        UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED,
                        sizeof(Size), &Size, nullptr);
                assert(Result == UR_RESULT_SUCCESS);
                context.logger.info("PVC PageSize: {}", Size);
                return Size;
            }();

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

                    const char Pattern[] = {0};

                    auto URes = context.urDdiTable.Enqueue.pfnUSMFill(
                        Queue, (void *)MappedPtr, 1, Pattern, PageSize, 0,
                        nullptr, nullptr);
                    if (URes != UR_RESULT_SUCCESS) {
                        context.logger.error("urEnqueueUSMFill(): {}", URes);
                        return URes;
                    }
                }
            }
        }

        const char Pattern[] = {(char)Value};
        auto URes = context.urDdiTable.Enqueue.pfnUSMFill(
            Queue, (void *)ShadowBegin, 1, Pattern, ShadowEnd - ShadowBegin + 1,
            0, nullptr, nullptr);
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
        ShadowByte = kDeviceGlobalRedZoneMagic;
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
    std::vector<ur_device_handle_t> Devices = getProgramDevices(Program);

    auto ContextInfo = getContextInfo(Context);

    for (auto Device : Devices) {
        ur_queue_handle_t Queue;
        ur_result_t Result = context.urDdiTable.Queue.pfnCreate(
            Context, Device, nullptr, &Queue);
        if (Result != UR_RESULT_SUCCESS) {
            context.logger.error("Failed to create command queue: {}", Result);
            return Result;
        }

        uint64_t NumOfDeviceGlobal;
        Result = context.urDdiTable.Enqueue.pfnDeviceGlobalVariableRead(
            Queue, Program, kSPIR_AsanDeviceGlobalCount, true,
            sizeof(NumOfDeviceGlobal), 0, &NumOfDeviceGlobal, 0, nullptr,
            nullptr);
        if (Result == UR_RESULT_ERROR_INVALID_ARGUMENT) {
            context.logger.info("No device globals");
            continue;
        } else if (Result != UR_RESULT_SUCCESS) {
            context.logger.error("Device Global[{}] Read Failed: {}",
                                 kSPIR_AsanDeviceGlobalCount, Result);
            return Result;
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
            auto AI = std::make_shared<AllocInfo>(AllocInfo{
                GVInfos[i].Addr, GVInfos[i].Addr,
                GVInfos[i].Addr + GVInfos[i].Size, GVInfos[i].SizeWithRedZone,
                AllocType::DEVICE_GLOBAL});

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
    DI->Type = getDeviceType(Device);
    if (DI->Type == DeviceType::UNKNOWN) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    // Query alignment
    UR_CALL(context.urDdiTable.Device.pfnGetInfo(
        Device, UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN, sizeof(DI->Alignment),
        &DI->Alignment, nullptr));

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

ur_result_t SanitizerInterceptor::prepareLaunch(
    ur_context_handle_t Context, std::shared_ptr<DeviceInfo> &DeviceInfo,
    ur_queue_handle_t Queue, ur_kernel_handle_t Kernel, LaunchInfo &LaunchInfo,
    uint32_t numWorkgroup) {
    auto Program = getProgram(Kernel);

    do {
        // Set global variable to program
        auto EnqueueWriteGlobal =
            [Queue, Program](const char *Name, const void *Value, size_t Size) {
                auto Result =
                    context.urDdiTable.Enqueue.pfnDeviceGlobalVariableWrite(
                        Queue, Program, Name, false, Size, 0, Value, 0, nullptr,
                        nullptr);
                if (Result != UR_RESULT_SUCCESS) {
                    context.logger.warning("Device Global[{}] Write Failed: {}",
                                           Name, Result);
                    return false;
                }
                return true;
            };

        // Write debug
        EnqueueWriteGlobal(kSPIR_AsanDebug, &cl_Debug, sizeof(cl_Debug));

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

        // Write shadow memory offset for local memory
        auto LocalMemorySize = getLocalMemorySize(DeviceInfo->Handle);
        auto LocalShadowMemorySize =
            (numWorkgroup * LocalMemorySize) >> ASAN_SHADOW_SCALE;

        context.logger.info("LocalInfo(WorkGroup={}, LocalMemorySize={}, "
                            "LocalShadowMemorySize={})",
                            numWorkgroup, LocalMemorySize,
                            LocalShadowMemorySize);

        ur_usm_desc_t Desc{UR_STRUCTURE_TYPE_USM_HOST_DESC, nullptr, 0, 0};
        auto Result = context.urDdiTable.USM.pfnDeviceAlloc(
            Context, DeviceInfo->Handle, &Desc, nullptr, LocalShadowMemorySize,
            (void **)&LaunchInfo.LocalShadowOffset);
        if (Result != UR_RESULT_SUCCESS) {
            context.logger.error(
                "Failed to allocate shadow memory for local memory: {}",
                numWorkgroup, Result);
            context.logger.error("Maybe the number of workgroup too large");
            return Result;
        }
        LaunchInfo.LocalShadowOffsetEnd =
            LaunchInfo.LocalShadowOffset + LocalShadowMemorySize - 1;

        EnqueueWriteGlobal(kSPIR_AsanShadowMemoryLocalStart,
                           &LaunchInfo.LocalShadowOffset,
                           sizeof(LaunchInfo.LocalShadowOffset));
        EnqueueWriteGlobal(kSPIR_AsanShadowMemoryLocalEnd,
                           &LaunchInfo.LocalShadowOffsetEnd,
                           sizeof(LaunchInfo.LocalShadowOffsetEnd));

        {
            const char Pattern[] = {0};

            auto URes = context.urDdiTable.Enqueue.pfnUSMFill(
                Queue, (void *)LaunchInfo.LocalShadowOffset, 1, Pattern,
                LocalShadowMemorySize, 0, nullptr, nullptr);
            if (URes != UR_RESULT_SUCCESS) {
                context.logger.error("urEnqueueUSMFill(): {}", URes);
                return URes;
            }
        }

        context.logger.info("ShadowMemory(Local, {} - {})",
                            (void *)LaunchInfo.LocalShadowOffset,
                            (void *)LaunchInfo.LocalShadowOffsetEnd);
    } while (false);

    return UR_RESULT_SUCCESS;
}

std::vector<std::shared_ptr<AllocInfo>>
SanitizerInterceptor::findAllocInfoByAddress(uptr Address) {
    std::vector<std::shared_ptr<AllocInfo>> Result;
    auto current = std::make_shared<AllocInfo>(AllocInfo{Address});

    std::shared_lock<ur_shared_mutex> Guard(m_AllocationsMapMutex);

    auto It = std::lower_bound(
        m_AllocationsMap.begin(), m_AllocationsMap.end(), Address,
        [](const std::shared_ptr<AllocInfo> &AllocInfo, uptr Addr) {
            return (AllocInfo->AllocBegin + AllocInfo->AllocSize) < Addr;
        });
    for (; It != m_AllocationsMap.end(); ++It) {
        auto AI = *It;
        if (AI->AllocBegin > Address) {
            break;
        }
        Result.emplace_back(*It);
    }
    return Result;
}

LaunchInfo::~LaunchInfo() {
    if (LocalShadowOffset) {
        [[maybe_unused]] auto Result =
            context.urDdiTable.USM.pfnFree(Context, (void *)LocalShadowOffset);
        assert(Result == UR_RESULT_SUCCESS);
    }
}

} // namespace ur_sanitizer_layer
