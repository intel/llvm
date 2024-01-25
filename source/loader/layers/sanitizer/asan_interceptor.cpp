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
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {

namespace {

// These magic values are written to shadow for better error
// reporting.
constexpr int kUsmDeviceRedzoneMagic = (char)0x81;
constexpr int kUsmHostRedzoneMagic = (char)0x82;
constexpr int kUsmSharedRedzoneMagic = (char)0x83;
constexpr int kMemBufferRedzoneMagic = (char)0x84;

constexpr auto kSPIR_AsanShadowMemoryGlobalStart =
    "__AsanShadowMemoryGlobalStart";
constexpr auto kSPIR_AsanShadowMemoryGlobalEnd = "__AsanShadowMemoryGlobalEnd";
constexpr auto kSPIR_AsanShadowMemoryLocalStart =
    "__AsanShadowMemoryLocalStart";
constexpr auto kSPIR_AsanShadowMemoryLocalEnd = "__AsanShadowMemoryLocalEnd";

constexpr auto kSPIR_DeviceType = "__DeviceType";

constexpr auto kSPIR_DeviceSanitizerReportMem = "__DeviceSanitizerReportMem";

DeviceSanitizerReport SPIR_DeviceSanitizerReportMem;

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

ur_context_handle_t getContext(ur_queue_handle_t Queue) {
    ur_context_handle_t Context;
    [[maybe_unused]] auto Result = context.urDdiTable.Queue.pfnGetInfo(
        Queue, UR_QUEUE_INFO_CONTEXT, sizeof(ur_context_handle_t), &Context,
        nullptr);
    assert(Result == UR_RESULT_SUCCESS);
    return Context;
}

ur_device_handle_t getDevice(ur_queue_handle_t Queue) {
    ur_device_handle_t Device;
    [[maybe_unused]] auto Result = context.urDdiTable.Queue.pfnGetInfo(
        Queue, UR_QUEUE_INFO_DEVICE, sizeof(ur_device_handle_t), &Device,
        nullptr);
    assert(Result == UR_RESULT_SUCCESS);
    return Device;
}

ur_program_handle_t getProgram(ur_kernel_handle_t Kernel) {
    ur_program_handle_t Program;
    [[maybe_unused]] auto Result = context.urDdiTable.Kernel.pfnGetInfo(
        Kernel, UR_KERNEL_INFO_PROGRAM, sizeof(ur_program_handle_t), &Program,
        nullptr);
    assert(Result == UR_RESULT_SUCCESS);
    return Program;
}

size_t getLocalMemorySize(ur_device_handle_t Device) {
    size_t LocalMemorySize;
    [[maybe_unused]] auto Result = context.urDdiTable.Device.pfnGetInfo(
        Device, UR_DEVICE_INFO_LOCAL_MEM_SIZE, sizeof(LocalMemorySize),
        &LocalMemorySize, nullptr);
    assert(Result == UR_RESULT_SUCCESS);
    return LocalMemorySize;
}

std::string getKernelName(ur_kernel_handle_t Kernel) {
    size_t KernelNameSize = 0;
    [[maybe_unused]] auto Res = context.urDdiTable.Kernel.pfnGetInfo(
        Kernel, UR_KERNEL_INFO_FUNCTION_NAME, 0, nullptr, &KernelNameSize);
    assert(Res == UR_RESULT_SUCCESS);

    std::vector<char> KernelNameBuf(KernelNameSize);
    Res = context.urDdiTable.Kernel.pfnGetInfo(
        Kernel, UR_KERNEL_INFO_FUNCTION_NAME, KernelNameSize,
        KernelNameBuf.data(), nullptr);
    assert(Res == UR_RESULT_SUCCESS);

    return std::string(KernelNameBuf.data(), KernelNameSize - 1);
}

} // namespace

SanitizerInterceptor::SanitizerInterceptor()
    : m_IsInASanContext(IsInASanContext()),
      m_ShadowMemInited(m_IsInASanContext) {}

SanitizerInterceptor::~SanitizerInterceptor() {
    if (!m_IsInASanContext && m_ShadowMemInited && !DestroyShadowMem()) {
        context.logger.error("Failed to destroy shadow memory");
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
    void **ResultPtr, USMMemoryType Type) {
    auto Alignment = Properties->align;
    assert(Alignment == 0 || IsPowerOfTwo(Alignment));

    auto ContextInfo = getContextInfo(Context);
    std::shared_ptr<DeviceInfo> DeviceInfo;
    if (Device) {
        DeviceInfo = ContextInfo->getDeviceInfo(Device);
    }

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

    if (Type == USMMemoryType::DEVICE) {
        UR_CALL(context.urDdiTable.USM.pfnDeviceAlloc(
            Context, Device, Properties, Pool, NeededSize, &Allocated));
    } else if (Type == USMMemoryType::HOST) {
        UR_CALL(context.urDdiTable.USM.pfnHostAlloc(Context, Properties, Pool,
                                                    NeededSize, &Allocated));
    } else if (Type == USMMemoryType::SHARE) {
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

    auto AllocInfo = std::make_shared<USMAllocInfo>(
        USMAllocInfo{AllocBegin, UserBegin, UserEnd, NeededSize, Type});

    // For updating shadow memory
    if (DeviceInfo) { // device/shared USM
        std::scoped_lock<ur_shared_mutex> Guard(DeviceInfo->Mutex);
        DeviceInfo->AllocInfos.emplace_back(AllocInfo);
    } else { // host USM's AllocInfo needs to insert into all devices
        for (auto &pair : ContextInfo->DeviceMap) {
            auto DeviceInfo = pair.second;
            std::scoped_lock<ur_shared_mutex> Guard(DeviceInfo->Mutex);
            DeviceInfo->AllocInfos.emplace_back(AllocInfo);
        }
    }

    // For memory release
    {
        std::scoped_lock<ur_shared_mutex> Guard(ContextInfo->Mutex);
        ContextInfo->AllocatedUSMMap[AllocBegin] = AllocInfo;
    }

    context.logger.info(
        "AllocInfos(AllocBegin={},  User={}-{}, NeededSize={}, Type={})",
        (void *)AllocBegin, (void *)UserBegin, (void *)UserEnd, NeededSize,
        Type);

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::releaseMemory(ur_context_handle_t Context,
                                                void *Ptr) {
    auto ContextInfo = getContextInfo(Context);

    std::shared_lock<ur_shared_mutex> Guard(ContextInfo->Mutex);

    auto Addr = reinterpret_cast<uptr>(Ptr);
    // Find the last element is not greater than key
    auto AllocInfoIt = ContextInfo->AllocatedUSMMap.upper_bound((uptr)Addr);
    if (AllocInfoIt == ContextInfo->AllocatedUSMMap.begin()) {
        context.logger.error(
            "Can't find release pointer({}) in AllocatedAddressesMap", Ptr);
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }
    --AllocInfoIt;
    auto &AllocInfo = AllocInfoIt->second;

    context.logger.debug("USMAllocInfo(AllocBegin={}, UserBegin={})",
                         AllocInfo->AllocBegin, AllocInfo->UserBegin);

    if (Addr != AllocInfo->UserBegin) {
        context.logger.error("Releasing pointer({}) is not match to {}", Ptr,
                             AllocInfo->UserBegin);
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    // TODO: Update shadow memory
    return context.urDdiTable.USM.pfnFree(Context,
                                          (void *)AllocInfo->AllocBegin);
}

ur_result_t SanitizerInterceptor::preLaunchKernel(ur_kernel_handle_t Kernel,
                                                  ur_queue_handle_t Queue,
                                                  ur_event_handle_t &Event,
                                                  LaunchInfo &LaunchInfo,
                                                  uint32_t numWorkgroup) {
    UR_CALL(prepareLaunch(Queue, Kernel, LaunchInfo, numWorkgroup));

    UR_CALL(updateShadowMemory(Queue));

    // Return LastEvent in QueueInfo
    auto Context = getContext(Queue);
    auto ContextInfo = getContextInfo(Context);
    auto QueueInfo = ContextInfo->getQueueInfo(Queue);

    std::scoped_lock<ur_mutex> Guard(QueueInfo->Mutex);
    Event = QueueInfo->LastEvent;
    QueueInfo->LastEvent = nullptr;

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

        const char *File = AH->File[0] ? AH->File : "<unknown file>";
        const char *Func = AH->Func[0] ? AH->Func : "<unknown func>";
        auto KernelName = getKernelName(Kernel);

        context.logger.always("\n====ERROR: DeviceSanitizer: {} on {}",
                              DeviceSanitizerFormat(AH->ErrorType),
                              DeviceSanitizerFormat(AH->MemoryType));
        context.logger.always(
            "{} of size {} at kernel <{}> LID({}, {}, {}) GID({}, "
            "{}, {})",
            AH->IsWrite ? "WRITE" : "READ", AH->AccessSize, KernelName.c_str(),
            AH->LID0, AH->LID1, AH->LID2, AH->GID0, AH->GID1, AH->GID2);
        context.logger.always("  #0 {} {}:{}", Func, File, AH->Line);
        if (!AH->IsRecover) {
            exit(1);
        }
    }
}

ur_result_t SanitizerInterceptor::allocShadowMemory(
    ur_context_handle_t Context, std::shared_ptr<DeviceInfo> &DeviceInfo) {
    if (DeviceInfo->Type == DeviceType::CPU) {
        if (!m_IsInASanContext) {
            static std::once_flag OnceFlag;
            bool Result = true;
            std::call_once(OnceFlag, [&]() {
                Result = m_ShadowMemInited = SetupShadowMem();
            });

            if (!Result) {
                context.logger.error("Failed to allocate shadow memory");
                return UR_RESULT_ERROR_OUT_OF_RESOURCES;
            }
        }

        DeviceInfo->ShadowOffset = LOW_SHADOW_BEGIN;
        DeviceInfo->ShadowOffsetEnd = HIGH_SHADOW_END;
    } else if (DeviceInfo->Type == DeviceType::GPU_PVC) {
        /// SHADOW MEMORY MAPPING (PVC, with CPU 47bit)
        ///   Host/Shared USM : 0x0              ~ 0x0fff_ffff_ffff
        ///   ?               : 0x1000_0000_0000 ~ 0x1fff_ffff_ffff
        ///   Device USM      : 0x2000_0000_0000 ~ 0x3fff_ffff_ffff
        constexpr size_t SHADOW_SIZE = 1ULL << 46;
        // FIXME: Currently, Level-Zero doesn't create independent VAs for each contexts,
        // which will cause out-of-resource error when users use multiple contexts
        static uptr ShadowOffset, ShadowOffsetEnd;

        if (!ShadowOffset) {
            // TODO: Protect Bad Zone
            auto Result = context.urDdiTable.VirtualMem.pfnReserve(
                Context, nullptr, SHADOW_SIZE, (void **)&ShadowOffset);
            if (Result != UR_RESULT_SUCCESS) {
                context.logger.error(
                    "Failed to allocate shadow memory on PVC: {}", Result);
                return Result;
            }
            ShadowOffsetEnd = ShadowOffset + SHADOW_SIZE;
        }

        DeviceInfo->ShadowOffset = ShadowOffset;
        DeviceInfo->ShadowOffsetEnd = ShadowOffsetEnd;
    } else {
        context.logger.error("Unsupport device type");
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }
    context.logger.info("ShadowMemory(Global): {} - {}",
                        (void *)DeviceInfo->ShadowOffset,
                        (void *)DeviceInfo->ShadowOffsetEnd);
    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::enqueueMemSetShadow(
    ur_context_handle_t Context, ur_device_handle_t Device,
    ur_queue_handle_t Queue, uptr Ptr, uptr Size, u8 Value,
    ur_event_handle_t DepEvent, ur_event_handle_t *OutEvent) {

    uint32_t NumEventsInWaitList = DepEvent ? 1 : 0;
    const ur_event_handle_t *EventsWaitList = DepEvent ? &DepEvent : nullptr;
    ur_event_handle_t InternalEvent{};
    ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;

    auto ContextInfo = getContextInfo(Context);
    auto DeviceInfo = ContextInfo->getDeviceInfo(Device);

    if (DeviceInfo->Type == DeviceType::CPU) {
        uptr ShadowBegin = MemToShadow_CPU(DeviceInfo->ShadowOffset, Ptr);
        uptr ShadowEnd =
            MemToShadow_CPU(DeviceInfo->ShadowOffset, Ptr + Size - 1);

        const char Pattern[] = {(char)Value};
        auto URes = context.urDdiTable.Enqueue.pfnUSMFill(
            Queue, (void *)ShadowBegin, 1, Pattern, ShadowEnd - ShadowBegin + 1,
            NumEventsInWaitList, EventsWaitList, Event);
        context.logger.debug(
            "enqueueMemSetShadow (addr={}, count={}, value={}): {}",
            (void *)ShadowBegin, ShadowEnd - ShadowBegin + 1,
            (void *)(size_t)Value, URes);
        if (URes != UR_RESULT_SUCCESS) {
            context.logger.error("urEnqueueUSMFill(): {}", URes);
            return URes;
        }
    } else if (DeviceInfo->Type == DeviceType::GPU_PVC) {
        uptr ShadowBegin = MemToShadow_PVC(DeviceInfo->ShadowOffset, Ptr);
        uptr ShadowEnd =
            MemToShadow_PVC(DeviceInfo->ShadowOffset, Ptr + Size - 1);

        {
            static const size_t PageSize = [Context, Device]() {
                size_t Size;
                [[maybe_unused]] auto Result =
                    context.urDdiTable.VirtualMem.pfnGranularityGetInfo(
                        Context, Device,
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
                        Context, Device, PageSize, &Desc, &PhysicalMem);
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
                        Queue, (void *)MappedPtr, 1, Pattern, PageSize,
                        NumEventsInWaitList, EventsWaitList, Event);
                    if (URes != UR_RESULT_SUCCESS) {
                        context.logger.error("urEnqueueUSMFill(): {}", URes);
                        return URes;
                    }

                    NumEventsInWaitList = 1;
                    EventsWaitList = Event;
                }
            }
        }

        const char Pattern[] = {(char)Value};
        auto URes = context.urDdiTable.Enqueue.pfnUSMFill(
            Queue, (void *)ShadowBegin, 1, Pattern, ShadowEnd - ShadowBegin + 1,
            NumEventsInWaitList, EventsWaitList, Event);
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
    ur_context_handle_t Context, ur_device_handle_t Device,
    ur_queue_handle_t Queue, std::shared_ptr<USMAllocInfo> &AllocInfo,
    ur_event_handle_t &LastEvent) {
    // Init zero
    UR_CALL(enqueueMemSetShadow(Context, Device, Queue, AllocInfo->AllocBegin,
                                AllocInfo->AllocSize, 0, LastEvent,
                                &LastEvent));

    uptr TailBegin = RoundUpTo(AllocInfo->UserEnd, ASAN_SHADOW_GRANULARITY);
    uptr TailEnd = AllocInfo->AllocBegin + AllocInfo->AllocSize;

    // User tail
    if (TailBegin != AllocInfo->UserEnd) {
        auto Value = AllocInfo->UserEnd -
                     RoundDownTo(AllocInfo->UserEnd, ASAN_SHADOW_GRANULARITY);
        UR_CALL(enqueueMemSetShadow(Context, Device, Queue, AllocInfo->UserEnd,
                                    1, static_cast<u8>(Value), LastEvent,
                                    &LastEvent));
    }

    int ShadowByte;
    switch (AllocInfo->Type) {
    case USMMemoryType::HOST:
        ShadowByte = kUsmHostRedzoneMagic;
        break;
    case USMMemoryType::DEVICE:
        ShadowByte = kUsmDeviceRedzoneMagic;
        break;
    case USMMemoryType::SHARE:
        ShadowByte = kUsmSharedRedzoneMagic;
        break;
    case USMMemoryType::MEM_BUFFER:
        ShadowByte = kMemBufferRedzoneMagic;
        break;
    default:
        ShadowByte = 0xff;
        assert(false && "Unknow AllocInfo Type");
    }

    // Left red zone
    UR_CALL(enqueueMemSetShadow(Context, Device, Queue, AllocInfo->AllocBegin,
                                AllocInfo->UserBegin - AllocInfo->AllocBegin,
                                ShadowByte, LastEvent, &LastEvent));

    // Right red zone
    UR_CALL(enqueueMemSetShadow(Context, Device, Queue, TailBegin,
                                TailEnd - TailBegin, ShadowByte, LastEvent,
                                &LastEvent));

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::updateShadowMemory(ur_queue_handle_t Queue) {
    auto Context = getContext(Queue);
    auto Device = getDevice(Queue);
    assert(Device != nullptr);

    auto ContextInfo = getContextInfo(Context);

    auto DeviceInfo = ContextInfo->getDeviceInfo(Device);
    auto QueueInfo = ContextInfo->getQueueInfo(Queue);

    std::unique_lock<ur_shared_mutex> DeviceGuard(DeviceInfo->Mutex,
                                                  std::defer_lock);
    std::scoped_lock<std::unique_lock<ur_shared_mutex>, ur_mutex> Guard(
        DeviceGuard, QueueInfo->Mutex);

    ur_event_handle_t LastEvent = QueueInfo->LastEvent;

    for (auto &AllocInfo : DeviceInfo->AllocInfos) {
        UR_CALL(enqueueAllocInfo(Context, Device, Queue, AllocInfo, LastEvent));
    }
    DeviceInfo->AllocInfos.clear();

    QueueInfo->LastEvent = LastEvent;

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::insertContext(ur_context_handle_t Context) {
    auto ContextInfo = std::make_shared<ur_sanitizer_layer::ContextInfo>();

    std::scoped_lock<ur_shared_mutex> Guard(m_ContextMapMutex);
    assert(m_ContextMap.find(Context) == m_ContextMap.end());
    m_ContextMap.emplace(Context, std::move(ContextInfo));

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::eraseContext(ur_context_handle_t Context) {
    std::scoped_lock<ur_shared_mutex> Guard(m_ContextMapMutex);
    assert(m_ContextMap.find(Context) != m_ContextMap.end());
    m_ContextMap.erase(Context);
    // TODO: Remove devices in each context
    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::insertDevice(ur_context_handle_t Context,
                                               ur_device_handle_t Device) {
    auto DeviceInfo = std::make_shared<ur_sanitizer_layer::DeviceInfo>();

    // Query device type
    ur_device_type_t DeviceType;
    UR_CALL(context.urDdiTable.Device.pfnGetInfo(
        Device, UR_DEVICE_INFO_TYPE, sizeof(DeviceType), &DeviceType, nullptr));
    switch (DeviceType) {
    case UR_DEVICE_TYPE_CPU:
        DeviceInfo->Type = DeviceType::CPU;
        break;
    case UR_DEVICE_TYPE_GPU:
        DeviceInfo->Type = DeviceType::GPU_PVC;
        break;
    default:
        DeviceInfo->Type = DeviceType::UNKNOWN;
    }

    // Query alignment
    UR_CALL(context.urDdiTable.Device.pfnGetInfo(
        Device, UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN,
        sizeof(DeviceInfo->Alignment), &DeviceInfo->Alignment, nullptr));

    // Allocate shadow memory
    UR_CALL(allocShadowMemory(Context, DeviceInfo));

    auto ContextInfo = getContextInfo(Context);
    std::scoped_lock<ur_shared_mutex> Guard(ContextInfo->Mutex);
    ContextInfo->DeviceMap.emplace(Device, std::move(DeviceInfo));

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::insertQueue(ur_context_handle_t Context,
                                              ur_queue_handle_t Queue) {
    auto QueueInfo = std::make_shared<ur_sanitizer_layer::QueueInfo>();
    QueueInfo->LastEvent = nullptr;

    auto ContextInfo = getContextInfo(Context);
    std::scoped_lock<ur_shared_mutex> Guard(ContextInfo->Mutex);
    ContextInfo->QueueMap.emplace(Queue, std::move(QueueInfo));

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::eraseQueue(ur_context_handle_t Context,
                                             ur_queue_handle_t Queue) {
    auto ContextInfo = getContextInfo(Context);
    std::scoped_lock<ur_shared_mutex> Guard(ContextInfo->Mutex);
    assert(ContextInfo->QueueMap.find(Queue) != ContextInfo->QueueMap.end());
    ContextInfo->QueueMap.erase(Queue);
    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::prepareLaunch(ur_queue_handle_t Queue,
                                                ur_kernel_handle_t Kernel,
                                                LaunchInfo &LaunchInfo,
                                                uint32_t numWorkgroup) {
    auto Context = getContext(Queue);
    auto Device = getDevice(Queue);
    auto Program = getProgram(Kernel);

    LaunchInfo.Context = Context;

    auto ContextInfo = getContextInfo(Context);
    auto DeviceInfo = ContextInfo->getDeviceInfo(Device);
    auto QueueInfo = ContextInfo->getQueueInfo(Queue);

    std::scoped_lock<ur_mutex> Guard(QueueInfo->Mutex);
    ur_event_handle_t LastEvent = QueueInfo->LastEvent;

    do {
        // Set global variable to program
        auto EnqueueWriteGlobal = [&](const char *Name, const void *Value) {
            ur_event_handle_t NewEvent{};
            uint32_t NumEvents = LastEvent ? 1 : 0;
            const ur_event_handle_t *EventsList =
                LastEvent ? &LastEvent : nullptr;
            auto Result =
                context.urDdiTable.Enqueue.pfnDeviceGlobalVariableWrite(
                    Queue, Program, Name, false, sizeof(uptr), 0, Value,
                    NumEvents, EventsList, &NewEvent);
            if (Result != UR_RESULT_SUCCESS) {
                context.logger.warning("Device Global[{}] Write Failed: {}",
                                       Name, Result);
                return false;
            }
            LastEvent = NewEvent;
            return true;
        };

        // Write shadow memory offset for global memory
        EnqueueWriteGlobal(kSPIR_AsanShadowMemoryGlobalStart,
                           &DeviceInfo->ShadowOffset);
        EnqueueWriteGlobal(kSPIR_AsanShadowMemoryGlobalEnd,
                           &DeviceInfo->ShadowOffsetEnd);

        // Write device type
        EnqueueWriteGlobal(kSPIR_DeviceType, &DeviceInfo->Type);

        if (DeviceInfo->Type == DeviceType::CPU) {
            break;
        }

        // Write shadow memory offset for local memory
        auto LocalMemorySize = getLocalMemorySize(Device);
        auto LocalShadowMemorySize =
            (numWorkgroup * LocalMemorySize) >> ASAN_SHADOW_SCALE;

        context.logger.info("LocalInfo(WorkGroup={}, LocalMemorySize={}, "
                            "LocalShadowMemorySize={})",
                            numWorkgroup, LocalMemorySize,
                            LocalShadowMemorySize);

        ur_usm_desc_t Desc{UR_STRUCTURE_TYPE_USM_HOST_DESC, nullptr, 0, 0};
        auto Result = context.urDdiTable.USM.pfnDeviceAlloc(
            Context, Device, &Desc, nullptr, LocalShadowMemorySize,
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
                           &LaunchInfo.LocalShadowOffset);
        EnqueueWriteGlobal(kSPIR_AsanShadowMemoryLocalEnd,
                           &LaunchInfo.LocalShadowOffsetEnd);

        {
            ur_event_handle_t NewEvent{};
            uint32_t NumEvents = LastEvent ? 1 : 0;
            const ur_event_handle_t *EventsList =
                LastEvent ? &LastEvent : nullptr;
            const char Pattern[] = {0};

            auto URes = context.urDdiTable.Enqueue.pfnUSMFill(
                Queue, (void *)LaunchInfo.LocalShadowOffset, 1, Pattern,
                LocalShadowMemorySize, NumEvents, EventsList, &NewEvent);
            if (URes != UR_RESULT_SUCCESS) {
                context.logger.error("urEnqueueUSMFill(): {}", URes);
                return URes;
            }
            LastEvent = NewEvent;
        }

        context.logger.info("ShadowMemory(Local, {} - {})",
                            (void *)LaunchInfo.LocalShadowOffset,
                            (void *)LaunchInfo.LocalShadowOffsetEnd);
    } while (false);

    QueueInfo->LastEvent = LastEvent;
    return UR_RESULT_SUCCESS;
}

LaunchInfo::~LaunchInfo() {
    if (LocalShadowOffset) {
        [[maybe_unused]] auto Result =
            context.urDdiTable.USM.pfnFree(Context, (void *)LocalShadowOffset);
        assert(Result == UR_RESULT_SUCCESS);
    }
}

} // namespace ur_sanitizer_layer
