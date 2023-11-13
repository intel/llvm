//==---------- sanitizer_interceptor.cpp - Sanitizer interceptor -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sanitizer_interceptor.hpp"
#include "device_sanitizer_report.hpp"
#include "ur_sanitizer_layer.hpp"

#include <cstdint>
#include <cstring>
#include <utility>

namespace ur_sanitizer_layer {

namespace {

// These magic values are written to shadow for better error
// reporting.
const int kUsmDeviceRedzoneMagic = 0x81;
const int kUsmHostRedzoneMagic = 0x82;
const int kUsmSharedRedzoneMagic = 0x83;
const int kMemBufferRedzoneMagic = 0x84;

const int kUsmDeviceDeallocatedMagic = 0x91;
const int kUsmHostDeallocatedMagic = 0x92;
const int kUsmSharedDeallocatedMagic = 0x93;

// Same with Asan Stack
const int kPrivateLeftRedzoneMagic = 0xf1;
const int kPrivateMidRedzoneMagic = 0xf2;
const int kPrivateRightRedzoneMagic = 0xf3;

// These magic values are written to shadow for better error
// reporting.
// const int kAsanHeapLeftRedzoneMagic = 0xfa;
// const int kAsanHeapFreeMagic = 0xfd;
// const int kAsanStackLeftRedzoneMagic = 0xf1;
// const int kAsanStackMidRedzoneMagic = 0xf2;
// const int kAsanStackRightRedzoneMagic = 0xf3;
// const int kAsanStackAfterReturnMagic = 0xf5;
// const int kAsanInitializationOrderMagic = 0xf6;
// const int kAsanUserPoisonedMemoryMagic = 0xf7;
// const int kAsanContiguousContainerOOBMagic = 0xfc;
// const int kAsanStackUseAfterScopeMagic = 0xf8;
// const int kAsanGlobalRedzoneMagic = 0xf9;
// const int kAsanInternalHeapMagic = 0xfe;
// const int kAsanArrayCookieMagic = 0xac;
// const int kAsanIntraObjectRedzone = 0xbb;
// const int kAsanAllocaLeftMagic = 0xca;
// const int kAsanAllocaRightMagic = 0xcb;

const auto kSPIR_AsanShadowMemoryGlobalStart = "__AsanShadowMemoryGlobalStart";
const auto kSPIR_AsanShadowMemoryGlobalEnd = "__AsanShadowMemoryGlobalEnd";

const auto kSPIR_DeviceSanitizerReportMem = "__DeviceSanitizerReportMem";

DeviceSanitizerReport SPIR_DeviceSanitizerReportMem;

} // namespace

ur_result_t SanitizerInterceptor::allocateMemory(
    ur_context_handle_t Context, ur_device_handle_t Device,
    const ur_usm_desc_t *Properties, ur_usm_pool_handle_t Pool, size_t Size,
    void **ResultPtr, USMMemoryType Type) {
    auto Alignment = Properties->align;
    assert(Alignment == 0 || IsPowerOfTwo(Alignment));

    auto &ContextInfo = getContextInfo(Context);
    if (!ContextInfo.Init) {
        initContext(Context);
    }

    if (Device) {
        auto &DeviceInfo = ContextInfo.getDeviceInfo(Device);
        if (!DeviceInfo.Init) {
            initDevice(Context, Device);
        }
        if (Alignment == 0) {
            Alignment = DeviceInfo.Alignment;
        }
    }

    if (Alignment == 0) {
        // FIXME: OS Defined?
        Alignment = 8;
    }

    // Calcuate Size + RZSize
    uptr RZLog = ComputeRZLog(Size);
    uptr RZSize = RZLog2Size(RZLog);
    uptr RoundedSize = RoundUpTo(Size, Alignment);
    uptr NeededSize = RoundedSize + RZSize * 2;

    void *Allocated = nullptr;
    ur_result_t Result = UR_RESULT_SUCCESS;
    if (Type == USMMemoryType::DEVICE) {
        Result = m_Dditable.USM.pfnDeviceAlloc(Context, Device, Properties,
                                               Pool, NeededSize, &Allocated);
    } else if (Type == USMMemoryType::HOST) {
        Result = m_Dditable.USM.pfnHostAlloc(Context, Properties, Pool,
                                             NeededSize, &Allocated);
    } else if (Type == USMMemoryType::SHARE) {
        Result = m_Dditable.USM.pfnSharedAlloc(Context, Device, Properties,
                                               Pool, NeededSize, &Allocated);
    } else {
        assert(false && "SanitizerInterceptor::allocateMemory not implemented");
    }
    if (Result != UR_RESULT_SUCCESS) {
        return Result;
    }

    // Enqueue Shadow Memory Init
    uptr AllocBegin = reinterpret_cast<uptr>(Allocated);
    uptr AllocEnd = AllocBegin + NeededSize;
    uptr UserBegin = AllocBegin + RZSize;
    if (!IsAligned(UserBegin, Alignment)) {
        UserBegin = RoundUpTo(UserBegin, Alignment);
    }
    uptr UserEnd = UserBegin + Size;
    assert(UserEnd <= AllocEnd);

    *ResultPtr = reinterpret_cast<void *>(UserBegin);

    auto MemoryInfo =
        USMMemoryInfo{AllocBegin, UserBegin, UserEnd, NeededSize, Type};
    if (Device) {
        MemoryInfo.Devices.emplace(Device);
    }

    // Update Shadow Memory
    if (Device) {
        auto &DeviceInfo = ContextInfo.getDeviceInfo(Device);
        std::lock_guard<std::mutex> Guard(DeviceInfo.Mutex);
        DeviceInfo.AllocInfos.emplace_back(MemoryInfo);
    } else {
        std::lock_guard<std::mutex> Guard(ContextInfo.Mutex);
        ContextInfo.AllocHostInfos.emplace_back(MemoryInfo);
    }

    // Save into AllocatedAddressesMap for releasing
    {
        std::lock_guard<std::mutex> Guard(ContextInfo.Mutex);
        ContextInfo.AllocatedAddressesMap[AllocBegin] = MemoryInfo;
    }

    context.logger.info("AllocInfos:\n  AllocBegin: {}\n  User: {}-{}\n  "
                        "NeededSize: {}\nType: {}",
                        AllocBegin, UserBegin, UserEnd, NeededSize, Type);

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::releaseMemory(ur_context_handle_t Context,
                                                void *Ptr) {
    auto &ContextInfo = getContextInfo(Context);
    assert(ContextInfo.Init);

    context.logger.debug("ReleaseMemory: {}", Ptr);

    std::lock_guard<std::mutex> Guard(ContextInfo.Mutex);
    auto Addr = (uptr)Ptr;
    // Find the last element is not greater than key
    auto AddressInfoIt =
        ContextInfo.AllocatedAddressesMap.upper_bound((uptr)Addr);
    if (AddressInfoIt == ContextInfo.AllocatedAddressesMap.begin()) {
        context.logger.error(
            "Can't find release pointer({}) in AllocatedAddressesMap", Ptr);
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }
    --AddressInfoIt;
    auto &AddressInfo = AddressInfoIt->second;
    std::cerr << "AddressInfo: " << AddressInfo.AllocBegin << " "
              << AddressInfo.UserBegin << "\n";
    if (Addr != AddressInfo.UserBegin) {
        std::cerr << "ERROR: releaseMemory failed! UserBegin\n";
        context.logger.error("Release pointer({}) is not match to {}", Ptr,
                             AddressInfo.UserBegin);
        return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    // TODO: Update shadow memory
    return m_Dditable.USM.pfnFree(Context, (void *)AddressInfo.AllocBegin);
}

void SanitizerInterceptor::addQueue(ur_context_handle_t Context,
                                    ur_device_handle_t Device,
                                    ur_queue_handle_t Queue) {
    auto &QueueInfo = getQueueInfo(Queue);
    QueueInfo.Device = Device;
    QueueInfo.Context = Context;
}

void SanitizerInterceptor::addKernel(ur_program_handle_t Program,
                                     ur_kernel_handle_t Kernel) {
    auto &KernelInfo = getKernelInfo(Kernel);
    KernelInfo.Program = Program;
}

bool SanitizerInterceptor::launchKernel(ur_kernel_handle_t Kernel,
                                        ur_queue_handle_t Queue,
                                        ur_event_handle_t &Event) {
    // KernelInfo &KernelInfo = getKernelInfo(Kernel);
    initKernel(Queue, Kernel);

    updateShadowMemory(Queue, Kernel);

    auto &QueueInfo = getQueueInfo(Queue);
    std::lock_guard<std::mutex> Guard(QueueInfo.Mutex);
    Event = QueueInfo.LastEvent;
    QueueInfo.LastEvent = nullptr;
    return true;
}

static void checkSanitizerReport(const char *KernelName) {
    auto AH = &SPIR_DeviceSanitizerReportMem;
    if (!AH->Flag) {
        return;
    }

    const char *File = AH->File[0] ? AH->File : "<unknown file>";
    const char *Func = AH->Func[0] ? AH->Func : "<unknown func>";

    fprintf(stderr, "\n====ERROR: DeviceSanitizer: %s on %s\n\n",
            DeviceSanitizerFormat(AH->ErrorType),
            DeviceSanitizerFormat(AH->MemoryType));
    fprintf(stderr,
            "%s of size %u at kernel <%s> LID(%lu, %lu, %lu) GID(%lu, "
            "%lu, %lu)\n",
            AH->IsWrite ? "WRITE" : "READ", AH->AccessSize, KernelName,
            AH->LID0, AH->LID1, AH->LID2, AH->GID0, AH->GID1, AH->GID2);
    fprintf(stderr, "  #0 %s %s:%d\n", Func, File, AH->Line);
    fflush(stderr);
    if (!AH->IsRecover) {
        abort();
    }
}

void SanitizerInterceptor::postLaunchKernel(ur_kernel_handle_t Kernel,
                                            ur_queue_handle_t Queue,
                                            ur_event_handle_t *Event,
                                            bool SetCallback) {
    auto &KernelInfo = getKernelInfo(Kernel);
    auto Program = KernelInfo.Program;

    ur_event_handle_t ReadEvent{};

    // If kernel has defined SPIR_DeviceSanitizerReportMem, then we try to read it
    // to host, but it's okay that it isn't defined
    auto Ret = m_Dditable.Enqueue.pfnDeviceGlobalVariableRead(
        Queue, Program, kSPIR_DeviceSanitizerReportMem, true,
        sizeof(SPIR_DeviceSanitizerReportMem), 0,
        &SPIR_DeviceSanitizerReportMem, 1, Event, &ReadEvent);

    if (Ret == UR_RESULT_SUCCESS) {
        *Event = ReadEvent;

        auto AH = &SPIR_DeviceSanitizerReportMem;
        if (!AH->Flag) {
            return;
        }

        const char *File = AH->File[0] ? AH->File : "<unknown file>";
        const char *Func = AH->Func[0] ? AH->Func : "<unknown func>";

        fprintf(stderr, "\n====ERROR: DeviceSanitizer: %s on %s\n\n",
                DeviceSanitizerFormat(AH->ErrorType),
                DeviceSanitizerFormat(AH->MemoryType));
        fprintf(stderr,
                "%s of size %u at kernel <%s> LID(%lu, %lu, %lu) GID(%lu, "
                "%lu, %lu)\n",
                AH->IsWrite ? "WRITE" : "READ", AH->AccessSize,
                KernelInfo.Name.c_str(), AH->LID0, AH->LID1, AH->LID2, AH->GID0,
                AH->GID1, AH->GID2);
        fprintf(stderr, "  #0 %s %s:%d\n", Func, File, AH->Line);
        fflush(stderr);
        if (!AH->IsRecover) {
            abort();
        }
    }
}

std::string SanitizerInterceptor::getKernelName(ur_kernel_handle_t Kernel) {
    size_t KernelNameSize = 0;
    auto Res = m_Dditable.Kernel.pfnGetInfo(
        Kernel, UR_KERNEL_INFO_FUNCTION_NAME, 0, nullptr, &KernelNameSize);
    assert(Res == UR_RESULT_SUCCESS);

    std::vector<char> KernelNameBuf(KernelNameSize + 1);
    Res = m_Dditable.Kernel.pfnGetInfo(Kernel, UR_KERNEL_INFO_FUNCTION_NAME,
                                       KernelNameSize, KernelNameBuf.data(),
                                       nullptr);
    assert(Res == UR_RESULT_SUCCESS);
    KernelNameBuf[KernelNameSize] = '\0';

    return std::string(KernelNameBuf.data(), KernelNameSize);
}

void SanitizerInterceptor::checkSanitizerError(ur_kernel_handle_t Kernel) {
    std::string KernelName = getKernelName(Kernel);
    checkSanitizerReport(KernelName.c_str());
}

bool SanitizerInterceptor::updateHostShadowMemory(ur_context_handle_t Context,
                                                  USMMemoryInfo AllocInfo) {
    auto &ContextInfo = getContextInfo(Context);
    auto ShadowOffset = ContextInfo.HostShadowOffset;

    uptr tail_beg = RoundUpTo(AllocInfo.UserEnd, ASAN_SHADOW_GRANULARITY);
    uptr tail_end = AllocInfo.AllocBegin + AllocInfo.AllocSize;
    // user tail
    if (tail_beg != AllocInfo.UserEnd) {
        auto Value = AllocInfo.UserEnd -
                     RoundDownTo(AllocInfo.UserEnd, ASAN_SHADOW_GRANULARITY);
        auto ShadowPtr = (u8 *)MemToShadow(AllocInfo.UserEnd, ShadowOffset);
        *ShadowPtr = Value;
    }

    u8 ShadowByte;
    switch (AllocInfo.Type) {
    case USMMemoryType::DEVICE:
        ShadowByte = kUsmDeviceRedzoneMagic;
        break;
    case USMMemoryType::HOST:
        ShadowByte = kUsmHostRedzoneMagic;
        break;
    case USMMemoryType::SHARE:
        ShadowByte = kUsmSharedRedzoneMagic;
        break;
    case USMMemoryType::MEM_BUFFER:
        ShadowByte = kMemBufferRedzoneMagic;
        break;
    default:
        ShadowByte = 0xff;
        assert(false && "Unknow AllocInfo.Type");
    }

    std::memset((void *)MemToShadow(AllocInfo.AllocBegin, ShadowOffset),
                ShadowByte, AllocInfo.UserBegin - AllocInfo.AllocBegin);
    std::memset((void *)MemToShadow(tail_beg, ShadowOffset), ShadowByte,
                tail_end - tail_beg);
    return true;
}

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

uptr MemToShadow_DG2(uptr USM_SHADOW_BASE, uptr UPtr) {
    if (UPtr & (~0xFFFFFFFFFFFFULL)) { // Device USM
        return USM_SHADOW_BASE + ((UPtr & 0xFFFFFFFFFFFFULL) >> 3);
    } else {
        return USM_SHADOW_BASE + (UPtr >> 3);
    }
}

ur_result_t
SanitizerInterceptor::piextMemAllocShadow(ur_context_handle_t Context,
                                          ur_device_handle_t Device) {
    auto &ContextInfo = getContextInfo(Context);
    auto &DeviceInfo = ContextInfo.getDeviceInfo(Device);
    if (DeviceInfo.Type == UR_DEVICE_TYPE_CPU) {
        DeviceInfo.ShadowOffset = 0x00007fff7fffULL;
        DeviceInfo.ShadowOffsetEnd = 0x10007fff7fffULL;
    } else if (DeviceInfo.Type == UR_DEVICE_TYPE_GPU) {
        /// SHADOW MEMORY MAPPING (PVC, with CPU 47bit)
        ///   Host/Shared USM : 0x0              ~ 0x0fff_ffff_ffff
        ///   ?               : 0x1000_0000_0000 ~ 0x1fff_ffff_ffff
        ///   Device USM      : 0x2000_0000_0000 ~ 0x3fff_ffff_ffff
        constexpr size_t SHADOW_SIZE = 1ULL << 46;

        // TODO: Protect Bad Zone
        auto URes = m_Dditable.VirtualMem.pfnReserve(
            Context, nullptr, SHADOW_SIZE, (void **)&DeviceInfo.ShadowOffset);
        if (URes != UR_RESULT_SUCCESS) {
            context.logger.error("urVirtualMemReserve: {}",
                                 getUrResultString(URes));
            return URes;
        }

        DeviceInfo.ShadowOffsetEnd = DeviceInfo.ShadowOffset + SHADOW_SIZE;
    } else {
        assert(false && "Unsupport device type");
    }
    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::piextEnqueueMemSetShadow(
    ur_context_handle_t Context, ur_device_handle_t Device,
    ur_queue_handle_t Queue, void *Ptr, size_t Size, uint8_t Value,
    size_t NumEventsInWaitList, const ur_event_handle_t *EventsWaitList,
    ur_event_handle_t *OutEvent) {
    auto &ContextInfo = getContextInfo(Context);
    auto &DeviceInfo = ContextInfo.getDeviceInfo(Device);
    if (DeviceInfo.Type == UR_DEVICE_TYPE_CPU) {
        assert(false && "Unsupport device type");
    } else if (DeviceInfo.Type == UR_DEVICE_TYPE_GPU) {
        const uptr UPtr = (uptr)Ptr;

        ur_event_handle_t InternalEvent{};
        ur_event_handle_t *Event = OutEvent ? OutEvent : &InternalEvent;

        uptr ShadowBegin = MemToShadow_PVC(DeviceInfo.ShadowOffset, UPtr);
        uptr ShadowEnd =
            MemToShadow_PVC(DeviceInfo.ShadowOffset, UPtr + Size - 1);

        // Maybe in future, we needn't to map physical memory manually
        const bool IsNeedMapPhysicalMem = true;

        if (IsNeedMapPhysicalMem) {
            // We use fixed GPU PageSize: 64KB
            const size_t PageSize = 64 * 1024u;

            ur_physical_mem_properties_t Desc{
                UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES, nullptr, 0};
            static ur_physical_mem_handle_t PhysicalMem{};

            // Make sure [Ptr, Ptr + Size] is mapped to physical memory
            for (auto MappedPtr = RoundDownTo(ShadowBegin, PageSize);
                 MappedPtr <= ShadowEnd; MappedPtr += PageSize) {
                if (!PhysicalMem) {
                    auto URes = m_Dditable.PhysicalMem.pfnCreate(
                        Context, Device, PageSize, &Desc, &PhysicalMem);
                    if (URes != UR_RESULT_SUCCESS) {
                        context.logger.error("zePhysicalMemCreate(): {}",
                                             getUrResultString(URes));
                        return URes;
                    }
                }

                context.logger.debug("zeVirtualMemMap: {} ~ {}",
                                     (void *)MappedPtr,
                                     (void *)(MappedPtr + PageSize - 1));

                // FIXME: No flag to check the failed reason is VA is already mapped
                auto URes = m_Dditable.VirtualMem.pfnMap(
                    Context, (void *)MappedPtr, PageSize, PhysicalMem, 0,
                    UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE);
                if (URes != UR_RESULT_SUCCESS) {
                    context.logger.debug("    zeVirtualMemMap(): %s\n",
                                         getUrResultString(URes));
                }

                // Initialize to zero
                if (URes == UR_RESULT_SUCCESS) {
                    // Reset PhysicalMem to null since it's been mapped
                    PhysicalMem = nullptr;

                    // FIXME: Maybe we needn't to initialize shadow memory to zero? Or it'd be better to be a negative value?
                    const char Pattern[] = {0};

                    auto URes = m_Dditable.Enqueue.pfnUSMFill(
                        Queue, (void *)MappedPtr, 1, Pattern, PageSize,
                        NumEventsInWaitList, EventsWaitList, Event);
                    if (URes != UR_RESULT_SUCCESS) {
                        context.logger.error("urEnqueueUSMFill(): {}",
                                             getUrResultString(URes));
                        return URes;
                    }

                    NumEventsInWaitList = 1;
                    EventsWaitList = Event;
                }
            }
        }

        const char Pattern[] = {(char)Value};
        auto URes = m_Dditable.Enqueue.pfnUSMFill(
            Queue, (void *)ShadowBegin, 1, Pattern,
            (ShadowEnd - ShadowBegin + 1), NumEventsInWaitList, EventsWaitList,
            Event);
        if (URes != UR_RESULT_SUCCESS) {
            context.logger.error("urEnqueueUSMFill(): {}",
                                 getUrResultString(URes));
            return URes;
        }
    } else {
        assert(false && "Unsupport device type");
    }
    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::enqueuePoisonShadow(
    ur_context_handle_t Context, ur_device_handle_t Device,
    ur_queue_handle_t Queue, uptr Addr, uptr Size, u8 Value,
    ur_event_handle_t DepEvent, ur_event_handle_t *OutEvent) {
    uint32_t NumEvents = DepEvent ? 1 : 0;
    const ur_event_handle_t *EventsList = DepEvent ? &DepEvent : nullptr;
    return piextEnqueueMemSetShadow(Context, Device, Queue, (void *)Addr, Size,
                                    Value, NumEvents, EventsList, OutEvent);
}

void SanitizerInterceptor::enqueueAllocInfo(ur_context_handle_t Context,
                                            ur_device_handle_t Device,
                                            ur_queue_handle_t Queue,
                                            USMMemoryInfo &AllocInfo,
                                            ur_event_handle_t &LastEvent) {
    // Init zero
    auto Res =
        enqueuePoisonShadow(Context, Device, Queue, AllocInfo.AllocBegin,
                            AllocInfo.AllocSize, 0, LastEvent, &LastEvent);
    assert(Res == UR_RESULT_SUCCESS);

    uptr TailBegin = RoundUpTo(AllocInfo.UserEnd, ASAN_SHADOW_GRANULARITY);
    uptr TailEnd = AllocInfo.AllocBegin + AllocInfo.AllocSize;

    // User tail
    if (TailBegin != AllocInfo.UserEnd) {
        auto Value = AllocInfo.UserEnd -
                     RoundDownTo(AllocInfo.UserEnd, ASAN_SHADOW_GRANULARITY);
        auto Res =
            enqueuePoisonShadow(Context, Device, Queue, AllocInfo.UserEnd, 1,
                                Value, LastEvent, &LastEvent);
        assert(Res == UR_RESULT_SUCCESS);
    }

    int ShadowByte;
    switch (AllocInfo.Type) {
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
        assert(false && "Unknow AllocInfo.Type");
    }

    // Left red zone
    Res = enqueuePoisonShadow(Context, Device, Queue, AllocInfo.AllocBegin,
                              AllocInfo.UserBegin - AllocInfo.AllocBegin,
                              ShadowByte, LastEvent, &LastEvent);
    assert(Res == UR_RESULT_SUCCESS);

    // Right red zone
    Res = enqueuePoisonShadow(Context, Device, Queue, TailBegin,
                              TailEnd - TailBegin, ShadowByte, LastEvent,
                              &LastEvent);
    assert(Res == UR_RESULT_SUCCESS);
}

bool SanitizerInterceptor::updateShadowMemory(ur_queue_handle_t Queue,
                                              ur_kernel_handle_t Kernel) {
    (void)Kernel;
    auto &QueueInfo = getQueueInfo(Queue);
    auto Context = QueueInfo.Context;
    auto Device = QueueInfo.Device;

    auto &ContextInfo = getContextInfo(Context);
    assert(ContextInfo.Init);
    auto &DeviceInfo = ContextInfo.getDeviceInfo(Device);
    assert(DeviceInfo.Init);

    std::lock_guard<std::mutex> QueueGuard(QueueInfo.Mutex);
    std::lock_guard<std::mutex> DeviceGuard(DeviceInfo.Mutex);

    ur_event_handle_t LastEvent = QueueInfo.LastEvent;

    // FIXME: Always update host USM, but it'd be better to update host USM
    // selectively, or each devices once
    for (auto &AllocInfo : ContextInfo.AllocHostInfos) {
        if (AllocInfo.Devices.find(Device) == AllocInfo.Devices.end()) {
            enqueueAllocInfo(Context, Device, Queue, AllocInfo, LastEvent);
            AllocInfo.Devices.emplace(Device);
        }
    }

    for (auto &AllocInfo : DeviceInfo.AllocInfos) {
        enqueueAllocInfo(Context, Device, Queue, AllocInfo, LastEvent);
    }
    DeviceInfo.AllocInfos.clear();

    QueueInfo.LastEvent = LastEvent;

    return true;
}

void SanitizerInterceptor::initContext(ur_context_handle_t Context) {
    auto &ContextInfo = getContextInfo(Context);
    std::lock_guard<std::mutex> Guard(ContextInfo.Mutex);

    if (ContextInfo.Init) {
        return;
    }

    ContextInfo.Init = true;
}

void SanitizerInterceptor::initDevice(ur_context_handle_t Context,
                                      ur_device_handle_t Device) {
    auto &ContextInfo = getContextInfo(Context);
    assert(ContextInfo.Init && "Context not inited");

    assert(Device);
    auto &DeviceInfo = ContextInfo.getDeviceInfo(Device);
    std::lock_guard<std::mutex> Guard(DeviceInfo.Mutex);

    if (DeviceInfo.Init) {
        return;
    }

    // Query Device Type
    auto Result = m_Dditable.Device.pfnGetInfo(Device, UR_DEVICE_INFO_TYPE,
                                               sizeof(DeviceInfo.Type),
                                               &DeviceInfo.Type, nullptr);
    assert(Result == UR_RESULT_SUCCESS);

    // Query alignment
    Result = m_Dditable.Device.pfnGetInfo(
        Device, UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN,
        sizeof(DeviceInfo.Alignment), &DeviceInfo.Alignment, nullptr);
    assert(Result == UR_RESULT_SUCCESS);

    // Allocate shadow memory
    Result = piextMemAllocShadow(Context, Device);
    assert(Result == UR_RESULT_SUCCESS);

    context.logger.info("Device ShadowOffset: {} - {}",
                        (void *)DeviceInfo.ShadowOffset,
                        (void *)DeviceInfo.ShadowOffsetEnd);

    DeviceInfo.Init = true;
}

bool SanitizerInterceptor::initKernel(ur_queue_handle_t Queue,
                                      ur_kernel_handle_t Kernel) {
    auto &KernelInfo = getKernelInfo(Kernel);
    auto Program = KernelInfo.Program;

    auto &QueueInfo = getQueueInfo(Queue);
    auto Device = QueueInfo.Device;
    auto Context = QueueInfo.Context;

    auto &ContextInfo = getContextInfo(Context);
    if (!ContextInfo.Init) {
        initContext(Context);
    }

    auto &DeviceInfo = ContextInfo.getDeviceInfo(Device);
    if (!DeviceInfo.Init) {
        initDevice(Context, Device);
    }

    // Get kernel name
    {
        std::lock_guard<std::mutex> KernelGuard(KernelInfo.Mutex);
        if (KernelInfo.Name.empty()) {
            KernelInfo.Name = getKernelName(Kernel);
        }
    }

    std::lock_guard<std::mutex> QueueGuard(QueueInfo.Mutex);
    ur_event_handle_t LastEvent = QueueInfo.LastEvent;

    do {
        // Set global variable to program
        auto EnqueueWriteGlobal = [&](const char *Name, const void *Value) {
            ur_event_handle_t NewEvent{};
            uint32_t NumEvents = LastEvent ? 1 : 0;
            const ur_event_handle_t *EventsList =
                LastEvent ? &LastEvent : nullptr;
            auto Result = m_Dditable.Enqueue.pfnDeviceGlobalVariableWrite(
                Queue, Program, Name, false, sizeof(uptr), 0, Value, NumEvents,
                EventsList, &NewEvent);
            if (Result != UR_RESULT_SUCCESS) {
                context.logger.warning("Device Global Write Failed[{}]: {}",
                                       Name, getUrResultString(Result));
                return false;
            }
            LastEvent = NewEvent;
            return true;
        };

        // Device shadow memory offset
        EnqueueWriteGlobal(kSPIR_AsanShadowMemoryGlobalStart,
                           &DeviceInfo.ShadowOffset);
        EnqueueWriteGlobal(kSPIR_AsanShadowMemoryGlobalEnd,
                           &DeviceInfo.ShadowOffsetEnd);
    } while (false);

    QueueInfo.LastEvent = LastEvent;

    return true;
}

ur_result_t SanitizerInterceptor::createMemoryBuffer(
    ur_context_handle_t Context, ur_mem_flags_t Flags, size_t Size,
    const ur_buffer_properties_t *Properties, ur_mem_handle_t *Buffer) {
    auto &ContextInfo = getContextInfo(Context);
    if (!ContextInfo.Init) {
        initContext(Context);
    }

    constexpr size_t Alignment = 8;

    // Calcuate Size + RZSize
    const uptr RZLog = ComputeRZLog(Size);
    const uptr RZSize = RZLog2Size(RZLog);
    const uptr RoundedSize = RoundUpTo(Size, Alignment);
    const uptr NeededSize = RoundedSize + RZSize * 2;

    auto Result = m_Dditable.Mem.pfnBufferCreate(Context, Flags, NeededSize,
                                                 Properties, Buffer);
    if (Result != UR_RESULT_SUCCESS) {
        return Result;
    }

    // Get Native Handle
    ur_native_handle_t NativeHandle{};
    Result = m_Dditable.Mem.pfnGetNativeHandle(*Buffer, &NativeHandle);
    if (Result != UR_RESULT_SUCCESS) {
        return Result;
    }
    void *Allocated = (void *)NativeHandle;

    ur_device_handle_t Device;
    Result = m_Dditable.USM.pfnGetMemAllocInfo(Context, Allocated, UR_USM_ALLOC_INFO_DEVICE, sizeof(Device), &Device, nullptr);

    uptr AllocBegin = reinterpret_cast<uptr>(Allocated);
    uptr AllocEnd = AllocBegin + NeededSize;
    uptr UserBegin = AllocBegin + RZSize;
    if (!IsAligned(UserBegin, Alignment)) {
        UserBegin = RoundUpTo(UserBegin, Alignment);
    }
    uptr UserEnd = UserBegin + Size;
    assert(UserEnd <= AllocEnd);

    auto MemoryInfo = USMMemoryInfo{AllocBegin, UserBegin, UserEnd, NeededSize,
                                    USMMemoryType::MEM_BUFFER};
    MemoryInfo.Devices.emplace(Device);

    // Update Shadow Memory
    auto &DeviceInfo = ContextInfo.getDeviceInfo(Device);
    std::lock_guard<std::mutex> Guard(DeviceInfo.Mutex);
    DeviceInfo.AllocInfos.emplace_back(MemoryInfo);

    // Save into AllocatedAddressesMap for releasing
    {
        std::lock_guard<std::mutex> Guard(ContextInfo.Mutex);
        ContextInfo.AllocatedAddressesMap[AllocBegin] = MemoryInfo;
    }

    // Create a subbuffer which is surrounded by red zone
    ur_buffer_region_t BufferRegion{UR_STRUCTURE_TYPE_BUFFER_REGION, nullptr, UserBegin - AllocBegin, Size};
    ur_mem_handle_t SubBuffer;
    Result = m_Dditable.Mem.pfnBufferPartition(*Buffer, Flags,
                                               UR_BUFFER_CREATE_TYPE_REGION,
                                               &BufferRegion, &SubBuffer);
    if (Result != UR_RESULT_SUCCESS) {
        return Result;
    }

    *Buffer = SubBuffer;
    return UR_RESULT_SUCCESS;
}

} // namespace ur_sanitizer_layer
