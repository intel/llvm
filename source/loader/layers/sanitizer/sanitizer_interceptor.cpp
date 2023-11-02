//==---------- sanitizer_interceptor.cpp - Sanitizer interceptor -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sanitizer_interceptor.hpp"
#include "device_sanitizer_report.hpp"

#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <utility>

#define ASAN_SHADOW_SCALE 3
#define ASAN_SHADOW_GRANULARITY (1ULL << ASAN_SHADOW_SCALE)

// These magic values are written to shadow for better error
// reporting.
const int kUsmDeviceRedzoneMagic = 0x81;
const int kUsmHostRedzoneMagic = 0x82;
const int kUsmSharedRedzoneMagic = 0x83;
const int kUsmDeviceDeallocatedMagic = 0x84;
const int kUsmHostDeallocatedMagic = 0x85;
const int kUsmSharedDeallocatedMagic = 0x86;

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

namespace {

DeviceSanitizerReport SPIR_DeviceSanitizerReportMem;

inline constexpr bool IsPowerOfTwo(uptr x) {
    return (x & (x - 1)) == 0 && x != 0;
}

inline constexpr uptr RoundUpTo(uptr Size, uptr boundary) {
    assert(IsPowerOfTwo(boundary));
    return (Size + boundary - 1) & ~(boundary - 1);
}

inline constexpr uptr RoundDownTo(uptr x, uptr boundary) {
    assert(IsPowerOfTwo(boundary));
    return x & ~(boundary - 1);
}

inline constexpr bool IsAligned(uptr a, uptr alignment) {
    return (a & (alignment - 1)) == 0;
}

/*
inline uptr LeastSignificantSetBitIndex(uptr x) {
  // CHECK_NE(x, 0U);
  unsigned long up;
#if !SANITIZER_WINDOWS || defined(__clang__) || defined(__GNUC__)
#ifdef _WIN64
  up = __builtin_ctzll(x);
#else
  up = __builtin_ctzl(x);
#endif
#elif defined(_WIN64)
  _BitScanForward64(&up, x);
#else
  _BitScanForward(&up, x);
#endif
  return up;
}

inline uptr Log2(uptr x) {
  // CHECK(IsPowerOfTwo(x));
  return LeastSignificantSetBitIndex(x);
}
*/

// Valid redzone sizes are 16, 32, 64, ... 2048, so we encode them in 3 bits.
// We use adaptive redzones: for larger allocation larger redzones are used.
static u32 RZLog2Size(u32 rz_log) {
    // CHECK_LT(rz_log, 8);
    return 16 << rz_log;
}

/*
static u32 RZSize2Log(u32 rz_size) {
  // CHECK_GE(rz_size, 16);
  // CHECK_LE(rz_size, 2048);
  // CHECK(IsPowerOfTwo(rz_size));
  u32 res = Log2(rz_size) - 4;
  // CHECK_EQ(rz_size, RZLog2Size(res));
  return res;
}
*/

uptr ComputeRZLog(uptr user_requested_size) {
    u32 rz_log = user_requested_size <= 64 - 16            ? 0
                 : user_requested_size <= 128 - 32         ? 1
                 : user_requested_size <= 512 - 64         ? 2
                 : user_requested_size <= 4096 - 128       ? 3
                 : user_requested_size <= (1 << 14) - 256  ? 4
                 : user_requested_size <= (1 << 15) - 512  ? 5
                 : user_requested_size <= (1 << 16) - 1024 ? 6
                                                           : 7;
    // u32 hdr_log = RZSize2Log(RoundUpToPowerOfTwo(sizeof(ChunkHeader)));
    // u32 min_log = RZSize2Log(atomic_load(&min_redzone, memory_order_acquire));
    // u32 max_log = RZSize2Log(atomic_load(&max_redzone, memory_order_acquire));
    // return Min(Max(rz_log, Max(min_log, hdr_log)), Max(max_log, hdr_log));
    return rz_log;
}

inline constexpr uptr MemToShadow(uptr Addr, uptr ShadowOffset) {
    return ShadowOffset + ((Addr) >> ASAN_SHADOW_SCALE);
}

static auto getUrResultString = [](ur_result_t Result) {
    switch (Result) {
    case UR_RESULT_SUCCESS:
        return "UR_RESULT_SUCCESS";
    case UR_RESULT_ERROR_INVALID_OPERATION:
        return "UR_RESULT_ERROR_INVALID_OPERATION";
    case UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES:
        return "UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES";
    case UR_RESULT_ERROR_INVALID_QUEUE:
        return "UR_RESULT_ERROR_INVALID_QUEUE";
    case UR_RESULT_ERROR_INVALID_VALUE:
        return "UR_RESULT_ERROR_INVALID_VALUE";
    case UR_RESULT_ERROR_INVALID_CONTEXT:
        return "UR_RESULT_ERROR_INVALID_CONTEXT";
    case UR_RESULT_ERROR_INVALID_PLATFORM:
        return "UR_RESULT_ERROR_INVALID_PLATFORM";
    case UR_RESULT_ERROR_INVALID_BINARY:
        return "UR_RESULT_ERROR_INVALID_BINARY";
    case UR_RESULT_ERROR_INVALID_PROGRAM:
        return "UR_RESULT_ERROR_INVALID_PROGRAM";
    case UR_RESULT_ERROR_INVALID_SAMPLER:
        return "UR_RESULT_ERROR_INVALID_SAMPLER";
    case UR_RESULT_ERROR_INVALID_BUFFER_SIZE:
        return "UR_RESULT_ERROR_INVALID_BUFFER_SIZE";
    case UR_RESULT_ERROR_INVALID_MEM_OBJECT:
        return "UR_RESULT_ERROR_INVALID_MEM_OBJECT";
    case UR_RESULT_ERROR_INVALID_EVENT:
        return "UR_RESULT_ERROR_INVALID_EVENT";
    case UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST:
        return "UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST";
    case UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET:
        return "UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET";
    case UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE:
        return "UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE";
    case UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE:
        return "UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE";
    case UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE:
        return "UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE";
    case UR_RESULT_ERROR_DEVICE_NOT_FOUND:
        return "UR_RESULT_ERROR_DEVICE_NOT_FOUND";
    case UR_RESULT_ERROR_INVALID_DEVICE:
        return "UR_RESULT_ERROR_INVALID_DEVICE";
    case UR_RESULT_ERROR_DEVICE_LOST:
        return "UR_RESULT_ERROR_DEVICE_LOST";
    case UR_RESULT_ERROR_DEVICE_REQUIRES_RESET:
        return "UR_RESULT_ERROR_DEVICE_REQUIRES_RESET";
    case UR_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
        return "UR_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE";
    case UR_RESULT_ERROR_DEVICE_PARTITION_FAILED:
        return "UR_RESULT_ERROR_DEVICE_PARTITION_FAILED";
    case UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT:
        return "UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT";
    case UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE:
        return "UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE";
    case UR_RESULT_ERROR_INVALID_WORK_DIMENSION:
        return "UR_RESULT_ERROR_INVALID_WORK_DIMENSION";
    case UR_RESULT_ERROR_INVALID_KERNEL_ARGS:
        return "UR_RESULT_ERROR_INVALID_KERNEL_ARGS";
    case UR_RESULT_ERROR_INVALID_KERNEL:
        return "UR_RESULT_ERROR_INVALID_KERNEL";
    case UR_RESULT_ERROR_INVALID_KERNEL_NAME:
        return "UR_RESULT_ERROR_INVALID_KERNEL_NAME";
    case UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
        return "UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
    case UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
        return "UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
    case UR_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
        return "UR_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
    case UR_RESULT_ERROR_INVALID_IMAGE_SIZE:
        return "UR_RESULT_ERROR_INVALID_IMAGE_SIZE";
    case UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED:
        return "UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED";
    case UR_RESULT_ERROR_MEM_OBJECT_ALLOCATION_FAILURE:
        return "UR_RESULT_ERROR_MEM_OBJECT_ALLOCATION_FAILURE";
    case UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE:
        return "UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE";
    case UR_RESULT_ERROR_UNINITIALIZED:
        return "UR_RESULT_ERROR_UNINITIALIZED";
    case UR_RESULT_ERROR_OUT_OF_HOST_MEMORY:
        return "UR_RESULT_ERROR_OUT_OF_HOST_MEMORY";
    case UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
        return "UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
    case UR_RESULT_ERROR_OUT_OF_RESOURCES:
        return "UR_RESULT_ERROR_OUT_OF_RESOURCES";
    case UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE:
        return "UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE";
    case UR_RESULT_ERROR_PROGRAM_LINK_FAILURE:
        return "UR_RESULT_ERROR_PROGRAM_LINK_FAILURE";
    case UR_RESULT_ERROR_UNSUPPORTED_VERSION:
        return "UR_RESULT_ERROR_UNSUPPORTED_VERSION";
    case UR_RESULT_ERROR_UNSUPPORTED_FEATURE:
        return "UR_RESULT_ERROR_UNSUPPORTED_FEATURE";
    case UR_RESULT_ERROR_INVALID_ARGUMENT:
        return "UR_RESULT_ERROR_INVALID_ARGUMENT";
    case UR_RESULT_ERROR_INVALID_NULL_HANDLE:
        return "UR_RESULT_ERROR_INVALID_NULL_HANDLE";
    case UR_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
        return "UR_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
    case UR_RESULT_ERROR_INVALID_NULL_POINTER:
        return "UR_RESULT_ERROR_INVALID_NULL_POINTER";
    case UR_RESULT_ERROR_INVALID_SIZE:
        return "UR_RESULT_ERROR_INVALID_SIZE";
    case UR_RESULT_ERROR_UNSUPPORTED_SIZE:
        return "UR_RESULT_ERROR_UNSUPPORTED_SIZE";
    case UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
        return "UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
    case UR_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
        return "UR_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
    case UR_RESULT_ERROR_INVALID_ENUMERATION:
        return "UR_RESULT_ERROR_INVALID_ENUMERATION";
    case UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
        return "UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
    case UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
        return "UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
    case UR_RESULT_ERROR_INVALID_NATIVE_BINARY:
        return "UR_RESULT_ERROR_INVALID_NATIVE_BINARY";
    case UR_RESULT_ERROR_INVALID_GLOBAL_NAME:
        return "UR_RESULT_ERROR_INVALID_GLOBAL_NAME";
    case UR_RESULT_ERROR_INVALID_FUNCTION_NAME:
        return "UR_RESULT_ERROR_INVALID_FUNCTION_NAME";
    case UR_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
        return "UR_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
    case UR_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
        return "UR_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
    case UR_RESULT_ERROR_PROGRAM_UNLINKED:
        return "UR_RESULT_ERROR_PROGRAM_UNLINKED";
    case UR_RESULT_ERROR_OVERLAPPING_REGIONS:
        return "UR_RESULT_ERROR_OVERLAPPING_REGIONS";
    case UR_RESULT_ERROR_INVALID_HOST_PTR:
        return "UR_RESULT_ERROR_INVALID_HOST_PTR";
    case UR_RESULT_ERROR_INVALID_USM_SIZE:
        return "UR_RESULT_ERROR_INVALID_USM_SIZE";
    case UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE:
        return "UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE";
    case UR_RESULT_ERROR_ADAPTER_SPECIFIC:
        return "UR_RESULT_ERROR_ADAPTER_SPECIFIC";
    default:
        return "UR_RESULT_ERROR_UNKNOWN";
    }
};

} // namespace

ur_result_t SanitizerInterceptor::allocateMemory(
    ur_context_handle_t Context, ur_device_handle_t Device,
    const ur_usm_desc_t *Properties, ur_usm_pool_handle_t Pool, size_t Size,
    void **ResultPtr, USMMemoryType Type) {
    (void)Context;

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
    uptr rz_log = ComputeRZLog(Size);
    uptr rz_size = RZLog2Size(rz_log);
    uptr rounded_size = RoundUpTo(Size, Alignment);
    uptr NeededSize = rounded_size + rz_size * 2;

    std::cerr << "allocateMemory:"
              << "\n  user_size: " << Size << "\n  rz_size: " << rz_size
              << "\n  rounded_size: " << rounded_size
              << "\n  NeededSize: " << NeededSize << std::endl;

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
    uptr UserBegin = AllocBegin + rz_size;
    if (!IsAligned(UserBegin, Alignment)) {
        UserBegin = RoundUpTo(UserBegin, Alignment);
    }
    uptr UserEnd = UserBegin + Size;
    assert(UserEnd <= AllocEnd);

    *ResultPtr = reinterpret_cast<void *>(UserBegin);

    auto MemoryInfo =
        AllocatedMemoryInfo{AllocBegin, UserBegin, UserEnd, NeededSize, Type};
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
        std::cout << "AllocatedAddressesMap: " << (void *)AllocBegin << "\n";
    }

    std::cout << "AllocInfos: " << (void *)AllocBegin << " "
              << (void *)UserBegin << "-" << (void *)UserEnd << " "
              << NeededSize << " " << (void *)Type << std::endl;

    return UR_RESULT_SUCCESS;
}

ur_result_t SanitizerInterceptor::releaseMemory(ur_context_handle_t Context,
                                                void *Ptr) {
    auto &ContextInfo = getContextInfo(Context);
    assert(ContextInfo.Init);

    std::cerr << "releaseMemory: " << Ptr << "\n";

    std::lock_guard<std::mutex> Guard(ContextInfo.Mutex);
    auto Addr = (uptr)Ptr;
    // Find the last element is not greater than key
    auto AddressInfoIt =
        ContextInfo.AllocatedAddressesMap.upper_bound((uptr)Addr);
    if (AddressInfoIt == ContextInfo.AllocatedAddressesMap.begin()) {
        std::cerr << "ERROR: releaseMemory failed! AllocatedAddressesMap\n";
        return UR_RESULT_SUCCESS;
    }
    --AddressInfoIt;
    auto &AddressInfo = AddressInfoIt->second;
    std::cerr << "AddressInfo: " << AddressInfo.AllocBegin << " "
              << AddressInfo.UserBegin << "\n";
    if (Addr != AddressInfo.UserBegin) {
        std::cerr << "ERROR: releaseMemory failed! UserBegin\n";
        return UR_RESULT_SUCCESS;
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

bool SanitizerInterceptor::updateHostShadowMemory(
    ur_context_handle_t Context, AllocatedMemoryInfo AllocInfo) {
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
    auto ShadowByte = AllocInfo.Type == USMMemoryType::DEVICE
                          ? kUsmDeviceRedzoneMagic
                          : kUsmSharedRedzoneMagic;
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
        /// SHADOW MAPPING (PVC, with CPU 47bit)
        ///   Host/Shared USM : 0x0              ~ 0x0fff_ffff_ffff
        ///   ?               : 0x1000_0000_0000 ~ 0x1fff_ffff_ffff
        ///   Device USM      : 0x2000_0000_0000 ~ 0x3fff_ffff_ffff
        constexpr size_t SHADOW_SIZE = 1ULL << 46;

        // TODO: Protect Bad Zone
        auto URes = m_Dditable.VirtualMem.pfnReserve(
            Context, nullptr, SHADOW_SIZE, (void **)&DeviceInfo.ShadowOffset);
        if (URes != UR_RESULT_SUCCESS) {
                printf("urVirtualMemReserve(): %s\n", getUrResultString(URes));
        }
        assert(URes == UR_RESULT_SUCCESS);

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
                        printf("    zePhysicalMemCreate(): %s\n",
                               getUrResultString(URes));
                    }
                    assert(URes == UR_RESULT_SUCCESS);
                }

                printf("  zeVirtualMemMap: %p ~ %p\n", (void *)MappedPtr,
                       (void *)(MappedPtr + PageSize - 1));

                // FIXME: No flag to check the failed reason is VA is already mapped
                auto URes = m_Dditable.VirtualMem.pfnMap(
                    Context, (void *)MappedPtr, PageSize, PhysicalMem, 0,
                    UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE);
                if (URes != UR_RESULT_SUCCESS) {
                    printf("    zeVirtualMemMap(): %s\n",
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
                        printf("    urEnqueueUSMFill(): %s\n",
                               getUrResultString(URes));
                    }
                    assert(URes == UR_RESULT_SUCCESS);

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
            printf("  urEnqueueUSMFill(): %s\n", getUrResultString(URes));
        }
        assert(URes == UR_RESULT_SUCCESS);
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
                                            AllocatedMemoryInfo &AllocInfo,
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

    int ShadowByte = 0;
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
    default:
        ShadowByte = 0xFF;
        assert(false && "Unknow AllocInfo.Type");
        break;
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

    std::cout << "Device ShadowOffset: " << (void *)DeviceInfo.ShadowOffset
              << " - " << (void *)DeviceInfo.ShadowOffsetEnd << "\n";

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

    bool Res = true;
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
                std::cerr << "WARNING: Device Global Write Failed [" << Name
                          << "] " << Result << std::endl;
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

    assert(Res && "Init Kernel Failed");

    QueueInfo.LastEvent = LastEvent;

    return Res;
}
