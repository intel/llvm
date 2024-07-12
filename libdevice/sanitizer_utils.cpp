//==--- sanitizer_utils.cpp - device sanitizer util inserted by compiler ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "atomic.hpp"
#include "device.h"
#include "spirv_vars.h"

#include "include/asan_libdevice.hpp"
#include "include/sanitizer_utils.hpp"

using uptr = uintptr_t;
using s8 = char;
using u8 = unsigned char;
using s16 = short;
using u16 = unsigned short;

DeviceGlobal<uptr> __AsanShadowMemoryGlobalStart;
DeviceGlobal<uptr> __AsanShadowMemoryGlobalEnd;
DeviceGlobal<DeviceType> __DeviceType;
DeviceGlobal<uint64_t> __AsanDebug;
// Save the pointer to LaunchInfo
__SYCL_GLOBAL__ uptr *__SYCL_LOCAL__ __AsanLaunchInfo;

#if defined(__SPIR__) || defined(__SPIRV__)

#if defined(__SYCL_DEVICE_ONLY__)

#define __USE_SPIR_BUILTIN__ 1

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif // SYCL_EXTERNAL

#else // __SYCL_DEVICE_ONLY__

#define __USE_SPIR_BUILTIN__

#endif // __SYCL_DEVICE_ONLY__

#if __USE_SPIR_BUILTIN__
extern SYCL_EXTERNAL int
__spirv_ocl_printf(const __SYCL_CONSTANT__ char *Format, ...);

extern SYCL_EXTERNAL __SYCL_GLOBAL__ void *
__spirv_GenericCastToPtrExplicit_ToGlobal(void *, int);
extern SYCL_EXTERNAL __SYCL_LOCAL__ void *
__spirv_GenericCastToPtrExplicit_ToLocal(void *, int);
extern SYCL_EXTERNAL __SYCL_PRIVATE__ void *
__spirv_GenericCastToPtrExplicit_ToPrivate(void *, int);
#endif // __USE_SPIR_BUILTIN__

static const __SYCL_CONSTANT__ char __asan_shadow_value_start[] =
    "[kernel] %p(%d) -> %p:";
static const __SYCL_CONSTANT__ char __asan_shadow_value[] = " %02X";
static const __SYCL_CONSTANT__ char __asan_current_shadow_value[] = ">%02X";
static const __SYCL_CONSTANT__ char __newline[] = "\n";

static const __SYCL_CONSTANT__ char __global_shadow_out_of_bound[] =
    "[kernel] Global shadow memory out-of-bound (ptr: %p -> %p, base: %p)\n";
static const __SYCL_CONSTANT__ char __local_shadow_out_of_bound[] =
    "[kernel] Local shadow memory out-of-bound (ptr: %p -> %p, wg: %d, base: "
    "%p)\n";
static const __SYCL_CONSTANT__ char __private_shadow_out_of_bound[] =
    "[kernel] Private shadow memory out-of-bound (ptr: %p -> %p, wg: %d, base: "
    "%p)\n";

static const __SYCL_CONSTANT__ char __asan_print_unsupport_device_type[] =
    "[kernel] Unsupport device type: %d\n";

static const __SYCL_CONSTANT__ char __asan_print_shadow_value1[] =
    "[kernel] %p(%d) -> %p: %02X\n";
static const __SYCL_CONSTANT__ char __asan_print_shadow_value2[] =
    "[kernel] %p(%d) -> %p: --\n";

static __SYCL_CONSTANT__ const char __generic_to[] =
    "[kernel] %p(4) - %p(%d)\n";

static __SYCL_CONSTANT__ const char __generic_to_fail[] =
    "[kernel] %p(4) - unknown address space\n";

static __SYCL_CONSTANT__ const char __mem_launch_info[] =
    "[kernel] launch_info: %p (local_shadow=%p~%p, numLocalArgs=%d, "
    "localArgs=%p)\n";

#define ASAN_REPORT_NONE 0
#define ASAN_REPORT_START 1
#define ASAN_REPORT_FINISH 2

enum ADDRESS_SPACE : uint32_t {
  ADDRESS_SPACE_PRIVATE = 0,
  ADDRESS_SPACE_GLOBAL = 1,
  ADDRESS_SPACE_CONSTANT = 2,
  ADDRESS_SPACE_LOCAL = 3,
  ADDRESS_SPACE_GENERIC = 4,
};

namespace {

bool __asan_report_unknown_device();
bool __asan_report_out_of_shadow_bounds();
void __asan_print_shadow_memory(uptr addr, uptr shadow_address, uint32_t as);

__SYCL_GLOBAL__ void *ToGlobal(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToGlobal(ptr, 5);
}
__SYCL_LOCAL__ void *ToLocal(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToLocal(ptr, 4);
}
__SYCL_PRIVATE__ void *ToPrivate(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToPrivate(ptr, 7);
}

inline bool ConvertGenericPointer(uptr &addr, uint32_t &as) {
  auto old = addr;
  if ((addr = (uptr)ToPrivate((void *)old))) {
    as = ADDRESS_SPACE_PRIVATE;
  } else if ((addr = (uptr)ToLocal((void *)old))) {
    as = ADDRESS_SPACE_LOCAL;
  } else if ((addr = (uptr)ToGlobal((void *)old))) {
    as = ADDRESS_SPACE_GLOBAL;
  } else {
    if (__AsanDebug)
      __spirv_ocl_printf(__generic_to_fail, old);
    return false;
  }
  if (__AsanDebug)
    __spirv_ocl_printf(__generic_to, old, addr, as);
  return true;
}

inline uptr MemToShadow_CPU(uptr addr) {
  return __AsanShadowMemoryGlobalStart + (addr >> ASAN_SHADOW_SCALE);
}

inline uptr MemToShadow_DG2(uptr addr, uint32_t as) {
  if (as == ADDRESS_SPACE_GENERIC) {
    if (!ConvertGenericPointer(addr, as)) {
      return 0;
    }
  }

  if (as == ADDRESS_SPACE_GLOBAL) {     // global
    if (addr & 0xFFFF000000000000ULL) { // Device USM
      return __AsanShadowMemoryGlobalStart + 0x80000000000ULL +
             ((addr & 0x7FFFFFFFFFFFULL) >> ASAN_SHADOW_SCALE);
    } else { // Host/Shared USM
      return __AsanShadowMemoryGlobalStart + (addr >> ASAN_SHADOW_SCALE);
    }
  } else if (as == ADDRESS_SPACE_LOCAL) { // local
    // The size of SLM is 64KB on DG2
    constexpr unsigned slm_size = 64 * 1024;
    const auto wg_lid =
        __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
            __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.z;

    auto launch_info = (__SYCL_GLOBAL__ const LaunchInfo *)__AsanLaunchInfo;
    const auto shadow_offset = launch_info->LocalShadowOffset;
    const auto shadow_offset_end = launch_info->LocalShadowOffsetEnd;

    if (shadow_offset == 0) {
      return 0;
    }

    if (__AsanDebug)
      __spirv_ocl_printf(__mem_launch_info, launch_info,
                         launch_info->LocalShadowOffset,
                         launch_info->LocalShadowOffsetEnd,
                         launch_info->NumLocalArgs, launch_info->LocalArgs);

    auto shadow_ptr = shadow_offset +
                      ((wg_lid * slm_size) >> ASAN_SHADOW_SCALE) +
                      ((addr & (slm_size - 1)) >> ASAN_SHADOW_SCALE);

    if (shadow_ptr > shadow_offset_end) {
      if (__asan_report_out_of_shadow_bounds()) {
        __spirv_ocl_printf(__local_shadow_out_of_bound, addr, shadow_ptr,
                           wg_lid, (uptr)shadow_offset);
      }
      return 0;
    }
    return shadow_ptr;
  } else if (as == ADDRESS_SPACE_PRIVATE) { // private
    // work-group linear id
    const auto WG_LID =
        __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
            __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.z;

    auto launch_info = (__SYCL_GLOBAL__ const LaunchInfo *)__AsanLaunchInfo;
    const auto shadow_offset = launch_info->PrivateShadowOffset;
    const auto shadow_offset_end = launch_info->LocalShadowOffsetEnd;

    if (shadow_offset == 0) {
      return 0;
    }

    if (__AsanDebug)
      __spirv_ocl_printf(__mem_launch_info, launch_info,
                         launch_info->PrivateShadowOffset, 0,
                         launch_info->NumLocalArgs, launch_info->LocalArgs);

    uptr shadow_ptr = shadow_offset +
                      ((WG_LID * ASAN_PRIVATE_SIZE) >> ASAN_SHADOW_SCALE) +
                      ((addr & (ASAN_PRIVATE_SIZE - 1)) >> ASAN_SHADOW_SCALE);

    if (shadow_ptr > shadow_offset_end) {
      if (__asan_report_out_of_shadow_bounds()) {
        __spirv_ocl_printf(__private_shadow_out_of_bound, addr, shadow_ptr,
                           WG_LID, (uptr)shadow_offset);
      }
      return 0;
    }
    return shadow_ptr;
  }

  return 0;
}

inline uptr MemToShadow_PVC(uptr addr, uint32_t as) {
  if (as == ADDRESS_SPACE_GENERIC) {
    if (!ConvertGenericPointer(addr, as)) {
      return 0;
    }
  }

  if (as == ADDRESS_SPACE_GLOBAL) { // global
    uptr shadow_ptr;
    if (addr & 0xFF00000000000000) { // Device USM
      shadow_ptr = __AsanShadowMemoryGlobalStart + 0x80000000000 +
                   ((addr & 0xFFFFFFFFFFFF) >> ASAN_SHADOW_SCALE);
    } else { // Only consider 47bit VA
      shadow_ptr = __AsanShadowMemoryGlobalStart +
                   ((addr & 0x7FFFFFFFFFFF) >> ASAN_SHADOW_SCALE);
    }

    if (shadow_ptr > __AsanShadowMemoryGlobalEnd) {
      if (__asan_report_out_of_shadow_bounds()) {
        __spirv_ocl_printf(__global_shadow_out_of_bound, addr, shadow_ptr,
                           (uptr)__AsanShadowMemoryGlobalStart);
      }
      return 0;
    }
    return shadow_ptr;
  } else if (as == ADDRESS_SPACE_LOCAL) { // local
    // The size of SLM is 128KB on PVC
    constexpr unsigned SLM_SIZE = 128 * 1024;
    // work-group linear id
    const auto wg_lid =
        __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
            __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.z;

    auto launch_info = (__SYCL_GLOBAL__ const LaunchInfo *)__AsanLaunchInfo;
    const auto shadow_offset = launch_info->LocalShadowOffset;
    const auto shadow_offset_end = launch_info->LocalShadowOffsetEnd;

    if (shadow_offset == 0) {
      return 0;
    }

    if (__AsanDebug)
      __spirv_ocl_printf(__mem_launch_info, launch_info,
                         launch_info->LocalShadowOffset,
                         launch_info->LocalShadowOffsetEnd,
                         launch_info->NumLocalArgs, launch_info->LocalArgs);

    uptr shadow_ptr = shadow_offset +
                      ((wg_lid * SLM_SIZE) >> ASAN_SHADOW_SCALE) +
                      ((addr & (SLM_SIZE - 1)) >> ASAN_SHADOW_SCALE);

    if (shadow_ptr > shadow_offset_end) {
      if (__asan_report_out_of_shadow_bounds()) {
        __spirv_ocl_printf(__local_shadow_out_of_bound, addr, shadow_ptr,
                           wg_lid, (uptr)shadow_offset);
      }
      return 0;
    }
    return shadow_ptr;
  } else if (as == ADDRESS_SPACE_PRIVATE) { // private
    // work-group linear id
    const auto WG_LID =
        __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
            __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.z;

    auto launch_info = (__SYCL_GLOBAL__ const LaunchInfo *)__AsanLaunchInfo;
    const auto shadow_offset = launch_info->PrivateShadowOffset;
    const auto shadow_offset_end = launch_info->PrivateShadowOffsetEnd;

    if (shadow_offset == 0) {
      return 0;
    }

    if (__AsanDebug)
      __spirv_ocl_printf(__mem_launch_info, launch_info,
                         launch_info->PrivateShadowOffset, 0,
                         launch_info->NumLocalArgs, launch_info->LocalArgs);

    uptr shadow_ptr = shadow_offset +
                      ((WG_LID * ASAN_PRIVATE_SIZE) >> ASAN_SHADOW_SCALE) +
                      ((addr & (ASAN_PRIVATE_SIZE - 1)) >> ASAN_SHADOW_SCALE);

    if (shadow_ptr > shadow_offset_end) {
      if (__asan_report_out_of_shadow_bounds()) {
        __spirv_ocl_printf(__private_shadow_out_of_bound, addr, shadow_ptr,
                           WG_LID, (uptr)shadow_offset);
      }
      return 0;
    }
    return shadow_ptr;
  }

  return 0;
}

inline uptr MemToShadow(uptr addr, uint32_t as) {
  uptr shadow_ptr = 0;

  if (__DeviceType == DeviceType::CPU) {
    shadow_ptr = MemToShadow_CPU(addr);
  } else if (__DeviceType == DeviceType::GPU_PVC) {
    shadow_ptr = MemToShadow_PVC(addr, as);
  } else if (__DeviceType == DeviceType::GPU_DG2) {
    shadow_ptr = MemToShadow_DG2(addr, as);
  } else {
    if (__asan_report_unknown_device() && __AsanDebug) {
      __spirv_ocl_printf(__asan_print_unsupport_device_type, (int)__DeviceType);
    }
    return shadow_ptr;
  }

// FIXME: OCL "O2" optimizer doesn't work well with following code
#if 0
  if (__AsanDebug) {
    if (shadow_ptr) {
      if (as == ADDRESS_SPACE_PRIVATE)
        __asan_print_shadow_memory(addr, shadow_ptr, as);
      else
        __spirv_ocl_printf(__asan_print_shadow_value1, addr, as, shadow_ptr,
                           *(u8 *)shadow_ptr);
    } else {
      __spirv_ocl_printf(__asan_print_shadow_value2, addr, as, shadow_ptr);
    }
  }
#endif

  return shadow_ptr;
}

inline constexpr uptr RoundUpTo(uptr size, uptr boundary) {
  return (size + boundary - 1) & ~(boundary - 1);
}

inline constexpr uptr RoundDownTo(uptr x, uptr boundary) {
  return x & ~(boundary - 1);
}

bool MemIsZero(__SYCL_GLOBAL__ const char *beg, uptr size) {
  __SYCL_GLOBAL__ const char *end = beg + size;
  auto *aligned_beg =
      (__SYCL_GLOBAL__ uptr *)RoundUpTo((uptr)beg, sizeof(uptr));
  auto *aligned_end =
      (__SYCL_GLOBAL__ uptr *)RoundDownTo((uptr)end, sizeof(uptr));
  uptr all = 0;
  // Prologue.
  for (__SYCL_GLOBAL__ const char *mem = beg;
       mem < (__SYCL_GLOBAL__ char *)aligned_beg && mem < end; mem++)
    all |= *mem;
  // Aligned loop.
  for (; aligned_beg < aligned_end; aligned_beg++)
    all |= *aligned_beg;
  // Epilogue.
  if ((__SYCL_GLOBAL__ char *)aligned_end >= beg) {
    for (__SYCL_GLOBAL__ const char *mem = (__SYCL_GLOBAL__ char *)aligned_end;
         mem < end; mem++)
      all |= *mem;
  }
  return all == 0;
}

///
/// ASAN Save Report
///

static __SYCL_CONSTANT__ const char __mem_sanitizer_report[] =
    "[kernel] SanitizerReport (ErrorType=%d, IsRecover=%d)\n";

bool __asan_internal_report_save(DeviceSanitizerErrorType error_type) {
  const int Expected = ASAN_REPORT_NONE;
  int Desired = ASAN_REPORT_START;

  // work-group linear id
  const auto WG_LID =
      __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
          __spirv_BuiltInNumWorkgroups.z +
      __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
      __spirv_BuiltInWorkgroupId.z;

  auto &SanitizerReport = ((__SYCL_GLOBAL__ LaunchInfo *)__AsanLaunchInfo)
                              ->SanitizerReport[WG_LID % ASAN_MAX_NUM_REPORTS];

  if (atomicCompareAndSet(&SanitizerReport.Flag, Desired, Expected) ==
      Expected) {
    SanitizerReport.ErrorType = error_type;
    SanitizerReport.IsRecover = false;

    // Show we've done copying
    atomicStore(&SanitizerReport.Flag, ASAN_REPORT_FINISH);

    if (__AsanDebug)
      __spirv_ocl_printf(__mem_sanitizer_report, SanitizerReport.ErrorType,
                         SanitizerReport.IsRecover);
    return true;
  }
  return false;
}

bool __asan_internal_report_save(
    uptr ptr, uint32_t as, const char __SYCL_CONSTANT__ *file, uint32_t line,
    const char __SYCL_CONSTANT__ *func, bool is_write, uint32_t access_size,
    DeviceSanitizerMemoryType memory_type, DeviceSanitizerErrorType error_type,
    bool is_recover = false) {

  const int Expected = ASAN_REPORT_NONE;
  int Desired = ASAN_REPORT_START;

  if (__AsanDebug) {
    auto *launch_info = (__SYCL_GLOBAL__ LaunchInfo *)__AsanLaunchInfo;
    __spirv_ocl_printf(__mem_launch_info, launch_info,
                       launch_info->LocalShadowOffset,
                       launch_info->LocalShadowOffsetEnd,
                       launch_info->NumLocalArgs, launch_info->LocalArgs);
  }

  // work-group linear id
  const auto WG_LID =
      __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
          __spirv_BuiltInNumWorkgroups.z +
      __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
      __spirv_BuiltInWorkgroupId.z;

  auto &SanitizerReport = ((__SYCL_GLOBAL__ LaunchInfo *)__AsanLaunchInfo)
                              ->SanitizerReport[WG_LID % ASAN_MAX_NUM_REPORTS];

  if (atomicCompareAndSet(&SanitizerReport.Flag, Desired, Expected) ==
      Expected) {

    int FileLength = 0;
    int FuncLength = 0;

    if (file)
      for (auto *C = file; *C != '\0'; ++C, ++FileLength)
        ;
    if (func)
      for (auto *C = func; *C != '\0'; ++C, ++FuncLength)
        ;

    int MaxFileIdx = sizeof(SanitizerReport.File) - 1;
    int MaxFuncIdx = sizeof(SanitizerReport.Func) - 1;

    if (FileLength < MaxFileIdx)
      MaxFileIdx = FileLength;
    if (FuncLength < MaxFuncIdx)
      MaxFuncIdx = FuncLength;

    for (int Idx = 0; Idx < MaxFileIdx; ++Idx)
      SanitizerReport.File[Idx] = file[Idx];
    SanitizerReport.File[MaxFileIdx] = '\0';

    for (int Idx = 0; Idx < MaxFuncIdx; ++Idx)
      SanitizerReport.Func[Idx] = func[Idx];
    SanitizerReport.Func[MaxFuncIdx] = '\0';

    SanitizerReport.Line = line;
    SanitizerReport.GID0 = __spirv_GlobalInvocationId_x();
    SanitizerReport.GID1 = __spirv_GlobalInvocationId_y();
    SanitizerReport.GID2 = __spirv_GlobalInvocationId_z();
    SanitizerReport.LID0 = __spirv_LocalInvocationId_x();
    SanitizerReport.LID1 = __spirv_LocalInvocationId_y();
    SanitizerReport.LID2 = __spirv_LocalInvocationId_z();

    SanitizerReport.Address = ptr;
    SanitizerReport.IsWrite = is_write;
    SanitizerReport.AccessSize = access_size;
    SanitizerReport.ErrorType = error_type;
    SanitizerReport.MemoryType = memory_type;
    SanitizerReport.IsRecover = is_recover;

    // Show we've done copying
    atomicStore(&SanitizerReport.Flag, ASAN_REPORT_FINISH);

    if (__AsanDebug)
      __spirv_ocl_printf(__mem_sanitizer_report, SanitizerReport.ErrorType,
                         SanitizerReport.IsRecover);
    return true;
  }
  return false;
}

///
/// ASAN Error Reporters
///

DeviceSanitizerMemoryType GetMemoryTypeByShadowValue(int shadow_value) {
  switch (shadow_value) {
  case kUsmDeviceRedzoneMagic:
  case kUsmDeviceDeallocatedMagic:
    return DeviceSanitizerMemoryType::USM_DEVICE;
  case kUsmHostRedzoneMagic:
  case kUsmHostDeallocatedMagic:
    return DeviceSanitizerMemoryType::USM_HOST;
  case kUsmSharedRedzoneMagic:
  case kUsmSharedDeallocatedMagic:
    return DeviceSanitizerMemoryType::USM_SHARED;
  case kPrivateLeftRedzoneMagic:
  case kPrivateMidRedzoneMagic:
  case kPrivateRightRedzoneMagic:
    return DeviceSanitizerMemoryType::PRIVATE;
  case kMemBufferRedzoneMagic:
    return DeviceSanitizerMemoryType::MEM_BUFFER;
  case kSharedLocalRedzoneMagic:
    return DeviceSanitizerMemoryType::LOCAL;
  case kDeviceGlobalRedzoneMagic:
    return DeviceSanitizerMemoryType::DEVICE_GLOBAL;
  default:
    return DeviceSanitizerMemoryType::UNKNOWN;
  }
}

void __asan_report_access_error(uptr addr, uint32_t as, size_t size,
                                bool is_write, uptr poisoned_addr,
                                const char __SYCL_CONSTANT__ *file,
                                uint32_t line,
                                const char __SYCL_CONSTANT__ *func,
                                bool is_recover = false) {
  // Check Error Type
  auto *shadow_address = (__SYCL_GLOBAL__ s8 *)MemToShadow(poisoned_addr, as);
  int shadow_value = *shadow_address;
  if (shadow_value > 0) {
    shadow_value = *(shadow_address + 1);
  }
  // FIXME: check if shadow_address out-of-bound

  DeviceSanitizerMemoryType memory_type =
      GetMemoryTypeByShadowValue(shadow_value);
  DeviceSanitizerErrorType error_type;

  switch (shadow_value) {
  case kUsmDeviceRedzoneMagic:
  case kUsmHostRedzoneMagic:
  case kUsmSharedRedzoneMagic:
  case kPrivateLeftRedzoneMagic:
  case kPrivateMidRedzoneMagic:
  case kPrivateRightRedzoneMagic:
  case kMemBufferRedzoneMagic:
  case kSharedLocalRedzoneMagic:
  case kDeviceGlobalRedzoneMagic:
    error_type = DeviceSanitizerErrorType::OUT_OF_BOUNDS;
    break;
  case kUsmDeviceDeallocatedMagic:
  case kUsmHostDeallocatedMagic:
  case kUsmSharedDeallocatedMagic:
    error_type = DeviceSanitizerErrorType::USE_AFTER_FREE;
    break;
  default:
    error_type = DeviceSanitizerErrorType::UNKNOWN;
  }

  __asan_internal_report_save(addr, as, file, line, func, is_write, size,
                              memory_type, error_type, is_recover);
}

void __asan_report_misalign_error(uptr addr, uint32_t as, size_t size,
                                  bool is_write, uptr poisoned_addr,
                                  const char __SYCL_CONSTANT__ *file,
                                  uint32_t line,
                                  const char __SYCL_CONSTANT__ *func,
                                  bool is_recover = false) {

  auto *shadow = (__SYCL_GLOBAL__ s8 *)MemToShadow(addr, as);
  while (*shadow >= 0) {
    ++shadow;
  }
  int shadow_value = *shadow;

  DeviceSanitizerErrorType error_type = DeviceSanitizerErrorType::MISALIGNED;
  DeviceSanitizerMemoryType memory_type =
      GetMemoryTypeByShadowValue(shadow_value);

  __asan_internal_report_save(addr, as, file, line, func, is_write, size,
                              memory_type, error_type, is_recover);
}

bool __asan_report_unknown_device() {
  return __asan_internal_report_save(DeviceSanitizerErrorType::UNKNOWN_DEVICE);
}

bool __asan_report_out_of_shadow_bounds() {
  return __asan_internal_report_save(
      DeviceSanitizerErrorType::OUT_OF_SHADOW_BOUNDS);
}

///
/// ASan utils
///

void __asan_print_shadow_memory(uptr addr, uptr shadow_address, uint32_t as) {
  uptr p = shadow_address & (~0xf);
  __spirv_ocl_printf(__asan_shadow_value_start, addr, as, p);
  for (int i = 0; i < 0xf; ++i) {
    u8 shadow_value = *(u8 *)(p + i);
    if (p + i == shadow_address) {
      __spirv_ocl_printf(__asan_current_shadow_value, shadow_value);
    } else {
      __spirv_ocl_printf(__asan_shadow_value, shadow_value);
    }
  }
  __spirv_ocl_printf(__newline);
}

bool __asan_region_is_value(uptr addr, uint32_t as, std::size_t size,
                            char value) {
  if (size == 0)
    return true;
  while (size--) {
    auto *shadow = (__SYCL_GLOBAL__ char *)MemToShadow(addr, as);
    if (*shadow != value) {
      return false;
    }
    ++addr;
  }
  return true;
}

// NOTE: size <= 16
inline int __asan_address_is_poisoned(uptr a, uint32_t as, size_t size = 1) {
  auto *shadow_address = (__SYCL_GLOBAL__ s8 *)MemToShadow(a, as);
  if (shadow_address) {
    auto shadow_value = *shadow_address;
    if (shadow_value) {
      if (size == ASAN_SHADOW_GRANULARITY)
        return true;
      s8 last_accessed_byte = (a & (ASAN_SHADOW_GRANULARITY - 1)) + size - 1;
      return (last_accessed_byte >= shadow_value);
    }
  }
  return false;
}

inline uptr __asan_region_is_poisoned(uptr beg, uint32_t as, size_t size) {
  if (!size)
    return 0;

  uptr end = beg + size;
  uptr aligned_b = RoundUpTo(beg, ASAN_SHADOW_GRANULARITY);
  uptr aligned_e = RoundDownTo(end, ASAN_SHADOW_GRANULARITY);

  uptr shadow_beg = MemToShadow(aligned_b, as);
  if (!shadow_beg) {
    return 0;
  }
  uptr shadow_end = MemToShadow(aligned_e, as);
  if (!shadow_end) {
    return 0;
  }

  // First check the first and the last application bytes,
  // then check the ASAN_SHADOW_GRANULARITY-aligned region by calling
  // MemIsZero on the corresponding shadow.
  if (!__asan_address_is_poisoned(beg, as) &&
      !__asan_address_is_poisoned(end - 1, as) &&
      (shadow_end <= shadow_beg ||
       MemIsZero((__SYCL_GLOBAL__ const char *)shadow_beg,
                 shadow_end - shadow_beg)))
    return 0;

  // The fast check failed, so we have a poisoned byte somewhere.
  // Find it slowly.
  for (; beg < end; beg++)
    if (__asan_address_is_poisoned(beg, as))
      return beg;

  return 0;
}

constexpr size_t AlignMask(size_t n) { return n - 1; }

} // namespace

///
/// ASAN Load/Store Report Built-ins
///

#define ASAN_REPORT_ERROR(type, is_write, size)                                \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##size(                           \
      uptr addr, uint32_t as, const char __SYCL_CONSTANT__ *file,              \
      uint32_t line, const char __SYCL_CONSTANT__ *func) {                     \
    if (addr & AlignMask(size)) {                                              \
      __asan_report_misalign_error(addr, as, size, is_write, addr, file, line, \
                                   func);                                      \
    }                                                                          \
    if (__asan_address_is_poisoned(addr, as, size)) {                          \
      __asan_report_access_error(addr, as, size, is_write, addr, file, line,   \
                                 func);                                        \
    }                                                                          \
  }                                                                            \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##size##_noabort(                 \
      uptr addr, uint32_t as, const char __SYCL_CONSTANT__ *file,              \
      uint32_t line, const char __SYCL_CONSTANT__ *func) {                     \
    if (addr & AlignMask(size)) {                                              \
      __asan_report_misalign_error(addr, as, size, is_write, addr, file, line, \
                                   func, true);                                \
    }                                                                          \
    if (__asan_address_is_poisoned(addr, as, size)) {                          \
      __asan_report_access_error(addr, as, size, is_write, addr, file, line,   \
                                 func, true);                                  \
    }                                                                          \
  }

ASAN_REPORT_ERROR(load, false, 1)
ASAN_REPORT_ERROR(load, false, 2)
ASAN_REPORT_ERROR(load, false, 4)
ASAN_REPORT_ERROR(load, false, 8)
ASAN_REPORT_ERROR(load, false, 16)
ASAN_REPORT_ERROR(store, true, 1)
ASAN_REPORT_ERROR(store, true, 2)
ASAN_REPORT_ERROR(store, true, 4)
ASAN_REPORT_ERROR(store, true, 8)
ASAN_REPORT_ERROR(store, true, 16)

#define ASAN_REPORT_ERROR_N(type, is_write)                                    \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##N(                              \
      uptr addr, size_t size, uint32_t as, const char __SYCL_CONSTANT__ *file, \
      uint32_t line, const char __SYCL_CONSTANT__ *func) {                     \
    if (auto poisoned_addr = __asan_region_is_poisoned(addr, as, size)) {      \
      __asan_report_access_error(addr, as, size, is_write, poisoned_addr,      \
                                 file, line, func);                            \
    }                                                                          \
  }                                                                            \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##N_noabort(                      \
      uptr addr, size_t size, uint32_t as, const char __SYCL_CONSTANT__ *file, \
      uint32_t line, const char __SYCL_CONSTANT__ *func) {                     \
    if (auto poisoned_addr = __asan_region_is_poisoned(addr, as, size)) {      \
      __asan_report_access_error(addr, as, size, is_write, poisoned_addr,      \
                                 file, line, func, true);                      \
    }                                                                          \
  }

ASAN_REPORT_ERROR_N(load, false)
ASAN_REPORT_ERROR_N(store, true)

///
/// ASAN convert memory address to shadow memory address
///

DEVICE_EXTERN_C_NOINLINE uptr __asan_mem_to_shadow(uptr ptr, uint32_t as) {
  return MemToShadow(ptr, as);
}

///
/// ASAN initialize shdadow memory of local memory
///

static __SYCL_CONSTANT__ const char __mem_set_shadow_local[] =
    "[kernel] set_shadow_local(beg=%p, end=%p, val:%02X)\n";

DEVICE_EXTERN_C_NOINLINE void
__asan_set_shadow_static_local(uptr ptr, size_t size,
                               size_t size_with_redzone) {
  // Since ptr is aligned to ASAN_SHADOW_GRANULARITY,
  // if size != aligned_size, then the buffer tail of ptr is not aligned
  uptr aligned_size = RoundUpTo(size, ASAN_SHADOW_GRANULARITY);

  // Set user zone to zero
  {
    auto shadow_begin = MemToShadow(ptr, ADDRESS_SPACE_LOCAL);
    auto shadow_end = MemToShadow(ptr + size, ADDRESS_SPACE_LOCAL);
    if (__AsanDebug)
      __spirv_ocl_printf(__mem_set_shadow_local, shadow_begin, shadow_end, 0);
    while (shadow_begin <= shadow_end) {
      *((__SYCL_GLOBAL__ u8 *)shadow_begin) = 0;
      ++shadow_begin;
    }
  }

  // Set left red zone
  {
    auto shadow_address = MemToShadow(ptr + aligned_size, ADDRESS_SPACE_LOCAL);
    auto count = (size_with_redzone - aligned_size) >> ASAN_SHADOW_SCALE;
    if (__AsanDebug)
      __spirv_ocl_printf(__mem_set_shadow_local, shadow_address,
                         shadow_address + count,
                         (unsigned char)kSharedLocalRedzoneMagic);
    for (size_t i = 0; i < count; ++i) {
      ((__SYCL_GLOBAL__ u8 *)shadow_address)[i] = kSharedLocalRedzoneMagic;
    }
  }

  // Set unaligned tail
  if (size != aligned_size) {
    auto user_end = ptr + size;
    auto *shadow_end =
        (__SYCL_GLOBAL__ s8 *)MemToShadow(user_end, ADDRESS_SPACE_LOCAL);
    auto value = user_end - RoundDownTo(user_end, ASAN_SHADOW_GRANULARITY) + 1;
    if (__AsanDebug)
      __spirv_ocl_printf(__mem_set_shadow_local, shadow_end, shadow_end, value);
    *shadow_end = value;
  }
}

static __SYCL_CONSTANT__ const char __mem_local_arg[] =
    "[kernel] local_arg(index=%d, size=%d, size_rz=%d)\n";

static __SYCL_CONSTANT__ const char __mem_set_shadow_dynamic_local_begin[] =
    "[kernel] BEGIN __asan_set_shadow_dynamic_local\n";
static __SYCL_CONSTANT__ const char __mem_set_shadow_dynamic_local_end[] =
    "[kernel] END   __asan_set_shadow_dynamic_local\n";
static __SYCL_CONSTANT__ const char __mem_report_arg_count_incorrect[] =
    "[kernel] ERROR: The number of local args is incorrect, expect %d, actual "
    "%d\n";

DEVICE_EXTERN_C_NOINLINE void
__asan_set_shadow_dynamic_local(uptr ptr, uint32_t num_args) {
  if (__AsanDebug)
    __spirv_ocl_printf(__mem_set_shadow_dynamic_local_begin);

  auto *launch_info = (__SYCL_GLOBAL__ const LaunchInfo *)__AsanLaunchInfo;
  if (num_args != launch_info->NumLocalArgs) {
    __spirv_ocl_printf(__mem_report_arg_count_incorrect, num_args,
                       launch_info->NumLocalArgs);
    return;
  }

  uptr *args = (uptr *)ptr;
  if (__AsanDebug)
    __spirv_ocl_printf(__mem_launch_info, launch_info,
                       launch_info->LocalShadowOffset,
                       launch_info->LocalShadowOffsetEnd,
                       launch_info->NumLocalArgs, launch_info->LocalArgs);

  for (uint32_t i = 0; i < num_args; ++i) {
    auto *local_arg = &launch_info->LocalArgs[i];
    if (__AsanDebug)
      __spirv_ocl_printf(__mem_local_arg, i, local_arg->Size,
                         local_arg->SizeWithRedZone);

    __asan_set_shadow_static_local(args[i], local_arg->Size,
                                   local_arg->SizeWithRedZone);
  }

  if (__AsanDebug)
    __spirv_ocl_printf(__mem_set_shadow_dynamic_local_end);
}

#endif // __SPIR__ || __SPIRV__
