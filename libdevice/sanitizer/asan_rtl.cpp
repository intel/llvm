//==--- asan_rtl.cpp - device address sanitizer runtime library ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/asan_rtl.hpp"
#include "asan/asan_libdevice.hpp"
#include "atomic.hpp"
#include "device.h"
#include "spirv_vars.h"

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

extern SYCL_EXTERNAL __attribute__((convergent)) void
__spirv_ControlBarrier(uint32_t Execution, uint32_t Memory, uint32_t Semantics);

extern "C" SYCL_EXTERNAL void __devicelib_exit();
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

#define ASAN_REPORT_NONE 0
#define ASAN_REPORT_START 1
#define ASAN_REPORT_FINISH 2

#define ASAN_DEBUG(X)                                                          \
  do {                                                                         \
    auto launch_info = (__SYCL_GLOBAL__ const LaunchInfo *)__AsanLaunchInfo;   \
    if (launch_info->Debug) {                                                  \
      X;                                                                       \
    }                                                                          \
  } while (false)

enum ADDRESS_SPACE : uint32_t {
  ADDRESS_SPACE_PRIVATE = 0,
  ADDRESS_SPACE_GLOBAL = 1,
  ADDRESS_SPACE_CONSTANT = 2,
  ADDRESS_SPACE_LOCAL = 3,
  ADDRESS_SPACE_GENERIC = 4,
};

namespace {

void __asan_report_unknown_device();
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

inline void ConvertGenericPointer(uptr &addr, uint32_t &as) {
  auto old = addr;
  if ((addr = (uptr)ToPrivate((void *)old))) {
    as = ADDRESS_SPACE_PRIVATE;
  } else if ((addr = (uptr)ToLocal((void *)old))) {
    as = ADDRESS_SPACE_LOCAL;
  } else {
    // FIXME: I'm not sure if we need to check ADDRESS_SPACE_CONSTANT,
    // but this can really simplify the generic pointer conversion logic
    as = ADDRESS_SPACE_GLOBAL;
    addr = old;
  }
  ASAN_DEBUG(__spirv_ocl_printf(__generic_to, old, addr, as));
}

inline uptr MemToShadow_CPU(uptr addr) {
  auto launch_info = (__SYCL_GLOBAL__ const LaunchInfo *)__AsanLaunchInfo;
  return launch_info->GlobalShadowOffset + (addr >> ASAN_SHADOW_SCALE);
}

inline uptr MemToShadow_DG2(uptr addr, uint32_t as) {
  if (as == ADDRESS_SPACE_GENERIC) {
    ConvertGenericPointer(addr, as);
  }

  auto launch_info = (__SYCL_GLOBAL__ const LaunchInfo *)__AsanLaunchInfo;
  if (as == ADDRESS_SPACE_GLOBAL) { // global
    uptr shadow_ptr;
    if (addr & 0xFFFF000000000000ULL) { // Device USM
      shadow_ptr = launch_info->GlobalShadowOffset + 0x80000000000ULL +
                   ((addr & 0x7FFFFFFFFFFFULL) >> ASAN_SHADOW_SCALE);
    } else { // Host/Shared USM
      shadow_ptr =
          launch_info->GlobalShadowOffset + (addr >> ASAN_SHADOW_SCALE);
    }

    ASAN_DEBUG(
        const auto shadow_offset_end = launch_info->GlobalShadowOffsetEnd;
        if (shadow_ptr > shadow_offset_end) {
          __spirv_ocl_printf(__global_shadow_out_of_bound, addr, shadow_ptr,
                             (uptr)launch_info->GlobalShadowOffset);
          return 0;
        });

    return shadow_ptr;
  } else if (as == ADDRESS_SPACE_LOCAL) { // local
    // The size of SLM is 64KB on DG2
    constexpr unsigned slm_size = 64 * 1024;
    const auto wg_lid =
        __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
            __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.z;

    const auto shadow_offset = launch_info->LocalShadowOffset;
    if (shadow_offset == 0) {
      return 0;
    }

    auto shadow_ptr = shadow_offset +
                      ((wg_lid * slm_size) >> ASAN_SHADOW_SCALE) +
                      ((addr & (slm_size - 1)) >> ASAN_SHADOW_SCALE);

    ASAN_DEBUG(const auto shadow_offset_end = launch_info->LocalShadowOffsetEnd;
               if (shadow_ptr > shadow_offset_end) {
                 __spirv_ocl_printf(__local_shadow_out_of_bound, addr,
                                    shadow_ptr, wg_lid, (uptr)shadow_offset);
                 return 0;
               });
    return shadow_ptr;
  } else if (as == ADDRESS_SPACE_PRIVATE) { // private
    // work-group linear id
    const auto WG_LID =
        __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
            __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.z;

    const auto shadow_offset = launch_info->PrivateShadowOffset;
    if (shadow_offset == 0) {
      return 0;
    }

    uptr shadow_ptr = shadow_offset +
                      ((WG_LID * ASAN_PRIVATE_SIZE) >> ASAN_SHADOW_SCALE) +
                      ((addr & (ASAN_PRIVATE_SIZE - 1)) >> ASAN_SHADOW_SCALE);

    ASAN_DEBUG(const auto shadow_offset_end =
                   launch_info->PrivateShadowOffsetEnd;
               if (shadow_ptr > shadow_offset_end) {
                 __spirv_ocl_printf(__private_shadow_out_of_bound, addr,
                                    shadow_ptr, WG_LID, (uptr)shadow_offset);
                 return 0;
               });
    return shadow_ptr;
  }

  return 0;
}

inline uptr MemToShadow_PVC(uptr addr, uint32_t as) {
  if (as == ADDRESS_SPACE_GENERIC) {
    ConvertGenericPointer(addr, as);
  }

  auto launch_info = (__SYCL_GLOBAL__ const LaunchInfo *)__AsanLaunchInfo;
  if (as == ADDRESS_SPACE_GLOBAL) { // global
    uptr shadow_ptr;
    if (addr & 0xFF00000000000000) { // Device USM
      shadow_ptr = launch_info->GlobalShadowOffset + 0x80000000000 +
                   ((addr & 0xFFFFFFFFFFFF) >> ASAN_SHADOW_SCALE);
    } else { // Only consider 47bit VA
      shadow_ptr = launch_info->GlobalShadowOffset +
                   ((addr & 0x7FFFFFFFFFFF) >> ASAN_SHADOW_SCALE);
    }

    ASAN_DEBUG(
        const auto shadow_offset_end = launch_info->GlobalShadowOffsetEnd;
        if (shadow_ptr > shadow_offset_end) {
          __spirv_ocl_printf(__global_shadow_out_of_bound, addr, shadow_ptr,
                             (uptr)launch_info->GlobalShadowOffset);
          return 0;
        });
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

    const auto shadow_offset = launch_info->LocalShadowOffset;

    if (shadow_offset == 0) {
      return 0;
    }

    uptr shadow_ptr = shadow_offset +
                      ((wg_lid * SLM_SIZE) >> ASAN_SHADOW_SCALE) +
                      ((addr & (SLM_SIZE - 1)) >> ASAN_SHADOW_SCALE);

    ASAN_DEBUG(const auto shadow_offset_end = launch_info->LocalShadowOffsetEnd;
               if (shadow_ptr > shadow_offset_end) {
                 __spirv_ocl_printf(__local_shadow_out_of_bound, addr,
                                    shadow_ptr, wg_lid, (uptr)shadow_offset);
                 return 0;
               });
    return shadow_ptr;
  } else if (as == ADDRESS_SPACE_PRIVATE) { // private
    // work-group linear id
    const auto WG_LID =
        __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
            __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.z;

    const auto shadow_offset = launch_info->PrivateShadowOffset;

    if (shadow_offset == 0) {
      return 0;
    }

    uptr shadow_ptr = shadow_offset +
                      ((WG_LID * ASAN_PRIVATE_SIZE) >> ASAN_SHADOW_SCALE) +
                      ((addr & (ASAN_PRIVATE_SIZE - 1)) >> ASAN_SHADOW_SCALE);

    ASAN_DEBUG(const auto shadow_offset_end =
                   launch_info->PrivateShadowOffsetEnd;
               if (shadow_ptr > shadow_offset_end) {
                 __spirv_ocl_printf(__private_shadow_out_of_bound, addr,
                                    shadow_ptr, WG_LID, (uptr)shadow_offset);
                 return 0;
               });
    return shadow_ptr;
  }

  return 0;
}

inline uptr MemToShadow(uptr addr, uint32_t as) {
  uptr shadow_ptr = 0;

#if defined(__LIBDEVICE_PVC__)
  shadow_ptr = MemToShadow_PVC(addr, as);
#elif defined(__LIBDEVICE_CPU__)
  shadow_ptr = MemToShadow_CPU(addr);
#elif defined(__LIBDEVICE_DG2__)
  shadow_ptr = MemToShadow_DG2(addr, as);
#else
  auto launch_info = (__SYCL_GLOBAL__ const LaunchInfo *)__AsanLaunchInfo;
  if (launch_info->DeviceTy == DeviceType::CPU) {
    shadow_ptr = MemToShadow_CPU(addr);
  } else if (launch_info->DeviceTy == DeviceType::GPU_PVC) {
    shadow_ptr = MemToShadow_PVC(addr, as);
  } else if (launch_info->DeviceTy == DeviceType::GPU_DG2) {
    shadow_ptr = MemToShadow_DG2(addr, as);
  } else {
    ASAN_DEBUG(__spirv_ocl_printf(__asan_print_unsupport_device_type,
                                  (int)launch_info->DeviceTy));
    __asan_report_unknown_device();
    return 0;
  }
#endif

  ASAN_DEBUG(
      if (shadow_ptr) {
        if (as == ADDRESS_SPACE_PRIVATE)
          __asan_print_shadow_memory(addr, shadow_ptr, as);
        else
          __spirv_ocl_printf(__asan_print_shadow_value1, addr, as, shadow_ptr,
                             *(u8 *)shadow_ptr);
      } else {
        __spirv_ocl_printf(__asan_print_shadow_value2, addr, as, shadow_ptr);
      });

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

void __asan_internal_report_save(ErrorType error_type) {
  const int Expected = ASAN_REPORT_NONE;
  int Desired = ASAN_REPORT_START;

  // work-group linear id
  const auto WG_LID =
      __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
          __spirv_BuiltInNumWorkgroups.z +
      __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
      __spirv_BuiltInWorkgroupId.z;

  auto &SanitizerReport = ((__SYCL_GLOBAL__ LaunchInfo *)__AsanLaunchInfo)
                              ->Report[WG_LID % ASAN_MAX_NUM_REPORTS];

  if (atomicCompareAndSet(
          &(((__SYCL_GLOBAL__ LaunchInfo *)__AsanLaunchInfo)->ReportFlag), 1,
          0) == 0 &&
      atomicCompareAndSet(&SanitizerReport.Flag, Desired, Expected) ==
          Expected) {
    SanitizerReport.ErrorTy = error_type;
    SanitizerReport.IsRecover = false;

    // Show we've done copying
    atomicStore(&SanitizerReport.Flag, ASAN_REPORT_FINISH);

    ASAN_DEBUG(__spirv_ocl_printf(__mem_sanitizer_report,
                                  SanitizerReport.ErrorTy,
                                  SanitizerReport.IsRecover));
  }
  __devicelib_exit();
}

void __asan_internal_report_save(
    uptr ptr, uint32_t as, const char __SYCL_CONSTANT__ *file, uint32_t line,
    const char __SYCL_CONSTANT__ *func, bool is_write, uint32_t access_size,
    MemoryType memory_type, ErrorType error_type, bool is_recover = false) {

  const int Expected = ASAN_REPORT_NONE;
  int Desired = ASAN_REPORT_START;

  // work-group linear id
  const auto WG_LID =
      __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
          __spirv_BuiltInNumWorkgroups.z +
      __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
      __spirv_BuiltInWorkgroupId.z;

  auto &SanitizerReport = ((__SYCL_GLOBAL__ LaunchInfo *)__AsanLaunchInfo)
                              ->Report[WG_LID % ASAN_MAX_NUM_REPORTS];

  if ((is_recover ||
       atomicCompareAndSet(
           &(((__SYCL_GLOBAL__ LaunchInfo *)__AsanLaunchInfo)->ReportFlag), 1,
           0) == 0) &&
      atomicCompareAndSet(&SanitizerReport.Flag, Desired, Expected) ==
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
    SanitizerReport.ErrorTy = error_type;
    SanitizerReport.MemoryTy = memory_type;
    SanitizerReport.IsRecover = is_recover;

    // Show we've done copying
    atomicStore(&SanitizerReport.Flag, ASAN_REPORT_FINISH);

    ASAN_DEBUG(__spirv_ocl_printf(__mem_sanitizer_report,
                                  SanitizerReport.ErrorTy,
                                  SanitizerReport.IsRecover));
  }
  __devicelib_exit();
}

///
/// ASAN Error Reporters
///

MemoryType GetMemoryTypeByShadowValue(int shadow_value) {
  switch (shadow_value) {
  case kUsmDeviceRedzoneMagic:
  case kUsmDeviceDeallocatedMagic:
    return MemoryType::USM_DEVICE;
  case kUsmHostRedzoneMagic:
  case kUsmHostDeallocatedMagic:
    return MemoryType::USM_HOST;
  case kUsmSharedRedzoneMagic:
  case kUsmSharedDeallocatedMagic:
    return MemoryType::USM_SHARED;
  case kPrivateLeftRedzoneMagic:
  case kPrivateMidRedzoneMagic:
  case kPrivateRightRedzoneMagic:
    return MemoryType::PRIVATE;
  case kMemBufferRedzoneMagic:
    return MemoryType::MEM_BUFFER;
  case kSharedLocalRedzoneMagic:
    return MemoryType::LOCAL;
  case kDeviceGlobalRedzoneMagic:
    return MemoryType::DEVICE_GLOBAL;
  default:
    return MemoryType::UNKNOWN;
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

  MemoryType memory_type = GetMemoryTypeByShadowValue(shadow_value);
  ErrorType error_type;

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
    error_type = ErrorType::OUT_OF_BOUNDS;
    break;
  case kUsmDeviceDeallocatedMagic:
  case kUsmHostDeallocatedMagic:
  case kUsmSharedDeallocatedMagic:
    error_type = ErrorType::USE_AFTER_FREE;
    break;
  case kNullPointerRedzoneMagic:
    error_type = ErrorType::NULL_POINTER;
    break;
  default:
    error_type = ErrorType::UNKNOWN;
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

  ErrorType error_type = ErrorType::MISALIGNED;
  MemoryType memory_type = GetMemoryTypeByShadowValue(shadow_value);

  __asan_internal_report_save(addr, as, file, line, func, is_write, size,
                              memory_type, error_type, is_recover);
}

void __asan_report_unknown_device() {
  __asan_internal_report_save(ErrorType::UNKNOWN_DEVICE);
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
/// NOTE:
///   if __AsanLaunchInfo equals 0, the sanitizer is disabled for this launch
///

#define ASAN_REPORT_ERROR_BASE(type, is_write, size, as)                       \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##size##_as##as(                  \
      uptr addr, const char __SYCL_CONSTANT__ *file, uint32_t line,            \
      const char __SYCL_CONSTANT__ *func) {                                    \
    if (!__AsanLaunchInfo)                                                     \
      return;                                                                  \
    if (addr & AlignMask(size)) {                                              \
      __asan_report_misalign_error(addr, as, size, is_write, addr, file, line, \
                                   func);                                      \
    }                                                                          \
    if (__asan_address_is_poisoned(addr, as, size)) {                          \
      __asan_report_access_error(addr, as, size, is_write, addr, file, line,   \
                                 func);                                        \
    }                                                                          \
  }                                                                            \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##size##_as##as##_noabort(        \
      uptr addr, const char __SYCL_CONSTANT__ *file, uint32_t line,            \
      const char __SYCL_CONSTANT__ *func) {                                    \
    if (!__AsanLaunchInfo)                                                     \
      return;                                                                  \
    if (addr & AlignMask(size)) {                                              \
      __asan_report_misalign_error(addr, as, size, is_write, addr, file, line, \
                                   func, true);                                \
    }                                                                          \
    if (__asan_address_is_poisoned(addr, as, size)) {                          \
      __asan_report_access_error(addr, as, size, is_write, addr, file, line,   \
                                 func, true);                                  \
    }                                                                          \
  }

#define ASAN_REPORT_ERROR(type, is_write, size)                                \
  ASAN_REPORT_ERROR_BASE(type, is_write, size, 0)                              \
  ASAN_REPORT_ERROR_BASE(type, is_write, size, 1)                              \
  ASAN_REPORT_ERROR_BASE(type, is_write, size, 2)                              \
  ASAN_REPORT_ERROR_BASE(type, is_write, size, 3)                              \
  ASAN_REPORT_ERROR_BASE(type, is_write, size, 4)

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

#define ASAN_REPORT_ERROR_N_BASE(type, is_write, as)                           \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##N_as##as(                       \
      uptr addr, size_t size, const char __SYCL_CONSTANT__ *file,              \
      uint32_t line, const char __SYCL_CONSTANT__ *func) {                     \
    if (!__AsanLaunchInfo)                                                     \
      return;                                                                  \
    if (auto poisoned_addr = __asan_region_is_poisoned(addr, as, size)) {      \
      __asan_report_access_error(addr, as, size, is_write, poisoned_addr,      \
                                 file, line, func);                            \
    }                                                                          \
  }                                                                            \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##N_as##as##_noabort(             \
      uptr addr, size_t size, const char __SYCL_CONSTANT__ *file,              \
      uint32_t line, const char __SYCL_CONSTANT__ *func) {                     \
    if (!__AsanLaunchInfo)                                                     \
      return;                                                                  \
    if (auto poisoned_addr = __asan_region_is_poisoned(addr, as, size)) {      \
      __asan_report_access_error(addr, as, size, is_write, poisoned_addr,      \
                                 file, line, func, true);                      \
    }                                                                          \
  }

#define ASAN_REPORT_ERROR_N(type, is_write)                                    \
  ASAN_REPORT_ERROR_N_BASE(type, is_write, 0)                                  \
  ASAN_REPORT_ERROR_N_BASE(type, is_write, 1)                                  \
  ASAN_REPORT_ERROR_N_BASE(type, is_write, 2)                                  \
  ASAN_REPORT_ERROR_N_BASE(type, is_write, 3)                                  \
  ASAN_REPORT_ERROR_N_BASE(type, is_write, 4)

ASAN_REPORT_ERROR_N(load, false)
ASAN_REPORT_ERROR_N(store, true)

///
/// ASAN convert memory address to shadow memory address
///

DEVICE_EXTERN_C_NOINLINE uptr __asan_mem_to_shadow(uptr ptr, uint32_t as) {
  if (!__AsanLaunchInfo)
    return 0;

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
  if (!__AsanLaunchInfo)
    return;

  // Since ptr is aligned to ASAN_SHADOW_GRANULARITY,
  // if size != aligned_size, then the buffer tail of ptr is not aligned
  uptr aligned_size = RoundUpTo(size, ASAN_SHADOW_GRANULARITY);

  // Set red zone
  {
    auto shadow_address = MemToShadow(ptr + aligned_size, ADDRESS_SPACE_LOCAL);
    auto count = (size_with_redzone - aligned_size) >> ASAN_SHADOW_SCALE;

    ASAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_local, shadow_address,
                                  shadow_address + count,
                                  (unsigned char)kSharedLocalRedzoneMagic));

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
    *shadow_end = value;

    ASAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_local, shadow_end,
                                  shadow_end, value));
  }
}

static __SYCL_CONSTANT__ const char __mem_unpoison_shadow_static_local_begin[] =
    "[kernel] BEGIN __asan_unpoison_shadow_static_local\n";
static __SYCL_CONSTANT__ const char __mem_unpoison_shadow_static_local_end[] =
    "[kernel] END   __asan_unpoison_shadow_static_local\n";

DEVICE_EXTERN_C_NOINLINE void
__asan_unpoison_shadow_static_local(uptr ptr, size_t size,
                                    size_t size_with_redzone) {
  if (!__AsanLaunchInfo)
    return;

  ASAN_DEBUG(__spirv_ocl_printf(__mem_unpoison_shadow_static_local_begin));

  auto shadow_begin = MemToShadow(ptr + size, ADDRESS_SPACE_LOCAL);
  auto shadow_end = MemToShadow(ptr + size_with_redzone, ADDRESS_SPACE_LOCAL);

  ASAN_DEBUG(
      __spirv_ocl_printf(__mem_set_shadow_local, shadow_begin, shadow_end, 0));

  __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
                         __spv::MemorySemanticsMask::SequentiallyConsistent |
                             __spv::MemorySemanticsMask::WorkgroupMemory);

  while (shadow_begin <= shadow_end) {
    *((__SYCL_GLOBAL__ u8 *)shadow_begin) = 0;
    ++shadow_begin;
  }

  ASAN_DEBUG(__spirv_ocl_printf(__mem_unpoison_shadow_static_local_end));
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
  if (!__AsanLaunchInfo)
    return;

  ASAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_dynamic_local_begin));

  auto *launch_info = (__SYCL_GLOBAL__ const LaunchInfo *)__AsanLaunchInfo;
  if (num_args != launch_info->NumLocalArgs) {
    __spirv_ocl_printf(__mem_report_arg_count_incorrect, num_args,
                       launch_info->NumLocalArgs);
    return;
  }

  uptr *args = (uptr *)ptr;

  for (uint32_t i = 0; i < num_args; ++i) {
    auto *local_arg = &launch_info->LocalArgs[i];
    ASAN_DEBUG(__spirv_ocl_printf(__mem_local_arg, i, local_arg->Size,
                                  local_arg->SizeWithRedZone));

    __asan_set_shadow_static_local(args[i], local_arg->Size,
                                   local_arg->SizeWithRedZone);
  }

  ASAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_dynamic_local_end));
}

static __SYCL_CONSTANT__ const char
    __mem_unpoison_shadow_dynamic_local_begin[] =
        "[kernel] BEGIN __asan_unpoison_shadow_dynamic_local\n";
static __SYCL_CONSTANT__ const char __mem_unpoison_shadow_dynamic_local_end[] =
    "[kernel] END   __asan_unpoison_shadow_dynamic_local\n";

DEVICE_EXTERN_C_NOINLINE void
__asan_unpoison_shadow_dynamic_local(uptr ptr, uint32_t num_args) {
  if (!__AsanLaunchInfo)
    return;

  ASAN_DEBUG(__spirv_ocl_printf(__mem_unpoison_shadow_dynamic_local_begin));

  auto *launch_info = (__SYCL_GLOBAL__ const LaunchInfo *)__AsanLaunchInfo;
  if (num_args != launch_info->NumLocalArgs) {
    __spirv_ocl_printf(__mem_report_arg_count_incorrect, num_args,
                       launch_info->NumLocalArgs);
    return;
  }

  uptr *args = (uptr *)ptr;

  for (uint32_t i = 0; i < num_args; ++i) {
    auto *local_arg = &launch_info->LocalArgs[i];
    ASAN_DEBUG(__spirv_ocl_printf(__mem_local_arg, i, local_arg->Size,
                                  local_arg->SizeWithRedZone));

    __asan_unpoison_shadow_static_local(args[i], local_arg->Size,
                                        local_arg->SizeWithRedZone);
  }

  ASAN_DEBUG(__spirv_ocl_printf(__mem_unpoison_shadow_dynamic_local_end));
}

///
/// ASAN initialize shdadow memory of private memory
///

static __SYCL_CONSTANT__ const char __mem_set_shadow_private_begin[] =
    "[kernel] BEGIN __asan_set_shadow_private\n";
static __SYCL_CONSTANT__ const char __mem_set_shadow_private_end[] =
    "[kernel] END   __asan_set_shadow_private\n";
static __SYCL_CONSTANT__ const char __mem_set_shadow_private[] =
    "[kernel] set_shadow_private(beg=%p, end=%p, val:%02X)\n";

DEVICE_EXTERN_C_NOINLINE void __asan_set_shadow_private(uptr begin, uptr size,
                                                        char val) {
  if (!__AsanLaunchInfo)
    return;

  ASAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_private_begin));

  auto *launch_info = (__SYCL_GLOBAL__ const LaunchInfo *)__AsanLaunchInfo;
  if (launch_info->PrivateShadowOffset == 0)
    return;

  ASAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_private, (void *)begin,
                                (void *)(begin + size), val & 0xFF));

  for (size_t i = 0; i < size; i++)
    ((__SYCL_GLOBAL__ u8 *)begin)[i] = val;

  ASAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_private_end));
}

#endif // __SPIR__ || __SPIRV__
