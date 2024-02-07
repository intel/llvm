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

#include "include/device-sanitizer-report.hpp"
#include "include/sanitizer_device_utils.hpp"

#include <cstddef>
#include <cstdint>

using uptr = uintptr_t;
using s8 = char;
using u8 = unsigned char;
using s16 = short;
using u16 = unsigned short;

#define ASAN_SHADOW_SCALE 3
#define ASAN_SHADOW_GRANULARITY (1ULL << ASAN_SHADOW_SCALE)

DeviceGlobal<uptr> __AsanShadowMemoryGlobalStart;
DeviceGlobal<uptr> __AsanShadowMemoryGlobalEnd;
DeviceGlobal<uptr> __AsanShadowMemoryLocalStart;
DeviceGlobal<uptr> __AsanShadowMemoryLocalEnd;

DeviceGlobal<DeviceSanitizerReport> __DeviceSanitizerReportMem;

DeviceGlobal<DeviceType> __DeviceType;

#if defined(__SPIR__)

#ifdef __SYCL_DEVICE_ONLY__
extern SYCL_EXTERNAL int
__spirv_ocl_printf(const __SYCL_CONSTANT__ char *Format, ...);

extern SYCL_EXTERNAL __SYCL_GLOBAL__ void *
__spirv_GenericCastToPtrExplicit_ToGlobal(void *, int);
extern SYCL_EXTERNAL __SYCL_LOCAL__ void *
__spirv_GenericCastToPtrExplicit_ToLocal(void *, int);
extern SYCL_EXTERNAL __SYCL_PRIVATE__ void *
__spirv_GenericCastToPtrExplicit_ToPrivate(void *, int);
#endif

// These magic values are written to shadow for better error
// reporting.
const int kUsmDeviceRedzoneMagic = (char)0x81;
const int kUsmHostRedzoneMagic = (char)0x82;
const int kUsmSharedRedzoneMagic = (char)0x83;
const int kMemBufferRedzoneMagic = (char)0x84;

const int kUsmDeviceDeallocatedMagic = (char)0x91;
const int kUsmHostDeallocatedMagic = (char)0x92;
const int kUsmSharedDeallocatedMagic = (char)0x93;

const int kSharedLocalRedzoneMagic = (char)0xa1;

// Same with Asan Stack
const int kPrivateLeftRedzoneMagic = (char)0xf1;
const int kPrivateMidRedzoneMagic = (char)0xf2;
const int kPrivateRightRedzoneMagic = (char)0xf3;

static const __SYCL_CONSTANT__ char __asan_shadow_value_start[] =
    "%p(%d) -> %p:";
static const __SYCL_CONSTANT__ char __asan_shadow_value[] = " %02X";
static const __SYCL_CONSTANT__ char __asan_current_shadow_value[] = ">%02X";
static const __SYCL_CONSTANT__ char __newline[] = "\n";

static const __SYCL_CONSTANT__ char __global_shadow_out_of_bound[] =
    "ERROR: Global shadow memory out-of-bound (ptr: %p -> %p, base: %p)\n";
static const __SYCL_CONSTANT__ char __local_shadow_out_of_bound[] =
    "ERROR: Local shadow memory out-of-bound (ptr: %p -> %p, wg: %d, base: "
    "%p)\n";

static const __SYCL_CONSTANT__ char __unsupport_device_type[] =
    "ERROR: Unsupport device type: %d\n";

#define ASAN_REPORT_NONE 0
#define ASAN_REPORT_START 1
#define ASAN_REPORT_FINISH 2

#define AS_PRIVATE 0
#define AS_GLOBAL 1
#define AS_CONSTANT 2
#define AS_LOCAL 3
#define AS_GENERIC 4

namespace {

__SYCL_GLOBAL__ void *ToGlobal(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToGlobal(ptr, 5);
}
__SYCL_LOCAL__ void *ToLocal(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToLocal(ptr, 4);
}
__SYCL_PRIVATE__ void *ToPrivate(void *ptr) {
  return __spirv_GenericCastToPtrExplicit_ToPrivate(ptr, 7);
}

inline uptr MemToShadow_CPU(uptr addr, int32_t as) {
  return __AsanShadowMemoryGlobalStart + (addr >> 3);
}

inline uptr MemToShadow_DG2(uptr addr, int32_t as) {
  uptr shadow_ptr = 0;
  if (addr & (~0xffffffffffff)) {
    shadow_ptr =
        (((addr & 0xffffffffffff) >> 3) + __AsanShadowMemoryGlobalStart) |
        (~0xffffffffffff);
  } else {
    shadow_ptr = (addr >> 3) + __AsanShadowMemoryGlobalStart;
  }

  if (shadow_ptr > __AsanShadowMemoryGlobalEnd) {
    __spirv_ocl_printf(__global_shadow_out_of_bound, addr, shadow_ptr);
  }

  return shadow_ptr;
}

inline uptr MemToShadow_PVC(uptr addr, int32_t as) {
  uptr shadow_ptr = 0;

  if (as == AS_GENERIC) {
    if ((shadow_ptr = (uptr)ToGlobal((void *)addr))) {
      as = AS_GLOBAL;
    } else if ((shadow_ptr = (uptr)ToPrivate((void *)addr))) {
      as = AS_PRIVATE;
    } else if ((shadow_ptr = (uptr)ToLocal((void *)addr))) {
      as = AS_LOCAL;
    } else {
      return 0;
    }
  }

  if (as == AS_PRIVATE) {            // private
  } else if (as == AS_GLOBAL) {      // global
    if (addr & 0xFF00000000000000) { // Device USM
      shadow_ptr = __AsanShadowMemoryGlobalStart + 0x200000000000 +
                   ((addr & 0xFFFFFFFFFFFF) >> 3);
    } else { // Only consider 47bit VA
      shadow_ptr =
          __AsanShadowMemoryGlobalStart + ((addr & 0x7FFFFFFFFFFF) >> 3);
    }

    if (shadow_ptr > __AsanShadowMemoryGlobalEnd) {
      __spirv_ocl_printf(__global_shadow_out_of_bound, addr, shadow_ptr,
                         (uptr)__AsanShadowMemoryGlobalStart);
      shadow_ptr = 0;
    }
  } else if (as == AS_CONSTANT) { // constant
  } else if (as == AS_LOCAL) {    // local
    // The size of SLM is 128KB on PVC
    constexpr unsigned slm_size = 128 * 1024;
    const auto wg_lid =
        __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
            __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.z;

    shadow_ptr = __AsanShadowMemoryLocalStart + ((wg_lid * slm_size) >> 3) +
                 ((addr & (slm_size - 1)) >> 3);

    if (shadow_ptr > __AsanShadowMemoryLocalEnd) {
      __spirv_ocl_printf(__local_shadow_out_of_bound, addr, shadow_ptr, wg_lid,
                         (uptr)__AsanShadowMemoryLocalStart);
      shadow_ptr = 0;
    }
  }

  return shadow_ptr;
}

static const __SYCL_CONSTANT__ char __asan_shadow_value[] =
    "%p(%d) -> %p: %02X\n";

inline uptr MemToShadow(uptr addr, int32_t as) {
  uptr shadow_ptr = 0;

  if (__DeviceType == DeviceType::CPU) {
    shadow_ptr = MemToShadow_CPU(addr, as);
  } else if (__DeviceType == DeviceType::GPU_PVC) {
    shadow_ptr = MemToShadow_PVC(addr, as);
  } else {
    __spirv_ocl_printf(__unsupport_device_type, (int)__DeviceType);
    return shadow_ptr;
  }

  // __spirv_ocl_printf(__asan_shadow_value, addr, as, shadow_ptr,
  //                    shadow_ptr ? *(u8 *)shadow_ptr : 0xff);

  return shadow_ptr;
}

inline constexpr uptr RoundUpTo(uptr size, uptr boundary) {
  return (size + boundary - 1) & ~(boundary - 1);
}

inline constexpr uptr RoundDownTo(uptr x, uptr boundary) {
  return x & ~(boundary - 1);
}

bool MemIsZero(const char *beg, uptr size) {
  const char *end = beg + size;
  uptr *aligned_beg = (uptr *)RoundUpTo((uptr)beg, sizeof(uptr));
  uptr *aligned_end = (uptr *)RoundDownTo((uptr)end, sizeof(uptr));
  uptr all = 0;
  // Prologue.
  for (const char *mem = beg; mem < (char *)aligned_beg && mem < end; mem++)
    all |= *mem;
  // Aligned loop.
  for (; aligned_beg < aligned_end; aligned_beg++)
    all |= *aligned_beg;
  // Epilogue.
  if ((char *)aligned_end >= beg) {
    for (const char *mem = (char *)aligned_end; mem < end; mem++)
      all |= *mem;
  }
  return all == 0;
}

void print_shadow_memory(uptr addr, int32_t as) {
  uptr shadow_address = MemToShadow(addr, as);
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

} // namespace

bool __asan_region_is_value(uptr addr, int32_t as, std::size_t size,
                            char value) {
  if (size == 0)
    return true;
  while (size--) {
    char *shadow = (char *)MemToShadow(addr, as);
    if (*shadow != value) {
      return false;
    }
    ++addr;
  }
  return true;
}

static void __asan_internal_report_save(
    uptr ptr, int32_t as, const char __SYCL_CONSTANT__ *file, int32_t line,
    const char __SYCL_CONSTANT__ *func, bool is_write, uint32_t access_size,
    DeviceSanitizerMemoryType memory_type, DeviceSanitizerErrorType error_type,
    bool is_recover = false) {

  const int Expected = ASAN_REPORT_NONE;
  int Desired = ASAN_REPORT_START;
  if (atomicCompareAndSet(&__DeviceSanitizerReportMem.get().Flag, Desired,
                          Expected) == Expected) {

    int FileLength = 0;
    int FuncLength = 0;

    if (file)
      for (auto *C = file; *C != '\0'; ++C, ++FileLength)
        ;
    if (func)
      for (auto *C = func; *C != '\0'; ++C, ++FuncLength)
        ;

    int MaxFileIdx = sizeof(__DeviceSanitizerReportMem.get().File) - 1;
    int MaxFuncIdx = sizeof(__DeviceSanitizerReportMem.get().Func) - 1;

    if (FileLength < MaxFileIdx)
      MaxFileIdx = FileLength;
    if (FuncLength < MaxFuncIdx)
      MaxFuncIdx = FuncLength;

    for (int Idx = 0; Idx < MaxFileIdx; ++Idx)
      __DeviceSanitizerReportMem.get().File[Idx] = file[Idx];
    __DeviceSanitizerReportMem.get().File[MaxFileIdx] = '\0';

    for (int Idx = 0; Idx < MaxFuncIdx; ++Idx)
      __DeviceSanitizerReportMem.get().Func[Idx] = func[Idx];
    __DeviceSanitizerReportMem.get().Func[MaxFuncIdx] = '\0';

    __DeviceSanitizerReportMem.get().Line = line;
    __DeviceSanitizerReportMem.get().GID0 = __spirv_GlobalInvocationId_x();
    __DeviceSanitizerReportMem.get().GID1 = __spirv_GlobalInvocationId_y();
    __DeviceSanitizerReportMem.get().GID2 = __spirv_GlobalInvocationId_z();
    __DeviceSanitizerReportMem.get().LID0 = __spirv_LocalInvocationId_x();
    __DeviceSanitizerReportMem.get().LID1 = __spirv_LocalInvocationId_y();
    __DeviceSanitizerReportMem.get().LID2 = __spirv_LocalInvocationId_z();

    __DeviceSanitizerReportMem.get().Addr = ptr;
    __DeviceSanitizerReportMem.get().IsWrite = is_write;
    __DeviceSanitizerReportMem.get().AccessSize = access_size;
    __DeviceSanitizerReportMem.get().ErrorType = error_type;
    __DeviceSanitizerReportMem.get().MemoryType = memory_type;
    __DeviceSanitizerReportMem.get().IsRecover = is_recover;

    // Show we've done copying
    atomicStore(&__DeviceSanitizerReportMem.get().Flag, ASAN_REPORT_FINISH);
  }
}

///
/// ASAN Error Reporters
///

void __asan_report_access_error(uptr addr, int32_t as, size_t size,
                                bool is_write, uptr poisoned_addr,
                                const char __SYCL_CONSTANT__ *file,
                                int32_t line,
                                const char __SYCL_CONSTANT__ *func,
                                bool is_recover = false) {
  // Check Error Type
  s8 *shadow_address = (s8 *)MemToShadow(poisoned_addr, as);
  int shadow_value = *shadow_address;
  if (shadow_value > 0) {
    shadow_value = *(shadow_address + 1);
  }
  // FIXME: check if shadow_address out-of-bound

  DeviceSanitizerMemoryType memory_type;
  DeviceSanitizerErrorType error_type;

  switch (shadow_value) {
  case kUsmDeviceRedzoneMagic:
    memory_type = DeviceSanitizerMemoryType::USM_DEVICE;
    error_type = DeviceSanitizerErrorType::OUT_OF_BOUNDS;
    break;
  case kUsmHostRedzoneMagic:
    memory_type = DeviceSanitizerMemoryType::USM_HOST;
    error_type = DeviceSanitizerErrorType::OUT_OF_BOUNDS;
    break;
  case kUsmSharedRedzoneMagic:
    memory_type = DeviceSanitizerMemoryType::USM_SHARED;
    error_type = DeviceSanitizerErrorType::OUT_OF_BOUNDS;
    break;
  case kUsmDeviceDeallocatedMagic:
    memory_type = DeviceSanitizerMemoryType::USM_DEVICE;
    error_type = DeviceSanitizerErrorType::USE_AFTER_FREE;
    break;
  case kUsmHostDeallocatedMagic:
    memory_type = DeviceSanitizerMemoryType::USM_HOST;
    error_type = DeviceSanitizerErrorType::USE_AFTER_FREE;
    break;
  case kUsmSharedDeallocatedMagic:
    memory_type = DeviceSanitizerMemoryType::USM_SHARED;
    error_type = DeviceSanitizerErrorType::USE_AFTER_FREE;
    break;
  case kPrivateLeftRedzoneMagic:
  case kPrivateMidRedzoneMagic:
  case kPrivateRightRedzoneMagic:
    memory_type = DeviceSanitizerMemoryType::PRIVATE;
    error_type = DeviceSanitizerErrorType::OUT_OF_BOUNDS;
    break;
  case kMemBufferRedzoneMagic:
    memory_type = DeviceSanitizerMemoryType::MEM_BUFFER;
    error_type = DeviceSanitizerErrorType::OUT_OF_BOUNDS;
    break;
  case kSharedLocalRedzoneMagic:
    memory_type = DeviceSanitizerMemoryType::LOCAL;
    error_type = DeviceSanitizerErrorType::OUT_OF_BOUNDS;
    break;
  default:
    memory_type = DeviceSanitizerMemoryType::UNKNOWN;
    error_type = DeviceSanitizerErrorType::UNKNOWN;
  }

  __asan_internal_report_save(addr, as, file, line, func, is_write, size,
                              memory_type, error_type, is_recover);
}

///
/// Check if memory is poisoned
///

// NOTE: size < 8
inline int __asan_address_is_poisoned(uptr a, int32_t as, size_t size) {
  auto *shadow_address = (s8 *)MemToShadow(a, as);
  if (shadow_address) {
    auto shadow_value = *shadow_address;
    if (shadow_value) {
      s8 last_accessed_byte = (a & (ASAN_SHADOW_GRANULARITY - 1)) + size - 1;
      return (last_accessed_byte >= shadow_value);
    }
  }
  return false;
}

// NOTE: size = 1
inline int __asan_address_is_poisoned(uptr a, int32_t as) {
  return __asan_address_is_poisoned(a, as, 1);
}

inline uptr __asan_region_is_poisoned(uptr beg, int32_t as, size_t size) {
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
       MemIsZero((const char *)shadow_beg, shadow_end - shadow_beg)))
    return 0;

  // The fast check failed, so we have a poisoned byte somewhere.
  // Find it slowly.
  for (; beg < end; beg++)
    if (__asan_address_is_poisoned(beg, as))
      return beg;

  return 0;
}

///
/// ASAN Load/Store Report Built-ins
///

#define ASAN_REPORT_ERROR(type, is_write, size)                                \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##size(                           \
      uptr addr, int32_t as, const char __SYCL_CONSTANT__ *file, int32_t line, \
      const char __SYCL_CONSTANT__ *func) {                                    \
    if (__asan_address_is_poisoned(addr, as, size)) {                          \
      __asan_report_access_error(addr, as, size, is_write, addr, file, line,   \
                                 func);                                        \
    }                                                                          \
  }                                                                            \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##size##_noabort(                 \
      uptr addr, int32_t as, const char __SYCL_CONSTANT__ *file, int32_t line, \
      const char __SYCL_CONSTANT__ *func) {                                    \
    if (__asan_address_is_poisoned(addr, as, size)) {                          \
      __asan_report_access_error(addr, as, size, is_write, addr, file, line,   \
                                 func, true);                                  \
    }                                                                          \
  }

ASAN_REPORT_ERROR(load, false, 1)
ASAN_REPORT_ERROR(load, false, 2)
ASAN_REPORT_ERROR(load, false, 4)
ASAN_REPORT_ERROR(store, true, 1)
ASAN_REPORT_ERROR(store, true, 2)
ASAN_REPORT_ERROR(store, true, 4)

#define ASAN_REPORT_ERROR_BYTE(type, is_write, size)                           \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##size(                           \
      uptr addr, int32_t as, const char __SYCL_CONSTANT__ *file, int32_t line, \
      const char __SYCL_CONSTANT__ *func) {                                    \
    u##size *shadow_address = (u##size *)MemToShadow(addr, as);                \
    if (shadow_address && *shadow_address) {                                   \
      __asan_report_access_error(addr, as, size, is_write, addr, file, line,   \
                                 func);                                        \
    }                                                                          \
  }                                                                            \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##size##_noabort(                 \
      uptr addr, int32_t as, const char __SYCL_CONSTANT__ *file, int32_t line, \
      const char __SYCL_CONSTANT__ *func) {                                    \
    u##size *shadow_address = (u##size *)MemToShadow(addr, as);                \
    if (shadow_address && *shadow_address) {                                   \
      __asan_report_access_error(addr, as, size, is_write, addr, file, line,   \
                                 func, true);                                  \
    }                                                                          \
  }

ASAN_REPORT_ERROR_BYTE(load, false, 8)
ASAN_REPORT_ERROR_BYTE(load, false, 16)
ASAN_REPORT_ERROR_BYTE(store, true, 8)
ASAN_REPORT_ERROR_BYTE(store, true, 16)

#define ASAN_REPORT_ERROR_N(type, is_write)                                    \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##N(                              \
      uptr addr, size_t size, int32_t as, const char __SYCL_CONSTANT__ *file,  \
      int32_t line, const char __SYCL_CONSTANT__ *func) {                      \
    if (auto poisoned_addr = __asan_region_is_poisoned(addr, as, size)) {      \
      __asan_report_access_error(addr, as, size, is_write, poisoned_addr,      \
                                 file, line, func);                            \
    }                                                                          \
  }                                                                            \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##N_noabort(                      \
      uptr addr, size_t size, int32_t as, const char __SYCL_CONSTANT__ *file,  \
      int32_t line, const char __SYCL_CONSTANT__ *func) {                      \
    if (auto poisoned_addr = __asan_region_is_poisoned(addr, as, size)) {      \
      __asan_report_access_error(addr, as, size, is_write, poisoned_addr,      \
                                 file, line, func, true);                      \
    }                                                                          \
  }

ASAN_REPORT_ERROR_N(load, false)
ASAN_REPORT_ERROR_N(store, true)

DEVICE_EXTERN_C_NOINLINE void
__asan_set_shadow_local_memory(uptr ptr, size_t size,
                               size_t size_with_redzone) {
  uptr aligned_size = RoundUpTo(size, ASAN_SHADOW_GRANULARITY);

  {
    auto shadow_address = MemToShadow(ptr + aligned_size, AS_LOCAL);
    auto count = (size_with_redzone - aligned_size) / ASAN_SHADOW_GRANULARITY;
    for (size_t i = 0; i < count; ++i) {
      ((u8 *)shadow_address)[i] = kSharedLocalRedzoneMagic;
    }
  }

  if (size != aligned_size) {
    auto user_end = ptr + size - 1;
    auto *shadow_end = (s8 *)MemToShadow(user_end, AS_LOCAL);
    *shadow_end = user_end - RoundDownTo(user_end, ASAN_SHADOW_GRANULARITY);
  }
}

#endif
