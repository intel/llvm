//==--- asan_rtl.cpp - device address sanitizer runtime library ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/asan_rtl.hpp"
#include "asan/asan_libdevice.hpp"

// Save the pointer to LaunchInfo
__SYCL_GLOBAL__ uptr *__SYCL_LOCAL__ __AsanLaunchInfo;

#if defined(__SPIR__) || defined(__SPIRV__)

static const __SYCL_CONSTANT__ char __asan_shadow_value_start[] =
    "[kernel] %p(%d) -> %p:";
static const __SYCL_CONSTANT__ char __asan_shadow_value[] = " %02X";
static const __SYCL_CONSTANT__ char __asan_current_shadow_value[] = ">%02X";
static const __SYCL_CONSTANT__ char __newline[] = "\n";

static const __SYCL_CONSTANT__ char __global_shadow_out_of_bound[] =
    "[kernel] Global shadow memory out-of-bound (ptr: %p -> %p, base: %p)\n";
static const __SYCL_CONSTANT__ char __local_shadow_out_of_bound[] =
    "[kernel] Local shadow memory out-of-bound (ptr: %p -> %p, wid: %llu, "
    "base: %p)\n";
static const __SYCL_CONSTANT__ char __private_shadow_out_of_bound[] =
    "[kernel] Private shadow memory out-of-bound (ptr: %p -> %p, sid: %llu, "
    "base: %p)\n";

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
    auto launch_info =                                                         \
        (__SYCL_GLOBAL__ const AsanRuntimeData *)__AsanLaunchInfo;             \
    if (launch_info->Debug) {                                                  \
      X;                                                                       \
    }                                                                          \
  } while (false)

namespace {

struct DebugInfo {
  uptr addr;
  uint32_t as;
  size_t access_size;
  bool is_write;
  const char __SYCL_CONSTANT__ *file;
  const char __SYCL_CONSTANT__ *func;
  uint32_t line;
};

void ReportUnknownDevice(const DebugInfo *debug);
void PrintShadowMemory(uptr addr, uptr shadow_address, uint32_t as);

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
    addr = (uptr)ToGlobal((void *)old);
  }
  ASAN_DEBUG(__spirv_ocl_printf(__generic_to, old, addr, as));
}

inline uptr MemToShadow_CPU(uptr addr) {
  auto launch_info = (__SYCL_GLOBAL__ const AsanRuntimeData *)__AsanLaunchInfo;
  return launch_info->GlobalShadowOffset + (addr >> ASAN_SHADOW_SCALE);
}

inline uptr MemToShadow_DG2(uptr addr, uint32_t as,
                            [[maybe_unused]] const DebugInfo *debug) {
  if (as == ADDRESS_SPACE_GENERIC) {
    ConvertGenericPointer(addr, as);
  }

  auto launch_info = (__SYCL_GLOBAL__ const AsanRuntimeData *)__AsanLaunchInfo;
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
    const auto shadow_offset = launch_info->LocalShadowOffset;
    const size_t wid = WorkGroupLinearId();
    if (shadow_offset == 0 || wid >= ASAN_MAX_WG_LOCAL) {
      return 0;
    }

    // The size of SLM is 64KB on DG2
    constexpr unsigned slm_size = 64 * 1024;

    auto shadow_ptr = shadow_offset + ((wid * slm_size) >> ASAN_SHADOW_SCALE) +
                      ((addr & (slm_size - 1)) >> ASAN_SHADOW_SCALE);

    ASAN_DEBUG(const auto shadow_offset_end = launch_info->LocalShadowOffsetEnd;
               if (shadow_ptr > shadow_offset_end) {
                 __spirv_ocl_printf(__local_shadow_out_of_bound, addr,
                                    shadow_ptr, wid, (uptr)shadow_offset);
                 return 0;
               });
    return shadow_ptr;
  } else if (as == ADDRESS_SPACE_PRIVATE) { // private
    const auto shadow_offset = launch_info->PrivateShadowOffset;
    const size_t sid = SubGroupLinearId();
    if (shadow_offset == 0 || sid >= ASAN_MAX_SG_PRIVATE) {
      return 0;
    }

    const uptr private_base = launch_info->PrivateBase[sid];

    // FIXME: The recorded private_base may not be the most bottom one,
    // ideally there should have a build-in to get this information
    if (addr < private_base) {
      return 0;
    }

    uptr shadow_ptr = shadow_offset +
                      ((sid * ASAN_PRIVATE_SIZE) >> ASAN_SHADOW_SCALE) +
                      ((addr - private_base) >> ASAN_SHADOW_SCALE);

    const auto shadow_offset_end = launch_info->PrivateShadowOffsetEnd;
    if (shadow_ptr > shadow_offset_end) {
      __spirv_ocl_printf(__private_shadow_out_of_bound, addr, shadow_ptr, sid,
                         private_base);
      return 0;
    };

    return shadow_ptr;
  }

  return 0;
}

inline uptr MemToShadow_PVC(uptr addr, uint32_t as,
                            [[maybe_unused]] const DebugInfo *debug) {
  if (as == ADDRESS_SPACE_GENERIC) {
    ConvertGenericPointer(addr, as);
  }

  auto launch_info = (__SYCL_GLOBAL__ const AsanRuntimeData *)__AsanLaunchInfo;
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
    const auto shadow_offset = launch_info->LocalShadowOffset;
    const auto wid = WorkGroupLinearId();
    if (shadow_offset == 0 || wid >= ASAN_MAX_WG_LOCAL) {
      return 0;
    }

    // The size of SLM is 128KB on PVC
    constexpr unsigned SLM_SIZE = 128 * 1024;

    uptr shadow_ptr = shadow_offset + ((wid * SLM_SIZE) >> ASAN_SHADOW_SCALE) +
                      ((addr & (SLM_SIZE - 1)) >> ASAN_SHADOW_SCALE);

    ASAN_DEBUG(const auto shadow_offset_end = launch_info->LocalShadowOffsetEnd;
               if (shadow_ptr > shadow_offset_end) {
                 __spirv_ocl_printf(__local_shadow_out_of_bound, addr,
                                    shadow_ptr, wid, (uptr)shadow_offset);
                 return 0;
               });
    return shadow_ptr;
  } else if (as == ADDRESS_SPACE_PRIVATE) { // private
    const auto shadow_offset = launch_info->PrivateShadowOffset;
    const size_t sid = SubGroupLinearId();
    if (shadow_offset == 0 || sid >= ASAN_MAX_SG_PRIVATE) {
      return 0;
    }

    const uptr private_base = launch_info->PrivateBase[sid];

    // FIXME: The recorded private_base may not be the most bottom one,
    // ideally there should have a build-in to get this information
    if (addr < private_base) {
      return 0;
    }

    uptr shadow_ptr = shadow_offset +
                      ((sid * ASAN_PRIVATE_SIZE) >> ASAN_SHADOW_SCALE) +
                      ((addr - private_base) >> ASAN_SHADOW_SCALE);

    const auto shadow_offset_end = launch_info->PrivateShadowOffsetEnd;
    if (shadow_ptr > shadow_offset_end) {
      __spirv_ocl_printf(__private_shadow_out_of_bound, addr, shadow_ptr, sid,
                         private_base);
      return 0;
    };

    return shadow_ptr;
  }

  return 0;
}

inline uptr MemToShadow(uptr addr, uint32_t as,
                        const DebugInfo *debug = nullptr) {
  uptr shadow_ptr = 0;

#if defined(__LIBDEVICE_PVC__)
  shadow_ptr = MemToShadow_PVC(addr, as, debug);
#elif defined(__LIBDEVICE_CPU__)
  shadow_ptr = MemToShadow_CPU(addr);
#elif defined(__LIBDEVICE_DG2__)
  shadow_ptr = MemToShadow_DG2(addr, as, debug);
#else
  if (GetDeviceTy() == DeviceType::CPU) {
    shadow_ptr = MemToShadow_CPU(addr);
  } else if (GetDeviceTy() == DeviceType::GPU_PVC) {
    shadow_ptr = MemToShadow_PVC(addr, as, debug);
  } else if (GetDeviceTy() == DeviceType::GPU_DG2) {
    shadow_ptr = MemToShadow_DG2(addr, as, debug);
  } else {
    ASAN_DEBUG(__spirv_ocl_printf(__asan_print_unsupport_device_type,
                                  (int)GetDeviceTy()));
    ReportUnknownDevice(debug);
    return 0;
  }
#endif

  ASAN_DEBUG(
      if (shadow_ptr) {
        if (as == ADDRESS_SPACE_PRIVATE)
          PrintShadowMemory(addr, shadow_ptr, as);
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

void Exit(ErrorType error_type) {
  // Exit the kernel when we really need it
  switch (error_type) {
  case ErrorType::UNKNOWN:
  case ErrorType::UNKNOWN_DEVICE:
  case ErrorType::NULL_POINTER:
    __devicelib_exit();
    break;
  default:
    break;
  }
}

void SaveReport(ErrorType error_type, MemoryType memory_type, bool is_recover,
                const DebugInfo *debug) {
  const int Expected = ASAN_REPORT_NONE;
  int Desired = ASAN_REPORT_START;

  const size_t wid = WorkGroupLinearId();
  auto &SanitizerReport = ((__SYCL_GLOBAL__ AsanRuntimeData *)__AsanLaunchInfo)
                              ->Report[wid % ASAN_MAX_NUM_REPORTS];

  if ((is_recover ||
       atomicCompareAndSet(
           &(((__SYCL_GLOBAL__ AsanRuntimeData *)__AsanLaunchInfo)->ReportFlag),
           1, 0) == 0) &&
      atomicCompareAndSet(&SanitizerReport.Flag, Desired, Expected) ==
          Expected) {

    const uptr ptr = debug ? debug->addr : 0;
    const uint32_t as = debug ? debug->as : 5;
    const char __SYCL_CONSTANT__ *file = debug ? debug->file : nullptr;
    const uint32_t line = debug ? debug->line : 0;
    const char __SYCL_CONSTANT__ *func = debug ? debug->func : nullptr;
    const bool is_write = debug ? debug->is_write : false;
    const uint32_t access_size = debug ? debug->access_size : 0;

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
    SanitizerReport.GID0 = __spirv_BuiltInGlobalInvocationId(0);
    SanitizerReport.GID1 = __spirv_BuiltInGlobalInvocationId(1);
    SanitizerReport.GID2 = __spirv_BuiltInGlobalInvocationId(2);
    SanitizerReport.LID0 = __spirv_BuiltInLocalInvocationId(0);
    SanitizerReport.LID1 = __spirv_BuiltInLocalInvocationId(1);
    SanitizerReport.LID2 = __spirv_BuiltInLocalInvocationId(2);

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
  Exit(error_type);
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

void ReportAccessError(uptr poisoned_addr, uint32_t as, bool is_recover,
                       const DebugInfo *debug) {
  // Check Error Type
  auto *shadow_address =
      (__SYCL_GLOBAL__ s8 *)MemToShadow(poisoned_addr, as, debug);
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

  SaveReport(error_type, memory_type, is_recover, debug);
}

void ReportMisalignError(uptr addr, uint32_t as, bool is_recover,
                         const DebugInfo *debug) {

  auto *shadow = (__SYCL_GLOBAL__ s8 *)MemToShadow(addr, as, debug);
  if (!shadow)
    return;

  while (*shadow >= 0) {
    ++shadow;
  }
  int shadow_value = *shadow;

  SaveReport(ErrorType::MISALIGNED, GetMemoryTypeByShadowValue(shadow_value),
             is_recover, debug);
}

void ReportUnknownDevice(const DebugInfo *debug) {
  SaveReport(ErrorType::UNKNOWN_DEVICE, MemoryType::UNKNOWN, false, debug);
}

///
/// ASan utils
///

void PrintShadowMemory(uptr addr, uptr shadow_address, uint32_t as) {
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

// NOTE: size <= 16
inline int IsAddressPoisoned(uptr a, uint32_t as, size_t size,
                             const DebugInfo *debug) {
  auto *shadow_address = (__SYCL_GLOBAL__ s8 *)MemToShadow(a, as, debug);
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

inline uptr IsRegionPoisoned(uptr beg, uint32_t as, size_t size,
                             const DebugInfo *debug) {
  if (!size)
    return 0;

  uptr end = beg + size;
  uptr aligned_b = RoundUpTo(beg, ASAN_SHADOW_GRANULARITY);
  uptr aligned_e = RoundDownTo(end, ASAN_SHADOW_GRANULARITY);

  uptr shadow_beg = MemToShadow(aligned_b, as, debug);
  if (!shadow_beg) {
    return 0;
  }
  uptr shadow_end = MemToShadow(aligned_e, as, debug);
  if (!shadow_end) {
    return 0;
  }

  // First check the first and the last application bytes,
  // then check the ASAN_SHADOW_GRANULARITY-aligned region by calling
  // MemIsZero on the corresponding shadow.
  if (!IsAddressPoisoned(beg, as, 1, debug) &&
      !IsAddressPoisoned(end - 1, as, 1, debug) &&
      (shadow_end <= shadow_beg ||
       MemIsZero((__SYCL_GLOBAL__ const char *)shadow_beg,
                 shadow_end - shadow_beg)))
    return 0;

  // The fast check failed, so we have a poisoned byte somewhere.
  // Find it slowly.
  for (; beg < end; beg++)
    if (IsAddressPoisoned(beg, as, 1, debug))
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
    DebugInfo debug{addr, as, size, is_write, file, func, line};               \
    if (addr & AlignMask(size)) {                                              \
      ReportMisalignError(addr, as, false, &debug);                            \
    }                                                                          \
    if (IsAddressPoisoned(addr, as, size, &debug)) {                           \
      ReportAccessError(addr, as, false, &debug);                              \
    }                                                                          \
  }                                                                            \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##size##_as##as##_noabort(        \
      uptr addr, const char __SYCL_CONSTANT__ *file, uint32_t line,            \
      const char __SYCL_CONSTANT__ *func) {                                    \
    if (!__AsanLaunchInfo)                                                     \
      return;                                                                  \
    DebugInfo debug{addr, as, size, is_write, file, func, line};               \
    if (addr & AlignMask(size)) {                                              \
      ReportMisalignError(addr, as, true, &debug);                             \
    }                                                                          \
    if (IsAddressPoisoned(addr, as, size, &debug)) {                           \
      ReportAccessError(addr, as, true, &debug);                               \
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
    DebugInfo debug{addr, as, size, is_write, file, func, line};               \
    if (auto poisoned_addr = IsRegionPoisoned(addr, as, size, &debug)) {       \
      ReportAccessError(poisoned_addr, as, false, &debug);                     \
    }                                                                          \
  }                                                                            \
  DEVICE_EXTERN_C_NOINLINE void __asan_##type##N_as##as##_noabort(             \
      uptr addr, size_t size, const char __SYCL_CONSTANT__ *file,              \
      uint32_t line, const char __SYCL_CONSTANT__ *func) {                     \
    if (!__AsanLaunchInfo)                                                     \
      return;                                                                  \
    DebugInfo debug{addr, as, size, is_write, file, func, line};               \
    if (auto poisoned_addr = IsRegionPoisoned(addr, as, size, &debug)) {       \
      ReportAccessError(poisoned_addr, as, true, &debug);                      \
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
/// This function is used only for getting shadow address of private memory so
/// that we can poison them later. And the current implmentation require the ptr
/// to be aligned with ASAN_SHADOW_GRANULARITY. If not aligned, the subsequent
/// poisoning will not work correctly.
///
static __SYCL_CONSTANT__ const char __asan_mem_to_shadow_unaligned_msg[] =
    "[kernel] __asan_mem_to_shadow() unaligned address: %p\n";

DEVICE_EXTERN_C_NOINLINE uptr __asan_mem_to_shadow(uptr ptr, uint32_t as) {
  if (!__AsanLaunchInfo)
    return 0;

  // If ptr is not aligned, then it should be considered as implementation
  // error. Print a warning for it.
  if (ptr & AlignMask(ASAN_SHADOW_GRANULARITY)) {
    __spirv_ocl_printf(__asan_mem_to_shadow_unaligned_msg, (void *)ptr);
  }

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
    if (!shadow_address)
      return;

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
    if (!shadow_end)
      return;

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
  if (!shadow_begin)
    return;
  auto shadow_end = MemToShadow(ptr + size_with_redzone, ADDRESS_SPACE_LOCAL);
  if (!shadow_end)
    return;

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

  auto *launch_info = (__SYCL_GLOBAL__ const AsanRuntimeData *)__AsanLaunchInfo;
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

  auto *launch_info = (__SYCL_GLOBAL__ const AsanRuntimeData *)__AsanLaunchInfo;
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

// We outline the function of setting shadow memory of private memory, because
// it may allocate failed on UR
DEVICE_EXTERN_C_NOINLINE void __asan_set_shadow_private(uptr shadow, uptr size,
                                                        char val) {
  auto *launch_info = (__SYCL_GLOBAL__ const AsanRuntimeData *)__AsanLaunchInfo;
  if (!launch_info || launch_info->PrivateShadowOffset == 0)
    return;

  // "__asan_mem_to_shadow" may return 0 although "PrivateShadowOffset != 0", in
  // this case, "shadow" may be out of range of private shadow
  if (shadow < launch_info->PrivateShadowOffset)
    return;

  ASAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_private_begin));

  ASAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_private, (void *)shadow,
                                (void *)(shadow + size), val & 0xFF));

  for (size_t i = 0; i < size; i++)
    ((__SYCL_GLOBAL__ u8 *)shadow)[i] = val;

  ASAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_private_end));
}

static __SYCL_CONSTANT__ const char __asan_print_private_base[] =
    "[kernel] set_private_base: %llu -> %p\n";

inline void SetPrivateBaseImpl(__SYCL_PRIVATE__ void *ptr) {
  auto launch_info = (__SYCL_GLOBAL__ const AsanRuntimeData *)__AsanLaunchInfo;
  const size_t sid = SubGroupLinearId();
  if (!launch_info || sid >= ASAN_MAX_SG_PRIVATE ||
      launch_info->PrivateShadowOffset == 0 || launch_info->PrivateBase == 0)
    return;
  // Only set on the first sub-group item
  if (__spirv_BuiltInSubgroupLocalInvocationId() == 0) {
    launch_info->PrivateBase[sid] = (uptr)ptr;
    ASAN_DEBUG(__spirv_ocl_printf(__asan_print_private_base, sid, ptr));
  }
  SubGroupBarrier();
}

DEVICE_EXTERN_C_NOINLINE void
__asan_set_private_base(__SYCL_PRIVATE__ void *ptr) {
#if defined(__LIBDEVICE_CPU__)
  return;
#elif defined(__LIBDEVICE_DG2__) || defined(__LIBDEVICE_PVC__)
  SetPrivateBaseImpl(ptr);
#else
  if (GetDeviceTy() == DeviceType::CPU)
    return;
  SetPrivateBaseImpl(ptr);
#endif
}

#endif // __SPIR__ || __SPIRV__
