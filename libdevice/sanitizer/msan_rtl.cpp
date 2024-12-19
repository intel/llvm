//==--- msan_rtl.cpp - device memory sanitizer runtime library -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/msan_rtl.hpp"
#include "atomic.hpp"
#include "device.h"
#include "msan/msan_libdevice.hpp"
#include "spirv_vars.h"

DeviceGlobal<void *> __MsanLaunchInfo;

constexpr int MSAN_REPORT_NONE = 0;
constexpr int MSAN_REPORT_START = 1;
constexpr int MSAN_REPORT_FINISH = 2;

static const __SYCL_CONSTANT__ char __msan_print_warning_return[] =
    "[kernel] !!! msan warning return\n";

static const __SYCL_CONSTANT__ char __msan_print_shadow[] =
    "[kernel] __msan_get_shadow(addr=%p, as=%d) = %p: %02X\n";

static const __SYCL_CONSTANT__ char __msan_print_warning_nolaunchinfo[] =
    "[kernel] !!! __mem_warning_nolaunchinfo\n";

static const __SYCL_CONSTANT__ char __msan_print_launchinfo[] =
    "[kernel] !!! launchinfo %p (GlobalShadow=%p)\n";

static const __SYCL_CONSTANT__ char __msan_print_report[] =
    "[kernel] %d bytes uninitialized at kernel %s\n";

static const __SYCL_CONSTANT__ char __msan_print_unsupport_device_type[] =
    "[kernel] Unsupport device type: %d\n";

static __SYCL_CONSTANT__ const char __msan_print_generic_to[] =
    "[kernel] %p(4) - %p(%d)\n";

#if defined(__SPIR__) || defined(__SPIRV__)

#if defined(__SYCL_DEVICE_ONLY__)
#define __USE_SPIR_BUILTIN__ 1
#endif

#if __USE_SPIR_BUILTIN__
extern SYCL_EXTERNAL int
__spirv_ocl_printf(const __SYCL_CONSTANT__ char *Format, ...);
extern "C" SYCL_EXTERNAL void __devicelib_exit();
#endif

#define MSAN_DEBUG(X)                                                          \
  do {                                                                         \
    auto launch_info =                                                         \
        (__SYCL_GLOBAL__ const MsanLaunchInfo *)__MsanLaunchInfo.get();        \
    if (launch_info->Debug) {                                                  \
      X;                                                                       \
    }                                                                          \
  } while (false)

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
  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_generic_to, old, addr, as));
}

void __msan_internal_report_save(const uint32_t size,
                                 const char __SYCL_CONSTANT__ *file,
                                 const uint32_t line,
                                 const char __SYCL_CONSTANT__ *func) {
  const int Expected = MSAN_REPORT_NONE;
  int Desired = MSAN_REPORT_START;

  auto &SanitizerReport =
      ((__SYCL_GLOBAL__ MsanLaunchInfo *)__MsanLaunchInfo.get())->Report;

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

    SanitizerReport.AccessSize = size;
    SanitizerReport.Line = line;
    SanitizerReport.GID0 = __spirv_GlobalInvocationId_x();
    SanitizerReport.GID1 = __spirv_GlobalInvocationId_y();
    SanitizerReport.GID2 = __spirv_GlobalInvocationId_z();
    SanitizerReport.LID0 = __spirv_LocalInvocationId_x();
    SanitizerReport.LID1 = __spirv_LocalInvocationId_y();
    SanitizerReport.LID2 = __spirv_LocalInvocationId_z();

    // Show we've done copying
    atomicStore(&SanitizerReport.Flag, MSAN_REPORT_FINISH);
  }
}

void __msan_report_error(const uint32_t size,
                         const char __SYCL_CONSTANT__ *file,
                         const uint32_t line,
                         const char __SYCL_CONSTANT__ *func) {
  __msan_internal_report_save(size, file, line, func);

  auto launch = (__SYCL_GLOBAL__ MsanLaunchInfo *)__MsanLaunchInfo.get();
  if (!launch->IsRecover) {
    __devicelib_exit();
  }
}

inline uptr __msan_get_shadow_cpu(uptr addr) {
  return addr ^ 0x500000000000ULL;
}

inline uptr __msan_get_shadow_pvc(uptr addr, uint32_t as) {
  if (as == ADDRESS_SPACE_GENERIC) {
    ConvertGenericPointer(addr, as);
    if (as != ADDRESS_SPACE_GLOBAL)
      return (uptr)((__SYCL_GLOBAL__ MsanLaunchInfo *)__MsanLaunchInfo.get())
          ->CleanShadow;
  }

  // Device USM only
  uptr shadow_ptr = ((__SYCL_GLOBAL__ MsanLaunchInfo *)__MsanLaunchInfo.get())
                        ->GlobalShadowOffset +
                    (addr & 0x3FFF'FFFF'FFFFULL);
  return shadow_ptr;
}

} // namespace

#define MSAN_MAYBE_WARNING(type, size)                                         \
  DEVICE_EXTERN_C_NOINLINE void __msan_maybe_warning_##size(                   \
      type s, u32 o, const char __SYCL_CONSTANT__ *file, uint32_t line,        \
      const char __SYCL_CONSTANT__ *func) {                                    \
    if (UNLIKELY(s)) {                                                         \
      __msan_report_error(size, file, line, func);                             \
    }                                                                          \
  }

MSAN_MAYBE_WARNING(u8, 1)
MSAN_MAYBE_WARNING(u16, 2)
MSAN_MAYBE_WARNING(u32, 4)
MSAN_MAYBE_WARNING(u64, 8)

DEVICE_EXTERN_C_NOINLINE uptr __msan_get_shadow(uptr addr, uint32_t as) {
  // Return clean shadow (0s) by default
  uptr shadow_ptr =
      (uptr)((__SYCL_GLOBAL__ MsanLaunchInfo *)__MsanLaunchInfo.get())
          ->CleanShadow;

  if (UNLIKELY(!__MsanLaunchInfo)) {
    __spirv_ocl_printf(__msan_print_warning_nolaunchinfo);
    return shadow_ptr;
  }

  auto launch_info = (__SYCL_GLOBAL__ MsanLaunchInfo *)__MsanLaunchInfo.get();
  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_launchinfo, (void *)launch_info,
                                launch_info->GlobalShadowOffset));

  if (LIKELY(launch_info->DeviceTy == DeviceType::CPU)) {
    shadow_ptr = __msan_get_shadow_cpu(addr);
  } else if (launch_info->DeviceTy == DeviceType::GPU_PVC) {
    shadow_ptr = __msan_get_shadow_pvc(addr, as);
  } else {
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_unsupport_device_type,
                                  launch_info->DeviceTy));
  }

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_shadow, (void *)addr, as,
                                (void *)shadow_ptr, *(u8 *)shadow_ptr));

  return shadow_ptr;
}

#endif // __SPIR__ || __SPIRV__
