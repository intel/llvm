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
#define GetMsanLaunchInfo                                                      \
  ((__SYCL_GLOBAL__ MsanLaunchInfo *)__MsanLaunchInfo.get())

namespace {

constexpr int MSAN_REPORT_NONE = 0;
constexpr int MSAN_REPORT_START = 1;
constexpr int MSAN_REPORT_FINISH = 2;

constexpr uptr PVC_DEVICE_USM_MASK = 0xff00'0000'0000'0000ULL;
constexpr uptr PVC_DEVICE_USM_BEGIN = 0xff00'0000'0000'0000ULL;
constexpr uptr PVC_DEVICE_USM_END = 0xff00'ffff'ffff'ffffULL;

constexpr uptr DG2_DEVICE_USM_MASK = 0xffff'0000'0000'0000ULL;
constexpr uptr DG2_DEVICE_USM_BEGIN = 0xffff'8000'0000'0000ULL;
constexpr uptr DG2_DEVICE_USM_END = 0xffff'ffff'ffff'ffffULL;

const __SYCL_CONSTANT__ char __msan_print_shadow[] =
    "[kernel] __msan_get_shadow(addr=%p, as=%d) = %p: %02X\n";

const __SYCL_CONSTANT__ char __msan_print_launchinfo[] =
    "[kernel] !!! launchinfo %p (GlobalShadow=%p)\n";

const __SYCL_CONSTANT__ char __msan_print_unsupport_device_type[] =
    "[kernel] Unsupport device type: %d\n";

const __SYCL_CONSTANT__ char __msan_print_generic_to[] =
    "[kernel] %p(4) - %p(%d)\n";

const __SYCL_CONSTANT__ char __msan_print_func_beg[] =
    "[kernel] ===== %s() begin\n";

const __SYCL_CONSTANT__ char __msan_print_func_end[] =
    "[kernel] ===== %s() end\n";

} // namespace

#if defined(__SPIR__) || defined(__SPIRV__)

#define MSAN_DEBUG(X)                                                          \
  do {                                                                         \
    if (GetMsanLaunchInfo->Debug) {                                            \
      X;                                                                       \
    }                                                                          \
  } while (false)

namespace {

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

  auto &SanitizerReport = GetMsanLaunchInfo->Report;

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

  if (!GetMsanLaunchInfo->IsRecover) {
    __devicelib_exit();
  }
}

inline uptr __msan_get_shadow_cpu(uptr addr) {
  return addr ^ 0x500000000000ULL;
}

inline uptr __msan_get_shadow_dg2(uptr addr, uint32_t as) {
  if (as == ADDRESS_SPACE_GENERIC) {
    ConvertGenericPointer(addr, as);
  }

  if (as != ADDRESS_SPACE_GLOBAL || !(addr & DG2_DEVICE_USM_MASK))
    return (uptr)((__SYCL_GLOBAL__ MsanLaunchInfo *)__MsanLaunchInfo.get())
        ->CleanShadow;

  // Device USM only
  auto shadow_begin = ((__SYCL_GLOBAL__ MsanLaunchInfo *)__MsanLaunchInfo.get())
                          ->GlobalShadowOffset;
  auto shadow_end = ((__SYCL_GLOBAL__ MsanLaunchInfo *)__MsanLaunchInfo.get())
                        ->GlobalShadowOffsetEnd;
  if (addr < shadow_begin) {
    return addr + (shadow_begin - DG2_DEVICE_USM_BEGIN);
  } else {
    return addr - (DG2_DEVICE_USM_END - shadow_end);
  }
}

inline uptr __msan_get_shadow_pvc(uptr addr, uint32_t as) {
  if (as == ADDRESS_SPACE_GENERIC) {
    ConvertGenericPointer(addr, as);
  }

  // Device USM only
  if (as == ADDRESS_SPACE_GLOBAL && (addr & DG2_DEVICE_USM_MASK)) {
    auto shadow_begin = GetMsanLaunchInfo->GlobalShadowOffset;
    auto shadow_end = GetMsanLaunchInfo->GlobalShadowOffsetEnd;
    if (addr < shadow_begin) {
      return addr + (shadow_begin - DG2_DEVICE_USM_BEGIN);
    } else {
      return addr - (DG2_DEVICE_USM_END - shadow_end);
    }
  } else if (as == ADDRESS_SPACE_LOCAL) {
    // The size of SLM is 128KB on PVC
    constexpr unsigned SLM_SIZE = 128 * 1024;
    // work-group linear id
    const auto wg_lid =
        __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
            __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
        __spirv_BuiltInWorkgroupId.z;

    const auto shadow_offset = GetMsanLaunchInfo->LocalShadowOffset;

    if (shadow_offset != 0) {
      return shadow_offset + (wg_lid * SLM_SIZE) + (addr & (SLM_SIZE - 1));
    }
  }

  return GetMsanLaunchInfo->CleanShadow;
}

} // namespace

#define MSAN_MAYBE_WARNING(type, size)                                         \
  DEVICE_EXTERN_C_NOINLINE void __msan_maybe_warning_##size(                   \
      type s, u32 o, const char __SYCL_CONSTANT__ *file, uint32_t line,        \
      const char __SYCL_CONSTANT__ *func) {                                    \
    if (!GetMsanLaunchInfo)                                                    \
      return;                                                                  \
    if (UNLIKELY(s)) {                                                         \
      __msan_report_error(size, file, line, func);                             \
    }                                                                          \
  }

MSAN_MAYBE_WARNING(u8, 1)
MSAN_MAYBE_WARNING(u16, 2)
MSAN_MAYBE_WARNING(u32, 4)
MSAN_MAYBE_WARNING(u64, 8)

DEVICE_EXTERN_C_NOINLINE void
__msan_warning(const char __SYCL_CONSTANT__ *file, uint32_t line,
               const char __SYCL_CONSTANT__ *func) {
  __msan_report_error(1, file, line, func);
}

DEVICE_EXTERN_C_NOINLINE void
__msan_warning_noreturn(const char __SYCL_CONSTANT__ *file, uint32_t line,
                        const char __SYCL_CONSTANT__ *func) {
  __msan_internal_report_save(1, file, line, func);
  __devicelib_exit();
}

// For mapping detail, ref to
// "unified-runtime/source/loader/layers/sanitizer/msan/msan_shadow.hpp"
DEVICE_EXTERN_C_NOINLINE uptr __msan_get_shadow(uptr addr, uint32_t as) {
  // Return clean shadow (0s) by default
  uptr shadow_ptr = GetMsanLaunchInfo->CleanShadow;

  if (!GetMsanLaunchInfo)
    return shadow_ptr;

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_launchinfo, GetMsanLaunchInfo,
                                GetMsanLaunchInfo->GlobalShadowOffset));

#if defined(__LIBDEVICE_PVC__)
  shadow_ptr = __msan_get_shadow_pvc(addr, as);
#elif defined(__LIBDEVICE_CPU__)
  shadow_ptr = __msan_get_shadow_cpu(addr);
#else
  if (LIKELY(GetMsanLaunchInfo->DeviceTy == DeviceType::CPU)) {
    shadow_ptr = __msan_get_shadow_cpu(addr);
  } else if (GetMsanLaunchInfo->DeviceTy == DeviceType::GPU_PVC) {
    shadow_ptr = __msan_get_shadow_pvc(addr, as);
  } else if (GetMsanLaunchInfo->DeviceTy == DeviceType::GPU_DG2) {
    shadow_ptr = __msan_get_shadow_dg2(addr, as);
  } else {
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_unsupport_device_type,
                                  GetMsanLaunchInfo->DeviceTy));
  }
#endif

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_shadow, (void *)addr, as,
                                (void *)shadow_ptr, *(u8 *)shadow_ptr));

  return shadow_ptr;
}

static __SYCL_CONSTANT__ const char __mem_memset[] =
    "[kernel] memset(beg=%p, shadow_beg=%p, shadow_end=%p)\n";

#define MSAN_MEMSET(as)                                                        \
  DEVICE_EXTERN_C_NOINLINE                                                     \
  __attribute__((address_space(as))) void *__msan_memset_p##as(                \
      __attribute__((address_space(as))) char *dest, int val, size_t size) {   \
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg, "__msan_memset"));    \
    uptr shadow = __msan_get_shadow((uptr)dest, as);                           \
    for (size_t i = 0; i < size; i++) {                                        \
      dest[i] = val;                                                           \
      ((__SYCL_GLOBAL__ char *)shadow)[i] = 0;                                 \
    }                                                                          \
    MSAN_DEBUG(                                                                \
        __spirv_ocl_printf(__mem_memset, dest, shadow, shadow + size - 1));    \
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end, "__msan_memset"));    \
    return dest;                                                               \
  }

MSAN_MEMSET(0)
MSAN_MEMSET(1)
MSAN_MEMSET(3)
MSAN_MEMSET(4)

#define MSAN_MEMMOVE_BASE(dst_as, src_as)                                      \
  DEVICE_EXTERN_C_NOINLINE __attribute__((address_space(dst_as))) void         \
      *__msan_memmove_p##dst_as##_p##src_as(                                   \
          __attribute__((address_space(dst_as))) char *dest,                   \
          __attribute__((address_space(src_as))) char *src, size_t size) {     \
    uptr dest_shadow = __msan_get_shadow((uptr)dest, dst_as);                  \
    uptr src_shadow = __msan_get_shadow((uptr)src, src_as);                    \
    if ((uptr)dest > (uptr)src) {                                              \
      for (size_t i = size - 1; i < size; i--) {                               \
        dest[i] = src[i];                                                      \
        ((__SYCL_GLOBAL__ char *)dest_shadow)[i] =                             \
            ((__SYCL_GLOBAL__ char *)src_shadow)[i];                           \
      }                                                                        \
    } else {                                                                   \
      for (size_t i = 0; i < size; i++) {                                      \
        dest[i] = src[i];                                                      \
        ((__SYCL_GLOBAL__ char *)dest_shadow)[i] =                             \
            ((__SYCL_GLOBAL__ char *)src_shadow)[i];                           \
      }                                                                        \
    }                                                                          \
    return dest;                                                               \
  }

#define MSAN_MEMMOVE(dst_as)                                                   \
  MSAN_MEMMOVE_BASE(dst_as, 0)                                                 \
  MSAN_MEMMOVE_BASE(dst_as, 1)                                                 \
  MSAN_MEMMOVE_BASE(dst_as, 2)                                                 \
  MSAN_MEMMOVE_BASE(dst_as, 3)                                                 \
  MSAN_MEMMOVE_BASE(dst_as, 4)

MSAN_MEMMOVE(0)
MSAN_MEMMOVE(1)
MSAN_MEMMOVE(3)
MSAN_MEMMOVE(4)

#define MSAN_MEMCPY_BASE(dst_as, src_as)                                       \
  DEVICE_EXTERN_C_NOINLINE __attribute__((address_space(dst_as))) void         \
      *__msan_memcpy_p##dst_as##_p##src_as(                                    \
          __attribute__((address_space(dst_as))) char *dest,                   \
          __attribute__((address_space(src_as))) char *src, size_t size) {     \
    uptr dest_shadow = __msan_get_shadow((uptr)dest, dst_as);                  \
    uptr src_shadow = __msan_get_shadow((uptr)src, src_as);                    \
    for (size_t i = 0; i < size; i++) {                                        \
      dest[i] = src[i];                                                        \
      ((__SYCL_GLOBAL__ char *)dest_shadow)[i] =                               \
          ((__SYCL_GLOBAL__ char *)src_shadow)[i];                             \
    }                                                                          \
    return dest;                                                               \
  }

#define MSAN_MEMCPY(dst_as)                                                    \
  MSAN_MEMCPY_BASE(dst_as, 0)                                                  \
  MSAN_MEMCPY_BASE(dst_as, 1)                                                  \
  MSAN_MEMCPY_BASE(dst_as, 2)                                                  \
  MSAN_MEMCPY_BASE(dst_as, 3)                                                  \
  MSAN_MEMCPY_BASE(dst_as, 4)

MSAN_MEMCPY(0)
MSAN_MEMCPY(1)
MSAN_MEMCPY(3)
MSAN_MEMCPY(4)

///
/// Initialize shdadow memory of local memory
///

static __SYCL_CONSTANT__ const char __mem_set_shadow_local[] =
    "[kernel] set_shadow_local(beg=%p, end=%p, val:%02X)\n";

DEVICE_EXTERN_C_NOINLINE void __msan_set_shadow_static_local(uptr ptr,
                                                             size_t size) {
  // Update shadow memory of local memory only on first work-item
  if (__spirv_LocalInvocationId_x() + __spirv_LocalInvocationId_y() +
          __spirv_LocalInvocationId_z() ==
      0) {
    if (!GetMsanLaunchInfo)
      return;

    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg,
                                  "__msan_set_shadow_static_local"));

    auto shadow_address = __msan_get_shadow(ptr, ADDRESS_SPACE_LOCAL);
    if (shadow_address == GetMsanLaunchInfo->CleanShadow)
      return;

    for (size_t i = 0; i < size; ++i) {
      ((__SYCL_GLOBAL__ u8 *)shadow_address)[i] = 0xff;
    }

    MSAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_local, shadow_address,
                                  shadow_address + size, 0xff));
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end,
                                  "__msan_set_shadow_static_local"));
  }
}

DEVICE_EXTERN_C_NOINLINE void __msan_unpoison_shadow_static_local(uptr ptr,
                                                                  size_t size) {
  // Update shadow memory of local memory only on first work-item
  if (__spirv_LocalInvocationId_x() + __spirv_LocalInvocationId_y() +
          __spirv_LocalInvocationId_z() ==
      0) {
    if (!GetMsanLaunchInfo)
      return;

    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg,
                                  "__msan_unpoison_shadow_static_local"));

    auto shadow_address = __msan_get_shadow(ptr, ADDRESS_SPACE_LOCAL);
    if (shadow_address == GetMsanLaunchInfo->CleanShadow)
      return;

    for (size_t i = 0; i < size; ++i) {
      ((__SYCL_GLOBAL__ u8 *)shadow_address)[i] = 0;
    }

    MSAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_local, shadow_address,
                                  shadow_address + size, 0));
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end,
                                  "__msan_unpoison_shadow_static_local"));
  }
}

DEVICE_EXTERN_C_INLINE void __msan_barrier() {
  __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
                         __spv::MemorySemanticsMask::SequentiallyConsistent |
                             __spv::MemorySemanticsMask::WorkgroupMemory);
}

static __SYCL_CONSTANT__ const char __msan_print_local_arg[] =
    "[kernel] local_arg(index=%d, size=%d)\n";

static __SYCL_CONSTANT__ const char
    __msan_print_set_shadow_dynamic_local_begin[] =
        "[kernel] BEGIN __msan_poison_shadow_dynamic_local\n";
static __SYCL_CONSTANT__ const char
    __msan_print_set_shadow_dynamic_local_end[] =
        "[kernel] END   __msan_poison_shadow_dynamic_local\n";
static __SYCL_CONSTANT__ const char __msan_print_report_arg_count_incorrect[] =
    "[kernel] ERROR: The number of local args is incorrect, expect %d, actual "
    "%d\n";

DEVICE_EXTERN_C_NOINLINE void
__msan_poison_shadow_dynamic_local(uptr ptr, uint32_t num_args) {
  if (!GetMsanLaunchInfo)
    return;

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_set_shadow_dynamic_local_begin));

  if (num_args != GetMsanLaunchInfo->NumLocalArgs) {
    __spirv_ocl_printf(__msan_print_report_arg_count_incorrect, num_args,
                       GetMsanLaunchInfo->NumLocalArgs);
    return;
  }

  uptr *args = (uptr *)ptr;

  for (uint32_t i = 0; i < num_args; ++i) {
    auto *local_arg = &GetMsanLaunchInfo->LocalArgs[i];
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_local_arg, i, local_arg->Size));

    __msan_set_shadow_static_local(args[i], local_arg->Size);
  }

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_set_shadow_dynamic_local_end));
}

static __SYCL_CONSTANT__ const char
    __mem_unpoison_shadow_dynamic_local_begin[] =
        "[kernel] BEGIN __msan_unpoison_shadow_dynamic_local\n";
static __SYCL_CONSTANT__ const char __mem_unpoison_shadow_dynamic_local_end[] =
    "[kernel] END   __msan_unpoison_shadow_dynamic_local\n";

DEVICE_EXTERN_C_NOINLINE void
__msan_unpoison_shadow_dynamic_local(uptr ptr, uint32_t num_args) {
  if (!GetMsanLaunchInfo)
    return;

  MSAN_DEBUG(__spirv_ocl_printf(__mem_unpoison_shadow_dynamic_local_begin));

  if (num_args != GetMsanLaunchInfo->NumLocalArgs) {
    __spirv_ocl_printf(__msan_print_report_arg_count_incorrect, num_args,
                       GetMsanLaunchInfo->NumLocalArgs);
    return;
  }

  uptr *args = (uptr *)ptr;

  for (uint32_t i = 0; i < num_args; ++i) {
    auto *local_arg = &GetMsanLaunchInfo->LocalArgs[i];
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_local_arg, i, local_arg->Size));

    __msan_unpoison_shadow_static_local(args[i], local_arg->Size);
  }

  MSAN_DEBUG(__spirv_ocl_printf(__mem_unpoison_shadow_dynamic_local_end));
}

#endif // __SPIR__ || __SPIRV__
