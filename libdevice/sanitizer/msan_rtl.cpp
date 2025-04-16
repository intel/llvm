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
    "[kernel] __msan_get_shadow(addr=%p, as=%d) = %p: %02X <%s>\n";

const __SYCL_CONSTANT__ char __msan_print_launchinfo[] =
    "[kernel] !!! launchinfo %p (GlobalShadow=%p, LocalShadow=%p, "
    "PrivateShadow=%p)\n";

const __SYCL_CONSTANT__ char __msan_print_unsupport_device_type[] =
    "[kernel] Unsupport device type: %d\n";

const __SYCL_CONSTANT__ char __msan_print_generic_to[] =
    "[kernel] %p(4) - %p(%d)\n";

const __SYCL_CONSTANT__ char __msan_print_func_beg[] =
    "[kernel] ===== BEGIN %s()\n";

const __SYCL_CONSTANT__ char __msan_print_func_end[] =
    "[kernel] ===== END   %s()\n";

const __SYCL_CONSTANT__ char __msan_print_private_shadow_out_of_bound[] =
    "[kernel] Private shadow memory out-of-bound (ptr: %p -> %p, wg: %d, base: "
    "%p)\n";

const __SYCL_CONSTANT__ char __msan_print_unknown[] = "unknown";
} // namespace

#if defined(__SPIR__) || defined(__SPIRV__)

#define MSAN_DEBUG(X)                                                          \
  do {                                                                         \
    if (GetMsanLaunchInfo->Debug) {                                            \
      X;                                                                       \
    }                                                                          \
  } while (false)

namespace {

inline size_t WorkGroupLinearId() {
  return __spirv_BuiltInWorkgroupId.x * __spirv_BuiltInNumWorkgroups.y *
             __spirv_BuiltInNumWorkgroups.z +
         __spirv_BuiltInWorkgroupId.y * __spirv_BuiltInNumWorkgroups.z +
         __spirv_BuiltInWorkgroupId.z;
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
                                 const char __SYCL_CONSTANT__ *func,
                                 const uptr origin) {
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
    SanitizerReport.Origin = origin;
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
                         const char __SYCL_CONSTANT__ *func, uptr origin = 0) {
  __msan_internal_report_save(size, file, line, func, origin);
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
    return addr - (DG2_DEVICE_USM_END - shadow_end + 1);
  }
}

inline uptr __msan_get_shadow_pvc(uptr addr, uint32_t as) {
  if (as == ADDRESS_SPACE_GENERIC) {
    ConvertGenericPointer(addr, as);
  }

  // Device USM only
  if (as == ADDRESS_SPACE_GLOBAL && (addr & PVC_DEVICE_USM_MASK)) {
    auto shadow_begin = GetMsanLaunchInfo->GlobalShadowOffset;
    auto shadow_end = GetMsanLaunchInfo->GlobalShadowOffsetEnd;
    if (addr < shadow_begin) {
      return addr + (shadow_begin - PVC_DEVICE_USM_BEGIN);
    } else {
      return addr - (PVC_DEVICE_USM_END - shadow_end + 1);
    }
  } else if (as == ADDRESS_SPACE_LOCAL) {
    const auto shadow_offset = GetMsanLaunchInfo->LocalShadowOffset;
    if (shadow_offset != 0) {
      // The size of SLM is 128KB on PVC
      constexpr unsigned SLM_SIZE = 128 * 1024;
      const auto wg_lid = WorkGroupLinearId();
      return shadow_offset + (wg_lid * SLM_SIZE) + (addr & (SLM_SIZE - 1));
    }
  } else if (as == ADDRESS_SPACE_PRIVATE) {
    const auto shadow_offset = GetMsanLaunchInfo->PrivateShadowOffset;
    if (shadow_offset != 0) {
      const auto wg_lid = WorkGroupLinearId();
      uptr shadow_ptr = shadow_offset + (wg_lid * MSAN_PRIVATE_SIZE) +
                        (addr & (MSAN_PRIVATE_SIZE - 1));
      MSAN_DEBUG(const auto shadow_offset_end =
                     GetMsanLaunchInfo->PrivateShadowOffsetEnd;
                 if (shadow_ptr > shadow_offset_end) {
                   __spirv_ocl_printf(__msan_print_private_shadow_out_of_bound,
                                      addr, shadow_ptr, wg_lid,
                                      (uptr)shadow_offset);
                   return 0;
                 });
      return shadow_ptr;
    }
  }

  return GetMsanLaunchInfo->CleanShadow;
}

} // namespace

#define MSAN_MAYBE_WARNING(type, size)                                         \
  DEVICE_EXTERN_C_NOINLINE void __msan_maybe_warning_##size(                   \
      type s, uptr o, const char __SYCL_CONSTANT__ *file, uint32_t line,       \
      const char __SYCL_CONSTANT__ *func) {                                    \
    if (!GetMsanLaunchInfo)                                                    \
      return;                                                                  \
    if (UNLIKELY(s)) {                                                         \
      __msan_report_error(size, file, line, func, o);                          \
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
  __msan_internal_report_save(1, file, line, func, 0);
}

// For mapping detail, ref to
// "unified-runtime/source/loader/layers/sanitizer/msan/msan_shadow.hpp"
DEVICE_EXTERN_C_NOINLINE uptr __msan_get_shadow(
    uptr addr, uint32_t as, const char __SYCL_CONSTANT__ *func = nullptr) {
  // Return clean shadow (0s) by default
  uptr shadow_ptr = GetMsanLaunchInfo->CleanShadow;

  if (!GetMsanLaunchInfo)
    return shadow_ptr;

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_launchinfo, GetMsanLaunchInfo,
                                GetMsanLaunchInfo->GlobalShadowOffset,
                                GetMsanLaunchInfo->LocalShadowOffset,
                                GetMsanLaunchInfo->PrivateShadowOffset));

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
                                (void *)shadow_ptr, *(u8 *)shadow_ptr,
                                func ? func : __msan_print_unknown));

  return shadow_ptr;
}

static __SYCL_CONSTANT__ const char __msan_print_memset[] =
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
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_memset, dest, shadow,           \
                                  shadow + size - 1));                         \
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end, "__msan_memset"));    \
    return dest;                                                               \
  }

MSAN_MEMSET(0)
MSAN_MEMSET(1)
MSAN_MEMSET(3)
MSAN_MEMSET(4)

static __SYCL_CONSTANT__ const char __msan_print_memmove[] =
    "[kernel] memmove(dst=%p, src=%p, shadow_dst=%p, shadow_src=%p, size=%p)\n";

#define MSAN_MEMMOVE_BASE(dst_as, src_as)                                      \
  DEVICE_EXTERN_C_NOINLINE __attribute__((address_space(dst_as))) void         \
      *__msan_memmove_p##dst_as##_p##src_as(                                   \
          __attribute__((address_space(dst_as))) char *dest,                   \
          __attribute__((address_space(src_as))) char *src, size_t size) {     \
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg, "__msan_memmove"));   \
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
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_memmove, dest, src,             \
                                  dest_shadow, src_shadow, size));             \
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end, "__msan_memmove"));   \
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

static __SYCL_CONSTANT__ const char __msan_print_memcpy[] =
    "[kernel] memcpy(dst=%p, src=%p, shadow_dst=%p, shadow_src=%p, size=%p)\n";

#define MSAN_MEMCPY_BASE(dst_as, src_as)                                       \
  DEVICE_EXTERN_C_NOINLINE __attribute__((address_space(dst_as))) void         \
      *__msan_memcpy_p##dst_as##_p##src_as(                                    \
          __attribute__((address_space(dst_as))) char *dest,                   \
          __attribute__((address_space(src_as))) char *src, size_t size) {     \
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg, "__msan_memcpy"));    \
    uptr dest_shadow = __msan_get_shadow((uptr)dest, dst_as);                  \
    uptr src_shadow = __msan_get_shadow((uptr)src, src_as);                    \
    for (size_t i = 0; i < size; i++) {                                        \
      dest[i] = src[i];                                                        \
      ((__SYCL_GLOBAL__ char *)dest_shadow)[i] =                               \
          ((__SYCL_GLOBAL__ char *)src_shadow)[i];                             \
    }                                                                          \
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_memmove, dest, src,             \
                                  dest_shadow, src_shadow, size));             \
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end, "__msan_memcpy"));    \
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
/// Initialize shadow memory of local memory
///

static __SYCL_CONSTANT__ const char __mem_set_shadow_local[] =
    "[kernel] set_shadow_local(beg=%p, end=%p, val:%02X)\n";

DEVICE_EXTERN_C_NOINLINE void __msan_poison_shadow_static_local(uptr ptr,
                                                                size_t size) {
  // Update shadow memory of local memory only on first work-item
  if (__spirv_LocalInvocationId_x() + __spirv_LocalInvocationId_y() +
          __spirv_LocalInvocationId_z() ==
      0) {
    if (!GetMsanLaunchInfo)
      return;

    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg,
                                  "__msan_poison_shadow_static_local"));

    auto shadow_address = __msan_get_shadow(ptr, ADDRESS_SPACE_LOCAL);
    if (shadow_address == GetMsanLaunchInfo->CleanShadow)
      return;

    for (size_t i = 0; i < size; ++i) {
      ((__SYCL_GLOBAL__ u8 *)shadow_address)[i] = 0xff;
    }

    MSAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_local, shadow_address,
                                  shadow_address + size, 0xff));
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end,
                                  "__msan_poison_shadow_static_local"));
  }
}

DEVICE_EXTERN_C_NOINLINE void __msan_unpoison_shadow_static_local(uptr ptr,
                                                                  size_t size) {
  // Update shadow memory of local memory only on first work-item
  if (__spirv_LocalInvocationId_x() + __spirv_LocalInvocationId_y() +
          __spirv_LocalInvocationId_z() ==
      0) {
    if (!GetMsanLaunchInfo || GetMsanLaunchInfo->LocalShadowOffset == 0)
      return;

    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg,
                                  "__msan_unpoison_shadow_static_local"));

    auto shadow_address = __msan_get_shadow(ptr, ADDRESS_SPACE_LOCAL);
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

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg,
                                "__msan_poison_shadow_dynamic_local"));

  if (num_args != GetMsanLaunchInfo->NumLocalArgs) {
    __spirv_ocl_printf(__msan_print_report_arg_count_incorrect, num_args,
                       GetMsanLaunchInfo->NumLocalArgs);
    return;
  }

  uptr *args = (uptr *)ptr;

  for (uint32_t i = 0; i < num_args; ++i) {
    auto *local_arg = &GetMsanLaunchInfo->LocalArgs[i];
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_local_arg, i, local_arg->Size));

    __msan_poison_shadow_static_local(args[i], local_arg->Size);
  }

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end,
                                "__msan_poison_shadow_dynamic_local"));
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

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg,
                                "__msan_unpoison_shadow_dynamic_local"));

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

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end,
                                "__msan_unpoison_shadow_dynamic_local"));
}

static __SYCL_CONSTANT__ const char __msan_print_set_shadow_private[] =
    "[kernel] set_shadow_private(beg=%p, end=%p, val:%02X)\n";

// We outline the function of setting shadow memory of private memory, because
// it may allocate failed on UR
DEVICE_EXTERN_C_NOINLINE void __msan_poison_stack(__SYCL_PRIVATE__ void *ptr,
                                                  uptr size) {
  if (!GetMsanLaunchInfo || GetMsanLaunchInfo->PrivateShadowOffset == 0)
    return;

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg, "__msan_poison_stack"));

  auto shadow_address = __msan_get_shadow((uptr)ptr, ADDRESS_SPACE_PRIVATE);
  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_set_shadow_private,
                                (void *)shadow_address,
                                (void *)(shadow_address + size), 0xff));

  for (size_t i = 0; i < size; i++)
    ((__SYCL_GLOBAL__ u8 *)shadow_address)[i] = 0xff;

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end, "__msan_poison_stack"));
}

DEVICE_EXTERN_C_NOINLINE void __msan_unpoison_stack(__SYCL_PRIVATE__ void *ptr,
                                                    uptr size) {
  if (!GetMsanLaunchInfo || GetMsanLaunchInfo->PrivateShadowOffset == 0)
    return;

  MSAN_DEBUG(
      __spirv_ocl_printf(__msan_print_func_beg, "__msan_unpoison_stack"));

  auto shadow_address = __msan_get_shadow((uptr)ptr, ADDRESS_SPACE_PRIVATE);
  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_set_shadow_private,
                                (void *)shadow_address,
                                (void *)(shadow_address + size), 0x0));

  for (size_t i = 0; i < size; i++)
    ((__SYCL_GLOBAL__ u8 *)shadow_address)[i] = 0;

  MSAN_DEBUG(
      __spirv_ocl_printf(__msan_print_func_end, "__msan_unpoison_stack"));
}

#endif // __SPIR__ || __SPIRV__
