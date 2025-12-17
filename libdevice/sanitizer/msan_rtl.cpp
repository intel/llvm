//==--- msan_rtl.cpp - device memory sanitizer runtime library -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/msan_rtl.hpp"
#include "msan/msan_libdevice.hpp"

DeviceGlobal<void *> __MsanLaunchInfo;
#define GetMsanLaunchInfo                                                      \
  ((__SYCL_GLOBAL__ MsanRuntimeData *)__MsanLaunchInfo.get())

extern "C" __attribute__((weak)) const int __msan_track_origins;

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

const __SYCL_CONSTANT__ char __msan_print_origin[] =
    "[kernel] __msan_get_origin(addr=%p, as=%d) = %p: %04x\n";

const __SYCL_CONSTANT__ char __msan_print_unsupport_device_type[] =
    "[kernel] Unsupport device type: %d\n";

const __SYCL_CONSTANT__ char __msan_print_generic_to[] =
    "[kernel] %p(4) - %p(%d)\n";

const __SYCL_CONSTANT__ char __msan_print_func_beg[] =
    "[kernel] ===== BEGIN %s()\n";

const __SYCL_CONSTANT__ char __msan_print_func_end[] =
    "[kernel] ===== END   %s()\n";

const __SYCL_CONSTANT__ char __msan_print_private_shadow_out_of_bound[] =
    "[kernel] Private shadow memory out-of-bound(ptr: %p -> %p, "
    "sid: %llu, base: "
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

inline bool IsTrackOriginsEnabled() { return __msan_track_origins; }

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
  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_generic_to, old, addr, as));
}

void SaveReport(const uint32_t size, const char __SYCL_CONSTANT__ *file,
                const uint32_t line, const char __SYCL_CONSTANT__ *func,
                const uint32_t origin = 0) {
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
    SanitizerReport.GID0 = __spirv_BuiltInGlobalInvocationId(0);
    SanitizerReport.GID1 = __spirv_BuiltInGlobalInvocationId(1);
    SanitizerReport.GID2 = __spirv_BuiltInGlobalInvocationId(2);
    SanitizerReport.LID0 = __spirv_BuiltInLocalInvocationId(0);
    SanitizerReport.LID1 = __spirv_BuiltInLocalInvocationId(1);
    SanitizerReport.LID2 = __spirv_BuiltInLocalInvocationId(2);

    // Show we've done copying
    atomicStore(&SanitizerReport.Flag, MSAN_REPORT_FINISH);
  }
}

// The CPU mapping is based on compiler-rt/msan
inline uptr MemToShadow_CPU(uptr addr) { return addr ^ 0x500000000000ULL; }

inline uptr MemToShadow_DG2(uptr addr, uint32_t as) {
  if (as == ADDRESS_SPACE_GENERIC) {
    ConvertGenericPointer(addr, as);
  }

  if (as != ADDRESS_SPACE_GLOBAL || !(addr & DG2_DEVICE_USM_MASK))
    return (uptr)GetMsanLaunchInfo->CleanShadow;

  // Device USM only
  auto shadow_begin = GetMsanLaunchInfo->GlobalShadowOffset;
  auto shadow_end = GetMsanLaunchInfo->GlobalShadowOffsetEnd;
  if (addr < shadow_begin) {
    return addr + (shadow_begin - DG2_DEVICE_USM_BEGIN);
  } else {
    return addr - (DG2_DEVICE_USM_END - shadow_end + 1);
  }
}

inline uptr MemToShadow_PVC(uptr addr, uint32_t as) {
  if (as == ADDRESS_SPACE_GENERIC) {
    ConvertGenericPointer(addr, as);
  }

  if (as == ADDRESS_SPACE_GLOBAL) {
    // device USM
    if (addr >> 52 == 0xff0) {
      return addr - 0x5000'0000'0000ULL;
    }
    // host/shared USM
    auto shadow_base = GetMsanLaunchInfo->GlobalShadowOffset;
    return (addr & 0xfff'ffff'ffffULL) + ((addr & 0x8000'0000'0000ULL) >> 3) +
           shadow_base;
  } else if (as == ADDRESS_SPACE_LOCAL) {
    const auto shadow_offset = GetMsanLaunchInfo->LocalShadowOffset;
    const size_t wid = WorkGroupLinearId();
    if (shadow_offset != 0 && wid < MSAN_MAX_WG_LOCAL) {
      // The size of SLM is 128KB on PVC
      constexpr unsigned SLM_SIZE = 128 * 1024;
      return shadow_offset + (wid * SLM_SIZE) + (addr & (SLM_SIZE - 1));
    }
  } else if (as == ADDRESS_SPACE_PRIVATE) {
    const auto shadow_offset = GetMsanLaunchInfo->PrivateShadowOffset;
    const size_t sid = SubGroupLinearId();
    if (shadow_offset != 0 && sid < MSAN_MAX_SG_PRIVATE) {
      const uptr private_base = GetMsanLaunchInfo->PrivateBase[sid];

      // FIXME: The recorded private_base may not be the most bottom one,
      // ideally there should have a build-in to get this information
      if (addr < private_base) {
        return GetMsanLaunchInfo->CleanShadow;
      }

      uptr shadow_ptr =
          shadow_offset + (sid * MSAN_PRIVATE_SIZE) + (addr - private_base);

      const auto shadow_offset_end = GetMsanLaunchInfo->PrivateShadowOffsetEnd;
      if (shadow_ptr > shadow_offset_end) {
        __spirv_ocl_printf(__msan_print_private_shadow_out_of_bound, addr,
                           shadow_ptr, sid, private_base);
        return GetMsanLaunchInfo->CleanShadow;
      };

      return shadow_ptr;
    }
  }

  return GetMsanLaunchInfo->CleanShadow;
}

inline uptr MemToShadow(uptr addr, uint32_t as) {
  // Return clean shadow (0s) by default
  uptr shadow_ptr;

#if defined(__LIBDEVICE_PVC__)
  shadow_ptr = MemToShadow_PVC(addr, as);
#elif defined(__LIBDEVICE_CPU__)
  shadow_ptr = MemToShadow_CPU(addr);
#else
  if (LIKELY(GetMsanLaunchInfo->DeviceTy == DeviceType::CPU)) {
    shadow_ptr = MemToShadow_CPU(addr);
  } else if (GetMsanLaunchInfo->DeviceTy == DeviceType::GPU_PVC) {
    shadow_ptr = MemToShadow_PVC(addr, as);
  } else if (GetMsanLaunchInfo->DeviceTy == DeviceType::GPU_DG2) {
    shadow_ptr = MemToShadow_DG2(addr, as);
  } else {
    shadow_ptr = GetMsanLaunchInfo->CleanShadow;
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_unsupport_device_type,
                                  GetMsanLaunchInfo->DeviceTy));
  }
#endif

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_shadow, (void *)addr, as,
                                (void *)shadow_ptr, *(u8 *)shadow_ptr));

  return shadow_ptr;
}

// The CPU mapping is based on compiler-rt/msan
inline uptr MemToOrigin_CPU(uptr addr) {
  return MemToShadow_CPU(addr) + 0x100000000000ULL;
}

inline uptr MemToOrigin_DG2(uptr addr, uint32_t as) {
  return GetMsanLaunchInfo->CleanShadow;
}

inline uptr MemToOrigin_PVC(uptr addr, uint32_t as) {
  if (as == ADDRESS_SPACE_GENERIC) {
    ConvertGenericPointer(addr, as);
  }

  if (as == ADDRESS_SPACE_GLOBAL) {
    // device USM
    if (addr >> 52 == 0xff0) {
      return addr - 0xa000'0000'0000ULL;
    }
    // host/shared USM
    uptr shadow_base = GetMsanLaunchInfo->GlobalShadowOffset;
    return (addr & 0xfff'ffff'ffffULL) + ((addr & 0x8000'0000'0000ULL) >> 3) +
           shadow_base + 0x2000'0000'0000ULL;
  }

  // Return clean shadow (0s) by default
  return GetMsanLaunchInfo->CleanShadow;
}

inline uptr MemToOrigin(uptr addr, uint32_t as) {
  uptr aligned_addr = addr & ~3ULL;
  uptr origin_ptr;

#if defined(__LIBDEVICE_PVC__)
  origin_ptr = MemToOrigin_PVC(addr, as);
#elif defined(__LIBDEVICE_CPU__)
  origin_ptr = MemToOrigin_CPU(addr);
#else
  if (LIKELY(GetMsanLaunchInfo->DeviceTy == DeviceType::CPU)) {
    origin_ptr = MemToOrigin_CPU(aligned_addr);
  } else if (GetMsanLaunchInfo->DeviceTy == DeviceType::GPU_PVC) {
    origin_ptr = MemToOrigin_PVC(aligned_addr, as);
  } else if (GetMsanLaunchInfo->DeviceTy == DeviceType::GPU_DG2) {
    origin_ptr = MemToOrigin_DG2(aligned_addr, as);
  } else {
    // Return clean shadow (0s) by default
    origin_ptr = GetMsanLaunchInfo->CleanShadow;
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_unsupport_device_type,
                                  GetMsanLaunchInfo->DeviceTy));
  }
#endif

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_origin, (void *)addr, as,
                                (void *)origin_ptr, *(uint32_t *)origin_ptr));

  return origin_ptr;
}

inline void Exit() {
  if (!GetMsanLaunchInfo->IsRecover)
    __devicelib_exit();
}

inline void ReportError(const uint32_t size, const char __SYCL_CONSTANT__ *file,
                        const uint32_t line, const char __SYCL_CONSTANT__ *func,
                        const uint32_t origin = 0) {
  SaveReport(size, file, line, func, origin);
  Exit();
}

// This function is only used for shadow propagation
template <typename T>
void GroupAsyncCopy(uptr Dest, uptr Src, size_t NumElements, size_t Stride,
                    bool StrideOnSrc) {
  auto DestPtr = (__SYCL_GLOBAL__ T *)Dest;
  auto SrcPtr = (const __SYCL_GLOBAL__ T *)Src;
  for (size_t i = 0; i < NumElements; i++) {
    if (StrideOnSrc)
      DestPtr[i] = SrcPtr[i * Stride];
    else
      DestPtr[i * Stride] = SrcPtr[i];
  }
}

static __SYCL_CONSTANT__ const char __msan_print_copy_shadow[] =
    "[kernel] CopyShadow(dst=%p(%d), src=%p(%d), shadow_dst=%p, shadow_src=%p, "
    "size=%p)\n";

// FIXME: The original implemention only copies the origin of poisoned memories
void CopyOrigin(uptr dst, uint32_t dst_as, uptr src, uint32_t src_as,
                uptr size) {
  auto *src_beg = (__SYCL_GLOBAL__ char *)MemToOrigin(src, src_as);
  auto *src_end = (__SYCL_GLOBAL__ char *)MemToOrigin(src + size - 1, src_as) +
                  MSAN_ORIGIN_GRANULARITY;
  auto *dst_beg = (__SYCL_GLOBAL__ char *)MemToOrigin(dst, dst_as);
  Memcpy(dst_beg, src_beg, src_end - src_beg);
}

inline void CopyShadowAndOrigin(uptr dst, uint32_t dst_as, uptr src,
                                uint32_t src_as, size_t size) {
  auto *shadow_dst = (__SYCL_GLOBAL__ char *)MemToShadow(dst, dst_as);
  if ((uptr)shadow_dst == GetMsanLaunchInfo->CleanShadow) {
    return;
  }
  auto *shadow_src = (__SYCL_GLOBAL__ char *)MemToShadow(src, src_as);
  Memcpy(shadow_dst, shadow_src, size);
  if (IsTrackOriginsEnabled())
    CopyOrigin(dst, dst_as, src, src_as, size);

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_copy_shadow, dst, dst_as, src,
                                src_as, shadow_dst, shadow_src, size));
}

static __SYCL_CONSTANT__ const char __msan_print_move_shadow[] =
    "[kernel] MoveShadow(dst=%p(%d), src=%p(%d), shadow_dst=%p, shadow_src=%p, "
    "size=%p)\n";

// FIXME: The original implemention only moves the origin of poisoned memories
void MoveOrigin(uptr dst, uint32_t dst_as, uptr src, uint32_t src_as,
                uptr size) {
  auto *dst_beg = (__SYCL_GLOBAL__ char *)MemToOrigin(dst, dst_as);
  if ((uptr)dst_beg == GetMsanLaunchInfo->CleanShadow) {
    return;
  }
  auto *src_beg = (__SYCL_GLOBAL__ char *)MemToOrigin(src, src_as);
  auto *src_end = (__SYCL_GLOBAL__ char *)MemToOrigin(src + size - 1, src_as) +
                  MSAN_ORIGIN_GRANULARITY;
  Memmove(dst_beg, src_beg, src_end - src_beg);
}

inline void MoveShadowAndOrigin(uptr dst, uint32_t dst_as, uptr src,
                                uint32_t src_as, size_t size) {
  auto *shadow_dst = (__SYCL_GLOBAL__ char *)MemToShadow(dst, dst_as);
  auto *shadow_src = (__SYCL_GLOBAL__ char *)MemToShadow(src, src_as);
  // MoveOrigin transfers origins by refering to their shadows
  if (IsTrackOriginsEnabled())
    MoveOrigin(dst, dst_as, src, src_as, size);
  Memmove(shadow_dst, shadow_src, size);

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_move_shadow, dst, dst_as, src,
                                src_as, shadow_dst, shadow_src, size));
}

inline void UnpoisonShadow(uptr addr, uint32_t as, size_t size) {
  auto *shadow_ptr = (__SYCL_GLOBAL__ char *)MemToShadow(addr, as);
  if ((uptr)shadow_ptr == GetMsanLaunchInfo->CleanShadow) {
    return;
  }
  Memset(shadow_ptr, 0, size);
}

// Check if the current work item is the first one in the work group
inline bool IsFirstWorkItemWthinWorkGroup() {
  return __spirv_BuiltInLocalInvocationId(0) +
             __spirv_BuiltInLocalInvocationId(1) +
             __spirv_BuiltInLocalInvocationId(2) ==
         0;
}

} // namespace

#define MSAN_MAYBE_WARNING(type, size)                                         \
  DEVICE_EXTERN_C_NOINLINE void __msan_maybe_warning_##size(                   \
      type s, uint32_t o, const char __SYCL_CONSTANT__ *file, uint32_t line,   \
      const char __SYCL_CONSTANT__ *func) {                                    \
    if (!GetMsanLaunchInfo)                                                    \
      return;                                                                  \
    if (UNLIKELY(s)) {                                                         \
      ReportError(size, file, line, func, o);                                  \
    }                                                                          \
  }

MSAN_MAYBE_WARNING(u8, 1)
MSAN_MAYBE_WARNING(u16, 2)
MSAN_MAYBE_WARNING(u32, 4)
MSAN_MAYBE_WARNING(u64, 8)

DEVICE_EXTERN_C_NOINLINE void
__msan_warning(const char __SYCL_CONSTANT__ *file, uint32_t line,
               const char __SYCL_CONSTANT__ *func) {
  if (!GetMsanLaunchInfo)
    return;
  ReportError(1, file, line, func);
}

DEVICE_EXTERN_C_NOINLINE void
__msan_warning_noreturn(const char __SYCL_CONSTANT__ *file, uint32_t line,
                        const char __SYCL_CONSTANT__ *func) {
  if (!GetMsanLaunchInfo)
    return;
  ReportError(1, file, line, func, 0);
}

DEVICE_EXTERN_C_NOINLINE void
__msan_warning_with_origin(uint32_t origin, const char __SYCL_CONSTANT__ *file,
                           uint32_t line, const char __SYCL_CONSTANT__ *func) {
  if (!GetMsanLaunchInfo)
    return;
  ReportError(1, file, line, func, origin);
}

DEVICE_EXTERN_C_NOINLINE void __msan_warning_with_origin_noreturn(
    uint32_t origin, const char __SYCL_CONSTANT__ *file, uint32_t line,
    const char __SYCL_CONSTANT__ *func) {
  if (!GetMsanLaunchInfo)
    return;
  ReportError(1, file, line, func, origin);
}

// For mapping detail, ref to
// "unified-runtime/source/loader/layers/sanitizer/msan/msan_shadow.hpp"
DEVICE_EXTERN_C_NOINLINE __SYCL_GLOBAL__ void *__msan_get_shadow(uptr addr,
                                                                 uint32_t as) {
  if (!GetMsanLaunchInfo)
    return nullptr;
  return (__SYCL_GLOBAL__ void *)MemToShadow(addr, as);
}

// For mapping detail, ref to
// "unified-runtime/source/loader/layers/sanitizer/msan/msan_shadow.hpp"
DEVICE_EXTERN_C_NOINLINE __SYCL_GLOBAL__ void *__msan_get_origin(uptr addr,
                                                                 uint32_t as) {
  if (!GetMsanLaunchInfo)
    return nullptr;
  return (__SYCL_GLOBAL__ void *)MemToOrigin(addr, as);
}

#define MSAN_MAYBE_STORE_ORIGIN(type, size)                                    \
  DEVICE_EXTERN_C_NOINLINE void __msan_maybe_store_origin_##size(              \
      type s, uptr addr, uint32_t as, uint32_t o) {                            \
    if (UNLIKELY(s)) {                                                         \
      *(__SYCL_GLOBAL__ u32 *)MemToOrigin(addr, as) = o;                       \
    }                                                                          \
  }

MSAN_MAYBE_STORE_ORIGIN(u8, 1)
MSAN_MAYBE_STORE_ORIGIN(u16, 2)
MSAN_MAYBE_STORE_ORIGIN(u32, 4)
MSAN_MAYBE_STORE_ORIGIN(u64, 8)

static __SYCL_CONSTANT__ const char __msan_print_memset[] =
    "[kernel] memset(beg=%p, shadow_beg=%p, shadow_end=%p)\n";

#define MSAN_MEMSET(as)                                                        \
  DEVICE_EXTERN_C_NOINLINE                                                     \
  __attribute__((address_space(as))) void *__msan_memset_p##as(                \
      __attribute__((address_space(as))) char *dest, int val, size_t size) {   \
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg, "__msan_memset"));    \
    auto res = Memset(dest, val, size);                                        \
    UnpoisonShadow((uptr)dest, as, size);                                      \
    return res;                                                                \
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
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg, "__msan_memmove"));   \
    auto res = Memmove(dest, src, size);                                       \
    MoveShadowAndOrigin((uptr)dest, dst_as, (uptr)src, src_as, size);          \
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end, "__msan_memmove"));   \
    return res;                                                                \
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
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg, "__msan_memcpy"));    \
    auto res = Memcpy(dest, src, size);                                        \
    CopyShadowAndOrigin((uptr)dest, dst_as, (uptr)src, src_as, size);          \
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end, "__msan_memcpy"));    \
    return res;                                                                \
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
  if (!IsFirstWorkItemWthinWorkGroup())
    return;

  if (!GetMsanLaunchInfo || GetMsanLaunchInfo->LocalShadowOffset == 0)
    return;

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg,
                                "__msan_poison_shadow_static_local"));

  auto shadow_address = MemToShadow(ptr, ADDRESS_SPACE_LOCAL);
  if (shadow_address != GetMsanLaunchInfo->CleanShadow) {
    Memset((__SYCL_GLOBAL__ char *)shadow_address, 0xff, size);
    MSAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_local, shadow_address,
                                  shadow_address + size, 0xff));
  }

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end,
                                "__msan_poison_shadow_static_local"));
}

DEVICE_EXTERN_C_NOINLINE void __msan_unpoison_shadow_static_local(uptr ptr,
                                                                  size_t size) {
  // Update shadow memory of local memory only on first work-item
  if (!IsFirstWorkItemWthinWorkGroup())
    return;

  if (!GetMsanLaunchInfo || GetMsanLaunchInfo->LocalShadowOffset == 0)
    return;

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg,
                                "__msan_unpoison_shadow_static_local"));
  UnpoisonShadow(ptr, ADDRESS_SPACE_LOCAL, size);
  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end,
                                "__msan_unpoison_shadow_static_local"));
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
  // Update shadow memory of local memory only on first work-item
  if (!IsFirstWorkItemWthinWorkGroup())
    return;

  if (!GetMsanLaunchInfo || GetMsanLaunchInfo->LocalShadowOffset == 0)
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

    auto shadow_address = MemToShadow(args[i], ADDRESS_SPACE_LOCAL);
    if (shadow_address != GetMsanLaunchInfo->CleanShadow) {
      Memset((__SYCL_GLOBAL__ char *)shadow_address, 0xff, local_arg->Size);
      MSAN_DEBUG(__spirv_ocl_printf(__mem_set_shadow_local, shadow_address,
                                    shadow_address + local_arg->Size, 0xff));
    }
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
  // Update shadow memory of local memory only on first work-item
  if (!IsFirstWorkItemWthinWorkGroup())
    return;

  if (!GetMsanLaunchInfo || GetMsanLaunchInfo->LocalShadowOffset == 0)
    return;

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg,
                                "__msan_unpoison_shadow_dynamic_local"));

  if (num_args != GetMsanLaunchInfo->NumLocalArgs) {
    return;
  }

  uptr *args = (uptr *)ptr;

  for (uint32_t i = 0; i < num_args; ++i) {
    auto *local_arg = &GetMsanLaunchInfo->LocalArgs[i];
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_local_arg, i, local_arg->Size));

    UnpoisonShadow(args[i], ADDRESS_SPACE_LOCAL, local_arg->Size);
  }

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end,
                                "__msan_unpoison_shadow_dynamic_local"));
}

static __SYCL_CONSTANT__ const char __msan_print_set_shadow[] =
    "[kernel] __msan_set_value(beg=%p, end=%p, val=%02X)\n";

// We outline the function of setting shadow memory of private memory, because
// it may allocate failed on UR
DEVICE_EXTERN_C_NOINLINE void __msan_poison_stack(__SYCL_PRIVATE__ void *ptr,
                                                  uptr size) {
  if (!GetMsanLaunchInfo || GetMsanLaunchInfo->PrivateShadowOffset == 0)
    return;

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg, "__msan_poison_stack"));

  auto shadow_address = MemToShadow((uptr)ptr, ADDRESS_SPACE_PRIVATE);
  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_set_shadow, (void *)shadow_address,
                                (void *)(shadow_address + size), 0xff));

  if (shadow_address != GetMsanLaunchInfo->CleanShadow) {
    Memset((__SYCL_GLOBAL__ char *)shadow_address, 0xff, size);
  }

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end, "__msan_poison_stack"));
}

DEVICE_EXTERN_C_NOINLINE void __msan_unpoison_stack(__SYCL_PRIVATE__ void *ptr,
                                                    uptr size) {
  if (!GetMsanLaunchInfo || GetMsanLaunchInfo->PrivateShadowOffset == 0)
    return;

  MSAN_DEBUG(
      __spirv_ocl_printf(__msan_print_func_beg, "__msan_unpoison_stack"));

  auto shadow_address = MemToShadow((uptr)ptr, ADDRESS_SPACE_PRIVATE);
  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_set_shadow, (void *)shadow_address,
                                (void *)(shadow_address + size), 0x0));

  if (shadow_address != GetMsanLaunchInfo->CleanShadow) {
    Memset((__SYCL_GLOBAL__ char *)shadow_address, 0, size);
  }

  MSAN_DEBUG(
      __spirv_ocl_printf(__msan_print_func_end, "__msan_unpoison_stack"));
}

DEVICE_EXTERN_C_NOINLINE void __msan_unpoison_shadow(uptr ptr, uint32_t as,
                                                     uptr size) {
  if (!GetMsanLaunchInfo)
    return;

  MSAN_DEBUG(
      __spirv_ocl_printf(__msan_print_func_beg, "__msan_unpoison_shadow"));

  auto shadow_address = MemToShadow(ptr, as);
  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_set_shadow, (void *)shadow_address,
                                (void *)(shadow_address + size), 0x0));

  if (shadow_address != GetMsanLaunchInfo->CleanShadow) {
    Memset((__SYCL_GLOBAL__ char *)shadow_address, 0, size);
  }

  MSAN_DEBUG(
      __spirv_ocl_printf(__msan_print_func_end, "__msan_unpoison_shadow"));
}

static __SYCL_CONSTANT__ const char __msan_print_private_base[] =
    "[kernel] __msan_set_private_base(sid=%llu): %p\n";

DEVICE_EXTERN_C_NOINLINE void
__msan_set_private_base(__SYCL_PRIVATE__ void *ptr) {
  const size_t sid = SubGroupLinearId();
  if (!GetMsanLaunchInfo || sid >= MSAN_MAX_SG_PRIVATE ||
      GetMsanLaunchInfo->PrivateShadowOffset == 0 ||
      GetMsanLaunchInfo->PrivateBase == 0)
    return;
  // Only set on the first sub-group item
  if (__spirv_BuiltInSubgroupLocalInvocationId() == 0) {
    GetMsanLaunchInfo->PrivateBase[sid] = (uptr)ptr;
    MSAN_DEBUG(__spirv_ocl_printf(__msan_print_private_base, sid, ptr));
  }
  SubGroupBarrier();
}

static __SYCL_CONSTANT__ const char __msan_print_strided_copy_unsupport_type[] =
    "[kernel] __msan_unpoison_strided_copy: unsupported type(%d)\n";

DEVICE_EXTERN_C_NOINLINE void
__msan_unpoison_strided_copy(uptr dest, uint32_t dest_as, uptr src,
                             uint32_t src_as, uint32_t element_size,
                             uptr counts, uptr stride) {
  if (!GetMsanLaunchInfo)
    return;

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg,
                                "__msan_unpoison_strided_copy"));

  uptr shadow_dest = MemToShadow(dest, dest_as);
  if (shadow_dest != GetMsanLaunchInfo->CleanShadow) {
    uptr shadow_src = MemToShadow(src, src_as);

    switch (element_size) {
    case 1:
      GroupAsyncCopy<int8_t>(shadow_dest, shadow_src, counts, stride,
                             src_as == ADDRESS_SPACE_GLOBAL);
      break;
    case 2:
      GroupAsyncCopy<int16_t>(shadow_dest, shadow_src, counts, stride,
                              src_as == ADDRESS_SPACE_GLOBAL);
      break;
    case 4:
      GroupAsyncCopy<int32_t>(shadow_dest, shadow_src, counts, stride,
                              src_as == ADDRESS_SPACE_GLOBAL);
      break;
    case 8:
      GroupAsyncCopy<int64_t>(shadow_dest, shadow_src, counts, stride,
                              src_as == ADDRESS_SPACE_GLOBAL);
      break;
    default:
      __spirv_ocl_printf(__msan_print_strided_copy_unsupport_type,
                         element_size);
    }
  }

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end,
                                "__msan_unpoison_strided_copy"));
}

static __SYCL_CONSTANT__ const char __msan_print_copy_unsupport_type[] =
    "[kernel] __msan_unpoison_copy: unsupported type(%d <- %d)\n";

DEVICE_EXTERN_C_NOINLINE void __msan_unpoison_copy(uptr dst, uint32_t dst_as,
                                                   uptr src, uint32_t src_as,
                                                   uint32_t dst_element_size,
                                                   uint32_t src_element_size,
                                                   uptr counts) {
  if (!GetMsanLaunchInfo)
    return;

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_beg, "__msan_unpoison_copy"));

  uptr shadow_dst = MemToShadow(dst, dst_as);
  if (shadow_dst != GetMsanLaunchInfo->CleanShadow) {
    uptr shadow_src = MemToShadow(src, src_as);

    if (dst_element_size == 1 && src_element_size == 1) {
      Memcpy<__SYCL_GLOBAL__ int8_t *, __SYCL_GLOBAL__ int8_t *>(
          (__SYCL_GLOBAL__ int8_t *)shadow_dst,
          (__SYCL_GLOBAL__ int8_t *)shadow_src, counts);
    } else if (dst_element_size == 4 && src_element_size == 2) {
      Memcpy<__SYCL_GLOBAL__ int32_t *, __SYCL_GLOBAL__ int16_t *>(
          (__SYCL_GLOBAL__ int32_t *)shadow_dst,
          (__SYCL_GLOBAL__ int16_t *)shadow_src, counts);
    } else if (dst_element_size == 2 && src_element_size == 4) {
      Memcpy<__SYCL_GLOBAL__ int16_t *, __SYCL_GLOBAL__ int32_t *>(
          (__SYCL_GLOBAL__ int16_t *)shadow_dst,
          (__SYCL_GLOBAL__ int32_t *)shadow_src, counts);
    } else {
      __spirv_ocl_printf(__msan_print_copy_unsupport_type, dst_element_size,
                         src_element_size);
    }
  }

  MSAN_DEBUG(__spirv_ocl_printf(__msan_print_func_end, "__msan_unpoison_copy"));
}

#endif // __SPIR__ || __SPIRV__
