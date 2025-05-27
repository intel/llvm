//==--- tsan_rtl.cpp - device thread sanitizer runtime library -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/tsan_rtl.hpp"
#include "atomic.hpp"
#include "device.h"
#include "spirv_vars.h"

DeviceGlobal<void *> __TsanLaunchInfo;

#define TsanLaunchInfo                                                         \
  ((__SYCL_GLOBAL__ TsanRuntimeData *)__TsanLaunchInfo.get())

#if defined(__SPIR__) || defined(__SPIRV__)

static const __SYCL_CONSTANT__ char __tsan_print_generic_to[] =
    "[kernel] %p(4) - %p(%d)\n";

static const __SYCL_CONSTANT__ char __tsan_print_raw_shadow[] =
    "[kernel] %p(%d) -> %p: {%x, %x}\n";

static const __SYCL_CONSTANT__ char __tsan_print_shadow_value[] =
    "[kernel] %p(%d) : {size: %d, access: %x, sid: %d, clock: %d, is_write: "
    "%d}\n";

static const __SYCL_CONSTANT__ char __tsan_print_cleanup_private[] =
    "[kernel] cleanup private shadow: %p ~ %p\n";

static const __SYCL_CONSTANT__ char __tsan_print_unsupport_device_type[] =
    "[kernel] Unsupport device type: %d\n";

static const __SYCL_CONSTANT__ char __tsan_report_race[] =
    "[kernel] data race (%s:%d) in %s\n";

#define TSAN_DEBUG(X)                                                          \
  do {                                                                         \
    if (TsanLaunchInfo->Debug) {                                               \
      X;                                                                       \
    }                                                                          \
  } while (false)

#define TSAN_REPORT_NONE 0
#define TSAN_REPORT_START 1
#define TSAN_REPORT_FINISH 2

namespace {

inline constexpr uptr RoundUpTo(uptr x, uptr boundary) {
  return (x + boundary - 1) & ~(boundary - 1);
}

inline constexpr uptr RoundDownTo(uptr x, uptr boundary) {
  return x & ~(boundary - 1);
}

inline constexpr uptr Min(uptr a, uptr b) { return a < b ? a : b; }

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
  TSAN_DEBUG(__spirv_ocl_printf(__tsan_print_generic_to, old, addr, as));
}

inline Epoch IncrementEpoch(Sid sid) {
  return TsanLaunchInfo->Clock[sid].clk_[sid]++;
}

inline __SYCL_GLOBAL__ RawShadow *MemToShadow_CPU(uptr addr, uint32_t) {
  return reinterpret_cast<__SYCL_GLOBAL__ RawShadow *>(
      ((addr) & ~(0x700000000000ULL | (kShadowCell - 1))) * kShadowMultiplier +
      TsanLaunchInfo->GlobalShadowOffset);
}

inline __SYCL_GLOBAL__ RawShadow *MemToShadow_PVC(uptr addr, uint32_t as) {
  if (as == ADDRESS_SPACE_GENERIC) {
    ConvertGenericPointer(addr, as);
  }

  addr = RoundDownTo(addr, kShadowCell);

  if (as == ADDRESS_SPACE_GLOBAL) {
    if (addr & 0xff00'0000'0000'0000ULL) {
      // device usm
      return addr < TsanLaunchInfo->GlobalShadowOffset
                 ? reinterpret_cast<__SYCL_GLOBAL__ RawShadow *>(
                       addr + (TsanLaunchInfo->GlobalShadowOffset +
                               0x200'0000'0000ULL - 0xff00'0000'0000'0000ULL))
                 : reinterpret_cast<__SYCL_GLOBAL__ RawShadow *>(
                       addr - (0xff00'ffff'ffff'ffffULL -
                               TsanLaunchInfo->GlobalShadowOffsetEnd + 1));
    } else {
      // host & shared usm
      return reinterpret_cast<__SYCL_GLOBAL__ RawShadow *>(
          (addr & 0xffffffffffULL) + TsanLaunchInfo->GlobalShadowOffset +
          ((addr & 0x800000000000ULL) >> 7));
    }
  } else if (as == ADDRESS_SPACE_LOCAL) {
    const auto shadow_offset = TsanLaunchInfo->LocalShadowOffset;
    if (shadow_offset != 0) {
      // The size of SLM is 128KB on PVC
      constexpr unsigned SLM_SIZE = 128 * 1024;
      const size_t wid = WorkGroupLinearId();
      return reinterpret_cast<__SYCL_GLOBAL__ RawShadow *>(
          shadow_offset + (wid * SLM_SIZE) + (addr & (SLM_SIZE - 1)));
    }
  }

  return nullptr;
}

inline __SYCL_GLOBAL__ RawShadow *MemToShadow(uptr addr, uint32_t as) {
  __SYCL_GLOBAL__ RawShadow *shadow_ptr = nullptr;

#if defined(__LIBDEVICE_CPU__)
  shadow_ptr = MemToShadow_CPU(addr, as);
#elif defined(__LIBDEVICE_PVC__)
  shadow_ptr = MemToShadow_PVC(addr, as);
#else
  if (TsanLaunchInfo->DeviceTy == DeviceType::CPU) {
    shadow_ptr = MemToShadow_CPU(addr, as);
  } else if (TsanLaunchInfo->DeviceTy == DeviceType::GPU_PVC) {
    shadow_ptr = MemToShadow_PVC(addr, as);
  } else {
    TSAN_DEBUG(__spirv_ocl_printf(__tsan_print_unsupport_device_type,
                                  (int)TsanLaunchInfo->DeviceTy));
    return nullptr;
  }
#endif

  return shadow_ptr;
}

inline Sid GetCurrentSid() {
  const auto lid = __spirv_BuiltInGlobalLinearId;
  return lid % kThreadSlotCount;
}

inline RawShadow LoadShadow(const __SYCL_GLOBAL__ RawShadow *p) {
  return static_cast<RawShadow>(
      __spirv_AtomicLoad((const __SYCL_GLOBAL__ int *)p, __spv::Scope::Device,
                         __spv::MemorySemanticsMask::None));
}

inline void StoreShadow(__SYCL_GLOBAL__ RawShadow *p, RawShadow s) {
  __spirv_AtomicStore((__SYCL_GLOBAL__ int *)p, __spv::Scope::Device,
                      __spv::MemorySemanticsMask::None, static_cast<int>(s));
}

inline void DoReportRace(__SYCL_GLOBAL__ RawShadow *s, AccessType type,
                         uptr addr, uint32_t size, uint32_t as,
                         const char __SYCL_CONSTANT__ *file, uint32_t line,
                         const char __SYCL_CONSTANT__ *func) {
  // This prevents trapping on this address in future.
  for (uptr i = 0; i < kShadowCnt; i++)
    StoreShadow(&s[i], i == 0 ? Shadow::kRodata : Shadow::kEmpty);

  // On GPU device, SpinLock is not working. So, we use LoopLock here to write
  // report sequentially.
  while (true) {
    if (atomicCompareAndSet(&TsanLaunchInfo->Lock, 1, 0)) {
      if (TsanLaunchInfo->RecordedReportCount >= TSAN_MAX_NUM_REPORTS) {
        atomicStore(&TsanLaunchInfo->Lock, 0);
        return;
      }

      // Check if current address already being recorded before.
      for (uint32_t i = 0; i < TsanLaunchInfo->RecordedReportCount; i++) {
        auto &SanitizerReport = TsanLaunchInfo->Report[i];
        if (addr == SanitizerReport.Address) {
          atomicStore(&TsanLaunchInfo->Lock, 0);
          return;
        }
      }

      auto &SanitizerReport =
          TsanLaunchInfo->Report[TsanLaunchInfo->RecordedReportCount++];

      if (as == ADDRESS_SPACE_GENERIC) {
        ConvertGenericPointer(addr, as);
      }

      SanitizerReport.Address = addr;
      SanitizerReport.Type =
          type | (as == ADDRESS_SPACE_LOCAL ? kAccessLocal : 0);
      SanitizerReport.AccessSize = size;

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

      atomicStore(&TsanLaunchInfo->Lock, 0);
      break;
    }
  }
}

inline bool CheckRace(__SYCL_GLOBAL__ RawShadow *s, Shadow cur, AccessType type,
                      uptr addr, uint32_t size, uint32_t as,
                      const char __SYCL_CONSTANT__ *file, uint32_t line,
                      const char __SYCL_CONSTANT__ *func) {
  bool stored = false;
  for (uptr i = 0; i < kShadowCnt; i++) {
    __SYCL_GLOBAL__ RawShadow *sp = &s[i];
    Shadow old(LoadShadow(sp));
    if (old.raw() == Shadow::kEmpty) {
      if (!stored)
        StoreShadow(sp, cur.raw());
      return false;
    }

    // access different region, no data race.
    if (!(cur.access() & old.access()))
      continue;

    // access same region with same thread, just update the shadow word.
    if (cur.sid() == old.sid()) {
      if (cur.access() == old.access()) {
        StoreShadow(sp, cur.raw());
        stored = true;
      }
      continue;
    }

    // both reads, no data race
    if (old.IsBothReads(type))
      continue;

    // check happen before
    if (TsanLaunchInfo->Clock[cur.sid()].clk_[old.sid()] >= old.clock())
      continue;

    DoReportRace(s, type, addr, size, as, file, line, func);
    return true;
  }

  // We did not find any races and had already stored
  // the current access info, so we are done.
  if (stored)
    return false;
  // Choose a random candidate slot (except first slot, it's used for rodata
  // marker) and replace it.
  uptr index = cur.clock() % (kShadowCnt - 1) + 1;
  StoreShadow(&s[index], cur.raw());
  return false;
}

inline bool ContainsSameAccess(__SYCL_GLOBAL__ RawShadow *s, Shadow cur,
                               AccessType type) {
  for (uptr i = 0; i < kShadowCnt; i++) {
    RawShadow old = LoadShadow(&s[i]);
    if (!(type & kAccessRead)) {
      if (old == cur.raw())
        return true;
      continue;
    }

    // already trapped on current address before, just ignore it.
    if (old == Shadow::kRodata)
      return true;
  }
  return false;
}

} // namespace

#define TSAN_CHECK(type, is_write, size)                                       \
  DEVICE_EXTERN_C_NOINLINE void __tsan_##type##size(                           \
      uptr addr, uint32_t as, const char __SYCL_CONSTANT__ *file,              \
      uint32_t line, const char __SYCL_CONSTANT__ *func) {                     \
    __SYCL_GLOBAL__ RawShadow *shadow_mem = MemToShadow(addr, as);             \
    if (!shadow_mem)                                                           \
      return;                                                                  \
    Sid sid = GetCurrentSid();                                                 \
    uint16_t current_clock = IncrementEpoch(sid) + 1;                          \
    TSAN_DEBUG(__spirv_ocl_printf(__tsan_print_raw_shadow, (void *)addr, as,   \
                                  (void *)shadow_mem, shadow_mem[0],           \
                                  shadow_mem[1]));                             \
    AccessType type = is_write ? kAccessWrite : kAccessRead;                   \
    Shadow cur(addr, size, current_clock, sid, type);                          \
    TSAN_DEBUG(__spirv_ocl_printf(__tsan_print_shadow_value, (void *)addr, as, \
                                  size, cur.access(), cur.sid(), cur.clock(),  \
                                  is_write));                                  \
    if (ContainsSameAccess(shadow_mem, cur, type))                             \
      return;                                                                  \
    CheckRace(shadow_mem, cur, type, addr, size, as, file, line, func);        \
  }

TSAN_CHECK(read, false, 1)
TSAN_CHECK(read, false, 2)
TSAN_CHECK(read, false, 4)
TSAN_CHECK(read, false, 8)
TSAN_CHECK(write, true, 1)
TSAN_CHECK(write, true, 2)
TSAN_CHECK(write, true, 4)
TSAN_CHECK(write, true, 8)

DEVICE_EXTERN_C_NOINLINE void
__tsan_write16(uptr addr, uint32_t as, const char __SYCL_CONSTANT__ *file,
               uint32_t line, const char __SYCL_CONSTANT__ *func) {
  __tsan_write8(addr, as, file, line, func);
  __tsan_write8(addr + 8, as, file, line, func);
}

DEVICE_EXTERN_C_NOINLINE void
__tsan_read16(uptr addr, uint32_t as, const char __SYCL_CONSTANT__ *file,
              uint32_t line, const char __SYCL_CONSTANT__ *func) {
  __tsan_read8(addr, as, file, line, func);
  __tsan_read8(addr + 8, as, file, line, func);
}

#define TSAN_UNALIGNED_CHECK(type, is_write, size)                             \
  DEVICE_EXTERN_C_NOINLINE void __tsan_unaligned_##type##size(                 \
      uptr addr, uint32_t as, const char __SYCL_CONSTANT__ *file,              \
      uint32_t line, const char __SYCL_CONSTANT__ *func) {                     \
    __SYCL_GLOBAL__ RawShadow *shadow_mem = MemToShadow(addr, as);             \
    if (!shadow_mem)                                                           \
      return;                                                                  \
    Sid sid = GetCurrentSid();                                                 \
    uint16_t current_clock = IncrementEpoch(sid) + 1;                          \
    AccessType type = is_write ? kAccessWrite : kAccessRead;                   \
    uptr size1 = Min(size, RoundUpTo(addr + 1, kShadowCell) - addr);           \
    {                                                                          \
      TSAN_DEBUG(__spirv_ocl_printf(__tsan_print_raw_shadow, (void *)addr, as, \
                                    (void *)shadow_mem, shadow_mem[0],         \
                                    shadow_mem[1]));                           \
      Shadow cur(addr, size1, current_clock, sid, type);                       \
      TSAN_DEBUG(__spirv_ocl_printf(__tsan_print_shadow_value, (void *)addr,   \
                                    as, size1, cur.access(), cur.sid(),        \
                                    cur.clock(), is_write));                   \
      if (ContainsSameAccess(shadow_mem, cur, type))                           \
        goto SECOND;                                                           \
      if (CheckRace(shadow_mem, cur, type, addr, size1, as, file, line, func)) \
        return;                                                                \
    }                                                                          \
  SECOND:                                                                      \
    uptr size2 = size - size1;                                                 \
    if (size2 == 0)                                                            \
      return;                                                                  \
    shadow_mem += kShadowCnt;                                                  \
    {                                                                          \
      TSAN_DEBUG(__spirv_ocl_printf(                                           \
          __tsan_print_raw_shadow, (void *)(addr + size1), as,                 \
          (void *)shadow_mem, shadow_mem[0], shadow_mem[1]));                  \
      Shadow cur(0, size2, current_clock, sid, type);                          \
      TSAN_DEBUG(__spirv_ocl_printf(                                           \
          __tsan_print_shadow_value, (void *)(addr + size1), as, size2,        \
          cur.access(), cur.sid(), cur.clock(), is_write));                    \
      if (ContainsSameAccess(shadow_mem, cur, type))                           \
        return;                                                                \
      CheckRace(shadow_mem, cur, type, addr + size1, size2, as, file, line,    \
                func);                                                         \
    }                                                                          \
  }

TSAN_UNALIGNED_CHECK(read, false, 1)
TSAN_UNALIGNED_CHECK(read, false, 2)
TSAN_UNALIGNED_CHECK(read, false, 4)
TSAN_UNALIGNED_CHECK(read, false, 8)
TSAN_UNALIGNED_CHECK(write, true, 1)
TSAN_UNALIGNED_CHECK(write, true, 2)
TSAN_UNALIGNED_CHECK(write, true, 4)
TSAN_UNALIGNED_CHECK(write, true, 8)

DEVICE_EXTERN_C_NOINLINE void
__tsan_unaligned_write16(uptr addr, uint32_t as,
                         const char __SYCL_CONSTANT__ *file, uint32_t line,
                         const char __SYCL_CONSTANT__ *func) {
  __tsan_unaligned_write8(addr, as, file, line, func);
  __tsan_unaligned_write8(addr + 8, as, file, line, func);
}

DEVICE_EXTERN_C_NOINLINE void
__tsan_unaligned_read16(uptr addr, uint32_t as,
                        const char __SYCL_CONSTANT__ *file, uint32_t line,
                        const char __SYCL_CONSTANT__ *func) {
  __tsan_unaligned_read8(addr, as, file, line, func);
  __tsan_unaligned_read8(addr + 8, as, file, line, func);
}

static inline void __tsan_cleanup_private_cpu_impl(uptr addr, uint32_t size) {
  if (size) {
    addr = RoundDownTo(addr, kShadowCell);
    size = RoundUpTo(size, kShadowCell);

    RawShadow *Begin = MemToShadow_CPU(addr, 0);
    TSAN_DEBUG(__spirv_ocl_printf(
        __tsan_print_cleanup_private, Begin,
        (uptr)Begin + size / kShadowCell * kShadowCnt * kShadowSize - 1));
    for (uptr i = 0; i < size / kShadowCell * kShadowCnt; i++)
      Begin[i] = 0;
  }
}

DEVICE_EXTERN_C_NOINLINE void __tsan_cleanup_private(uptr addr, size_t size) {
#if defined(__LIBDEVICE_CPU__)
  __tsan_cleanup_private_cpu_impl(addr, size);
#elif defined(__LIBDEVICE_PVC__)
  return;
#else
  if (TsanLaunchInfo->DeviceTy != DeviceType::CPU)
    return;

  __tsan_cleanup_private_cpu_impl(addr, size);
#endif
}

static __SYCL_CONSTANT__ const char __tsan_print_cleanup_local[] =
    "[kernel] cleanup shadow (%p ~ %p) for local %p\n";

DEVICE_EXTERN_C_NOINLINE void __tsan_cleanup_static_local(uptr addr,
                                                          size_t size) {
  // Update shadow memory of local memory only on first work-item
  if (__spirv_LocalInvocationId_x() + __spirv_LocalInvocationId_y() +
          __spirv_LocalInvocationId_z() ==
      0) {
    if (TsanLaunchInfo->LocalShadowOffset == 0)
      return;

    addr = RoundDownTo(addr, kShadowCell);
    size = RoundUpTo(size, kShadowCell);

    RawShadow *Begin = MemToShadow(addr, ADDRESS_SPACE_LOCAL);
    for (uptr i = 0; i < size / kShadowCell * kShadowCnt; i++)
      Begin[i] = 0;

    TSAN_DEBUG(__spirv_ocl_printf(
        __tsan_print_cleanup_local, addr, Begin,
        (uptr)Begin + size / kShadowCell * kShadowCnt * kShadowSize - 1));
  }
}

static __SYCL_CONSTANT__ const char __tsan_print_report_arg_count_incorrect[] =
    "[kernel] ERROR: The number of local args is incorrect, expect %d, actual "
    "%d\n";

DEVICE_EXTERN_C_NOINLINE void __tsan_cleanup_dynamic_local(uptr ptr,
                                                           uint32_t num_args) {
  if (!TsanLaunchInfo->LocalShadowOffset)
    return;

  if (num_args != TsanLaunchInfo->NumLocalArgs) {
    __spirv_ocl_printf(__tsan_print_report_arg_count_incorrect, num_args,
                       TsanLaunchInfo->NumLocalArgs);
    return;
  }

  uptr *args = (uptr *)ptr;

  for (uint32_t i = 0; i < num_args; ++i) {
    auto *local_arg = &TsanLaunchInfo->LocalArgs[i];

    __tsan_cleanup_static_local(args[i], local_arg->Size);
  }
}

DEVICE_EXTERN_C_INLINE void __tsan_device_barrier() {
  Sid sid = GetCurrentSid();

  // sync current thread clock to global state
  TsanLaunchInfo->Clock[kThreadSlotCount].clk_[sid] =
      TsanLaunchInfo->Clock[sid].clk_[sid];

  __spirv_ControlBarrier(__spv::Scope::Device, __spv::Scope::Device,
                         __spv::MemorySemanticsMask::SequentiallyConsistent |
                             __spv::MemorySemanticsMask::CrossWorkgroupMemory |
                             __spv::MemorySemanticsMask::WorkgroupMemory);

  // sync global state back
  for (uptr i = 0; i < kThreadSlotCount; i++)
    TsanLaunchInfo->Clock[sid].clk_[i] =
        TsanLaunchInfo->Clock[kThreadSlotCount].clk_[i];
}

DEVICE_EXTERN_C_INLINE void __tsan_group_barrier() {
  Sid sid = GetCurrentSid();

  // sync current thread clock to global state
  TsanLaunchInfo->Clock[kThreadSlotCount].clk_[sid] =
      TsanLaunchInfo->Clock[sid].clk_[sid];

  __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
                         __spv::MemorySemanticsMask::SequentiallyConsistent |
                             __spv::MemorySemanticsMask::CrossWorkgroupMemory |
                             __spv::MemorySemanticsMask::WorkgroupMemory);

  // sync global state back
  for (uptr i = 0; i < kThreadSlotCount; i++)
    TsanLaunchInfo->Clock[sid].clk_[i] =
        TsanLaunchInfo->Clock[kThreadSlotCount].clk_[i];
}

#endif // __SPIR__ || __SPIRV__
