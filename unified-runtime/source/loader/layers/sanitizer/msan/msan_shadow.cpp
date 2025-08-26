/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_shadow.cpp
 *
 */

#include "msan_shadow.hpp"
#include "msan_interceptor.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"
#include "ur_api.h"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {
namespace msan {

namespace {

//
// The CPU part of shadow mapping is based on llvm/compiler-rt/lib/msan/msan.h
//
struct MappingDesc {
  uptr start;
  uptr end;
  enum Type {
    INVALID = 1,
    ALLOCATOR = 2,
    APP = 4,
    SHADOW = 8,
    ORIGIN = 16,
  } type;
  const char *name;
};

const MappingDesc kMemoryLayout[] = {
    {0x000000000000ULL, 0x010000000000ULL, MappingDesc::APP, "app-1"},
    {0x010000000000ULL, 0x100000000000ULL, MappingDesc::SHADOW, "shadow-2"},
    {0x100000000000ULL, 0x110000000000ULL, MappingDesc::INVALID, "invalid"},
    {0x110000000000ULL, 0x200000000000ULL, MappingDesc::ORIGIN, "origin-2"},
    {0x200000000000ULL, 0x300000000000ULL, MappingDesc::SHADOW, "shadow-3"},
    {0x300000000000ULL, 0x400000000000ULL, MappingDesc::ORIGIN, "origin-3"},
    {0x400000000000ULL, 0x500000000000ULL, MappingDesc::INVALID, "invalid"},
    {0x500000000000ULL, 0x510000000000ULL, MappingDesc::SHADOW, "shadow-1"},
    {0x510000000000ULL, 0x600000000000ULL, MappingDesc::APP, "app-2"},
    {0x600000000000ULL, 0x610000000000ULL, MappingDesc::ORIGIN, "origin-1"},
    {0x610000000000ULL, 0x700000000000ULL, MappingDesc::INVALID, "invalid"},
    {0x700000000000ULL, 0x740000000000ULL, MappingDesc::ALLOCATOR, "allocator"},
    {0x740000000000ULL, 0x800000000000ULL, MappingDesc::APP, "app-3"}};

const uptr kMemoryLayoutSize = sizeof(kMemoryLayout) / sizeof(kMemoryLayout[0]);

#define MEM_TO_SHADOW(mem) (((uptr)(mem)) ^ 0x500000000000ULL)
#define SHADOW_TO_ORIGIN(mem) (((uptr)(mem)) + 0x100000000000ULL)

} // namespace

std::shared_ptr<MsanShadowMemory>
GetMsanShadowMemory(ur_context_handle_t Context, ur_device_handle_t Device,
                    DeviceType Type) {
  if (Type == DeviceType::CPU) {
    static std::shared_ptr<MsanShadowMemory> ShadowCPU =
        std::make_shared<MsanShadowMemoryCPU>(Context, Device);
    return ShadowCPU;
  } else if (Type == DeviceType::GPU_PVC) {
    static std::shared_ptr<MsanShadowMemory> ShadowPVC =
        std::make_shared<MsanShadowMemoryPVC>(Context, Device);
    return ShadowPVC;
  } else if (Type == DeviceType::GPU_DG2) {
    static std::shared_ptr<MsanShadowMemory> ShadowDG2 =
        std::make_shared<MsanShadowMemoryDG2>(Context, Device);
    return ShadowDG2;
  }

  die("GetMsanShadowMemory: Unsupport device type");
  return nullptr;
}

ur_result_t MsanShadowMemoryCPU::Setup() {
  static ur_result_t Result = [this]() {
    for (unsigned i = 0; i < kMemoryLayoutSize; ++i) {
      uptr Start = kMemoryLayout[i].start;
      uptr End = kMemoryLayout[i].end;
      uptr Size = End - Start;
      MappingDesc::Type Type = kMemoryLayout[i].type;

      bool IsMap = Type == MappingDesc::SHADOW || Type == MappingDesc::ORIGIN;
      bool IsProtect = Type == MappingDesc::INVALID;

      if (IsMap) {
        if (MmapFixedNoReserve(Start, Size) == 0) {
          return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
        DontCoredumpRange(Start, Size);
      }
      if (IsProtect) {
        if (ProtectMemoryRange(Start, Size) == 0) {
          return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
      }
    }
    ShadowBegin = kMemoryLayout[1].start;
    ShadowEnd = kMemoryLayout[9].end;
    return UR_RESULT_SUCCESS;
  }();
  return Result;
}

ur_result_t MsanShadowMemoryCPU::Destory() {
  if (ShadowBegin == 0 && ShadowEnd == 0) {
    return UR_RESULT_SUCCESS;
  }
  static ur_result_t Result = [this]() {
    for (unsigned i = 0; i < kMemoryLayoutSize; ++i) {
      uptr Start = kMemoryLayout[i].start;
      uptr End = kMemoryLayout[i].end;
      uptr Size = End - Start;
      MappingDesc::Type Type = kMemoryLayout[i].type;
      bool IsMap = Type == MappingDesc::SHADOW || Type == MappingDesc::ORIGIN;
      if (IsMap) {
        if (Munmap(Start, Size)) {
          return UR_RESULT_ERROR_UNKNOWN;
        }
      }
    }
    ShadowBegin = ShadowEnd = 0;
    return UR_RESULT_SUCCESS;
  }();
  return Result;
}

uptr MsanShadowMemoryCPU::MemToShadow(uptr Ptr) { return MEM_TO_SHADOW(Ptr); }

uptr MsanShadowMemoryCPU::MemToOrigin(uptr Ptr) {
  uptr AlignedPtr = RoundDownTo(Ptr, MSAN_ORIGIN_GRANULARITY);
  return SHADOW_TO_ORIGIN(MEM_TO_SHADOW(AlignedPtr));
}

ur_result_t MsanShadowMemoryCPU::EnqueuePoisonShadow(
    ur_queue_handle_t Queue, uptr Ptr, uptr Size, u8 Value, uint32_t NumEvents,
    const ur_event_handle_t *EventWaitList, ur_event_handle_t *OutEvent) {
  return EnqueuePoisonShadowWithOrigin(Queue, Ptr, Size, Value, 0, NumEvents,
                                       EventWaitList, OutEvent);
}

ur_result_t MsanShadowMemoryCPU::EnqueuePoisonShadowWithOrigin(
    ur_queue_handle_t Queue, uptr Ptr, uptr Size, u8 Value, uint32_t Origin,
    uint32_t NumEvents, const ur_event_handle_t *EventWaitList,
    ur_event_handle_t *OutEvent) {
  if (Size) {
    {
      const uptr ShadowBegin = MemToShadow(Ptr);
      const uptr ShadowEnd = MemToShadow(Ptr + Size - 1);
      assert(ShadowBegin <= ShadowEnd);
      UR_LOG_L(getContext()->logger, DEBUG,
               "EnqueuePoisonShadow(addr={}, count={}, value={})",
               (void *)ShadowBegin, ShadowEnd - ShadowBegin + 1,
               (void *)(uptr)Value);
      memset((void *)ShadowBegin, Value, ShadowEnd - ShadowBegin + 1);
    }
    {
      const uptr OriginBegin = MemToOrigin(Ptr);
      const uptr OriginEnd =
          MemToOrigin(Ptr + Size - 1) + MSAN_ORIGIN_GRANULARITY;
      assert(OriginBegin <= OriginEnd);
      UR_LOG_L(getContext()->logger, DEBUG,
               "EnqueuePoisonOrigin(addr={}, count={}, value={})",
               (void *)OriginBegin, OriginEnd - OriginBegin + 1,
               (void *)(uptr)Origin);
      // memset((void *)OriginBegin, Value, OriginEnd - OriginBegin + 1);
      std::fill((uint32_t *)OriginBegin, (uint32_t *)OriginEnd, Origin);
    }
  }

  if (OutEvent) {
    UR_CALL(getContext()->urDdiTable.Enqueue.pfnEventsWait(
        Queue, NumEvents, EventWaitList, OutEvent));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t MsanShadowMemoryGPU::Setup() {
  // Currently, Level-Zero doesn't create independent VAs for each contexts, if
  // we reserve shadow memory for each contexts, this will cause out-of-resource
  // error when user uses multiple contexts. Therefore, we just create one
  // shadow memory here.
  static ur_result_t Result = [this]() {
    const size_t ShadowSize = GetShadowSize();
    // To reserve very large amount of GPU virtual memroy, the pStart param
    // should be beyond the SVM range, so that GFX driver will automatically
    // switch to reservation on the GPU heap.
    const void *StartAddress = (void *)GetStartAddress();
    // TODO: Protect Bad Zone
    auto Result = getContext()->urDdiTable.VirtualMem.pfnReserve(
        Context, StartAddress, ShadowSize, (void **)&ShadowBegin);
    if (Result != UR_RESULT_SUCCESS) {
      UR_LOG_L(getContext()->logger, ERR,
               "Shadow memory reserved failed with size {}: {}",
               (void *)ShadowSize, Result);
      return Result;
    }
    ShadowEnd = ShadowBegin + ShadowSize;
    // Retain the context which reserves shadow memory
    getContext()->urDdiTable.Context.pfnRetain(Context);
    return UR_RESULT_SUCCESS;
  }();
  return Result;
}

ur_result_t MsanShadowMemoryGPU::Destory() {
  if (ShadowBegin == 0) {
    return UR_RESULT_SUCCESS;
  }
  static ur_result_t Result = [this]() {
    auto Result = getContext()->urDdiTable.VirtualMem.pfnFree(
        Context, (const void *)ShadowBegin, GetShadowSize());

    if (PrivateShadowOffset != 0) {
      UR_CALL(getContext()->urDdiTable.USM.pfnFree(
          Context, (void *)PrivateShadowOffset));
      PrivateShadowOffset = 0;
    }

    if (PrivateBasePtr != 0) {
      UR_CALL(getContext()->urDdiTable.USM.pfnFree(Context,
                                                   (void *)PrivateBasePtr));
      PrivateBasePtr = 0;
    }

    if (LocalShadowOffset != 0) {
      UR_CALL(getContext()->urDdiTable.USM.pfnFree(Context,
                                                   (void *)LocalShadowOffset));
      LocalShadowOffset = 0;
    }
    getContext()->urDdiTable.Context.pfnRelease(Context);
    return Result;
  }();
  return Result;
}

ur_result_t MsanShadowMemoryGPU::EnqueueVirtualMemMap(
    uptr VirtualBegin, uptr VirtualEnd,
    std::vector<ur_event_handle_t> &EventWaitList,
    ur_event_handle_t *OutEvent) {
  const size_t PageSize = GetVirtualMemGranularity(Context, Device);
  // Make sure [Ptr, Ptr + Size] is mapped to physical memory
  for (auto MappedPtr = RoundDownTo(VirtualBegin, PageSize);
       MappedPtr <= VirtualEnd; MappedPtr += PageSize) {
    std::scoped_lock<ur_mutex> Guard(VirtualMemMapsMutex);
    if (VirtualMemMaps.find(MappedPtr) == VirtualMemMaps.end()) {
      ur_physical_mem_handle_t PhysicalMem{};
      auto URes = getContext()->urDdiTable.PhysicalMem.pfnCreate(
          Context, Device, PageSize, nullptr, &PhysicalMem);
      if (URes != UR_RESULT_SUCCESS) {
        UR_LOG_L(getContext()->logger, ERR, "urPhysicalMemCreate(): {}", URes);
        return URes;
      }

      URes = getContext()->urDdiTable.VirtualMem.pfnMap(
          Context, (void *)MappedPtr, PageSize, PhysicalMem, 0,
          UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE);
      if (URes != UR_RESULT_SUCCESS) {
        UR_LOG_L(getContext()->logger, ERR, "urVirtualMemMap({}, {}): {}",
                 (void *)MappedPtr, PageSize, URes);
        return URes;
      }

      UR_LOG_L(getContext()->logger, DEBUG, "urVirtualMemMap: {} ~ {}",
               (void *)MappedPtr, (void *)(MappedPtr + PageSize - 1));

      EventWaitList.clear();
      if (OutEvent) {
        EventWaitList.push_back(*OutEvent);
      }

      VirtualMemMaps[MappedPtr].first = PhysicalMem;
    }
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t MsanShadowMemoryGPU::EnqueuePoisonShadow(
    ur_queue_handle_t Queue, uptr Ptr, uptr Size, u8 Value, uint32_t NumEvents,
    const ur_event_handle_t *EventWaitList, ur_event_handle_t *OutEvent) {
  return EnqueuePoisonShadowWithOrigin(Queue, Ptr, Size, Value, 0, NumEvents,
                                       EventWaitList, OutEvent);
}

ur_result_t MsanShadowMemoryGPU::EnqueuePoisonShadowWithOrigin(
    ur_queue_handle_t Queue, uptr Ptr, uptr Size, u8 Value, uint32_t Origin,
    uint32_t NumEvents, const ur_event_handle_t *EventWaitList,
    ur_event_handle_t *OutEvent) {
  if (Size == 0) {
    if (OutEvent) {
      UR_CALL(getContext()->urDdiTable.Enqueue.pfnEventsWait(
          Queue, NumEvents, EventWaitList, OutEvent));
    }
    return UR_RESULT_SUCCESS;
  }

  std::vector<ur_event_handle_t> Events(EventWaitList,
                                        EventWaitList + NumEvents);
  {
    uptr ShadowBegin = MemToShadow(Ptr);
    uptr ShadowEnd = MemToShadow(Ptr + Size - 1);
    assert(ShadowBegin <= ShadowEnd);

    UR_CALL(EnqueueVirtualMemMap(ShadowBegin, ShadowEnd, Events, OutEvent));

    UR_LOG_L(getContext()->logger, DEBUG,
             "EnqueuePoisonShadow(addr={}, size={}, value={})",
             (void *)ShadowBegin, ShadowEnd - ShadowBegin + 1,
             (void *)(size_t)Value);

    UR_CALL(EnqueueUSMSet(Queue, (void *)ShadowBegin, Value,
                          ShadowEnd - ShadowBegin + 1, Events.size(),
                          Events.data(), OutEvent));
  }

  {
    uptr OriginBegin = MemToOrigin(Ptr);
    uptr OriginEnd = MemToOrigin(Ptr + Size - 1) + sizeof(Origin) - 1;
    UR_CALL(EnqueueVirtualMemMap(OriginBegin, OriginEnd, Events, OutEvent));

    UR_LOG_L(getContext()->logger, DEBUG,
             "EnqueuePoisonOrigin(addr={}, size={}, value={})",
             (void *)OriginBegin, OriginEnd - OriginBegin + 1,
             (void *)(uptr)Origin);

    UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMFill(
        Queue, (void *)OriginBegin, sizeof(Origin), &Origin,
        OriginEnd - OriginBegin + 1, NumEvents, EventWaitList, OutEvent));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t
MsanShadowMemoryGPU::ReleaseShadow(std::shared_ptr<MsanAllocInfo> AI) {
  uptr ShadowBegin = MemToShadow(AI->AllocBegin);
  uptr ShadowEnd = MemToShadow(AI->AllocBegin + AI->AllocSize);
  assert(ShadowBegin <= ShadowEnd);

  static const size_t PageSize = GetVirtualMemGranularity(Context, Device);

  for (auto MappedPtr = RoundDownTo(ShadowBegin, PageSize);
       MappedPtr <= ShadowEnd; MappedPtr += PageSize) {
    std::scoped_lock<ur_mutex> Guard(VirtualMemMapsMutex);
    if (VirtualMemMaps.find(MappedPtr) == VirtualMemMaps.end()) {
      continue;
    }
    VirtualMemMaps[MappedPtr].second.erase(AI);
    if (VirtualMemMaps[MappedPtr].second.empty()) {
      UR_CALL(getContext()->urDdiTable.VirtualMem.pfnUnmap(
          Context, (void *)MappedPtr, PageSize));
      UR_CALL(getContext()->urDdiTable.PhysicalMem.pfnRelease(
          VirtualMemMaps[MappedPtr].first));
      UR_LOG_L(getContext()->logger, DEBUG, "urVirtualMemUnmap: {} ~ {}",
               (void *)MappedPtr, (void *)(MappedPtr + PageSize - 1));
    }
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t MsanShadowMemoryGPU::AllocLocalShadow(ur_queue_handle_t Queue,
                                                  uint32_t NumWG, uptr &Begin,
                                                  uptr &End) {
  const size_t LocalMemorySize = GetDeviceLocalMemorySize(Device);
  const size_t RequiredShadowSize =
      std::min(NumWG, MSAN_MAX_WG_LOCAL) * LocalMemorySize;
  static size_t LastAllocedSize = 0;
  if (RequiredShadowSize > LastAllocedSize) {
    auto ContextInfo = getMsanInterceptor()->getContextInfo(Context);
    if (LocalShadowOffset) {
      UR_CALL(getContext()->urDdiTable.USM.pfnFree(Context,
                                                   (void *)LocalShadowOffset));
      LocalShadowOffset = 0;
      LastAllocedSize = 0;
    }

    UR_CALL(getContext()->urDdiTable.USM.pfnDeviceAlloc(
        Context, Device, nullptr, nullptr, RequiredShadowSize,
        (void **)&LocalShadowOffset));

    // Initialize shadow memory
    ur_result_t URes = EnqueueUSMSet(Queue, (void *)LocalShadowOffset, (char)0,
                                     RequiredShadowSize);
    if (URes != UR_RESULT_SUCCESS) {
      UR_CALL(getContext()->urDdiTable.USM.pfnFree(Context,
                                                   (void *)LocalShadowOffset));
      LocalShadowOffset = 0;
      LastAllocedSize = 0;

      return URes;
    }

    LastAllocedSize = RequiredShadowSize;
  }

  Begin = LocalShadowOffset;
  End = LocalShadowOffset + RequiredShadowSize - 1;
  return UR_RESULT_SUCCESS;
}

ur_result_t MsanShadowMemoryGPU::AllocPrivateShadow(ur_queue_handle_t Queue,
                                                    uint32_t NumSG, uptr *&Base,
                                                    uptr &Begin, uptr &End) {
  // Trying to allocate private base array and private shadow, and any one of
  // them fail to allocate would be a failure
  static size_t LastPrivateBaseAllocedSize = 0;
  static size_t LastPrivateShadowAllocedSize = 0;

  NumSG = std::min(NumSG, MSAN_MAX_SG_PRIVATE);

  try {
    const size_t NewPrivateBaseSize = NumSG * sizeof(uptr);
    if (NewPrivateBaseSize > LastPrivateBaseAllocedSize) {
      if (PrivateBasePtr) {
        UR_CALL_THROWS(getContext()->urDdiTable.USM.pfnFree(
            Context, (void *)PrivateBasePtr));
        PrivateBasePtr = 0;
        LastPrivateBaseAllocedSize = 0;
      }

      ur_usm_desc_t PrivateBaseProps{UR_STRUCTURE_TYPE_USM_DESC, nullptr,
                                     UR_USM_ADVICE_FLAG_DEFAULT, sizeof(uptr)};
      UR_CALL_THROWS(getContext()->urDdiTable.USM.pfnDeviceAlloc(
          Context, Device, &PrivateBaseProps, nullptr, NewPrivateBaseSize,
          (void **)&PrivateBasePtr));

      // No need to clean the shadow base, their should be set by work item on
      // launch

      // FIXME: we should add private base to statistic
      LastPrivateBaseAllocedSize = NewPrivateBaseSize;
    }

    const size_t NewPrivateShadowSize = NumSG * MSAN_PRIVATE_SIZE;
    if (NewPrivateShadowSize > LastPrivateShadowAllocedSize) {

      if (PrivateShadowOffset) {
        UR_CALL_THROWS(getContext()->urDdiTable.USM.pfnFree(
            Context, (void *)PrivateShadowOffset));
        PrivateShadowOffset = 0;
        LastPrivateShadowAllocedSize = 0;
      }

      UR_CALL_THROWS(getContext()->urDdiTable.USM.pfnDeviceAlloc(
          Context, Device, nullptr, nullptr, NewPrivateShadowSize,
          (void **)&PrivateShadowOffset));
      LastPrivateShadowAllocedSize = NewPrivateShadowSize;
      UR_CALL_THROWS(EnqueueUSMSet(Queue, (void *)PrivateShadowOffset, (char)0,
                                   NewPrivateShadowSize));
    }

    Base = (uptr *)PrivateBasePtr;
    Begin = PrivateShadowOffset;
    End = PrivateShadowOffset + NewPrivateShadowSize - 1;

  } catch (ur_result_t &UrRes) {
    assert(UrRes != UR_RESULT_SUCCESS);

    if (PrivateBasePtr) {
      UR_CALL_NOCHECK(getContext()->urDdiTable.USM.pfnFree(
          Context, (void *)PrivateBasePtr));
      PrivateBasePtr = 0;
      LastPrivateBaseAllocedSize = 0;
    }

    if (PrivateShadowOffset) {
      UR_CALL_NOCHECK(getContext()->urDdiTable.USM.pfnFree(
          Context, (void *)PrivateShadowOffset));
      PrivateShadowOffset = 0;
      LastPrivateShadowAllocedSize = 0;
    }

    return UrRes;
  }

  return UR_RESULT_SUCCESS;
}

uptr MsanShadowMemoryPVC::MemToShadow(uptr Ptr) {
  if (MsanShadowMemoryPVC::IsDeviceUSM(Ptr)) {
    return Ptr - 0x5000'0000'0000ULL;
  }
  // host/shared USM
  return (Ptr & 0xfff'ffff'ffffULL) + ((Ptr & 0x8000'0000'0000ULL) >> 3) +
         ShadowBegin;
}

uptr MsanShadowMemoryPVC::MemToOrigin(uptr Ptr) {
  uptr AlignedPtr = RoundDownTo(Ptr, MSAN_ORIGIN_GRANULARITY);
  if (MsanShadowMemoryPVC::IsDeviceUSM(AlignedPtr)) {
    return AlignedPtr - 0xA000'0000'0000ULL;
  }
  // host/shared USM
  return (AlignedPtr & 0xfff'ffff'ffffULL) +
         ((AlignedPtr & 0x8000'0000'0000ULL) >> 3) + ShadowBegin +
         0x2000'0000'0000ULL;
}

uptr MsanShadowMemoryDG2::MemToShadow(uptr Ptr) {
  assert(MsanShadowMemoryDG2::IsDeviceUSM(Ptr) && "Ptr must be device USM");
  if (Ptr < ShadowBegin) {
    return Ptr + (ShadowBegin - 0xffff'8000'0000'0000ULL);
  } else {
    return Ptr - (0xffff'ffff'ffff'ffffULL - ShadowEnd + 1);
  }
}

uptr MsanShadowMemoryDG2::MemToOrigin(uptr Ptr) {
  assert(MsanShadowMemoryDG2::IsDeviceUSM(Ptr) && "Ptr must be device USM");
  if (Ptr < ShadowBegin) {
    return Ptr + (ShadowBegin - 0xffff'8000'0000'0000ULL);
  } else {
    return Ptr - (0xffff'ffff'ffff'ffffULL - ShadowEnd + 1);
  }
}

} // namespace msan
} // namespace ur_sanitizer_layer
