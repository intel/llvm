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

#define CPU_SHADOW1_BEGIN 0x010000000000ULL
#define CPU_SHADOW1_END 0x100000000000ULL
#define CPU_SHADOW2_BEGIN 0x200000000000ULL
#define CPU_SHADOW2_END 0x300000000000ULL
#define CPU_SHADOW3_BEGIN 0x500000000000ULL
#define CPU_SHADOW3_END 0x510000000000ULL

#define CPU_SHADOW_MASK 0x500000000000ULL

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
  } else {
    UR_LOG_L(getContext()->logger, ERR, "Unsupport device type");
    return nullptr;
  }
}

ur_result_t MsanShadowMemoryCPU::Setup() {
  static ur_result_t Result = [this]() {
    if (MmapFixedNoReserve(CPU_SHADOW1_BEGIN,
                           CPU_SHADOW1_END - CPU_SHADOW1_BEGIN) == 0) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (ProtectMemoryRange(CPU_SHADOW1_END,
                           CPU_SHADOW2_BEGIN - CPU_SHADOW1_END) == 0) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (MmapFixedNoReserve(CPU_SHADOW2_BEGIN,
                           CPU_SHADOW2_END - CPU_SHADOW2_BEGIN) == 0) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (ProtectMemoryRange(CPU_SHADOW2_END,
                           CPU_SHADOW3_BEGIN - CPU_SHADOW2_END) == 0) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (MmapFixedNoReserve(CPU_SHADOW3_BEGIN,
                           CPU_SHADOW3_END - CPU_SHADOW3_BEGIN) == 0) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }
    ShadowBegin = CPU_SHADOW1_BEGIN;
    ShadowEnd = CPU_SHADOW3_END;
    DontCoredumpRange(ShadowBegin, ShadowEnd - ShadowBegin);
    return UR_RESULT_SUCCESS;
  }();
  return Result;
}

ur_result_t MsanShadowMemoryCPU::Destory() {
  if (ShadowBegin == 0 && ShadowEnd == 0) {
    return UR_RESULT_SUCCESS;
  }
  static ur_result_t Result = [this]() {
    if (!Munmap(CPU_SHADOW1_BEGIN, CPU_SHADOW1_END - CPU_SHADOW1_BEGIN)) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
    if (!Munmap(CPU_SHADOW1_END, CPU_SHADOW2_BEGIN - CPU_SHADOW1_END)) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
    if (!Munmap(CPU_SHADOW2_BEGIN, CPU_SHADOW2_END - CPU_SHADOW2_BEGIN) == 0) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
    if (!Munmap(CPU_SHADOW2_END, CPU_SHADOW3_BEGIN - CPU_SHADOW2_END)) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
    if (!Munmap(CPU_SHADOW3_BEGIN, CPU_SHADOW3_END - CPU_SHADOW3_BEGIN) == 0) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
    ShadowBegin = ShadowEnd = 0;
    return UR_RESULT_SUCCESS;
  }();
  return Result;
}

uptr MsanShadowMemoryCPU::MemToShadow(uptr Ptr) {
  return Ptr ^ CPU_SHADOW_MASK;
}

ur_result_t MsanShadowMemoryCPU::EnqueuePoisonShadow(
    ur_queue_handle_t Queue, uptr Ptr, uptr Size, u8 Value, uint32_t NumEvents,
    const ur_event_handle_t *EventWaitList, ur_event_handle_t *OutEvent) {

  if (Size) {
    const uptr ShadowBegin = MemToShadow(Ptr);
    const uptr ShadowEnd = MemToShadow(Ptr + Size - 1);
    assert(ShadowBegin <= ShadowEnd);
    UR_LOG_L(getContext()->logger, DEBUG,
             "EnqueuePoisonShadow(addr={}, count={}, value={})",
             (void *)ShadowBegin, ShadowEnd - ShadowBegin + 1,
             (void *)(size_t)Value);
    memset((void *)ShadowBegin, Value, ShadowEnd - ShadowBegin + 1);
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

ur_result_t MsanShadowMemoryGPU::EnqueueMapShadow(
    ur_queue_handle_t Queue, uptr Ptr, uptr Size,
    std::vector<ur_event_handle_t> &EventWaitList,
    ur_event_handle_t *OutEvent) {

  const size_t PageSize = GetVirtualMemGranularity(Context, Device);

  const uptr ShadowBegin = MemToShadow(Ptr);
  const uptr ShadowEnd = MemToShadow(Ptr + Size - 1);
  assert(ShadowBegin <= ShadowEnd);

  // Make sure [Ptr, Ptr + Size] is mapped to physical memory
  for (auto MappedPtr = RoundDownTo(ShadowBegin, PageSize);
       MappedPtr <= ShadowEnd; MappedPtr += PageSize) {
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

      // Initialize to zero
      URes = EnqueueUSMBlockingSet(Queue, (void *)MappedPtr, 0, PageSize,
                                   EventWaitList.size(), EventWaitList.data(),
                                   OutEvent);
      if (URes != UR_RESULT_SUCCESS) {
        UR_LOG_L(getContext()->logger, ERR, "EnqueueUSMSet(): {}", URes);
        return URes;
      }

      EventWaitList.clear();
      if (OutEvent) {
        EventWaitList.push_back(*OutEvent);
      }

      VirtualMemMaps[MappedPtr].first = PhysicalMem;
    }

    auto AllocInfoItOp = getMsanInterceptor()->findAllocInfoByAddress(Ptr);
    if (AllocInfoItOp) {
      VirtualMemMaps[MappedPtr].second.insert((*AllocInfoItOp)->second);
    }
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t MsanShadowMemoryGPU::EnqueuePoisonShadow(
    ur_queue_handle_t Queue, uptr Ptr, uptr Size, u8 Value, uint32_t NumEvents,
    const ur_event_handle_t *EventWaitList, ur_event_handle_t *OutEvent) {
  if (Size == 0) {
    if (OutEvent) {
      UR_CALL(getContext()->urDdiTable.Enqueue.pfnEventsWait(
          Queue, NumEvents, EventWaitList, OutEvent));
    }
    return UR_RESULT_SUCCESS;
  }

  std::vector<ur_event_handle_t> Events(EventWaitList,
                                        EventWaitList + NumEvents);
  UR_CALL(EnqueueMapShadow(Queue, Ptr, Size, Events, OutEvent));

  const uptr ShadowBegin = MemToShadow(Ptr);
  const uptr ShadowEnd = MemToShadow(Ptr + Size - 1);
  assert(ShadowBegin <= ShadowEnd);

  auto Result = EnqueueUSMBlockingSet(Queue, (void *)ShadowBegin, Value,
                                      ShadowEnd - ShadowBegin + 1,
                                      Events.size(), Events.data(), OutEvent);

  UR_LOG_L(getContext()->logger, DEBUG,
           "EnqueuePoisonShadow(addr={}, count={}, value={}): {}",
           (void *)ShadowBegin, ShadowEnd - ShadowBegin + 1,
           (void *)(size_t)Value, Result);

  return Result;
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
  const size_t RequiredShadowSize = NumWG * LocalMemorySize;
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
    ur_result_t URes = EnqueueUSMBlockingSet(Queue, (void *)LocalShadowOffset,
                                             0, RequiredShadowSize);
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
                                                    uint64_t NumWI,
                                                    uint32_t NumWG, uptr *&Base,
                                                    uptr &Begin, uptr &End) {
  {
    const size_t Size = NumWI * sizeof(uptr);
    ur_usm_desc_t Properties{UR_STRUCTURE_TYPE_USM_DESC, nullptr,
                             UR_USM_ADVICE_FLAG_DEFAULT, sizeof(uptr)};
    UR_CALL(getContext()->urDdiTable.USM.pfnDeviceAlloc(
        Context, Device, &Properties, nullptr, Size, (void **)&Base));
  }

  {
    const size_t RequiredShadowSize = NumWG * MSAN_PRIVATE_SIZE;
    static size_t LastAllocedSize = 0;
    if (RequiredShadowSize > LastAllocedSize) {
      auto ContextInfo = getMsanInterceptor()->getContextInfo(Context);
      if (PrivateShadowOffset) {
        UR_CALL(getContext()->urDdiTable.USM.pfnFree(
            Context, (void *)PrivateShadowOffset));
        PrivateShadowOffset = 0;
        LastAllocedSize = 0;
      }

      UR_CALL(getContext()->urDdiTable.USM.pfnDeviceAlloc(
          Context, Device, nullptr, nullptr, RequiredShadowSize,
          (void **)&PrivateShadowOffset));

      // Initialize shadow memory
      ur_result_t URes = EnqueueUSMBlockingSet(
          Queue, (void *)PrivateShadowOffset, 0, RequiredShadowSize);
      if (URes != UR_RESULT_SUCCESS) {
        UR_CALL(getContext()->urDdiTable.USM.pfnFree(
            Context, (void *)PrivateShadowOffset));
        PrivateShadowOffset = 0;
        LastAllocedSize = 0;
      }

      LastAllocedSize = RequiredShadowSize;
    }

    Begin = PrivateShadowOffset;
    End = PrivateShadowOffset + RequiredShadowSize - 1;
  }

  return UR_RESULT_SUCCESS;
}

uptr MsanShadowMemoryPVC::MemToShadow(uptr Ptr) {
  assert(MsanShadowMemoryPVC::IsDeviceUSM(Ptr) && "Ptr must be device USM");
  if (Ptr < ShadowBegin) {
    return Ptr + (ShadowBegin - 0xff00'0000'0000'0000ULL);
  } else {
    return Ptr - (0xff00'ffff'ffff'ffffULL - ShadowEnd + 1);
  }
}

uptr MsanShadowMemoryDG2::MemToShadow(uptr Ptr) {
  assert(MsanShadowMemoryDG2::IsDeviceUSM(Ptr) && "Ptr must be device USM");
  if (Ptr < ShadowBegin) {
    return Ptr + (ShadowBegin - 0xffff'8000'0000'0000ULL);
  } else {
    return Ptr - (0xffff'ffff'ffff'ffffULL - ShadowEnd + 1);
  }
}

} // namespace msan
} // namespace ur_sanitizer_layer
