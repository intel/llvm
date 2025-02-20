/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_shadow.cpp
 *
 */

#include "asan_shadow.hpp"
#include "asan_interceptor.hpp"
#include "asan_libdevice.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {
namespace asan {

std::shared_ptr<ShadowMemory> GetShadowMemory(ur_context_handle_t Context,
                                              ur_device_handle_t Device,
                                              DeviceType Type) {
  if (Type == DeviceType::CPU) {
    static std::shared_ptr<ShadowMemory> ShadowCPU =
        std::make_shared<ShadowMemoryCPU>(Context, Device);
    return ShadowCPU;
  } else if (Type == DeviceType::GPU_PVC) {
    static std::shared_ptr<ShadowMemory> ShadowPVC =
        std::make_shared<ShadowMemoryPVC>(Context, Device);
    return ShadowPVC;
  } else if (Type == DeviceType::GPU_DG2) {
    static std::shared_ptr<ShadowMemory> ShadowDG2 =
        std::make_shared<ShadowMemoryDG2>(Context, Device);
    return ShadowDG2;
  } else {
    getContext()->logger.error("Unsupport device type");
    return nullptr;
  }
}

ur_result_t ShadowMemoryCPU::Setup() {
  static ur_result_t Result = [this]() {
    size_t ShadowSize = GetShadowSize();
    ShadowBegin = MmapNoReserve(0, ShadowSize);
    if (ShadowBegin == 0) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }
    DontCoredumpRange(ShadowBegin, ShadowSize);
    ShadowEnd = ShadowBegin + ShadowSize;

    // Set shadow memory for null pointer
    // For CPU, we use a typical page size of 4K bytes.
    constexpr size_t NullptrRedzoneSize = 4096;
    auto URes = EnqueuePoisonShadow({}, 0, NullptrRedzoneSize,
                                    kNullPointerRedzoneMagic);
    if (URes != UR_RESULT_SUCCESS) {
      getContext()->logger.error("EnqueuePoisonShadow(NullPointerRZ): {}",
                                 URes);
      return URes;
    }
    return URes;
  }();
  return Result;
}

ur_result_t ShadowMemoryCPU::Destory() {
  if (ShadowBegin == 0) {
    return UR_RESULT_SUCCESS;
  }
  static ur_result_t Result = [this]() {
    if (!Munmap(ShadowBegin, GetShadowSize())) {
      return UR_RESULT_ERROR_UNKNOWN;
    }
    return UR_RESULT_SUCCESS;
  }();
  return Result;
}

uptr ShadowMemoryCPU::MemToShadow(uptr Ptr) {
  return ShadowBegin + (Ptr >> ASAN_SHADOW_SCALE);
}

ur_result_t ShadowMemoryCPU::EnqueuePoisonShadow(ur_queue_handle_t, uptr Ptr,
                                                 uptr Size, u8 Value) {
  if (Size == 0) {
    return UR_RESULT_SUCCESS;
  }

  uptr ShadowBegin = MemToShadow(Ptr);
  uptr ShadowEnd = MemToShadow(Ptr + Size - 1);
  assert(ShadowBegin <= ShadowEnd);
  getContext()->logger.debug("EnqueuePoisonShadow(addr={}, count={}, value={})",
                             (void *)ShadowBegin, ShadowEnd - ShadowBegin + 1,
                             (void *)(size_t)Value);
  memset((void *)ShadowBegin, Value, ShadowEnd - ShadowBegin + 1);

  return UR_RESULT_SUCCESS;
}

ur_result_t ShadowMemoryGPU::Setup() {
  // Currently, Level-Zero doesn't create independent VAs for each contexts, if
  // we reserve shadow memory for each contexts, this will cause out-of-resource
  // error when user uses multiple contexts. Therefore, we just create one
  // shadow memory here.
  static ur_result_t Result = [this]() {
    const size_t ShadowSize = GetShadowSize();
    // To reserve very large amount of GPU virtual memroy, the pStart param
    // should be beyond the SVM range, so that GFX driver will automatically
    // switch to reservation on the GPU heap.
    const void *StartAddress = (void *)(0x100'0000'0000'0000ULL);
    // TODO: Protect Bad Zone
    auto Result = getContext()->urDdiTable.VirtualMem.pfnReserve(
        Context, StartAddress, ShadowSize, (void **)&ShadowBegin);
    if (Result != UR_RESULT_SUCCESS) {
      getContext()->logger.error(
          "Shadow memory reserved failed with size {}: {}", (void *)ShadowSize,
          Result);
      return Result;
    }
    ShadowEnd = ShadowBegin + ShadowSize;
    // Retain the context which reserves shadow memory
    getContext()->urDdiTable.Context.pfnRetain(Context);

    // Set shadow memory for null pointer
    // For GPU, wu use up to 1 page of shadow memory
    const size_t NullptrRedzoneSize = GetVirtualMemGranularity(Context, Device)
                                      << ASAN_SHADOW_SCALE;
    ManagedQueue Queue(Context, Device);
    Result = EnqueuePoisonShadow(Queue, 0, NullptrRedzoneSize,
                                 kNullPointerRedzoneMagic);
    if (Result != UR_RESULT_SUCCESS) {
      getContext()->logger.error("EnqueuePoisonShadow(NullPointerRZ): {}",
                                 Result);
      return Result;
    }
    return Result;
  }();
  return Result;
}

ur_result_t ShadowMemoryGPU::Destory() {
  if (PrivateShadowOffset != 0) {
    UR_CALL(getContext()->urDdiTable.USM.pfnFree(Context,
                                                 (void *)PrivateShadowOffset));
    PrivateShadowOffset = 0;
  }

  static ur_result_t Result = [this]() {
    const size_t PageSize = GetVirtualMemGranularity(Context, Device);
    for (auto [MappedPtr, PhysicalMem] : VirtualMemMaps) {
      UR_CALL(getContext()->urDdiTable.VirtualMem.pfnUnmap(
          Context, (void *)MappedPtr, PageSize));
      UR_CALL(getContext()->urDdiTable.PhysicalMem.pfnRelease(PhysicalMem));
    }
    UR_CALL(getContext()->urDdiTable.VirtualMem.pfnFree(
        Context, (const void *)ShadowBegin, GetShadowSize()));
    UR_CALL(getContext()->urDdiTable.Context.pfnRelease(Context));
    return UR_RESULT_SUCCESS;
  }();
  if (!Result) {
    return Result;
  }

  if (LocalShadowOffset != 0) {
    UR_CALL(getContext()->urDdiTable.USM.pfnFree(Context,
                                                 (void *)LocalShadowOffset));
    LocalShadowOffset = 0;
  }
  if (ShadowBegin != 0) {
    UR_CALL(getContext()->urDdiTable.VirtualMem.pfnFree(
        Context, (const void *)ShadowBegin, GetShadowSize()));
    UR_CALL(getContext()->urDdiTable.Context.pfnRelease(Context));
    ShadowBegin = ShadowEnd = 0;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ShadowMemoryGPU::EnqueuePoisonShadow(ur_queue_handle_t Queue,
                                                 uptr Ptr, uptr Size,
                                                 u8 Value) {
  if (Size == 0) {
    return UR_RESULT_SUCCESS;
  }

  uptr ShadowBegin = MemToShadow(Ptr);
  uptr ShadowEnd = MemToShadow(Ptr + Size - 1);
  assert(ShadowBegin <= ShadowEnd);
  {
    static const size_t PageSize = GetVirtualMemGranularity(Context, Device);

    ur_physical_mem_properties_t Desc{UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES,
                                      nullptr, 0};

    // Make sure [Ptr, Ptr + Size] is mapped to physical memory
    for (auto MappedPtr = RoundDownTo(ShadowBegin, PageSize);
         MappedPtr <= ShadowEnd; MappedPtr += PageSize) {
      std::scoped_lock<ur_mutex> Guard(VirtualMemMapsMutex);
      if (VirtualMemMaps.find(MappedPtr) == VirtualMemMaps.end()) {
        ur_physical_mem_handle_t PhysicalMem{};
        auto URes = getContext()->urDdiTable.PhysicalMem.pfnCreate(
            Context, Device, PageSize, &Desc, &PhysicalMem);
        if (URes != UR_RESULT_SUCCESS) {
          getContext()->logger.error("urPhysicalMemCreate(): {}", URes);
          return URes;
        }

        URes = getContext()->urDdiTable.VirtualMem.pfnMap(
            Context, (void *)MappedPtr, PageSize, PhysicalMem, 0,
            UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE);
        if (URes != UR_RESULT_SUCCESS) {
          getContext()->logger.error("urVirtualMemMap({}, {}): {}",
                                     (void *)MappedPtr, PageSize, URes);
          return URes;
        }

        getContext()->logger.debug("urVirtualMemMap: {} ~ {}",
                                   (void *)MappedPtr,
                                   (void *)(MappedPtr + PageSize - 1));

        // Initialize to zero
        URes = EnqueueUSMBlockingSet(Queue, (void *)MappedPtr, 0, PageSize);
        if (URes != UR_RESULT_SUCCESS) {
          getContext()->logger.error("EnqueueUSMBlockingSet(): {}", URes);
          return URes;
        }

        VirtualMemMaps[MappedPtr] = PhysicalMem;
      }
    }
  }

  auto URes = EnqueueUSMBlockingSet(Queue, (void *)ShadowBegin, Value,
                                    ShadowEnd - ShadowBegin + 1);
  getContext()->logger.debug(
      "EnqueuePoisonShadow (addr={}, count={}, value={}): {}",
      (void *)ShadowBegin, ShadowEnd - ShadowBegin + 1, (void *)(size_t)Value,
      URes);
  if (URes != UR_RESULT_SUCCESS) {
    getContext()->logger.error("EnqueueUSMBlockingSet(): {}", URes);
    return URes;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ShadowMemoryGPU::AllocLocalShadow(ur_queue_handle_t Queue,
                                              uint32_t NumWG, uptr &Begin,
                                              uptr &End) {
  const size_t LocalMemorySize = GetDeviceLocalMemorySize(Device);
  const size_t RequiredShadowSize =
      (NumWG * LocalMemorySize) >> ASAN_SHADOW_SCALE;
  static size_t LastAllocedSize = 0;
  if (RequiredShadowSize > LastAllocedSize) {
    auto ContextInfo = getAsanInterceptor()->getContextInfo(Context);
    if (LocalShadowOffset) {
      UR_CALL(getContext()->urDdiTable.USM.pfnFree(Context,
                                                   (void *)LocalShadowOffset));
      ContextInfo->Stats.UpdateShadowFreed(LastAllocedSize);
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
    }

    ContextInfo->Stats.UpdateShadowMalloced(RequiredShadowSize);

    LastAllocedSize = RequiredShadowSize;
  }

  Begin = LocalShadowOffset;
  End = LocalShadowOffset + RequiredShadowSize - 1;
  return UR_RESULT_SUCCESS;
}

ur_result_t ShadowMemoryGPU::AllocPrivateShadow(ur_queue_handle_t Queue,
                                                uint32_t NumWG, uptr &Begin,
                                                uptr &End) {
  const size_t RequiredShadowSize =
      (NumWG * ASAN_PRIVATE_SIZE) >> ASAN_SHADOW_SCALE;
  static size_t LastAllocedSize = 0;
  if (RequiredShadowSize > LastAllocedSize) {
    auto ContextInfo = getAsanInterceptor()->getContextInfo(Context);
    if (PrivateShadowOffset) {
      UR_CALL(getContext()->urDdiTable.USM.pfnFree(
          Context, (void *)PrivateShadowOffset));
      ContextInfo->Stats.UpdateShadowFreed(LastAllocedSize);
      PrivateShadowOffset = 0;
      LastAllocedSize = 0;
    }

    UR_CALL(getContext()->urDdiTable.USM.pfnDeviceAlloc(
        Context, Device, nullptr, nullptr, RequiredShadowSize,
        (void **)&PrivateShadowOffset));

    // Initialize shadow memory
    ur_result_t URes = EnqueueUSMBlockingSet(Queue, (void *)PrivateShadowOffset,
                                             0, RequiredShadowSize);
    if (URes != UR_RESULT_SUCCESS) {
      UR_CALL(getContext()->urDdiTable.USM.pfnFree(
          Context, (void *)PrivateShadowOffset));
      PrivateShadowOffset = 0;
      LastAllocedSize = 0;
    }

    ContextInfo->Stats.UpdateShadowMalloced(RequiredShadowSize);

    LastAllocedSize = RequiredShadowSize;
  }

  Begin = PrivateShadowOffset;
  End = PrivateShadowOffset + RequiredShadowSize - 1;
  return UR_RESULT_SUCCESS;
}

uptr ShadowMemoryPVC::MemToShadow(uptr Ptr) {
  if (Ptr & 0xFF00000000000000ULL) { // Device USM
    return ShadowBegin + 0x80000000000ULL +
           ((Ptr & 0xFFFFFFFFFFFFULL) >> ASAN_SHADOW_SCALE);
  } else { // Only consider 47bit VA
    return ShadowBegin + ((Ptr & 0x7FFFFFFFFFFFULL) >> ASAN_SHADOW_SCALE);
  }
}

uptr ShadowMemoryDG2::MemToShadow(uptr Ptr) {
  if (Ptr & 0xFFFF000000000000ULL) { // Device USM
    return ShadowBegin + 0x80000000000ULL +
           ((Ptr & 0x7FFFFFFFFFFFULL) >> ASAN_SHADOW_SCALE);
  } else { // Host/Shared USM
    return ShadowBegin + (Ptr >> ASAN_SHADOW_SCALE);
  }
}

} // namespace asan
} // namespace ur_sanitizer_layer
