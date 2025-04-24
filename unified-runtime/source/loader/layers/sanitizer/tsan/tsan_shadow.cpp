/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file tsan_shadow.cpp
 *
 */

#include "tsan_shadow.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"

namespace ur_sanitizer_layer {
namespace tsan {

std::shared_ptr<ShadowMemory> GetShadowMemory(ur_context_handle_t Context,
                                              ur_device_handle_t Device,
                                              DeviceType Type) {
  // For CPU device, we only allocate once. But for GPU device, each device will
  // have its own one.
  if (Type == DeviceType::CPU) {
    static std::shared_ptr<ShadowMemory> ShadowCPU =
        std::make_shared<ShadowMemoryCPU>(Context, Device);
    return ShadowCPU;
  } else if (Type == DeviceType::GPU_PVC) {
    return std::make_shared<ShadowMemoryPVC>(Context, Device);
  } else {
    UR_LOG_L(getContext()->logger, ERR, "Unsupport device type");
    return nullptr;
  }
}

ur_result_t ShadowMemoryCPU::Setup() {
  static ur_result_t URes = [this]() {
    ShadowBegin = 0x100000000000ULL;
    ShadowEnd = 0x300000000000ULL;
    if (MmapFixedNoReserve(ShadowBegin, ShadowEnd - ShadowBegin) == 0)
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    DontCoredumpRange(ShadowBegin, ShadowEnd - ShadowBegin);
    return UR_RESULT_SUCCESS;
  }();
  return URes;
}

ur_result_t ShadowMemoryCPU::Destory() {
  if (ShadowBegin == 0 && ShadowEnd == 0)
    return UR_RESULT_SUCCESS;
  static ur_result_t URes = [this]() {
    if (!Munmap(ShadowBegin, ShadowEnd - ShadowBegin))
      return UR_RESULT_ERROR_UNKNOWN;
    ShadowBegin = ShadowEnd = 0;
    return UR_RESULT_SUCCESS;
  }();
  return URes;
}

RawShadow *ShadowMemoryCPU::MemToShadow(uptr Addr) {
  return reinterpret_cast<RawShadow *>(
      ((Addr) & ~(0x700000000000ULL | (kShadowCell - 1))) * kShadowMultiplier +
      ShadowBegin);
}

ur_result_t ShadowMemoryCPU::CleanShadow(ur_queue_handle_t, uptr Ptr,
                                         uptr Size) {
  if (Size) {
    Ptr = RoundDownTo(Ptr, kShadowCell);
    Size = RoundUpTo(Size, kShadowCell);

    RawShadow *Begin = MemToShadow(Ptr);
    UR_LOG_L(getContext()->logger, DEBUG, "CleanShadow(addr={}, count={})",
             (void *)Begin, Size / kShadowCell);
    memset((void *)Begin, 0, Size / kShadowCell * kShadowCnt * kShadowSize);
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t ShadowMemoryGPU::Setup() {
  if (ShadowBegin != 0)
    return UR_RESULT_SUCCESS;

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
}

ur_result_t ShadowMemoryGPU::Destory() {
  if (ShadowBegin == 0) {
    return UR_RESULT_SUCCESS;
  }

  const size_t PageSize = GetVirtualMemGranularity(Context, Device);
  for (auto [MappedPtr, PhysicalMem] : VirtualMemMaps) {
    UR_CALL(getContext()->urDdiTable.VirtualMem.pfnUnmap(
        Context, (void *)MappedPtr, PageSize));
    UR_CALL(getContext()->urDdiTable.PhysicalMem.pfnRelease(PhysicalMem));
  }
  UR_CALL(getContext()->urDdiTable.VirtualMem.pfnFree(
      Context, (const void *)ShadowBegin, GetShadowSize()));
  UR_CALL(getContext()->urDdiTable.Context.pfnRelease(Context));
  ShadowEnd = ShadowBegin = 0;
  return UR_RESULT_SUCCESS;
}

ur_result_t ShadowMemoryGPU::CleanShadow(ur_queue_handle_t Queue, uptr Ptr,
                                         uptr Size) {
  if (Size == 0) {
    return UR_RESULT_SUCCESS;
  }

  Ptr = RoundDownTo(Ptr, kShadowCell);
  Size = RoundUpTo(Size, kShadowCell);

  uptr Begin = (uptr)MemToShadow(Ptr);
  uptr End = Begin + Size / kShadowCell * kShadowCnt * kShadowSize - 1;

  {
    static const size_t PageSize = GetVirtualMemGranularity(Context, Device);

    ur_physical_mem_properties_t Desc{UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES,
                                      nullptr, 0};

    // Make sure [Ptr, Ptr + Size] is mapped to physical memory
    for (auto MappedPtr = RoundDownTo((uptr)Begin, PageSize); MappedPtr <= End;
         MappedPtr += PageSize) {
      std::scoped_lock<ur_mutex> Guard(VirtualMemMapsMutex);
      if (VirtualMemMaps.find(MappedPtr) == VirtualMemMaps.end()) {
        ur_physical_mem_handle_t PhysicalMem{};
        auto URes = getContext()->urDdiTable.PhysicalMem.pfnCreate(
            Context, Device, PageSize, &Desc, &PhysicalMem);
        if (URes != UR_RESULT_SUCCESS) {
          UR_LOG_L(getContext()->logger, ERR, "urPhysicalMemCreate(): {}",
                   URes);
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
        URes = EnqueueUSMBlockingSet(Queue, (void *)MappedPtr, 0, PageSize);
        if (URes != UR_RESULT_SUCCESS) {
          UR_LOG_L(getContext()->logger, ERR, "EnqueueUSMBlockingSet(): {}",
                   URes);
          return URes;
        }

        VirtualMemMaps[MappedPtr] = PhysicalMem;
      }
    }
  }

  auto URes = EnqueueUSMBlockingSet(
      Queue, (void *)Begin, 0, Size / kShadowCell * kShadowCnt * kShadowSize);
  if (URes != UR_RESULT_SUCCESS) {
    UR_LOG_L(getContext()->logger, ERR, "EnqueueUSMBlockingSet(): {}", URes);
    return URes;
  }

  UR_LOG_L(getContext()->logger, DEBUG, "CleanShadow(addr={}, count={})",
           (void *)Begin, Size / kShadowCell);

  return UR_RESULT_SUCCESS;
}

RawShadow *ShadowMemoryPVC::MemToShadow(uptr Ptr) {
  Ptr = RoundDownTo(Ptr, kShadowCell);
  if (Ptr & 0xff00'0000'0000'0000ULL) {
    // device usm
    return Ptr < ShadowBegin
               ? reinterpret_cast<RawShadow *>(Ptr + (ShadowBegin +
                                                      0x200'0000'0000ULL -
                                                      0xff00'0000'0000'0000ULL))
               : reinterpret_cast<RawShadow *>(
                     Ptr - (0xff00'ffff'ffff'ffffULL - ShadowEnd + 1));
  } else {
    // host & shared usm
    return reinterpret_cast<RawShadow *>((Ptr & 0xffffffffffULL) + ShadowBegin +
                                         ((Ptr & 0x800000000000ULL) >> 7));
  }
}

} // namespace tsan
} // namespace ur_sanitizer_layer
