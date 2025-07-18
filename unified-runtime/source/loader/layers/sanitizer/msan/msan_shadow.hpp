/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_shadow.hpp
 *
 */

#pragma once

#include "msan_allocator.hpp"
#include "sanitizer_common/sanitizer_libdevice.hpp"

#include <unordered_set>

namespace ur_sanitizer_layer {
namespace msan {

struct MsanShadowMemory {
  MsanShadowMemory(ur_context_handle_t Context, ur_device_handle_t Device)
      : Context(Context), Device(Device) {}

  virtual ~MsanShadowMemory() {}

  virtual ur_result_t Setup() = 0;

  virtual ur_result_t Destory() = 0;

  virtual uptr MemToShadow(uptr Ptr) = 0;
  virtual uptr MemToOrigin(uptr Ptr) = 0;

  virtual ur_result_t
  EnqueuePoisonShadow(ur_queue_handle_t Queue, uptr Ptr, uptr Size, u8 Value,
                      uint32_t NumEvents = 0,
                      const ur_event_handle_t *EventWaitList = nullptr,
                      ur_event_handle_t *OutEvent = nullptr) = 0;

  virtual ur_result_t EnqueuePoisonShadowWithOrigin(
      ur_queue_handle_t Queue, uptr Ptr, uptr Size, u8 Value, uint32_t Origin,
      uint32_t NumEvents = 0, const ur_event_handle_t *EventWaitList = nullptr,
      ur_event_handle_t *OutEvent = nullptr) = 0;

  virtual ur_result_t ReleaseShadow(std::shared_ptr<MsanAllocInfo>) {
    return UR_RESULT_SUCCESS;
  }

  virtual ur_result_t AllocLocalShadow(ur_queue_handle_t Queue, uint32_t NumWG,
                                       uptr &Begin, uptr &End) = 0;

  virtual ur_result_t AllocPrivateShadow(ur_queue_handle_t Queue,
                                         uint32_t NumSG, uptr *&Base,
                                         uptr &Begin, uptr &End) = 0;

  ur_context_handle_t Context{};

  ur_device_handle_t Device{};

  uptr ShadowBegin = 0;

  uptr ShadowEnd = 0;
};

// clang-format off
/// Shadow Memory layout of CPU device
///
/// 0x000000000000 ~ 0x010000000000 "app-1"
/// 0x010000000000 ~ 0x100000000000 "shadow-2"
/// 0x100000000000 ~ 0x110000000000 "invalid"
/// 0x110000000000 ~ 0x200000000000 "origin-2"
/// 0x200000000000 ~ 0x300000000000 "shadow-3"
/// 0x300000000000 ~ 0x400000000000 "origin-3"
/// 0x400000000000 ~ 0x500000000000 "invalid"
/// 0x500000000000 ~ 0x510000000000 "shadow-1"
/// 0x510000000000 ~ 0x600000000000 "app-2"
/// 0x600000000000 ~ 0x610000000000 "origin-1"
/// 0x610000000000 ~ 0x700000000000 "invalid"
/// 0x700000000000 ~ 0x740000000000 "allocator"
/// 0x740000000000 ~ 0x800000000000 "app-3"
///
// clang-format on
struct MsanShadowMemoryCPU final : public MsanShadowMemory {
  MsanShadowMemoryCPU(ur_context_handle_t Context, ur_device_handle_t Device)
      : MsanShadowMemory(Context, Device) {}

  ur_result_t Setup() override;

  ur_result_t Destory() override;

  uptr MemToShadow(uptr Ptr) override;
  uptr MemToOrigin(uptr Ptr) override;

  ur_result_t
  EnqueuePoisonShadow(ur_queue_handle_t Queue, uptr Ptr, uptr Size, u8 Value,
                      uint32_t NumEvents = 0,
                      const ur_event_handle_t *EventWaitList = nullptr,
                      ur_event_handle_t *OutEvent = nullptr) override;

  ur_result_t EnqueuePoisonShadowWithOrigin(
      ur_queue_handle_t Queue, uptr Ptr, uptr Size, u8 Value, uint32_t Origin,
      uint32_t NumEvents = 0, const ur_event_handle_t *EventWaitList = nullptr,
      ur_event_handle_t *OutEvent = nullptr) override;

  ur_result_t AllocLocalShadow(ur_queue_handle_t, uint32_t, uptr &Begin,
                               uptr &End) override {
    Begin = ShadowBegin;
    End = ShadowEnd;
    return UR_RESULT_SUCCESS;
  }

  ur_result_t AllocPrivateShadow(ur_queue_handle_t, uint32_t, uptr *&,
                                 uptr &Begin, uptr &End) override {
    // This is necessary as msan_rtl use it to check whether detecting private
    // is enabled
    Begin = ShadowBegin;
    End = ShadowEnd;
    return UR_RESULT_SUCCESS;
  }
};

struct MsanShadowMemoryGPU : public MsanShadowMemory {
  MsanShadowMemoryGPU(ur_context_handle_t Context, ur_device_handle_t Device)
      : MsanShadowMemory(Context, Device) {}

  ur_result_t Setup() override;

  ur_result_t Destory() override;

  ur_result_t
  EnqueuePoisonShadow(ur_queue_handle_t Queue, uptr Ptr, uptr Size, u8 Value,
                      uint32_t NumEvents = 0,
                      const ur_event_handle_t *EventWaitList = nullptr,
                      ur_event_handle_t *OutEvent = nullptr) override final;

  ur_result_t EnqueuePoisonShadowWithOrigin(
      ur_queue_handle_t Queue, uptr Ptr, uptr Size, u8 Value, uint32_t Origin,
      uint32_t NumEvents = 0, const ur_event_handle_t *EventWaitList = nullptr,
      ur_event_handle_t *OutEvent = nullptr) override;

  ur_result_t ReleaseShadow(std::shared_ptr<MsanAllocInfo> AI) override final;

  ur_result_t AllocLocalShadow(ur_queue_handle_t Queue, uint32_t NumWG,
                               uptr &Begin, uptr &End) override final;

  ur_result_t AllocPrivateShadow(ur_queue_handle_t Queue, uint32_t NumWG,
                                 uptr *&Base, uptr &Begin,
                                 uptr &End) override final;

  virtual size_t GetShadowSize() = 0;

  virtual uptr GetStartAddress() { return 0; }

private:
  ur_result_t
  EnqueueVirtualMemMap(uptr VirtualBegin, uptr VirtualEnd,
                       std::vector<ur_event_handle_t> &EventWaitList,
                       ur_event_handle_t *OutEvent);

  std::unordered_map<
      uptr, std::pair<ur_physical_mem_handle_t,
                      std::unordered_set<std::shared_ptr<MsanAllocInfo>>>>
      VirtualMemMaps;
  ur_mutex VirtualMemMapsMutex;

  uptr LocalShadowOffset = 0;
  uptr PrivateShadowOffset = 0;
  uptr PrivateBasePtr = 0;
};

// clang-format off
/// Shadow Memory layout of GPU PVC device
///
/// USM Allocation Range (56 bits)
///   Host   USM : 0x00ff_f000_0000_0000 ~ 0x00ff_ffff_ffff_ffff
///   Shared USM : 0x0000_7000_0000_0000 ~ 0x0000_7fff_ffff_ffff
///   Device USM : 0xff00_0000_0000_0000 ~ 0xff00_ffff_ffff_ffff
///
/// Shadow Memory Mapping
///     0xff00_0000_0000_0000 - MSAN_SHADOW_BASE       : "invalid"
///     MSAN_SHADOW_BASE      - MSAN_SHADOW_END1       : "shadow-1" (MSAN_SHADOW_END1 - MSAN_SHADOW_BASE = 0x2000_0000_0000)
///     MSAN_SHADOW_END1      - MSAN_SHADOW_END2       : "origin-1" (MSAN_SHADOW_END1 - MSAN_SHADOW_END2 = 0x2000_0000_0000)
///     (gap)                                          :                                                  (0x1000_0000_0000)
///     MSAN_SHADOW_END3      - MSAN_SHADOW_END4       : "origin-2" (MSAN_SHADOW_END4 - MSAN_SHADOW_END3 = 0x5000_0000_0000)
///     MSAN_SHADOW_END4      - MSAN_SHADOW_END5       : "shadow-2" (MSAN_SHADOW_END5 - MSAN_SHADOW_END4 = 0x5000_0000_0000)
///     MSAN_SHADOW_END5      - 0xff00_ffff_ffff_ffff  : "app"      (MSAN_SHADOW_END5 - MSAN_SHADOW_BASE = 0xF000_0000_0000)
///
///  here,
///    - We assume "invalid" is not usable for user application (by observation)    
///    - "shadow-1" and "origin-1" is use for host/shared USM
///    - "shadow-2" and "origin-2" is used for device USM
///    - "app" is device USM, the size of "app" is less than 0x5000_0000_0000_0000, so that it can be fully mapped to its shadow
///    - "gap" is necessary, so that "app" can be mapped to its shadow
// clang-format on
struct MsanShadowMemoryPVC final : public MsanShadowMemoryGPU {
  MsanShadowMemoryPVC(ur_context_handle_t Context, ur_device_handle_t Device)
      : MsanShadowMemoryGPU(Context, Device) {}

  static bool IsDeviceUSM(uptr Ptr) { return Ptr >> 52 == 0xff0; }

  uptr MemToShadow(uptr Ptr) override;
  uptr MemToOrigin(uptr Ptr) override;

  size_t GetShadowSize() override { return 0xf000'0000'0000ULL; }

  uptr GetStartAddress() override { return 0x100'0000'0000'0000ULL; }
};

// clang-format off
/// Shadow Memory layout of GPU DG2 device
///
/// USM Allocation Range (48 bits)
///   Host/Shared USM : 0x0000_0000_0000_0000 ~ 0x0000_7fff_ffff_ffff
///   Device      USM : 0xffff_8000_0000_0000 ~ 0xffff_ffff_ffff_ffff
///
/// Shadow Memory Mapping is similar to PVC
///
// clang-format on
struct MsanShadowMemoryDG2 final : public MsanShadowMemoryGPU {
  MsanShadowMemoryDG2(ur_context_handle_t Context, ur_device_handle_t Device)
      : MsanShadowMemoryGPU(Context, Device) {}

  static bool IsDeviceUSM(uptr Ptr) { return Ptr >> 48; }

  uptr MemToShadow(uptr Ptr) override;
  uptr MemToOrigin(uptr Ptr) override;

  size_t GetShadowSize() override { return 0x4000'0000'0000ULL; }
};

std::shared_ptr<MsanShadowMemory>
GetMsanShadowMemory(ur_context_handle_t Context, ur_device_handle_t Device,
                    DeviceType Type);

} // namespace msan
} // namespace ur_sanitizer_layer
