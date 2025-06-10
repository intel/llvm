/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file tsan_shadow.hpp
 *
 */

#pragma once

#include "sanitizer_common/sanitizer_common.hpp"
#include "tsan_libdevice.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {
namespace tsan {

struct ShadowMemory {
  ShadowMemory(ur_context_handle_t Context, ur_device_handle_t Device)
      : Context(Context), Device(Device) {}

  virtual ~ShadowMemory() {}

  virtual ur_result_t Setup() = 0;

  virtual ur_result_t Destory() = 0;

  virtual RawShadow *MemToShadow(uptr Ptr) = 0;

  virtual ur_result_t CleanShadow(ur_queue_handle_t Queue, uptr Ptr,
                                  uptr Size) = 0;

  virtual ur_result_t AllocLocalShadow(ur_queue_handle_t Queue, uint32_t NumWG,
                                       uptr &Begin, uptr &End) = 0;

  virtual size_t GetShadowSize() = 0;

  ur_context_handle_t Context{};

  ur_device_handle_t Device{};

  uptr ShadowBegin = 0;

  uptr ShadowEnd = 0;
};

// clang-format off
// Shadow Memory layout of CPU device
//
// 0000 0000 1000 - 0200 0000 0000: main binary and/or MAP_32BIT mappings (2TB)
// 0200 0000 0000 - 1000 0000 0000: -
// 1000 0000 0000 - 3000 0000 0000: shadow (32TB)
// 3000 0000 0000 - 3800 0000 0000: metainfo (memory blocks and sync objects; 8TB)
// 3800 0000 0000 - 5500 0000 0000: -
// 5500 0000 0000 - 5a00 0000 0000: pie binaries without ASLR or on 4.1+ kernels
// 5a00 0000 0000 - 7200 0000 0000: -
// 7200 0000 0000 - 7300 0000 0000: heap (1TB)
// 7300 0000 0000 - 7a00 0000 0000: -
// 7a00 0000 0000 - 8000 0000 0000: modules and main thread stack (6TB)
// clang-format on
struct ShadowMemoryCPU final : public ShadowMemory {
  ShadowMemoryCPU(ur_context_handle_t Context, ur_device_handle_t Device)
      : ShadowMemory(Context, Device) {}

  ur_result_t Setup() override;

  ur_result_t Destory() override;

  RawShadow *MemToShadow(uptr Ptr) override;

  ur_result_t CleanShadow(ur_queue_handle_t Queue, uptr Ptr,
                          uptr Size) override;

  ur_result_t AllocLocalShadow(ur_queue_handle_t, uint32_t, uptr &Begin,
                               uptr &End) override {
    Begin = ShadowBegin;
    End = ShadowEnd;
    return UR_RESULT_SUCCESS;
  }

  size_t GetShadowSize() override { return 0x2000'0000'0000ULL; }
};

struct ShadowMemoryGPU : public ShadowMemory {
  ShadowMemoryGPU(ur_context_handle_t Context, ur_device_handle_t Device)
      : ShadowMemory(Context, Device) {}

  ur_result_t Setup() override;

  ur_result_t Destory() override;

  ur_result_t CleanShadow(ur_queue_handle_t Queue, uptr Ptr,
                          uptr Size) override;

  ur_result_t AllocLocalShadow(ur_queue_handle_t Queue, uint32_t NumWG,
                               uptr &Begin, uptr &End) override final;

  virtual uptr GetStartAddress() { return 0; }

  ur_mutex VirtualMemMapsMutex;

  std::unordered_map<uptr, ur_physical_mem_handle_t> VirtualMemMaps;

  uptr LocalShadowOffset = 0;
};

// clang-format off
// Shadow Memory layout of GPU PVC device
// We only support limited memory range for host/shared usm
// USM Allocation Range (56 bits)
//   Host   USM : 0x00ff_ff00_0000_0000 ~ 0x00ff_ffff_ffff_ffff
//   Shared USM : 0x0000_7f00_0000_0000 ~ 0x0000_7fff_ffff_ffff
//   Device USM : 0xff00_0000_0000_0000 ~ 0xff00_ffff_ffff_ffff
//
// clang-format on
struct ShadowMemoryPVC : public ShadowMemoryGPU {
  ShadowMemoryPVC(ur_context_handle_t Context, ur_device_handle_t Device)
      : ShadowMemoryGPU(Context, Device) {}

  RawShadow *MemToShadow(uptr Ptr) override;

  size_t GetShadowSize() override { return 0x8200'0000'0000ULL; }

  uptr GetStartAddress() override { return 0x100'0000'0000'0000ULL; }
};

std::shared_ptr<ShadowMemory> GetShadowMemory(ur_context_handle_t Context,
                                              ur_device_handle_t Device,
                                              DeviceType Type);
} // namespace tsan
} // namespace ur_sanitizer_layer
