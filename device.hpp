//===--------- device.hpp - CUDA Adapter -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <ur/ur.hpp>

struct ur_device_handle_t_ {
private:
  using native_type = CUdevice;

  native_type CuDevice;
  CUcontext CuContext;
  CUevent EvBase; // CUDA event used as base counter
  std::atomic_uint32_t RefCount;
  ur_platform_handle_t Platform;

  static constexpr uint32_t MaxWorkItemDimensions = 3u;
  size_t MaxWorkItemSizes[MaxWorkItemDimensions];
  int MaxWorkGroupSize;

public:
  ur_device_handle_t_(native_type cuDevice, CUcontext cuContext, CUevent evBase,
                      ur_platform_handle_t platform)
      : CuDevice(cuDevice), CuContext(cuContext), EvBase(evBase), RefCount{1},
        Platform(platform) {}

  ur_device_handle_t_() { cuDevicePrimaryCtxRelease(CuDevice); }

  native_type get() const noexcept { return CuDevice; };

  CUcontext getContext() const noexcept { return CuContext; };

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  ur_platform_handle_t getPlatform() const noexcept { return Platform; };

  uint64_t getElapsedTime(CUevent) const;

  void saveMaxWorkItemSizes(size_t Size,
                            size_t *SaveMaxWorkItemSizes) noexcept {
    memcpy(MaxWorkItemSizes, SaveMaxWorkItemSizes, Size);
  };

  void saveMaxWorkGroupSize(int Value) noexcept { MaxWorkGroupSize = Value; };

  void getMaxWorkItemSizes(size_t RetSize,
                           size_t *RetMaxWorkItemSizes) const noexcept {
    memcpy(RetMaxWorkItemSizes, MaxWorkItemSizes, RetSize);
  };

  int getMaxWorkGroupSize() const noexcept { return MaxWorkGroupSize; };
};

int getAttribute(ur_device_handle_t Device, CUdevice_attribute Attribute);
