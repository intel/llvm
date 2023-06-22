//===---------- physical_mem.hpp - CUDA Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <ur/ur.hpp>

#include <cuda.h>

/// UR queue mapping on physical memory allocations used in virtual memory
/// management.
///
struct ur_physical_mem_t_ {
  using native_type = CUmemGenericAllocationHandle;

  std::atomic_uint32_t RefCount;
  native_type PhysicalMem;
  ur_context_handle_t_ Context;

  ur_physical_mem_t_(native_type PhysMem, ur_context_handle_t_ Ctx)
      : RefCount(1), PhysicalMem(PhysMem), Context(Ctx) {
    urContextRetain(Context);
  }

  ~ur_physical_mem_t_() { urContextRelease(Context); }

  native_type get() const noexcept { return PhysicalMem; }

  ur_context_handle_t_ getContext() const noexcept { return Context; }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }
};

// Find a device ordinal of a device.
static ur_result_t GetDeviceOrdinal(ur_device_handle_t Device, int &Ordinal) {
  // Get list of platforms
  uint32_t NumPlatforms;
  UR_ASSERT(urPlatformGet(0, nullptr, &NumPlatforms),
            UR_RESULT_ERROR_INVALID_ARGUMENT);
  UR_ASSERT(NumPlatforms, UR_RESULT_ERROR_UNKNOWN);

  std::vector<ur_platform_handle_t> Platforms{NumPlatforms};
  UR_ASSERT(urPlatformGet(NumPlatforms, Platforms.data(), nullptr),
            UR_RESULT_ERROR_INVALID_ARGUMENT);

  // Ordinal corresponds to the platform ID as each device has its own platform.
  CUdevice NativeDevice = Device->get();
  for (Ordinal = 0; size_t(Ordinal) < Platforms.size(); ++Ordinal)
    if (Platforms[Ordinal]->Devices[0]->get() == NativeDevice)
      return UR_RESULT_SUCCESS;
  return UR_RESULT_ERROR_INVALID_DEVICE;
}
