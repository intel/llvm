//===--------- device.hpp - HIP Adapter -----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"

#include <ur/ur.hpp>

/// UR device mapping to a hipDevice_t.
/// Includes an observer pointer to the platform,
/// and implements the reference counting semantics since
/// HIP objects are not refcounted.
struct ur_device_handle_t_ {
private:
  using native_type = hipDevice_t;

  native_type HIPDevice;
  std::atomic_uint32_t RefCount;
  ur_platform_handle_t Platform;
  hipEvent_t EvBase; // HIP event used as base counter
  uint32_t DeviceIndex;

  int MaxWorkGroupSize{0};
  int MaxBlockDimX{0};
  int MaxBlockDimY{0};
  int MaxBlockDimZ{0};
  int DeviceMaxLocalMem{0};
  int ManagedMemSupport{0};
  int ConcurrentManagedAccess{0};
  bool HardwareImageSupport{false};

public:
  ur_device_handle_t_(native_type HipDevice, hipEvent_t EvBase,
                      ur_platform_handle_t Platform, uint32_t DeviceIndex)
      : HIPDevice(HipDevice), RefCount{1}, Platform(Platform), EvBase(EvBase),
        DeviceIndex(DeviceIndex) {

    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &MaxWorkGroupSize, hipDeviceAttributeMaxThreadsPerBlock, HIPDevice));
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &MaxBlockDimX, hipDeviceAttributeMaxBlockDimX, HIPDevice));
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &MaxBlockDimY, hipDeviceAttributeMaxBlockDimY, HIPDevice));
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &MaxBlockDimZ, hipDeviceAttributeMaxBlockDimZ, HIPDevice));
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &DeviceMaxLocalMem, hipDeviceAttributeMaxSharedMemoryPerBlock,
        HIPDevice));
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &ManagedMemSupport, hipDeviceAttributeManagedMemory, HIPDevice));
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &ConcurrentManagedAccess, hipDeviceAttributeConcurrentManagedAccess,
        HIPDevice));
    // Check if texture functions are supported in the HIP host runtime.
    int Ret{};
    UR_CHECK_ERROR(
        hipDeviceGetAttribute(&Ret, hipDeviceAttributeImageSupport, HIPDevice));
    detail::ur::assertion(Ret == 0 || Ret == 1);
    HardwareImageSupport = Ret == 1;
  }

  ~ur_device_handle_t_() noexcept(false) {}

  native_type get() const noexcept { return HIPDevice; };

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  ur_platform_handle_t getPlatform() const noexcept { return Platform; };

  uint64_t getElapsedTime(hipEvent_t) const;

  // Returns the index of the device relative to the other devices in the same
  // platform
  uint32_t getIndex() const noexcept { return DeviceIndex; };

  int getMaxWorkGroupSize() const noexcept { return MaxWorkGroupSize; };

  int getMaxBlockDimX() const noexcept { return MaxBlockDimX; };

  int getMaxBlockDimY() const noexcept { return MaxBlockDimY; };

  int getMaxBlockDimZ() const noexcept { return MaxBlockDimZ; };

  int getDeviceMaxLocalMem() const noexcept { return DeviceMaxLocalMem; };

  int getManagedMemSupport() const noexcept { return ManagedMemSupport; };

  int getConcurrentManagedAccess() const noexcept {
    return ConcurrentManagedAccess;
  };

  bool supportsHardwareImages() const noexcept { return HardwareImageSupport; }
};

int getAttribute(ur_device_handle_t Device, hipDeviceAttribute_t Attribute);

namespace {
/// Scoped Device is used across all UR HIP plugin implementation to activate
/// the native Device on the current thread. The ScopedDevice does not
/// reinstate the previous device as all operations in the HIP adapter that
/// require an active device, set the active device and don't rely on device
/// reinstation
class ScopedDevice {
public:
  ScopedDevice(ur_device_handle_t hDevice) {
    if (!hDevice) {
      throw UR_RESULT_ERROR_INVALID_DEVICE;
    }
    UR_CHECK_ERROR(hipSetDevice(hDevice->getIndex()));
  }
};
} // namespace
