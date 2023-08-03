//===--------- device.hpp - HIP Adapter -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
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
  hipCtx_t HIPContext;

public:
  ur_device_handle_t_(native_type HipDevice, hipCtx_t Context,
                      ur_platform_handle_t Platform)
      : HIPDevice(HipDevice), RefCount{1}, Platform(Platform),
        HIPContext(Context) {}

  ~ur_device_handle_t_() {
    UR_CHECK_ERROR(hipDevicePrimaryCtxRelease(HIPDevice));
  }

  native_type get() const noexcept { return HIPDevice; };

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  ur_platform_handle_t getPlatform() const noexcept { return Platform; };

  hipCtx_t getNativeContext() { return HIPContext; };
};

int getAttribute(ur_device_handle_t Device, hipDeviceAttribute_t Attribute);

namespace {
/// RAII type to guarantee recovering original HIP context
/// Scoped context is used across all UR HIP plugin implementation
/// to activate the UR Context on the current thread, matching the
/// HIP driver semantics where the context used for the HIP Driver
/// API is the one active on the thread.
/// The implementation tries to avoid replacing the hipCtx_t if it cans
class ScopedDevice {
  hipCtx_t Original;
  bool NeedToRecover;

public:
  ScopedDevice(ur_device_handle_t hDevice) : NeedToRecover{false} {

    if (!hDevice) {
      throw UR_RESULT_ERROR_INVALID_DEVICE;
    }

    // FIXME when multi device context are supported in HIP adapter
    hipCtx_t Desired = hDevice->getNativeContext();
    UR_CHECK_ERROR(hipCtxGetCurrent(&Original));
    if (Original != Desired) {
      // Sets the desired context as the active one for the thread
      UR_CHECK_ERROR(hipCtxSetCurrent(Desired));
      if (Original == nullptr) {
        // No context is installed on the current thread
        // This is the most common case. We can activate the context in the
        // thread and leave it there until all the UR context referring to the
        // same underlying HIP context are destroyed. This emulates
        // the behaviour of the HIP runtime api, and avoids costly context
        // switches. No action is required on this side of the if.
      } else {
        NeedToRecover = true;
      }
    }
  }

  ~ScopedDevice() {
    if (NeedToRecover) {
      UR_CHECK_ERROR(hipCtxSetCurrent(Original));
    }
  }
};
} // namespace
