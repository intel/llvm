//===--------- device.hpp - HIP Adapter -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
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
  ur_context_handle_t Context;

public:
  ur_device_handle_t_(native_type HipDevice, ur_platform_handle_t Platform)
      : HIPDevice(HipDevice), RefCount{1}, Platform(Platform) {}

  native_type get() const noexcept { return HIPDevice; };

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  ur_platform_handle_t getPlatform() const noexcept { return Platform; };

  void setContext(ur_context_handle_t Ctxt) { Context = Ctxt; };

  ur_context_handle_t getContext() { return Context; };
};

int getAttribute(ur_device_handle_t Device, hipDeviceAttribute_t Attribute);
