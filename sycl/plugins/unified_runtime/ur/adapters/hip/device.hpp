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
///
struct ur_device_handle_t_ : public _ur_object {
private:
  using native_type = hipDevice_t;

  native_type hipDevice_;
  std::atomic_uint32_t refCount_;
  ur_platform_handle_t platform_;
  ur_context_handle_t context_;

public:
  ur_device_handle_t_(native_type hipDevice, ur_platform_handle_t platform)
      : hipDevice_(hipDevice), refCount_{1}, platform_(platform) {}

  native_type get() const noexcept { return hipDevice_; };

  uint32_t get_reference_count() const noexcept { return refCount_; }

  ur_platform_handle_t get_platform() const noexcept { return platform_; };

  void set_context(ur_context_handle_t ctx) { context_ = ctx; };

  ur_context_handle_t get_context() { return context_; };
};

int getAttribute(ur_device_handle_t device, hipDeviceAttribute_t attribute);
