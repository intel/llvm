//==-- sanitizer_device_utils.hpp - Declaration for sanitizer global var ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "spir_global_var.hpp"
#include <cstdint>

template <typename T>
class
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::global_variable_allowed, __sycl_detail__::device_global]]
#endif
    DeviceGlobal {
public:
  DeviceGlobal() = default;
  DeviceGlobal(const DeviceGlobal &) = delete;
  DeviceGlobal(const DeviceGlobal &&) = delete;
  DeviceGlobal &operator=(const DeviceGlobal &) = delete;
  DeviceGlobal &operator=(const DeviceGlobal &&) = delete;

  DeviceGlobal &operator=(const T newValue) noexcept {
    val = newValue;
    return *this;
  }

  operator T &() noexcept { return val; }

  operator const T &() const noexcept { return val; }

  T &get() noexcept { return val; }

  const T &get() const noexcept { return val; }

private:
  T val;
};

enum DeviceType : uintptr_t { UNKNOWN, CPU, GPU_PVC, GPU_DG2 };
