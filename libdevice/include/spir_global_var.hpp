//==- spir_global_var.hpp - Declaration for device global variable support -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

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

#ifndef SPIR_GLOBAL_VAR
#ifdef __SYCL_DEVICE_ONLY__
#define SPIR_GLOBAL_VAR __attribute__((sycl_global_var))
#else
#define SPIR_GLOBAL_VAR
#endif
#endif
