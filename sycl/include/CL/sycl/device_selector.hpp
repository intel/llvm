//==------ device_selector.hpp - SYCL device selector ---------*- C++ --*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/export.hpp>

// 4.6.1 Device selection class

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declarations
class device;

class SYCL_API device_selector {
public:
  virtual ~device_selector() = default;

  device select_device() const;

  virtual int operator()(const device &device) const = 0;
};

class SYCL_API default_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

class SYCL_API gpu_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

class SYCL_API cpu_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

class SYCL_API accelerator_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

class SYCL_API host_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
