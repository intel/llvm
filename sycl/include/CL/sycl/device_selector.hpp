//==------ device_selector.hpp - SYCL device selector ---------*- C++ --*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/export.hpp>

// 4.6.1 Device selection class

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declarations
class device;

class __SYCL_EXPORT device_selector {
protected:
  // SYCL 1.2.1 defines a negative score to reject a device from selection
  static constexpr int REJECT_DEVICE_SCORE = -1;

public:
  virtual ~device_selector() = default;

  device select_device() const;

  virtual int operator()(const device &device) const = 0;
};

class __SYCL_EXPORT default_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

class __SYCL_EXPORT gpu_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

class __SYCL_EXPORT cpu_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

class __SYCL_EXPORT accelerator_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

class __SYCL_EXPORT host_selector : public device_selector {
public:
  int operator()(const device &dev) const override;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
