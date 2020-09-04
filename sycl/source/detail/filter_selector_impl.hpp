//==-- filter_selector_impl.hpp - ONEAPI filter selector impl--*- C++ --*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/device_selector.hpp>

#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declarations
class device;

namespace ONEAPI {
namespace detail {

struct filter {
  backend Backend = backend::host;
  RT::PiDeviceType DeviceType = PI_DEVICE_TYPE_ALL;
  int DeviceNum = 0;
  bool HasBackend = false;
  bool HasDeviceType = false;
  bool HasDeviceNum = false;
  int MatchesSeen = 0;
};

class filter_selector_impl {
public:
  filter_selector_impl(const std::string &filter);
  int operator()(const device &dev) const;
  void reset() const;

private:
  static constexpr int REJECT_DEVICE_SCORE = -1;
  mutable std::vector<filter> mFilters;
  default_selector mRanker;
  mutable int mNumDevicesSeen;
  int mNumTotalDevices;
  mutable bool mMatchFound;
};
} // namespace detail
} // namespace ONEAPI
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
