//==------ filter_selector.hpp - ONEAPI filter selector -------*- C++ --*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/device_selector.hpp>

#include <memory>
#include <string>

// 4.6.1 Device selection class

namespace __sycl_internal {
inline namespace __v1 {

// Forward declarations
class device;
class device_selector;

namespace ONEAPI {
namespace detail {
class filter_selector_impl;
} // namespace detail

class __SYCL_EXPORT filter_selector : public device_selector {
public:
  filter_selector(const std::string &filter);
  int operator()(const device &dev) const override;
  void reset() const;
  device select_device() const override;

private:
  std::shared_ptr<detail::filter_selector_impl> impl;
};
} // namespace ONEAPI
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace sycl {
  using namespace __sycl_internal::__v1;
}
