//==---------------- opencl.hpp - SYCL OpenCL backend ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/export.hpp> // for __SYCL_EXPORT
#include <sycl/detail/ur.hpp>     // for cast
#include <sycl/device.hpp>        // for device
#include <sycl/platform.hpp>      // for platform

#include <string>      // for string
#include <type_traits> // for enable_if_t

namespace sycl {
inline namespace _V1 {
namespace opencl {
namespace detail {
using namespace sycl::detail;
__SYCL_EXPORT bool has_extension(const sycl::platform &SyclPlatform,
                                 detail::string_view Extension);
__SYCL_EXPORT bool has_extension(const sycl::device &SyclDevice,
                                 detail::string_view Extension);
} // namespace detail
inline bool has_extension(const sycl::platform &SyclPlatform,
                          const std::string &Extension) {
  return detail::has_extension(SyclPlatform, detail::string_view{Extension});
}
inline bool has_extension(const sycl::device &SyclDevice,
                          const std::string &Extension) {
  return detail::has_extension(SyclDevice, detail::string_view{Extension});
}
} // namespace opencl
} // namespace _V1
} // namespace sycl
