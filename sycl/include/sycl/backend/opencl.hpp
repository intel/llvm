//==---------------- opencl.hpp - SYCL OpenCL backend ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT
#include <sycl/device.hpp>                    // for device
#include <sycl/platform.hpp>                  // for platform

#include <string>      // for string
#include <type_traits> // for enable_if_t

namespace sycl {
inline namespace _V1 {
namespace opencl {
__SYCL_EXPORT bool has_extension(const sycl::platform &SyclPlatform,
                                 const std::string &Extension);
__SYCL_EXPORT bool has_extension(const sycl::device &SyclDevice,
                                 const std::string &Extension);
} // namespace opencl
} // namespace _V1
} // namespace sycl
