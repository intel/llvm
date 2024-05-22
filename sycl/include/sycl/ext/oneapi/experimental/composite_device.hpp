//==---------- composite_device.hpp - SYCL Composite Device ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/device.hpp>

#include <vector>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
__SYCL_EXPORT std::vector<device> get_composite_devices();
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
