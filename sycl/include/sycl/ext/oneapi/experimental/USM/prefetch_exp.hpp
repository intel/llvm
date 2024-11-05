//==-------- usm_prefetch_exp.hpp --- SYCL USM prefetch extensions ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace sycl {
inline namespace _V1 {

namespace ext::oneapi::experimental {

  /// @brief Indicates USM memory migration direction: either from host to device, or device to host.
  enum class migration_direction { 
    HOST_TO_DEVICE, /// Move data from host USM to device USM
    DEVICE_TO_HOST  /// Move data from device USM to host USM
  };

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl