//==--------------- enqueue_types.hpp ---- SYCL enqueue types --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

/// @brief Indicates the destination device for USM data to be prefetched to.
enum class prefetch_type { device, host };

inline std::string prefetchTypeToString(prefetch_type value) {
  switch (value) {
  case sycl::ext::oneapi::experimental::prefetch_type::device:
    return "prefetch_type::device";
  case sycl::ext::oneapi::experimental::prefetch_type::host:
    return "prefetch_type::host";
  default:
    return "prefetch_type::unknown";
  }
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
