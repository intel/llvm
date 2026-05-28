//==-- device.hpp - oneapi device info traits (non-experimental) -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/info_desc_traits.hpp>
#include <unified-runtime/ur_api.h>

#include <cstddef>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::info::device {

struct num_compute_units {
  using return_type = size_t;
  using info_class = sycl::detail::info_class::device;
  static constexpr ur_device_info_t ur_code =
      UR_DEVICE_INFO_NUM_COMPUTE_UNITS;
};

} // namespace ext::oneapi::info::device
} // namespace _V1
} // namespace sycl
