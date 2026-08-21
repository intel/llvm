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

struct num_compute_units
    : sycl::detail::ur_traits_base<sycl::detail::info_class::device,
                                   UR_DEVICE_INFO_NUM_COMPUTE_UNITS> {
  using return_type = size_t;
};

struct max_threads_per_compute_unit
    : sycl::detail::ur_traits_base<
          sycl::detail::info_class::device,
          UR_DEVICE_INFO_MAX_THREADS_PER_COMPUTE_UNIT> {
  using return_type = size_t;
};

} // namespace ext::oneapi::info::device
} // namespace _V1
} // namespace sycl
