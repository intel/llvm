//==-- max_registers_query.hpp - Codeplay max_registers_per_work_group ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/info_desc_traits.hpp>
#include <unified-runtime/ur_api.h>

#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace ext::codeplay::experimental::info::device {

struct max_registers_per_work_group {
  using return_type = uint32_t;
  using info_class = sycl::detail::info_class::device;
  static constexpr ur_device_info_t ur_code =
      UR_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP;
};

} // namespace ext::codeplay::experimental::info::device
} // namespace _V1
} // namespace sycl
