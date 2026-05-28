//==-- kernel.hpp - Intel kernel_device_specific extension info traits -----==//
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
namespace ext::intel::info::kernel_device_specific {

struct spill_memory_size {
  using return_type = size_t;
  using info_class = sycl::detail::info_class::kernel_device_specific;
  static constexpr ur_kernel_info_t ur_code = UR_KERNEL_INFO_SPILL_MEM_SIZE;
};

} // namespace ext::intel::info::kernel_device_specific
} // namespace _V1
} // namespace sycl
