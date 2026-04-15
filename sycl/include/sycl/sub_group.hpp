//==----------- sub_group.hpp --- SYCL sub-group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/sub_group_core.hpp>
#include <sycl/detail/sub_group_extra.hpp>
#include <sycl/detail/sub_group_load_store.hpp>
#include <sycl/nd_item.hpp>

namespace sycl {
inline namespace _V1 {
template <int Dimensions> sub_group nd_item<Dimensions>::get_sub_group() const {
  return sub_group();
}

} // namespace _V1
} // namespace sycl
