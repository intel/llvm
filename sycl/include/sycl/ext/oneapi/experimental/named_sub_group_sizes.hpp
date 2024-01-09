//== named_sub_group_sizes.hpp --- SYCL extension for named sub-group sizes ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/kernel_properties/properties.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

struct named_sub_group_size {
  static constexpr uint32_t primary = 0;
  static constexpr uint32_t automatic = -1;
};

inline constexpr sub_group_size_key::value_t<named_sub_group_size::primary>
    sub_group_size_primary;

inline constexpr sub_group_size_key::value_t<named_sub_group_size::automatic>
    sub_group_size_automatic;

namespace detail {
template <>
struct PropertyMetaInfo<
    sub_group_size_key::value_t<named_sub_group_size::automatic>> {
  // sub_group_size_automatic means that the kernel can be compiled with
  // any sub-group size. That is, if the kernel has the sub_group_size_automatic
  // property, then no sycl-sub-group-size IR attribute needs to be attached.
  // Specializing PropertyMetaInfo for sub_group_size_automatic and setting
  // name to an empty string will result in no sycl-sub-group-size IR being
  // attached.
  static constexpr const char *name = "";
  static constexpr const char *value = 0;
};
} // namespace detail

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
