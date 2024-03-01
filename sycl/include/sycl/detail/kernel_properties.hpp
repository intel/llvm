//==---------------- kernel_properties.hpp - SYCL Kernel Properties -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// APIs for setting kernel properties interpreted by GPU software stack.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
enum class register_alloc_mode_enum : uint32_t {
  automatic = 0,
  large = 2,
};

struct register_alloc_mode_key
    : ext::oneapi::experimental::detail::compile_time_property_key<
          ext::oneapi::experimental::detail::PropKind::RegisterAllocMode> {
  template <register_alloc_mode_enum Mode>
  using value_t = sycl::ext::oneapi::experimental::property_value<
      register_alloc_mode_key,
      std::integral_constant<register_alloc_mode_enum, Mode>>;
};

template <register_alloc_mode_enum Mode>
inline constexpr register_alloc_mode_key::value_t<Mode> register_alloc_mode
    __SYCL_DEPRECATED("register_alloc_mode is deprecated, "
                      "use sycl::ext::intel::experimental::grf_size or "
                      "sycl::ext::intel::experimental::grf_size_automatic");
} // namespace detail

namespace ext::oneapi::experimental {
namespace detail {
template <sycl::detail::register_alloc_mode_enum Mode>
struct PropertyMetaInfo<sycl::detail::register_alloc_mode_key::value_t<Mode>> {
  static constexpr const char *name = "sycl-register-alloc-mode";
  static constexpr sycl::detail::register_alloc_mode_enum value = Mode;
};
} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
