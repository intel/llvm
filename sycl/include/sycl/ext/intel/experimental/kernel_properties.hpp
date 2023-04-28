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
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace intel {
namespace experimental {

enum class RegisterAllocMode : uint32_t {
  AUTO = 0,
  LARGE = 2,
};

struct register_alloc_mode_key {
  template <RegisterAllocMode Mode>
  using value_t = oneapi::experimental::property_value<
      register_alloc_mode_key, std::integral_constant<RegisterAllocMode, Mode>>;
};

template <RegisterAllocMode Mode>
inline constexpr register_alloc_mode_key::value_t<Mode> register_alloc_mode;
} // namespace experimental
} // namespace intel

namespace oneapi {
namespace experimental {

template <>
struct is_property_key<intel::experimental::register_alloc_mode_key>
    : std::true_type {};

namespace detail {
template <>
struct PropertyToKind<intel::experimental::register_alloc_mode_key> {
  static constexpr PropKind Kind = PropKind::RegisterAllocMode;
};

template <>
struct IsCompileTimeProperty<intel::experimental::register_alloc_mode_key>
    : std::true_type {};

template <intel::experimental::RegisterAllocMode Mode>
struct PropertyMetaInfo<
    intel::experimental::register_alloc_mode_key::value_t<Mode>> {
  static constexpr const char *name = "RegisterAllocMode";
  static constexpr intel::experimental::RegisterAllocMode value = Mode;
};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
