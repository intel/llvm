//==---------------- maximum_registers_properties.hpp --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/free_function_kernel_properties.hpp>
#include <sycl/ext/oneapi/properties.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#define SYCL_EXT_INTEL_MAXIMUM_REGISTERS 1

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {
struct maximum_registers_key
    : oneapi::experimental::detail::compile_time_property_key<
          oneapi::experimental::detail::PropKind::MaximumRegisters> {
  template <unsigned int Size>
  using value_t = oneapi::experimental::property_value<
      maximum_registers_key, std::integral_constant<unsigned int, Size>>;
};

struct maximum_registers_automatic_key
    : oneapi::experimental::detail::compile_time_property_key<
          oneapi::experimental::detail::PropKind::MaximumRegistersAutomatic> {
  using value_t =
      oneapi::experimental::property_value<maximum_registers_automatic_key>;
};

template <unsigned int Size>
inline constexpr maximum_registers_key::value_t<Size> maximum_registers;

inline constexpr maximum_registers_automatic_key::value_t
    maximum_registers_automatic;

} // namespace ext::intel::experimental
namespace ext::oneapi::experimental::detail {
template <unsigned int Size>
struct PropertyMetaInfo<
    sycl::ext::intel::experimental::maximum_registers_key::value_t<Size>> {
  static_assert(Size == 128 || Size == 256 || Size == 512,
                "Unsupported maximum registers");
  static constexpr const char *name = "sycl-maximum-registers";
  static constexpr unsigned int value = Size;
};
template <>
struct PropertyMetaInfo<
    sycl::ext::intel::experimental::maximum_registers_automatic_key::value_t> {
  static constexpr const char *name = "sycl-maximum-registers";
  static constexpr unsigned int value = 0;
};

} // namespace ext::oneapi::experimental::detail
} // namespace _V1
} // namespace sycl
