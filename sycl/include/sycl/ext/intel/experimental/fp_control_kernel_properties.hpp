//=- fp_control_kernel_properties.hpp - SYCL kernel property for fp control -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=-------------------------------------------------------------------------=//

#pragma once

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#include <cstdint>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {

template <typename T, typename PropertyListT> class kernel_attribute;

enum class fp_mode : std::uint32_t {
  rte = 0,      // Round to nearest or even
  rtp = 1 << 4, // Round towards +ve inf
  rtn = 2 << 4, // Round towards -ve inf
  rtz = 3 << 4, // Round towards zero

  denorm_ftz = 0,            // Denorm mode flush to zero
  denorm_d_allow = 1 << 6,   // Denorm mode double allow
  denorm_f_allow = 1 << 7,   // Denorm mode float allow
  denorm_hf_allow = 1 << 10, // Denorm mode half allow
  denorm_allow = denorm_d_allow | denorm_f_allow |
                 denorm_hf_allow, // Denorm mode double/float/half allow

  float_mode_ieee = 0, // Single precision float IEEE mode
  float_mode_alt = 1   // Single precision float ALT mode
};

constexpr fp_mode operator|(const fp_mode &a, const fp_mode &b) {
  return static_cast<fp_mode>(static_cast<std::underlying_type_t<fp_mode>>(a) |
                              static_cast<std::underlying_type_t<fp_mode>>(b));
}

struct fp_control_key {
  template <fp_mode option>
  using value_t = ext::oneapi::experimental::property_value<
      fp_control_key, std::integral_constant<fp_mode, option>>;
};

template <fp_mode option = fp_mode::rte>
inline constexpr fp_control_key::value_t<option> fp_control;

} // namespace ext::intel::experimental

namespace ext::oneapi::experimental {
template <>
struct is_property_key<intel::experimental::fp_control_key> : std::true_type {};

template <typename T, typename PropertyListT>
struct is_property_key_of<
    intel::experimental::fp_control_key,
    intel::experimental::kernel_attribute<T, PropertyListT>> : std::true_type {
};

namespace detail {
template <> struct PropertyToKind<intel::experimental::fp_control_key> {
  static constexpr PropKind Kind = FloatingPointControls;
};

template <>
struct IsCompileTimeProperty<intel::experimental::fp_control_key>
    : std::true_type {};

template <intel::experimental::fp_mode FPMode>
struct PropertyMetaInfo<intel::experimental::fp_control_key::value_t<FPMode>> {
  static constexpr const char *name = "sycl-floating-point-control";
  static constexpr intel::experimental::fp_mode value = FPMode;
};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
