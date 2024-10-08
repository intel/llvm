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

// Masks defined here must be in sync with the LLVM pass for handling compile
// time properties (CompileTimePropertiesPass).
enum class fp_mode : std::uint32_t {
  round_to_nearest = 1,       // Round to nearest or even
  round_upward = 1 << 1,      // Round towards +ve inf
  round_downward = 1 << 2,    // Round towards -ve inf
  round_toward_zero = 1 << 3, // Round towards zero

  denorm_ftz = 1 << 4,      // Denorm mode flush to zero
  denorm_d_allow = 1 << 5,  // Denorm mode double allow
  denorm_f_allow = 1 << 6,  // Denorm mode float allow
  denorm_hf_allow = 1 << 7, // Denorm mode half allow
  denorm_allow = denorm_d_allow | denorm_f_allow |
                 denorm_hf_allow // Denorm mode double/float/half allow
};

constexpr fp_mode operator|(const fp_mode &a, const fp_mode &b) {
  return static_cast<fp_mode>(static_cast<std::underlying_type_t<fp_mode>>(a) |
                              static_cast<std::underlying_type_t<fp_mode>>(b));
}

namespace detail {
constexpr fp_mode operator&(const fp_mode &a, const fp_mode &b) {
  return static_cast<fp_mode>(static_cast<std::underlying_type_t<fp_mode>>(a) &
                              static_cast<std::underlying_type_t<fp_mode>>(b));
}
constexpr fp_mode operator^(const fp_mode &a, const fp_mode &b) {
  return static_cast<fp_mode>(static_cast<std::underlying_type_t<fp_mode>>(a) ^
                              static_cast<std::underlying_type_t<fp_mode>>(b));
}
constexpr bool isSet(const fp_mode &mode, const fp_mode &flag) {
  return (mode & flag) == flag;
}
constexpr bool checkMutuallyExclusive(const fp_mode &mode) {
  // Check that either none of the flags is set or only one flag is set.
  fp_mode roundMask = fp_mode::round_to_nearest | fp_mode::round_upward |
                      fp_mode::round_downward | fp_mode::round_toward_zero;
  bool isCorrectRoundingMode = ((mode & roundMask) == fp_mode(0)) ||
                               (isSet(mode, fp_mode::round_to_nearest) ^
                                isSet(mode, fp_mode::round_upward) ^
                                isSet(mode, fp_mode::round_downward) ^
                                isSet(mode, fp_mode::round_toward_zero));
  // Check that if denorm_ftz is set then other denorm flags are not set.
  fp_mode denormAllowMask = fp_mode::denorm_hf_allow | fp_mode::denorm_f_allow |
                            fp_mode::denorm_d_allow;
  bool isCorrectDenormMode = !isSet(mode, fp_mode::denorm_ftz) ||
                             ((mode & denormAllowMask) == fp_mode(0));
  return isCorrectRoundingMode && isCorrectDenormMode;
}

constexpr fp_mode setDefaultValuesIfNeeded(fp_mode mode) {
  fp_mode roundMask = fp_mode::round_to_nearest | fp_mode::round_upward |
                      fp_mode::round_downward | fp_mode::round_toward_zero;
  if ((mode & roundMask) == fp_mode(0))
    mode = mode | fp_mode::round_to_nearest;

  auto denormMask = fp_mode::denorm_ftz | fp_mode::denorm_hf_allow |
                    fp_mode::denorm_f_allow | fp_mode::denorm_d_allow;
  if ((mode & denormMask) == fp_mode(0))
    mode = mode | fp_mode::denorm_ftz;

  return mode;
}
} // namespace detail

struct fp_control_key
    : oneapi::experimental::detail::compile_time_property_key<
          oneapi::experimental::detail::PropKind::FloatingPointControls> {
  template <fp_mode option>
  using value_t = ext::oneapi::experimental::property_value<
      fp_control_key, std::integral_constant<fp_mode, option>>;
};

template <fp_mode option>
inline constexpr fp_control_key::value_t<option> fp_control;

} // namespace ext::intel::experimental

namespace ext::oneapi::experimental {
template <typename T, typename PropertyListT>
struct is_property_key_of<
    intel::experimental::fp_control_key,
    intel::experimental::kernel_attribute<T, PropertyListT>> : std::true_type {
};

namespace detail {
template <intel::experimental::fp_mode FPMode>
struct PropertyMetaInfo<intel::experimental::fp_control_key::value_t<FPMode>> {
  static_assert(intel::experimental::detail::checkMutuallyExclusive(FPMode),
                "Mutually exclusive fp modes are specified for the kernel.");
  static constexpr const char *name = "sycl-floating-point-control";
  static constexpr intel::experimental::fp_mode value =
      intel::experimental::detail::setDefaultValuesIfNeeded(FPMode);
};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
