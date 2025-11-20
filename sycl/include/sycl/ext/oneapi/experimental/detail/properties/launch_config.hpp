//==------ launch_config.hpp ------- SYCL kernel launch configuration -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/ext/oneapi/properties/property.hpp>

namespace sycl {
inline namespace _V1 {
template <int Dimensions>
class nd_range;
template<int Dimensions>
class range;

namespace ext::oneapi::experimental {
namespace detail {
struct AllowCTADTag;
// Trait for identifying sycl::range and sycl::nd_range.
template <typename RangeT> struct is_range_or_nd_range : std::false_type {};
template <int Dimensions>
struct is_range_or_nd_range<range<Dimensions>> : std::true_type {};
template <int Dimensions>
struct is_range_or_nd_range<nd_range<Dimensions>> : std::true_type {};

template <typename RangeT>
constexpr bool is_range_or_nd_range_v = is_range_or_nd_range<RangeT>::value;

template <typename LCRangeT, typename LCPropertiesT> struct LaunchConfigAccess;

// Checks that none of the properties in the property list has compile-time
// effects on the kernel.
template <typename T>
struct NoPropertyHasCompileTimeKernelEffect : std::false_type {};
template <typename... Ts>
struct NoPropertyHasCompileTimeKernelEffect<properties_t<Ts...>> {
  static constexpr bool value =
      !(HasCompileTimeEffect<Ts>::value || ... || false);
};
} // namespace detail

// Available only when Range is range or nd_range
template <
    typename RangeT, typename PropertiesT = empty_properties_t,
    typename = std::enable_if_t<
        ext::oneapi::experimental::detail::is_range_or_nd_range_v<RangeT>>>
class launch_config {
  static_assert(ext::oneapi::experimental::detail::
                    NoPropertyHasCompileTimeKernelEffect<PropertiesT>::value,
                "launch_config does not allow properties with compile-time "
                "kernel effects.");

public:
  launch_config(RangeT Range, PropertiesT Properties = {})
      : MRange{Range}, MProperties{Properties} {}

private:
  RangeT MRange;
  PropertiesT MProperties;

  const RangeT &getRange() const noexcept { return MRange; }

  const PropertiesT &getProperties() const noexcept { return MProperties; }

  template <typename LCRangeT, typename LCPropertiesT>
  friend struct detail::LaunchConfigAccess;
};

#ifdef __cpp_deduction_guides
// CTAD work-around to avoid warning from GCC when using default deduction
// guidance.
launch_config(detail::AllowCTADTag)
    -> launch_config<void, empty_properties_t, void>;
#endif // __cpp_deduction_guides

namespace detail {
// Helper for accessing the members of launch_config.
template <typename LCRangeT, typename LCPropertiesT> struct LaunchConfigAccess {
  LaunchConfigAccess(const launch_config<LCRangeT, LCPropertiesT> &LaunchConfig)
      : MLaunchConfig{LaunchConfig} {}

  const launch_config<LCRangeT, LCPropertiesT> &MLaunchConfig;

  const LCRangeT &getRange() const noexcept { return MLaunchConfig.getRange(); }

  const LCPropertiesT &getProperties() const noexcept {
    return MLaunchConfig.getProperties();
  }
};
} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
