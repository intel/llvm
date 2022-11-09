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

#include <sycl/ext/intel/experimental/detail/misc_intrin.hpp>

#include <sycl/detail/boost/mp11.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::intel::experimental {

namespace kernel_properties {

/// <summary>
///  This namespace contains APIs to set kernel properties.
/// </summary>

// Implementation note: <property_class>::value fields should match property IDs
// specified in llvm/lib/SYCLLowerIR/LowerKernelProps.cpp

namespace detail {
// Proxy to access private property classes' fields from the API code.
template <class T> struct proxy {
  static constexpr int value = T::value;
};
} // namespace detail

/// A boolean property which requests the compiler to use large register
/// allocation mode at the expense of reducing the amount of available hardware
/// threads.
struct use_large_grf_tag {
  template <class T> friend struct detail::proxy;

private:
  // Property identifier
  static constexpr int value = 0;
};

__SYCL_DEPRECATED("use_double_grf is deprecated, use use_large_grf instead")
inline constexpr use_large_grf_tag use_double_grf = {};
inline constexpr use_large_grf_tag use_large_grf = {};

} // namespace kernel_properties

namespace __MP11_NS = sycl::detail::boost::mp11;

// TODO this should be replaced with the generic SYCL compile-time properites
// mechanism once implementaion is available.
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_kernel_properties.asciidoc

template <class... KernelProps>
void set_kernel_properties(KernelProps... props) {
  // TODO check for duplicates
  using Props = __MP11_NS::mp_list<KernelProps...>;
  __MP11_NS::mp_for_each<Props>([&](auto Prop) {
    using PropT = decltype(Prop);
    constexpr bool IsLargeGRF =
        std::is_same_v<PropT, kernel_properties::use_large_grf_tag>;
    if constexpr (IsLargeGRF) {
      __sycl_set_kernel_properties(
          kernel_properties::detail::proxy<
              kernel_properties::use_large_grf_tag>::value);
    } else {
      static_assert(IsLargeGRF &&
                    "set_kernel_properties: invalid kernel property");
    }
  });
}

} // namespace ext::intel::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
