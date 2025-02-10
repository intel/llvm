//==--- builtins_utils_scalar.hpp - SYCL built-in function utilities -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/aliases.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/half_type.hpp>
#include <sycl/multi_ptr.hpp>

#include <algorithm>
#include <cstring>

namespace sycl {
inline namespace _V1 {

namespace detail {
#ifdef __FAST_MATH__
template <typename T>
struct use_fast_math
    : std::is_same<std::remove_cv_t<get_elem_type_t<T>>, float> {};
#else
template <typename> struct use_fast_math : std::false_type {};
#endif
template <typename T> constexpr bool use_fast_math_v = use_fast_math<T>::value;

// Utility for converting a swizzle to a vector or preserve the type if it isn't
// a swizzle.
template <typename T> struct simplify_if_swizzle {
  using type = T;
};

template <typename T>
using simplify_if_swizzle_t = typename simplify_if_swizzle<T>::type;

// Utility trait for getting the decoration of a multi_ptr.
template <typename T> struct get_multi_ptr_decoration;
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
struct get_multi_ptr_decoration<
    multi_ptr<ElementType, Space, DecorateAddress>> {
  static constexpr access::decorated value = DecorateAddress;
};

template <typename T>
constexpr access::decorated get_multi_ptr_decoration_v =
    get_multi_ptr_decoration<T>::value;

// Utility trait for checking if a multi_ptr has a "writable" address space,
// i.e. global, local, private or generic.
template <typename T> struct has_writeable_addr_space : std::false_type {};
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
struct has_writeable_addr_space<multi_ptr<ElementType, Space, DecorateAddress>>
    : std::bool_constant<Space == access::address_space::global_space ||
                         Space == access::address_space::local_space ||
                         Space == access::address_space::private_space ||
                         Space == access::address_space::generic_space> {};

template <typename T>
constexpr bool has_writeable_addr_space_v = has_writeable_addr_space<T>::value;

} // namespace detail
} // namespace _V1
} // namespace sycl
