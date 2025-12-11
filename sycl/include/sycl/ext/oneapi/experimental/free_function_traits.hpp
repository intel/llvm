//==-------- free_function_traits.hpp - SYCL free function queries --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/detail/kernel_desc.hpp>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template <auto *Func, int Dims> struct is_nd_range_kernel {
  static constexpr bool value = false;
};

template <auto *Func> struct is_single_task_kernel {
  static constexpr bool value = false;
};

template <auto *Func, int Dims>
inline constexpr bool is_nd_range_kernel_v =
    is_nd_range_kernel<Func, Dims>::value;

template <auto *Func>
inline constexpr bool is_single_task_kernel_v =
    is_single_task_kernel<Func>::value;

template <auto *Func> struct is_kernel {
// During device compilation mode the compiler does not yet know
// what the kernels are named because that is exactly what its trying to
// figure out during this phase. Therefore, we set the is_kernel trait to true
// by default during device compilation in order to not get missing functions
// errors.
#ifdef __SYCL_DEVICE_ONLY__
  static constexpr bool value = true;
#else
  static constexpr bool value = false;
#endif
};

template <auto *Func>
inline constexpr bool is_kernel_v = is_kernel<Func>::value;

namespace detail {
// A struct with special type is a struct type that contains special types
// passed as a paremeter to a free function kernel. It is decomposed into its
// consituents by the frontend which puts the relevant informaton about each of
// them into the struct below, namely offset, size and parameter kind for each
// one of them. The runtime then calls the addArg function to add each one of
// them as kernel arguments. The value bool is used to distinguish these structs
// from ordinary e.g standard layout structs.
template <typename T> struct is_struct_with_special_type {
  static constexpr bool value = false;
  static constexpr int offsets[] = {-1};
  static constexpr int sizes[] = {-1};
  static constexpr sycl::detail::kernel_param_kind_t kinds[] = {
      sycl::detail::kernel_param_kind_t::kind_invalid};
};

} // namespace detail
} // namespace ext::oneapi::experimental

template <typename T> struct is_device_copyable;

} // namespace _V1
} // namespace sycl
