//==-------- free_function_traits.hpp - SYCL free function queries --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <type_traits>
#include <sycl/detail/kernel_desc.hpp>

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
// A struct with special type is a struct type that contains special types.
// The frontend defines this trait to be true after analyzing the struct at
// compile time and contains information about Offset, Size and kind(accessor,
// etc...) inside the struct
template <typename T> struct is_struct_with_special_type {
  static constexpr bool value = false;
  static constexpr size_t offsetsSizes[2][];
  static constexpr detail::kernel_param_kind_t kinds[];
};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
