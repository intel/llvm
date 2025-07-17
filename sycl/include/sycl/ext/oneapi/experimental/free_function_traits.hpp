//==-------- free_function_traits.hpp - SYCL free function queries --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
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
// A special type wrapper is a struct type that contains special types.
// The frontend defines this trait to be true after analyzing the struct at compile time.
template <typename T> struct is_special_type_wrapper {
  inline static constexpr bool value = false;
};

// This struct is made to be specialized in the integration header.
// It calls set_arg for every special type contained in the struct regardless of
// the level of nesting. So if type Foo contains two accessors inside and the
// user calls set_arg(Foo), that call will call this function which will call
// set_arg for each of those two accessors.
template <typename T> struct special_type_wrapper_info {
  template <typename ArgT, typename HandlerT>
  static void set_arg(int ArgIndex, ArgT &arg, HandlerT &cgh, int &NumArgs) {}
};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
