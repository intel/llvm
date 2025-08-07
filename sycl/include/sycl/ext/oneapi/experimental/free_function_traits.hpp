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
// A struct with special type is a struct type that contains special types.
// The frontend defines this trait to be true after analyzing the struct at
// compile time.
template <typename T> struct is_struct_with_special_type {
  inline static constexpr bool value = false;
};

// This struct is made to be specialized in the integration header.
// It calls set_arg for every member of contained in the struct at
// any level of composition. So if type Foo contains two accessors and an integer
// inside and the user calls set_arg(Foo) which calls this function with T = Foo
// which will call set_arg for each of those two accessors and the int.
// The function stores in NumArgs the number of set_arg calls that it made so
// that subsequent set_arg calls initiated by the user can have the correct
// index.
template <typename T> struct struct_with_special_type_info {
  template <typename ArgT, typename HandlerT>
  static void set_arg([[maybe_unused]] int ArgIndex, [[maybe_unused]] ArgT &arg,
                      [[maybe_unused]] HandlerT &cgh,
                      [[maybe_unused]] int &NumArgs) {}
};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
