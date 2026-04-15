//==------------- access_base.hpp --- SYCL access base declarations -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
inline namespace _V1 {
namespace access {

enum class target {
  device = 2014,
  global_buffer __SYCL2020_DEPRECATED("use 'target::device' instead") = device,
  constant_buffer __SYCL2020_DEPRECATED("use 'target::device' instead") = 2015,
  local __SYCL2020_DEPRECATED("use `local_accessor` instead") = 2016,
  image = 2017,
  host_buffer __SYCL2020_DEPRECATED("use 'host_accessor' instead") = 2018,
  host_image = 2019,
  image_array = 2020,
  host_task = 2021,
};

enum class mode {
  read = 1024,
  write = 1025,
  read_write = 1026,
  discard_write = 1027,
  discard_read_write = 1028,
  atomic = 1029
};

enum class fence_space {
  local_space = 0,
  global_space = 1,
  global_and_local = 2
};

enum class placeholder { false_t = 0, true_t = 1 };

enum class address_space : int {
  private_space = 0,
  global_space = 1,
  constant_space __SYCL2020_DEPRECATED("sycl::access::address_space::constant_"
                                       "space is deprecated since SYCL 2020") =
      2,
  local_space = 3,
  generic_space = 4,
};

enum class decorated : int { no = 0, yes = 1, legacy = 2 };
} // namespace access

namespace detail {

#ifdef __SYCL_DEVICE_ONLY__
#define __OPENCL_GLOBAL_AS__ __attribute__((opencl_global))
#define __OPENCL_LOCAL_AS__ __attribute__((opencl_local))
#define __OPENCL_CONSTANT_AS__ __attribute__((opencl_constant))
#define __OPENCL_PRIVATE_AS__ __attribute__((opencl_private))
#else
#define __OPENCL_GLOBAL_AS__
#define __OPENCL_LOCAL_AS__
#define __OPENCL_CONSTANT_AS__
#define __OPENCL_PRIVATE_AS__
#endif

template <typename T> struct remove_decoration_impl {
  using type = T;
};

#ifdef __SYCL_DEVICE_ONLY__
template <typename T> struct remove_decoration_impl<__OPENCL_GLOBAL_AS__ T> {
  using type = T;
};

template <typename T> struct remove_decoration_impl<__OPENCL_PRIVATE_AS__ T> {
  using type = T;
};

template <typename T> struct remove_decoration_impl<__OPENCL_LOCAL_AS__ T> {
  using type = T;
};

template <typename T> struct remove_decoration_impl<__OPENCL_CONSTANT_AS__ T> {
  using type = T;
};
#endif
} // namespace detail

template <typename T> struct remove_decoration {
  using type = typename detail::remove_decoration_impl<T>::type;
};

template <typename T> struct remove_decoration<const T> {
  using type = const typename remove_decoration<T>::type;
};

template <typename T> struct remove_decoration<T *> {
  using type = typename remove_decoration<T>::type *;
};

template <typename T> struct remove_decoration<const T *> {
  using type = const typename remove_decoration<T>::type *;
};

template <typename T> struct remove_decoration<T &> {
  using type = typename remove_decoration<T>::type &;
};

template <typename T> struct remove_decoration<const T &> {
  using type = const typename remove_decoration<T>::type &;
};

template <typename T>
using remove_decoration_t = typename remove_decoration<T>::type;

#undef __OPENCL_GLOBAL_AS__
#undef __OPENCL_LOCAL_AS__
#undef __OPENCL_CONSTANT_AS__
#undef __OPENCL_PRIVATE_AS__

} // namespace _V1
} // namespace sycl