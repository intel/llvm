//==---------------- access.hpp --- SYCL access ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <CL/sycl/detail/defines.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace access {

enum class target {
  global_buffer = 2014,
  constant_buffer,
  local,
  image,
  host_buffer,
  host_image,
  image_array
};

enum class mode {
  read = 1024,
  write,
  read_write,
  discard_write,
  discard_read_write,
  atomic
};

enum class fence_space {
  local_space,
  global_space,
  global_and_local
};

enum class placeholder { false_t, true_t };

enum class address_space : int {
  private_space = 0,
  global_space,
  constant_space,
  local_space
};

}  // namespace access

namespace detail {

constexpr bool isTargetHostAccess(access::target T) {
  return T == access::target::host_buffer || T == access::target::host_image;
}

constexpr bool modeNeedsOldData(access::mode m) {
  return m == access::mode::read || m == access::mode::write ||
         m == access::mode::read_write || m == access::mode::atomic;
}

constexpr bool modeWritesNewData(access::mode m) {
  return m != access::mode::read;
}

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

template <access::target accessTarget> struct TargetToAS {
  constexpr static access::address_space AS =
      access::address_space::global_space;
};

template <> struct TargetToAS<access::target::local> {
  constexpr static access::address_space AS =
      access::address_space::local_space;
};

template <> struct TargetToAS<access::target::constant_buffer> {
  constexpr static access::address_space AS =
      access::address_space::constant_space;
};

template <typename ElementType, access::address_space addressSpace>
struct PtrValueType;

template <typename ElementType>
struct PtrValueType<ElementType, access::address_space::private_space> {
  using type = __OPENCL_PRIVATE_AS__ ElementType;
};

template <typename ElementType>
struct PtrValueType<ElementType, access::address_space::global_space> {
  using type = __OPENCL_GLOBAL_AS__ ElementType;
};

template <typename ElementType>
struct PtrValueType<ElementType, access::address_space::constant_space> {
  // Current implementation of address spaces handling leads to possibility
  // of emitting incorrect (in terms of OpenCL) address space casts from
  // constant to generic (and vise-versa). So, global address space is used here
  // instead of constant to avoid incorrect address space casts in the produced
  // device code.
#if defined(RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR)
  using type = const __OPENCL_GLOBAL_AS__ ElementType;
#else
  using type = __OPENCL_GLOBAL_AS__ ElementType;
#endif
};

template <typename ElementType>
struct PtrValueType<ElementType, access::address_space::local_space> {
  using type = __OPENCL_LOCAL_AS__ ElementType;
};

template <class T>
struct remove_AS {
  typedef T type;
};

#ifdef __SYCL_DEVICE_ONLY__
template <class T>
struct remove_AS<__OPENCL_GLOBAL_AS__ T> {
  typedef T type;
};

template <class T>
struct remove_AS<__OPENCL_PRIVATE_AS__ T> {
  typedef T type;
};

template <class T>
struct remove_AS<__OPENCL_LOCAL_AS__ T> {
  typedef T type;
};

template <class T>
struct remove_AS<__OPENCL_CONSTANT_AS__ T> {
  typedef T type;
};
#endif

#undef __OPENCL_GLOBAL_AS__
#undef __OPENCL_LOCAL_AS__
#undef __OPENCL_CONSTANT_AS__
#undef __OPENCL_PRIVATE_AS__

} // namespace detail

}  // namespace sycl
}  // __SYCL_INLINE_NAMESPACE(cl)
