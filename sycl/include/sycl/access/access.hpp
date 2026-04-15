//==---------------- access.hpp --- SYCL access ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access_base.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#include <type_traits>
#endif

namespace sycl {
inline namespace _V1 {
using access::target;
using access_mode = access::mode;

enum class image_target : unsigned int { device = 0, host_task = 1 };

template <access_mode mode> struct mode_tag_t {
  explicit mode_tag_t() = default;
};

template <access_mode mode, target trgt> struct mode_target_tag_t {
  explicit mode_target_tag_t() = default;
};

inline constexpr mode_tag_t<access_mode::read> read_only{};
inline constexpr mode_tag_t<access_mode::read_write> read_write{};
inline constexpr mode_tag_t<access_mode::write> write_only{};
inline constexpr mode_target_tag_t<access_mode::read, target::constant_buffer>
    read_constant{};
inline constexpr mode_target_tag_t<access_mode::read, target::host_task>
    read_only_host_task;
inline constexpr mode_target_tag_t<access_mode::read_write, target::host_task>
    read_write_host_task;
inline constexpr mode_target_tag_t<access_mode::write, target::host_task>
    write_only_host_task;

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

template <access::decorated Decorated> struct NegateDecorated;
template <> struct NegateDecorated<access::decorated::yes> {
  static constexpr access::decorated value = access::decorated::no;
};
template <> struct NegateDecorated<access::decorated::no> {
  static constexpr access::decorated value = access::decorated::yes;
};

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
struct DecoratedType;

template <typename ElementType>
struct DecoratedType<ElementType, access::address_space::private_space> {
  using type = __OPENCL_PRIVATE_AS__ ElementType;
};

template <typename ElementType>
struct DecoratedType<ElementType, access::address_space::generic_space> {
  using type = ElementType;
};

template <typename ElementType>
struct DecoratedType<ElementType, access::address_space::global_space> {
  using type = __OPENCL_GLOBAL_AS__ ElementType;
};

template <typename ElementType>
struct DecoratedType<ElementType, access::address_space::constant_space> {
  // Current implementation of address spaces handling leads to possibility
  // of emitting incorrect (in terms of OpenCL) address space casts from
  // constant to generic (and vise-versa). So, global address space is used
  // here instead of constant to avoid incorrect address space casts in the
  // produced device code.
#if defined(RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR)
  using type = const __OPENCL_GLOBAL_AS__ ElementType;
#else
  using type = __OPENCL_GLOBAL_AS__ ElementType;
#endif
};

template <typename ElementType>
struct DecoratedType<ElementType, access::address_space::local_space> {
  using type = __OPENCL_LOCAL_AS__ ElementType;
};

#ifdef __SYCL_DEVICE_ONLY__
template <class T> struct deduce_AS_impl {
  // Undecorated pointers are considered generic.
  // TODO: This assumes that the implementation uses generic as default. If
  //       address space inference is used this may need to change.
  static constexpr access::address_space value =
      access::address_space::generic_space;
};

template <class T> struct deduce_AS_impl<__OPENCL_GLOBAL_AS__ T> {
  static constexpr access::address_space value =
      access::address_space::global_space;
};

template <class T> struct deduce_AS_impl<__OPENCL_PRIVATE_AS__ T> {
  static constexpr access::address_space value =
      access::address_space::private_space;
};

template <class T> struct deduce_AS_impl<__OPENCL_LOCAL_AS__ T> {
  static constexpr access::address_space value =
      access::address_space::local_space;
};

template <class T> struct deduce_AS_impl<__OPENCL_CONSTANT_AS__ T> {
  static constexpr access::address_space value =
      access::address_space::constant_space;
};

template <class T>
struct deduce_AS
    : deduce_AS_impl<
          std::remove_pointer_t<std::remove_reference_t<std::remove_cv_t<T>>>> {
};
#endif
} // namespace detail

#undef __OPENCL_GLOBAL_AS__
#undef __OPENCL_LOCAL_AS__
#undef __OPENCL_CONSTANT_AS__
#undef __OPENCL_PRIVATE_AS__

} // namespace _V1
} // namespace sycl
