//==---------------- access.hpp --- SYCL access ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/defines.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace access {

enum class target {
  global_buffer __SYCL2020_DEPRECATED("use 'target::device' instead") = 2014,
  constant_buffer = 2015,
  local __SYCL2020_DEPRECATED("use `local_accessor` instead") = 2016,
  image = 2017,
  host_buffer = 2018,
  host_image = 2019,
  image_array = 2020,
  device = global_buffer,
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
  ext_intel_global_device_space = 4,
  ext_intel_global_host_space = 5,
  generic_space = 6, // TODO generic_space address space is not supported yet
};

enum class decorated : int { no = 0, yes = 1, legacy = 2 };
} // namespace access

using access::target;
using access_mode = access::mode;

template <access_mode mode> struct mode_tag_t {
  explicit mode_tag_t() = default;
};

template <access_mode mode, target trgt> struct mode_target_tag_t {
  explicit mode_target_tag_t() = default;
};

#if __cplusplus >= 201703L

inline constexpr mode_tag_t<access_mode::read> read_only{};
inline constexpr mode_tag_t<access_mode::read_write> read_write{};
inline constexpr mode_tag_t<access_mode::write> write_only{};
inline constexpr mode_target_tag_t<access_mode::read, target::constant_buffer>
    read_constant{};

#else

namespace {

constexpr const auto &read_only =
    sycl::detail::InlineVariableHelper<mode_tag_t<access_mode::read>>::value;
constexpr const auto &read_write = sycl::detail::InlineVariableHelper<
    mode_tag_t<access_mode::read_write>>::value;
constexpr const auto &write_only =
    sycl::detail::InlineVariableHelper<mode_tag_t<access_mode::write>>::value;
constexpr const auto &read_constant = sycl::detail::InlineVariableHelper<
    mode_target_tag_t<access_mode::read, target::constant_buffer>>::value;

} // namespace

#endif

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
#ifdef __ENABLE_USM_ADDR_SPACE__
#define __OPENCL_GLOBAL_DEVICE_AS__ __attribute__((opencl_global_device))
#define __OPENCL_GLOBAL_HOST_AS__ __attribute__((opencl_global_host))
#else
#define __OPENCL_GLOBAL_DEVICE_AS__ __attribute__((opencl_global))
#define __OPENCL_GLOBAL_HOST_AS__ __attribute__((opencl_global))
#endif // __ENABLE_USM_ADDR_SPACE__
#define __OPENCL_LOCAL_AS__ __attribute__((opencl_local))
#define __OPENCL_CONSTANT_AS__ __attribute__((opencl_constant))
#define __OPENCL_PRIVATE_AS__ __attribute__((opencl_private))
#else
#define __OPENCL_GLOBAL_AS__
#define __OPENCL_GLOBAL_DEVICE_AS__
#define __OPENCL_GLOBAL_HOST_AS__
#define __OPENCL_LOCAL_AS__
#define __OPENCL_CONSTANT_AS__
#define __OPENCL_PRIVATE_AS__
#endif

template <access::target accessTarget> struct TargetToAS {
  constexpr static access::address_space AS =
      access::address_space::global_space;
};

#ifdef __ENABLE_USM_ADDR_SPACE__
template <> struct TargetToAS<access::target::device> {
  constexpr static access::address_space AS =
      access::address_space::ext_intel_global_device_space;
};
#endif // __ENABLE_USM_ADDR_SPACE__

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
struct DecoratedType<ElementType,
                     access::address_space::ext_intel_global_device_space> {
  using type = __OPENCL_GLOBAL_DEVICE_AS__ ElementType;
};

template <typename ElementType>
struct DecoratedType<ElementType,
                     access::address_space::ext_intel_global_host_space> {
  using type = __OPENCL_GLOBAL_HOST_AS__ ElementType;
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

#ifdef __ENABLE_USM_ADDR_SPACE__
template <class T> struct deduce_AS_impl<__OPENCL_GLOBAL_DEVICE_AS__ T> {
  static constexpr access::address_space value =
      access::address_space::ext_intel_global_device_space;
};

template <class T> struct deduce_AS_impl<__OPENCL_GLOBAL_HOST_AS__ T> {
  static constexpr access::address_space value =
      access::address_space::ext_intel_global_host_space;
};
#endif // __ENABLE_USM_ADDR_SPACE__

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

template <typename T> struct remove_decoration {
  using type = T;
};

// Propagate through const qualifier.
template <typename T> struct remove_decoration<const T> {
  using type = const typename remove_decoration<T>::type;
};

// Propagate through pointer.
template <typename T> struct remove_decoration<T *> {
  using type = typename remove_decoration<T>::type *;
};

// Propagate through const qualified pointer.
template <typename T> struct remove_decoration<const T *> {
  using type = const typename remove_decoration<T>::type *;
};

// Propagate through reference.
template <typename T> struct remove_decoration<T &> {
  using type = typename remove_decoration<T>::type &;
};

// Propagate through const qualified reference.
template <typename T> struct remove_decoration<const T &> {
  using type = const typename remove_decoration<T>::type &;
};

#ifdef __SYCL_DEVICE_ONLY__
template <typename T> struct remove_decoration<__OPENCL_GLOBAL_AS__ T> {
  using type = T;
};

#ifdef __ENABLE_USM_ADDR_SPACE__
template <typename T> struct remove_decoration<__OPENCL_GLOBAL_DEVICE_AS__ T> {
  using type = T;
};

template <typename T> struct remove_decoration<__OPENCL_GLOBAL_HOST_AS__ T> {
  using type = T;
};

#endif // __ENABLE_USM_ADDR_SPACE__

template <typename T> struct remove_decoration<__OPENCL_PRIVATE_AS__ T> {
  using type = T;
};

template <typename T> struct remove_decoration<__OPENCL_LOCAL_AS__ T> {
  using type = T;
};

template <typename T> struct remove_decoration<__OPENCL_CONSTANT_AS__ T> {
  using type = T;
};
#endif // __SYCL_DEVICE_ONLY__

template <typename T>
using remove_decoration_t = typename remove_decoration<T>::type;

namespace detail {

// Helper function for selecting appropriate casts between address spaces.
template <typename ToT, typename FromT> inline ToT cast_AS(FromT from) {
#ifdef __SYCL_DEVICE_ONLY__
  constexpr access::address_space ToAS = deduce_AS<ToT>::value;
  constexpr access::address_space FromAS = deduce_AS<FromT>::value;
  if constexpr (FromAS == access::address_space::generic_space) {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    // TODO: NVPTX and AMDGCN backends do not currently support the
    //       __spirv_GenericCastToPtrExplicit_* builtins, so to work around this
    //       we do C-style casting. This may produce warnings when targetting
    //       these backends.
    return (ToT)from;
#else
    using ToElemT = std::remove_pointer_t<remove_decoration_t<ToT>>;
    if constexpr (ToAS == access::address_space::global_space)
      return __SYCL_GenericCastToPtrExplicit_ToGlobal<ToElemT>(from);
    else if constexpr (ToAS == access::address_space::local_space)
      return __SYCL_GenericCastToPtrExplicit_ToLocal<ToElemT>(from);
    else if constexpr (ToAS == access::address_space::private_space)
      return __SYCL_GenericCastToPtrExplicit_ToPrivate<ToElemT>(from);
#ifdef __ENABLE_USM_ADDR_SPACE__
    else if constexpr (ToAS == access::address_space::
                                   ext_intel_global_device_space ||
                       ToAS ==
                           access::address_space::ext_intel_global_host_space)
      // For extended address spaces we do not currently have a SPIR-V
      // conversion function, so we do a C-style cast. This may produce
      // warnings.
      return (ToT)from;
#endif // __ENABLE_USM_ADDR_SPACE__
    else
      return reinterpret_cast<ToT>(from);
#endif // defined(__NVPTX__) || defined(__AMDGCN__)
  } else
#endif // __SYCL_DEVICE_ONLY__
  {
    return reinterpret_cast<ToT>(from);
  }
}

} // namespace detail

#undef __OPENCL_GLOBAL_AS__
#undef __OPENCL_GLOBAL_DEVICE_AS__
#undef __OPENCL_GLOBAL_HOST_AS__
#undef __OPENCL_LOCAL_AS__
#undef __OPENCL_CONSTANT_AS__
#undef __OPENCL_PRIVATE_AS__

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
