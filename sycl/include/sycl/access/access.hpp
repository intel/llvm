//==---------------- access.hpp --- SYCL access ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/__spirv/spirv_ops.hpp>
#include <type_traits>
#endif

namespace sycl {
inline namespace _V1 {
namespace access {

enum class target {
  global_buffer __SYCL2020_DEPRECATED("use 'target::device' instead") = 2014,
  constant_buffer __SYCL2020_DEPRECATED("use 'target::device' instead") = 2015,
  local __SYCL2020_DEPRECATED("use `local_accessor` instead") = 2016,
  image = 2017,
  host_buffer __SYCL2020_DEPRECATED("use 'host_accessor' instead") = 2018,
  host_image = 2019,
  image_array = 2020,
  host_task = 2021,
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

template <typename T> struct remove_decoration_impl {
  using type = T;
};

#ifdef __SYCL_DEVICE_ONLY__
template <typename T> struct remove_decoration_impl<__OPENCL_GLOBAL_AS__ T> {
  using type = T;
};

#ifdef __ENABLE_USM_ADDR_SPACE__
template <typename T>
struct remove_decoration_impl<__OPENCL_GLOBAL_DEVICE_AS__ T> {
  using type = T;
};

template <typename T>
struct remove_decoration_impl<__OPENCL_GLOBAL_HOST_AS__ T> {
  using type = T;
};

#endif // __ENABLE_USM_ADDR_SPACE__

template <typename T> struct remove_decoration_impl<__OPENCL_PRIVATE_AS__ T> {
  using type = T;
};

template <typename T> struct remove_decoration_impl<__OPENCL_LOCAL_AS__ T> {
  using type = T;
};

template <typename T> struct remove_decoration_impl<__OPENCL_CONSTANT_AS__ T> {
  using type = T;
};
#endif // __SYCL_DEVICE_ONLY__
} // namespace detail

template <typename T> struct remove_decoration {
  using type = typename detail::remove_decoration_impl<T>::type;
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

template <typename T>
using remove_decoration_t = typename remove_decoration<T>::type;

namespace detail {
#ifdef __SYCL_DEVICE_ONLY__
inline constexpr bool
address_space_cast_is_possible(access::address_space Src,
                               access::address_space Dst) {
  // constant_space is unique and is not interchangeable with any other.
  auto constant_space = access::address_space::constant_space;
  if (Src == constant_space || Dst == constant_space)
    return Src == Dst;

  auto generic_space = access::address_space::generic_space;
  if (Src == Dst || Src == generic_space || Dst == generic_space)
    return true;

  // global_host/global_device could be casted to/from global
  auto global_space = access::address_space::global_space;
  auto global_device = access::address_space::ext_intel_global_device_space;
  auto global_host = access::address_space::ext_intel_global_host_space;

  if (Src == global_space || Dst == global_space) {
    auto Other = Src == global_space ? Dst : Src;
    if (Other == global_device || Other == global_host)
      return true;
  }

  // No more compatible combinations.
  return false;
}

template <access::address_space Space, typename ElementType>
auto static_address_cast(ElementType *Ptr) {
  constexpr auto generic_space = access::address_space::generic_space;
  constexpr auto global_space = access::address_space::global_space;
  constexpr auto local_space = access::address_space::local_space;
  constexpr auto private_space = access::address_space::private_space;
  constexpr auto global_device =
      access::address_space::ext_intel_global_device_space;
  constexpr auto global_host =
      access::address_space::ext_intel_global_host_space;

  constexpr auto SrcAS = deduce_AS<ElementType *>::value;
  static_assert(address_space_cast_is_possible(SrcAS, Space));

  using dst_type = typename DecoratedType<
      std::remove_pointer_t<remove_decoration_t<ElementType *>>, Space>::type *;

  // Note: reinterpret_cast isn't enough for some of the casts between different
  // address spaces, use C-style cast instead.
#if !defined(__SPIR__)
  return (dst_type)Ptr;
#else
  if constexpr (SrcAS != generic_space) {
    return (dst_type)Ptr;
  } else if constexpr (Space == global_space) {
    return (dst_type)__spirv_GenericCastToPtr_ToGlobal(
        Ptr, __spv::StorageClass::CrossWorkgroup);
  } else if constexpr (Space == local_space) {
    return (dst_type)__spirv_GenericCastToPtr_ToLocal(
        Ptr, __spv::StorageClass::Workgroup);
  } else if constexpr (Space == private_space) {
    return (dst_type)__spirv_GenericCastToPtr_ToPrivate(
        Ptr, __spv::StorageClass::Function);
#if !defined(__ENABLE_USM_ADDR_SPACE__)
  } else if constexpr (Space == global_device || Space == global_host) {
    // If __ENABLE_USM_ADDR_SPACE__ isn't defined then both
    // global_device/global_host are just aliases for global_space.
    return (dst_type)__spirv_GenericCastToPtr_ToGlobal(
        Ptr, __spv::StorageClass::CrossWorkgroup);
#endif
  } else {
    return (dst_type)Ptr;
  }
#endif
}

// Previous implementation (`castAS`, used in `multi_ptr` ctors among other
// places), used C-style cast instead of a proper dynamic check for some
// backends/spaces. `SupressNotImplementedAssert = true` parameter is emulating
// that previous behavior until the proper support is added for compatibility
// reasons.
template <access::address_space Space, bool SupressNotImplementedAssert = false,
          typename ElementType>
auto dynamic_address_cast(ElementType *Ptr) {
  constexpr auto generic_space = access::address_space::generic_space;
  constexpr auto global_space = access::address_space::global_space;
  constexpr auto local_space = access::address_space::local_space;
  constexpr auto private_space = access::address_space::private_space;
  constexpr auto global_device =
      access::address_space::ext_intel_global_device_space;
  constexpr auto global_host =
      access::address_space::ext_intel_global_host_space;

  constexpr auto SrcAS = deduce_AS<ElementType *>::value;
  using dst_type = typename DecoratedType<
      std::remove_pointer_t<remove_decoration_t<ElementType *>>, Space>::type *;

  if constexpr (!address_space_cast_is_possible(SrcAS, Space)) {
    return (dst_type) nullptr;
  } else if constexpr (Space == generic_space) {
    return (dst_type)Ptr;
  } else if constexpr (Space == global_space &&
                       (SrcAS == global_device || SrcAS == global_host)) {
    return (dst_type)Ptr;
  } else if constexpr (SrcAS == global_space &&
                       (Space == global_device || Space == global_host)) {
#if defined(__ENABLE_USM_ADDR_SPACE__)
    static_assert(SupressNotImplementedAssert || Space != Space,
                  "Not supported yet!");
    return static_address_cast<Space>(Ptr);
#else
    // If __ENABLE_USM_ADDR_SPACE__ isn't defined then both
    // global_device/global_host are just aliases for global_space.
    static_assert(std::is_same_v<dst_type, ElementType *>);
    return (dst_type)Ptr;
#endif
#if defined(__SPIR__)
  } else if constexpr (Space == global_space) {
    return (dst_type)__spirv_GenericCastToPtrExplicit_ToGlobal(
        Ptr, __spv::StorageClass::CrossWorkgroup);
  } else if constexpr (Space == local_space) {
    return (dst_type)__spirv_GenericCastToPtrExplicit_ToLocal(
        Ptr, __spv::StorageClass::Workgroup);
  } else if constexpr (Space == private_space) {
    return (dst_type)__spirv_GenericCastToPtrExplicit_ToPrivate(
        Ptr, __spv::StorageClass::Function);
#if !defined(__ENABLE_USM_ADDR_SPACE__)
  } else if constexpr (SrcAS == generic_space &&
                       (Space == global_device || Space == global_host)) {
    return (dst_type)__spirv_GenericCastToPtrExplicit_ToGlobal(
        Ptr, __spv::StorageClass::CrossWorkgroup);
#endif
#endif
  } else {
    static_assert(SupressNotImplementedAssert || Space != Space,
                  "Not supported yet!");
    return static_address_cast<Space>(Ptr);
  }
}
#else  // __SYCL_DEVICE_ONLY__
template <access::address_space Space, typename ElementType>
auto static_address_cast(ElementType *Ptr) {
  return Ptr;
}
template <access::address_space Space, bool SupressNotImplementedAssert = false,
          typename ElementType>
auto dynamic_address_cast(ElementType *Ptr) {
  return Ptr;
}
#endif // __SYCL_DEVICE_ONLY__
} // namespace detail

#undef __OPENCL_GLOBAL_AS__
#undef __OPENCL_GLOBAL_DEVICE_AS__
#undef __OPENCL_GLOBAL_HOST_AS__
#undef __OPENCL_LOCAL_AS__
#undef __OPENCL_CONSTANT_AS__
#undef __OPENCL_PRIVATE_AS__

} // namespace _V1
} // namespace sycl
