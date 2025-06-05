//==------- address_space_cast.hpp --- Implementation of AS casts ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/__spirv/spirv_types.hpp>

#include <type_traits>

namespace sycl {
inline namespace _V1 {

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
  constexpr auto SrcAS = deduce_AS<ElementType *>::value;
  static_assert(address_space_cast_is_possible(SrcAS, Space));

  using dst_type = typename DecoratedType<
      std::remove_pointer_t<remove_decoration_t<ElementType *>>, Space>::type *;

  // Note: reinterpret_cast isn't enough for some of the casts between different
  // address spaces, use C-style cast instead.
  return (dst_type)Ptr;
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
  using RemoveCvT = std::remove_cv_t<ElementType>;

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
    return detail::static_address_cast<Space>(Ptr);
#else
    // If __ENABLE_USM_ADDR_SPACE__ isn't defined then both
    // global_device/global_host are just aliases for global_space.
    static_assert(std::is_same_v<dst_type, ElementType *>);
    return (dst_type)Ptr;
#endif
  } else if constexpr (Space == global_space) {
    return (dst_type)__spirv_GenericCastToPtrExplicit_ToGlobal(
        const_cast<RemoveCvT *>(Ptr), __spv::StorageClass::CrossWorkgroup);
  } else if constexpr (Space == local_space) {
    return (dst_type)__spirv_GenericCastToPtrExplicit_ToLocal(
        const_cast<RemoveCvT *>(Ptr), __spv::StorageClass::Workgroup);
  } else if constexpr (Space == private_space) {
    return (dst_type)__spirv_GenericCastToPtrExplicit_ToPrivate(
        const_cast<RemoveCvT *>(Ptr), __spv::StorageClass::Function);
#if !defined(__ENABLE_USM_ADDR_SPACE__)
  } else if constexpr (SrcAS == generic_space &&
                       (Space == global_device || Space == global_host)) {
    return (dst_type)__spirv_GenericCastToPtrExplicit_ToGlobal(
        const_cast<RemoveCvT *>(Ptr), __spv::StorageClass::CrossWorkgroup);
#endif
  } else {
    static_assert(SupressNotImplementedAssert || Space != Space,
                  "Not supported yet!");
    return detail::static_address_cast<Space>(Ptr);
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

} // namespace _V1
} // namespace sycl
