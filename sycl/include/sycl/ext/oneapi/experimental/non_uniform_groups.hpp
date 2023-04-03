//==--- non_uniform_groups.hpp --- SYCL extension for non-uniform groups ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_vars.hpp>
#include <sycl/ext/oneapi/sub_group_mask.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {

template <class T> struct is_fixed_topology_group : std::false_type {};

template <class T>
inline constexpr bool is_fixed_topology_group_v =
    is_fixed_topology_group<T>::value;

#ifdef SYCL_EXT_ONEAPI_ROOT_GROUP
template <> struct is_fixed_topology_group<root_group> : std::true_type {};
#endif

template <int Dimensions>
struct is_fixed_topology_group<sycl::group<Dimensions>> : std::true_type {};

template <> struct is_fixed_topology_group<sycl::sub_group> : std::true_type {};

template <class T> struct is_user_constructed_group : std::false_type {};

template <class T>
inline constexpr bool is_user_constructed_group_v =
    is_user_constructed_group<T>::value;

#ifdef __SYCL_DEVICE_ONLY__
// TODO: This may need to be generalized beyond uint32_t for big masks
namespace detail {
uint32_t CallerPositionInMask(sub_group_mask Mask) {
  // FIXME: It would be nice to be able to jump straight to an __ocl_vec_t
  sycl::marray<unsigned, 4> TmpMArray;
  Mask.extract_bits(TmpMArray);
  sycl::vec<unsigned, 4> MemberMask;
  for (int i = 0; i < 4; ++i) {
    MemberMask[i] = TmpMArray[i];
  }
  auto OCLMask =
      sycl::detail::ConvertToOpenCLType_t<sycl::vec<unsigned, 4>>(MemberMask);
  return __spirv_GroupNonUniformBallotBitCount(
      __spv::Scope::Subgroup, (int)__spv::GroupOperation::ExclusiveScan,
      OCLMask);
}
} // namespace detail
#endif

} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
