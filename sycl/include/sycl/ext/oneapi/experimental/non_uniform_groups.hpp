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
#include <sycl/types.hpp>
#include <sycl/ext/oneapi/sub_group_mask.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace detail {

inline sycl::vec<unsigned, 4> ExtractMask(ext::oneapi::sub_group_mask Mask) {
  sycl::marray<unsigned, 4> TmpMArray;
  Mask.extract_bits(TmpMArray);
  sycl::vec<unsigned, 4> MemberMask;
  for (int i = 0; i < 4; ++i) {
    MemberMask[i] = TmpMArray[i];
  }
  return MemberMask;
}

#ifdef __SYCL_DEVICE_ONLY__
// TODO: This may need to be generalized beyond uint32_t for big masks
inline uint32_t CallerPositionInMask(ext::oneapi::sub_group_mask Mask) {
  sycl::vec<unsigned, 4> MemberMask = ExtractMask(Mask);
  auto OCLMask =
      sycl::detail::ConvertToOpenCLType_t<sycl::vec<unsigned, 4>>(MemberMask);
  return __spirv_GroupNonUniformBallotBitCount(
      __spv::Scope::Subgroup, (int)__spv::GroupOperation::ExclusiveScan,
      OCLMask);
}
#endif

template <typename NonUniformGroup>
inline uint32_t IdToMaskPosition(NonUniformGroup Group, uint32_t Id) {
  // TODO: This will need to be optimized
  sycl::vec<unsigned, 4> MemberMask = ExtractMask(Group.Mask);
  uint32_t Count = 0;
  for (int i = 0; i < 4; ++i) {
    for (int b = 0; b < 32; ++b) {
      if (MemberMask[i] & (1 << b)) {
        if (Count == Id) {
          return i * 32 + b;
        }
        Count++;
      }
    }
  }
  __builtin_unreachable();
  return Count;
}

} // namespace detail

namespace ext::oneapi::experimental {

// Forward declarations of non-uniform group types for algorithm definitions
template <typename ParentGroup> class ballot_group;

} // namespace ext::oneapi::experimental

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
