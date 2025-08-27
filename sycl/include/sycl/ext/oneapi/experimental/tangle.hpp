//==--------- tangle.hpp --- SYCL extension for non-uniform groups ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aspects.hpp>
#include <sycl/detail/spirv.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/experimental/non_uniform_groups.hpp>
#include <sycl/ext/oneapi/sub_group_mask.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/id.hpp>           // for id
#include <sycl/memory_enums.hpp> // for memory_scope
#include <sycl/range.hpp>
#include <sycl/sub_group.hpp>
#include <type_traits> // for enable_if_t, decay_t

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template <typename ParentGroup> class tangle;
template <typename ParentGroup>
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::__uses_aspects__(sycl::aspect::ext_oneapi_tangle)]]
#endif
inline std::enable_if_t<std::is_same_v<ParentGroup, sycl::sub_group>,
                        tangle<ParentGroup>> entangle(ParentGroup parent);

template <typename ParentGroup> class tangle {
public:
  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = typename ParentGroup::linear_id_type;
  static constexpr int dimensions = 1;
  static constexpr sycl::memory_scope fence_scope = ParentGroup::fence_scope;

  id_type get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<id_type>(0);
#else
    return id_type(0);
#endif
  }

  id_type get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::CallerPositionInMask(Mask);
#else
    return id_type(0);
#endif
  }

  range_type get_group_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return 1;
#else
    return range_type(0);
#endif
  }

  range_type get_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return Mask.count();
#else
    return range_type(0);
#endif
  }

  linear_id_type get_group_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_id()[0]);
#else
    return linear_id_type(0);
#endif
  }

  linear_id_type get_local_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_id()[0]);
#else
    return linear_id_type(0);
#endif
  }

  linear_id_type get_group_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_range()[0]);
#else
    return linear_id_type(0);
#endif
  }

  linear_id_type get_local_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_range()[0]);
#else
    return linear_id_type(0);
#endif
  }

  bool leader() const {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t Lowest = static_cast<uint32_t>(Mask.find_low()[0]);
    return __spirv_BuiltInSubgroupLocalInvocationId() == Lowest;
#else
    return false;
#endif
  }

protected:
#ifdef __SYCL_DEVICE_ONLY__
  sub_group_mask Mask;

  tangle(ext::oneapi::sub_group_mask mask) : Mask(mask) {}

  ext::oneapi::sub_group_mask getMask() const { return Mask; }
#else
  tangle() {}
#endif

  friend tangle<ParentGroup> entangle<ParentGroup>(ParentGroup);

  friend sub_group_mask
  sycl::detail::GetMask<tangle<ParentGroup>>(tangle<ParentGroup> Group);
};

template <typename ParentGroup>
inline std::enable_if_t<std::is_same_v<ParentGroup, sycl::sub_group>,
                        tangle<ParentGroup>>
entangle([[maybe_unused]] ParentGroup parent) {
#ifdef __SYCL_DEVICE_ONLY__
  // sync all work-items in parent group here
  sycl::group_barrier(parent);

#if defined(__SPIR__) || defined(__SPIRV__)
  // All SPIR-V devices that we currently target execute in SIMD fashion,
  // and so the group of work-items in converged control flow is implicit.
  // We store the mask here because it is required to calculate IDs, not
  // because it is required to construct the group.
  sub_group_mask mask = sycl::ext::oneapi::group_ballot(parent, true);
  return tangle<ParentGroup>(mask);
#elif defined(__NVPTX__)
  // TODO: CUDA devices will report false for the tangle
  //       support aspect so kernels launch should ensure this is never run.
  return tangle<ParentGroup>(0);
#endif
#else
  return tangle<ParentGroup>();
#endif
}

template <typename ParentGroup>
struct is_user_constructed_group<tangle<ParentGroup>> : std::true_type {};

} // namespace ext::oneapi::experimental

template <typename ParentGroup>
struct is_group<ext::oneapi::experimental::tangle<ParentGroup>>
    : std::true_type {};

} // namespace _V1
} // namespace sycl
