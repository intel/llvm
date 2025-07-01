//==---------- fragment.hpp --- SYCL extension for non-uniform groups ------==//
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
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/sub_group_mask.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/id.hpp>
#include <sycl/memory_enums.hpp>
#include <sycl/range.hpp>
#include <sycl/sub_group.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/ext/oneapi/functional.hpp>
#endif

#include <stdint.h>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template <typename ParentGroup> class fragment;

template <typename ParentGroup>
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::__uses_aspects__(sycl::aspect::ext_oneapi_fragment)]]
#endif
inline std::enable_if_t<sycl::is_group_v<std::decay_t<ParentGroup>> &&
                            std::is_same_v<ParentGroup, sycl::sub_group>,
                        fragment<ParentGroup>>
binary_partition(ParentGroup parent, bool predicate);

namespace this_work_item {
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::__uses_aspects__(sycl::aspect::ext_oneapi_fragment)]]
#endif
inline fragment<sycl::sub_group> get_opportunistic_group();
} // namespace this_work_item

template <typename ParentGroup> class fragment {
public:
  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = typename ParentGroup::linear_id_type;
  static constexpr int dimensions = 1;
  static constexpr sycl::memory_scope fence_scope = ParentGroup::fence_scope;

  id_type get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return GroupID;
#else
    throw exception(make_error_code(errc::runtime),
                    "Non-uniform groups are not supported on host.");
#endif
  }

  id_type get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::CallerPositionInMask(Mask);
#else
    throw exception(make_error_code(errc::runtime),
                    "Non-uniform groups are not supported on host.");
#endif
  }

  range_type get_group_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return GroupRange;
#else
    throw exception(make_error_code(errc::runtime),
                    "Non-uniform groups are not supported on host.");
#endif
  }

  range_type get_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return Mask.count();
#else
    throw exception(make_error_code(errc::runtime),
                    "Non-uniform groups are not supported on host.");
#endif
  }

  linear_id_type get_group_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_id()[0]);
#else
    throw exception(make_error_code(errc::runtime),
                    "Non-uniform groups are not supported on host.");
#endif
  }

  linear_id_type get_local_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_id()[0]);
#else
    throw exception(make_error_code(errc::runtime),
                    "Non-uniform groups are not supported on host.");
#endif
  }

  linear_id_type get_group_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_range()[0]);
#else
    throw exception(make_error_code(errc::runtime),
                    "Non-uniform groups are not supported on host.");
#endif
  }

  linear_id_type get_local_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_range()[0]);
#else
    throw exception(make_error_code(errc::runtime),
                    "Non-uniform groups are not supported on host.");
#endif
  }

  bool leader() const {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t Lowest = static_cast<uint32_t>(Mask.find_low()[0]);
    return __spirv_SubgroupLocalInvocationId() == Lowest;
#else
    throw exception(make_error_code(errc::runtime),
                    "Non-uniform groups are not supported on host.");
#endif
  }

protected:
  sub_group_mask Mask;
  id_type GroupID;
  range_type GroupRange;

  fragment(sub_group_mask m, id_type group_id, range_type group_range)
      : Mask(m), GroupID(group_id), GroupRange(group_range) {}

  friend fragment<ParentGroup> binary_partition<ParentGroup>(ParentGroup parent,
                                                             bool predicate);

  friend fragment<sycl::sub_group> this_work_item::get_opportunistic_group();

  friend sub_group_mask
  sycl::detail::GetMask<fragment<ParentGroup>>(fragment<ParentGroup> Group);
};

template <typename ParentGroup>
inline std::enable_if_t<sycl::is_group_v<std::decay_t<ParentGroup>> &&
                            std::is_same_v<ParentGroup, sycl::sub_group>,
                        fragment<ParentGroup>>
binary_partition(ParentGroup parent, bool predicate) {
  (void)parent;
#ifdef __SYCL_DEVICE_ONLY__
  // sync all work-items in parent group before partitioning
  sycl::group_barrier(parent);

#if defined(__SPIR__) || defined(__SPIRV__) || defined(__NVPTX__)
  sub_group_mask mask = sycl::ext::oneapi::group_ballot(parent, predicate);
  id<1> group_id = predicate ? 1 : 0;
  range<1> group_range = 2; // 2 groupds based on predicate by binary_partition

  if (!predicate) {
    sub_group_mask::BitsType participant_filter =
        (~sub_group_mask::BitsType{0}) >>
        (sub_group_mask::max_bits - parent.get_local_linear_range());
    mask = (~mask) & participant_filter;
  }

  return fragment<ParentGroup>(mask, group_id, group_range);
#endif
#else
  (void)predicate;
  throw exception(make_error_code(errc::runtime),
                  "Non-uniform groups are not supported on host.");
#endif
}

namespace this_work_item {

inline fragment<sycl::sub_group> get_opportunistic_group() {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__SPIR__) || defined(__SPIRV__)
  sycl::sub_group sg = sycl::ext::oneapi::experimental::this_sub_group();
  sub_group_mask mask = sycl::ext::oneapi::group_ballot(sg, true);
  return fragment<sycl::sub_group>(mask, 0, 1);
#elif defined(__NVPTX__)
  uint32_t active_mask;
  asm volatile("activemask.b32 %0;" : "=r"(active_mask));
  sub_group_mask mask =
      sycl::detail::Builder::createSubGroupMask<ext::oneapi::sub_group_mask>(
          active_mask, 32);
  return fragment<sycl::sub_group>(mask, 0, 1);
#endif
#else
  throw exception(make_error_code(errc::runtime),
                  "Non-uniform groups are not supported on host.");
#endif
}

} // namespace this_work_item

template <typename ParentGroup>
struct is_user_constructed_group<fragment<ParentGroup>> : std::true_type {};

template <typename ParentGroup>
struct is_fragment<fragment<ParentGroup>> : std::true_type {};

} // namespace ext::oneapi::experimental

template <typename ParentGroup>
struct is_group<ext::oneapi::experimental::fragment<ParentGroup>>
    : std::true_type {};

} // namespace _V1
} // namespace sycl