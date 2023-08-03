//==------ tangle_group.hpp --- SYCL extension for non-uniform groups ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/pi.h>                   // for PI_ERROR_INVALID_DEVICE
#include <sycl/detail/type_traits.hpp>        // for is_group, is_user_cons...
#include <sycl/exception.hpp>                 // for runtime_error
#include <sycl/ext/oneapi/sub_group_mask.hpp> // for sub_group_mask
#include <sycl/id.hpp>                        // for id
#include <sycl/memory_enums.hpp>              // for memory_scope
#include <sycl/range.hpp>                     // for range
#include <sycl/sub_group.hpp>                 // for sub_group

#include <type_traits> // for enable_if_t, decay_t

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template <typename ParentGroup> class tangle_group;

template <typename Group>
inline std::enable_if_t<sycl::is_group_v<std::decay_t<Group>> &&
                            std::is_same_v<Group, sycl::sub_group>,
                        tangle_group<Group>>
get_tangle_group(Group group);

template <typename ParentGroup> class tangle_group {
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
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

  id_type get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::CallerPositionInMask(Mask);
#else
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

  range_type get_group_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return 1;
#else
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

  range_type get_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return Mask.count();
#else
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

  linear_id_type get_group_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_id()[0]);
#else
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

  linear_id_type get_local_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_id()[0]);
#else
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

  linear_id_type get_group_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_range()[0]);
#else
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

  linear_id_type get_local_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_range()[0]);
#else
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

  bool leader() const {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t Lowest = static_cast<uint32_t>(Mask.find_low()[0]);
    return __spirv_SubgroupLocalInvocationId() == Lowest;
#else
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

protected:
  sub_group_mask Mask;

  tangle_group(sub_group_mask m) : Mask(m) {}

  friend tangle_group<ParentGroup> get_tangle_group<ParentGroup>(ParentGroup);

  friend sub_group_mask sycl::detail::GetMask<tangle_group<ParentGroup>>(
      tangle_group<ParentGroup> Group);
};

template <typename Group>
inline std::enable_if_t<sycl::is_group_v<std::decay_t<Group>> &&
                            std::is_same_v<Group, sycl::sub_group>,
                        tangle_group<Group>>
get_tangle_group(Group group) {
  (void)group;
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__SPIR__)
  // All SPIR-V devices that we currently target execute in SIMD fashion,
  // and so the group of work-items in converged control flow is implicit.
  // We store the mask here because it is required to calculate IDs, not
  // because it is required to construct the group.
  sub_group_mask mask = sycl::ext::oneapi::group_ballot(group, true);
  return tangle_group<sycl::sub_group>(mask);
#elif defined(__NVPTX__)
  // TODO: Construct from compiler-generated mask
#endif
#else
  throw runtime_error("Non-uniform groups are not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif

} // namespace this_kernel

template <typename ParentGroup>
struct is_user_constructed_group<tangle_group<ParentGroup>> : std::true_type {};

} // namespace ext::oneapi::experimental

template <typename ParentGroup>
struct is_group<ext::oneapi::experimental::tangle_group<ParentGroup>>
    : std::true_type {};

} // namespace _V1
} // namespace sycl
