//==------ ballot_group.hpp --- SYCL extension for non-uniform groups ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/ext/oneapi/experimental/non_uniform_groups.hpp>
#include <sycl/ext/oneapi/sub_group_mask.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {

template <typename ParentGroup> class ballot_group;

template <typename Group>
inline std::enable_if_t<sycl::is_group_v<std::decay_t<Group>> &&
                            std::is_same_v<Group, sycl::sub_group>,
                        ballot_group<Group>>
get_ballot_group(Group group, bool predicate);

template <typename ParentGroup> class ballot_group {
public:
  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = typename ParentGroup::linear_id_type;
  static constexpr int dimensions = 1;
  static constexpr sycl::memory_scope fence_scope = ParentGroup::fence_scope;

  id_type get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return (Predicate) ? 1 : 0;
#else
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

  id_type get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return detail::CallerPositionInMask(Mask);
#else
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

  range_type get_group_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return 2;
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

private:
  sub_group_mask Mask;
  bool Predicate;

protected:
  ballot_group(sub_group_mask m, bool p) : Mask(m), Predicate(p) {}

  friend ballot_group<ParentGroup>
  get_ballot_group<ParentGroup>(ParentGroup g, bool predicate);
};

template <typename Group>
inline std::enable_if_t<sycl::is_group_v<std::decay_t<Group>> &&
                            std::is_same_v<Group, sycl::sub_group>,
                        ballot_group<Group>>
get_ballot_group(Group group, bool predicate) {
  (void)group;
#ifdef __SYCL_DEVICE_ONLY__
  // ballot_group partitions into two groups using the predicate
  // Membership mask for one group is negation of the other
  sub_group_mask mask = sycl::ext::oneapi::group_ballot(group, predicate);
  if (predicate) {
    return ballot_group<sycl::sub_group>(mask, predicate);
  } else {
    return ballot_group<sycl::sub_group>(~mask, predicate);
  }
#else
  (void)predicate;
  throw runtime_error("Non-uniform groups are not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename ParentGroup>
struct is_user_constructed_group<ballot_group<ParentGroup>> : std::true_type {};

} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
