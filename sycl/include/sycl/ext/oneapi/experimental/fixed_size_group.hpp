//==--- fixed_size_group.hpp --- SYCL extension for non-uniform groups -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aspects.hpp>
#include <sycl/detail/pi.h>            // for PI_ERROR_INVALID_DEVICE
#include <sycl/detail/type_traits.hpp> // for is_fixed_size_group, is_group
#include <sycl/exception.hpp>          // for runtime_error
#include <sycl/ext/oneapi/sub_group_mask.hpp> // for sub_group_mask
#include <sycl/id.hpp>                        // for id
#include <sycl/memory_enums.hpp>              // for memory_scope
#include <sycl/range.hpp>                     // for range
#include <sycl/sub_group.hpp>                 // for sub_group

#include <stddef.h>    // for size_t
#include <type_traits> // for enable_if_t, true_type, dec...

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template <size_t PartitionSize, typename ParentGroup> class fixed_size_group;

template <size_t PartitionSize, typename Group>
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::__uses_aspects__(sycl::aspect::ext_oneapi_fixed_size_group)]]
#endif
inline std::enable_if_t<sycl::is_group_v<std::decay_t<Group>> &&
                            std::is_same_v<Group, sycl::sub_group>,
                        fixed_size_group<PartitionSize, Group>>
get_fixed_size_group(Group group);

template <size_t PartitionSize, typename ParentGroup> class fixed_size_group {
public:
  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = typename ParentGroup::linear_id_type;
  static constexpr int dimensions = 1;
  static constexpr sycl::memory_scope fence_scope = ParentGroup::fence_scope;

  id_type get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupLocalInvocationId() / PartitionSize;
#else
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

  id_type get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupLocalInvocationId() % PartitionSize;
#else
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

  range_type get_group_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupMaxSize() / PartitionSize;
#else
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

  range_type get_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return PartitionSize;
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
    return get_local_linear_id() == 0;
#else
    throw runtime_error("Non-uniform groups are not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }

protected:
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  sub_group_mask Mask;
#endif

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  fixed_size_group(ext::oneapi::sub_group_mask mask) : Mask(mask) {}
#else
  fixed_size_group() {}
#endif

  friend fixed_size_group<PartitionSize, ParentGroup>
  get_fixed_size_group<PartitionSize, ParentGroup>(ParentGroup g);

  friend sub_group_mask
  sycl::detail::GetMask<fixed_size_group<PartitionSize, ParentGroup>>(
      fixed_size_group<PartitionSize, ParentGroup> Group);
};

template <size_t PartitionSize, typename Group>
inline std::enable_if_t<sycl::is_group_v<std::decay_t<Group>> &&
                            std::is_same_v<Group, sycl::sub_group>,
                        fixed_size_group<PartitionSize, Group>>
get_fixed_size_group(Group group) {
  (void)group;
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
  uint32_t loc_id = group.get_local_linear_id();
  uint32_t loc_size = group.get_local_linear_range();
  uint32_t bits = PartitionSize == 32
                      ? 0xffffffff
                      : ((1 << PartitionSize) - 1)
                            << ((loc_id / PartitionSize) * PartitionSize);

  return fixed_size_group<PartitionSize, sycl::sub_group>(
      sycl::detail::Builder::createSubGroupMask<ext::oneapi::sub_group_mask>(
          bits, loc_size));
#else
  return fixed_size_group<PartitionSize, sycl::sub_group>();
#endif
#else
  throw runtime_error("Non-uniform groups are not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <size_t PartitionSize, typename ParentGroup>
struct is_user_constructed_group<fixed_size_group<PartitionSize, ParentGroup>>
    : std::true_type {};

} // namespace ext::oneapi::experimental

namespace detail {
template <size_t PartitionSize, typename ParentGroup>
struct is_fixed_size_group<
    ext::oneapi::experimental::fixed_size_group<PartitionSize, ParentGroup>>
    : std::true_type {};
} // namespace detail

template <size_t PartitionSize, typename ParentGroup>
struct is_group<
    ext::oneapi::experimental::fixed_size_group<PartitionSize, ParentGroup>>
    : std::true_type {};

} // namespace _V1
} // namespace sycl
