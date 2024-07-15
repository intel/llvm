//==--- opportunistic_group.hpp --- SYCL extension for non-uniform groups --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aspects.hpp>
#include <sycl/detail/spirv.hpp>
#include <sycl/detail/type_traits.hpp> // for is_group, is_user_cons...
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/experimental/non_uniform_groups.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp> // for this_sub_group
#include <sycl/ext/oneapi/sub_group_mask.hpp> // for sub_group_mask
#include <sycl/id.hpp>                        // for id
#include <sycl/memory_enums.hpp>              // for memory_scope
#include <sycl/range.hpp>                     // for range
#include <sycl/sub_group.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/ext/oneapi/functional.hpp>
#endif

#include <stdint.h>    // for uint32_t
#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

class opportunistic_group;

namespace this_kernel {
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::__uses_aspects__(
    sycl::aspect::ext_oneapi_opportunistic_group)]]
#endif
inline opportunistic_group get_opportunistic_group();
} // namespace this_kernel

class opportunistic_group {
public:
  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = uint32_t;
  static constexpr int dimensions = 1;
  static constexpr sycl::memory_scope fence_scope =
      sycl::memory_scope::sub_group;

  id_type get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<id_type>(0);
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
    return 1;
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

  opportunistic_group(sub_group_mask m) : Mask(m) {}

  friend opportunistic_group this_kernel::get_opportunistic_group();

  friend sub_group_mask
  sycl::detail::GetMask<opportunistic_group>(opportunistic_group Group);
};

namespace this_kernel {

inline opportunistic_group get_opportunistic_group() {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__SPIR__) || defined(__SPIRV__)
  // TODO: It may be wiser to call the intrinsic than rely on this_group()
  sycl::sub_group sg = sycl::ext::oneapi::experimental::this_sub_group();
  sub_group_mask mask = sycl::ext::oneapi::group_ballot(sg, true);
  return opportunistic_group(mask);
#elif defined(__NVPTX__)
  uint32_t active_mask;
  asm volatile("activemask.b32 %0;" : "=r"(active_mask));
  sub_group_mask mask =
      sycl::detail::Builder::createSubGroupMask<ext::oneapi::sub_group_mask>(
          active_mask, 32);
  return opportunistic_group(mask);
#endif
#else
  throw exception(make_error_code(errc::runtime),
                  "Non-uniform groups are not supported on host.");
#endif
}

} // namespace this_kernel

template <>
struct is_user_constructed_group<opportunistic_group> : std::true_type {};

} // namespace ext::oneapi::experimental

template <>
struct is_group<ext::oneapi::experimental::opportunistic_group>
    : std::true_type {};

} // namespace _V1
} // namespace sycl
