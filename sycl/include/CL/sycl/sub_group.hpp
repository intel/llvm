//==----------- sub_group.hpp --- SYCL sub-group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_vars.hpp>
#include <CL/sycl/ONEAPI/functional.hpp>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/spirv.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/types.hpp>

#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
class sub_group {
public:
  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = uint32_t;
  static constexpr int dimensions = 1;
  static constexpr memory_scope fence_scope = memory_scope::sub_group;

  id_type get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupId();
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  id_type get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupLocalInvocationId();
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  range_type get_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupSize();
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  range_type get_group_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_NumSubgroups();
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  range_type get_max_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupMaxSize();
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  linear_id_type get_group_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_id()[0]);
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  linear_id_type get_local_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_id()[0]);
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  linear_id_type get_group_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_range()[0]);
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  linear_id_type get_local_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_range()[0]);
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  bool leader() const {
#ifdef __SYCL_DEVICE_ONLY__
    using namespace sycl::detail::spirv;
    return GroupNonUniformElect<group_scope<sub_group>>();
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }
};
}
}
