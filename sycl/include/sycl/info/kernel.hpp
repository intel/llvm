//==----- kernel.hpp - SYCL kernel information descriptors -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/info_desc_traits.hpp>
#include <sycl/range.hpp>
#include <unified-runtime/ur_api.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace sycl {
inline namespace _V1 {

class context;

namespace info {

// A.5 Kernel information desctiptors
namespace kernel {
template <ur_kernel_info_t UrCode>
using kernel_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::kernel, UrCode>;

struct num_args : kernel_traits<UR_KERNEL_INFO_NUM_ARGS> {
  using return_type = uint32_t;
};
struct attributes : kernel_traits<UR_KERNEL_INFO_ATTRIBUTES> {
  using return_type = std::string;
};
struct function_name : kernel_traits<UR_KERNEL_INFO_FUNCTION_NAME> {
  using return_type = std::string;
};
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
struct __SYCL_DEPRECATED("info::kernel::reference_count is not part of "
                         "SYCL 2020") reference_count
    : kernel_traits<UR_KERNEL_INFO_REFERENCE_COUNT> {
  using return_type = uint32_t;
};
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
struct context : kernel_traits<UR_KERNEL_INFO_CONTEXT> {
  using return_type = sycl::context;
};
} // namespace kernel

namespace kernel_device_specific {
// kernel_device_specific traits dispatch through three UR APIs and so mix
// three native UR enum families; info_class::kernel_device_specific has
// ur_code_type = void to permit the mismatch. Use the matching alias for
// each family.
template <ur_kernel_group_info_t UrCode>
using group_traits = sycl::detail::ur_traits_base<
    sycl::detail::info_class::kernel_device_specific, UrCode>;
template <ur_kernel_sub_group_info_t UrCode>
using sub_group_traits = sycl::detail::ur_traits_base<
    sycl::detail::info_class::kernel_device_specific, UrCode>;
template <ur_kernel_info_t UrCode>
using kernel_info_traits = sycl::detail::ur_traits_base<
    sycl::detail::info_class::kernel_device_specific, UrCode>;

struct global_work_size : group_traits<UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE> {
  using return_type = sycl::range<3>;
};
struct work_group_size : group_traits<UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE> {
  using return_type = size_t;
};
struct compile_work_group_size
    : group_traits<UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE> {
  using return_type = sycl::range<3>;
};
struct preferred_work_group_size_multiple
    : group_traits<UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> {
  using return_type = size_t;
};
struct private_mem_size : group_traits<UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE> {
  using return_type = size_t;
};
struct max_num_sub_groups
    : sub_group_traits<UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS> {
  using return_type = uint32_t;
};
struct compile_num_sub_groups
    : sub_group_traits<UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS> {
  using return_type = uint32_t;
};
struct max_sub_group_size
    : sub_group_traits<UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE> {
  using return_type = uint32_t;
};
struct compile_sub_group_size
    : sub_group_traits<UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL> {
  using return_type = uint32_t;
};
struct ext_codeplay_num_regs : kernel_info_traits<UR_KERNEL_INFO_NUM_REGS> {
  using return_type = uint32_t;
};
} // namespace kernel_device_specific

} // namespace info

namespace detail {
// SFINAE predicates confining `kernel::get_info<T>()` to kernel traits and
// `kernel::get_info<T>(device)` to kernel-device-specific traits. The
// `return_type` alias is load-bearing for ABI symbol mangling — keep stable.
template <typename T>
struct is_kernel_info_desc : is_info_desc_for<T, info_class::kernel> {};
template <typename T>
struct is_kernel_device_specific_info_desc
    : is_info_desc_for<T, info_class::kernel_device_specific> {};
} // namespace detail
} // namespace _V1
} // namespace sycl
