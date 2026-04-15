//==---- detail/sub_group_extra.hpp --- SYCL sub_group deprecated extras --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops.hpp>
#include <sycl/detail/spirv_memory_semantics.hpp>
#include <sycl/detail/sub_group_core.hpp>

namespace sycl {
inline namespace _V1 {

inline void sub_group::barrier() const {
#ifdef __SYCL_DEVICE_ONLY__
  __spirv_ControlBarrier(__spv::Scope::Subgroup, __spv::Scope::Subgroup,
                         __spv::MemorySemanticsMask::AcquireRelease |
                             __spv::MemorySemanticsMask::SubgroupMemory |
                             __spv::MemorySemanticsMask::WorkgroupMemory |
                             __spv::MemorySemanticsMask::CrossWorkgroupMemory);
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

inline void sub_group::barrier(access::fence_space accessSpace) const {
#ifdef __SYCL_DEVICE_ONLY__
  int32_t flags = sycl::detail::getSPIRVMemorySemanticsMask(accessSpace);
  __spirv_ControlBarrier(__spv::Scope::Subgroup, __spv::Scope::Subgroup, flags);
#else
  (void)accessSpace;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

} // namespace _V1
} // namespace sycl