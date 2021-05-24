//==----------- sub_group.hpp --- SYCL sub-group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/ONEAPI/sub_group.hpp>
#include <CL/sycl/group.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
using ONEAPI::sub_group;
// TODO move the entire sub_group class implementation to this file once
// breaking changes are allowed.

template <>
inline void group_barrier<sub_group>(sub_group Group, memory_scope FenceScope) {
  (void)Group;
  (void)FenceScope;
#ifdef __SYCL_DEVICE_ONLY__
  __spirv_ControlBarrier(__spv::Scope::Subgroup,
                         detail::spirv::getScope(FenceScope),
                         __spv::MemorySemanticsMask::AcquireRelease |
                             __spv::MemorySemanticsMask::SubgroupMemory |
                             __spv::MemorySemanticsMask::WorkgroupMemory |
                             __spv::MemorySemanticsMask::CrossWorkgroupMemory);
#else
  throw runtime_error("Sub-groups are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
