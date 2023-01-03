
//==------------------------- group_barrier.hpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_types.hpp>
#include <CL/__spirv/spirv_vars.hpp>
#include <sycl/detail/spirv.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/group.hpp>
#include <sycl/sub_group.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace detail {
template <typename G> struct group_barrier_scope {};
template <> struct group_barrier_scope<sycl::sub_group> {
  constexpr static auto Scope = __spv::Scope::Subgroup;
};
template <int D> struct group_barrier_scope<sycl::group<D>> {
  constexpr static auto Scope = __spv::Scope::Workgroup;
};
} // namespace detail

template <typename Group>
typename std::enable_if<is_group_v<Group>>::type
group_barrier(Group, memory_scope FenceScope = Group::fence_scope) {
  (void)FenceScope;
#ifdef __SYCL_DEVICE_ONLY__
  // Per SYCL spec, group_barrier must perform both control barrier and memory
  // fence operations. All work-items execute a release fence prior to
  // barrier and acquire fence afterwards. The rest of semantics flags specify
  // which type of memory this behavior is applied to.
  constexpr auto SPIRVScope = sycl::detail::spirv::getScope(FenceScope);
  __spirv_ControlBarrier(detail::group_barrier_scope<Group>::Scope, SPIRVScope,
                         __spv::MemorySemanticsMask::SequentiallyConsistent |
                             __spv::MemorySemanticsMask::SubgroupMemory |
                             __spv::MemorySemanticsMask::WorkgroupMemory |
                             __spv::MemorySemanticsMask::CrossWorkgroupMemory);
#else
  throw sycl::runtime_error("Barriers are not supported on host device",
                            PI_ERROR_INVALID_DEVICE);
#endif
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
