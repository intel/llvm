
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

template <typename Group>
std::enable_if_t<is_group_v<Group>>
group_barrier(Group G, memory_scope FenceScope = Group::fence_scope) {
  // Per SYCL spec, group_barrier must perform both control barrier and memory
  // fence operations. All work-items execute a release fence prior to
  // barrier and acquire fence afterwards.
#ifdef __SYCL_DEVICE_ONLY__
  detail::spirv::ControlBarrier(G, FenceScope, memory_order::seq_cst);
#else
  (void)G;
  (void)FenceScope;
  throw sycl::runtime_error("Barriers are not supported on host device",
                            PI_ERROR_INVALID_DEVICE);
#endif
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
