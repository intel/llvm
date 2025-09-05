
//==------------------------- group_barrier.hpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/spirv.hpp>       // for ControlBarrier
#include <sycl/detail/type_traits.hpp> // for is_group
#include <sycl/exception.hpp>          // for make_error_code, errc, exception
#include <sycl/memory_enums.hpp>       // for memory_scope

#include <type_traits> // for enable_if_t

namespace sycl {
inline namespace _V1 {

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
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Barriers are not supported on host");
#endif
}

} // namespace _V1
} // namespace sycl
