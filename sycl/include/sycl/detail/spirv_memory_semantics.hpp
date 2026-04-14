//==--------- spirv_memory_semantics.hpp - SYCL memory semantics ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_types.hpp> // for MemorySemanticsMask
#include <sycl/access/access.hpp>       // for fence_space
#include <sycl/memory_enums.hpp>        // for memory_order

#include <stdint.h>

namespace sycl {
inline namespace _V1 {
namespace detail {

inline constexpr __spv::MemorySemanticsMask::Flag
getSPIRVMemorySemanticsMask(memory_order) {
  return __spv::MemorySemanticsMask::None;
}

inline constexpr uint32_t
getSPIRVMemorySemanticsMask(const access::fence_space AccessSpace,
                            const __spv::MemorySemanticsMask LocalScopeMask =
                                __spv::MemorySemanticsMask::WorkgroupMemory) {
  // Huge ternary operator below is a workaround for constexpr function
  // requirement that such function can only contain return statement and
  // nothing more
  //
  // It is equivalent to the following code:
  //
  // uint32_t Flags =
  //     static_cast<uint32_t>(__spv::MemorySemanticsMask::SequentiallyConsistent);
  // switch (AccessSpace) {
  // case access::fence_space::global_space:
  //   Flags |=
  //       static_cast<uint32_t>(__spv::MemorySemanticsMask::CrossWorkgroupMemory);
  //   break;
  // case access::fence_space::local_space:
  //   Flags |= static_cast<uint32_t>(LocalScopeMask);
  //   break;
  // case access::fence_space::global_and_local:
  // default:
  //   Flags |= static_cast<uint32_t>(
  //                __spv::MemorySemanticsMask::CrossWorkgroupMemory) |
  //            static_cast<uint32_t>(LocalScopeMask);
  //   break;
  // }
  // return Flags;

  return (AccessSpace == access::fence_space::global_space)
             ? static_cast<uint32_t>(
                   __spv::MemorySemanticsMask::SequentiallyConsistent |
                   __spv::MemorySemanticsMask::CrossWorkgroupMemory)
         : (AccessSpace == access::fence_space::local_space)
             ? static_cast<uint32_t>(
                   __spv::MemorySemanticsMask::SequentiallyConsistent |
                   LocalScopeMask)
             : /* default: (AccessSpace ==
                  access::fence_space::global_and_local) */
             static_cast<uint32_t>(
                 __spv::MemorySemanticsMask::SequentiallyConsistent |
                 __spv::MemorySemanticsMask::CrossWorkgroupMemory |
                 LocalScopeMask);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
