//==---------------- helpers.hpp - SYCL helpers ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_types.hpp>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/common.hpp>

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cl {
namespace sycl {
class context;
class event;
template <int dimensions, bool with_offset> struct item;
template <int dimensions> class group;
template <int dimensions> class range;
template <int dimensions> struct id;
template <int dimensions> class nd_item;
enum class memory_order;
namespace detail {
class context_impl;
// The function returns list of events that can be passed to OpenCL API as
// dependency list and waits for others.
std::vector<cl_event>
getOrWaitEvents(std::vector<cl::sycl::event> DepEvents,
                std::shared_ptr<cl::sycl::detail::context_impl> Context);

void waitEvents(std::vector<cl::sycl::event> DepEvents);

struct Builder {
  Builder() = delete;
  template <int dimensions>
  static group<dimensions> createGroup(const cl::sycl::range<dimensions> &G,
                                       const cl::sycl::range<dimensions> &L,
                                       const cl::sycl::id<dimensions> &I) {
    return cl::sycl::group<dimensions>(G, L, I);
  }

  template <int dimensions, bool with_offset>
  static item<dimensions, with_offset> createItem(
      typename std::enable_if<(with_offset == true),
                              const cl::sycl::range<dimensions>>::type &R,
      const cl::sycl::id<dimensions> &I, const cl::sycl::id<dimensions> &O) {
    return cl::sycl::item<dimensions, with_offset>(R, I, O);
  }

  template <int dimensions, bool with_offset>
  static item<dimensions, with_offset> createItem(
      typename std::enable_if<(with_offset == false),
                              const cl::sycl::range<dimensions>>::type &R,
      const cl::sycl::id<dimensions> &I) {
    return cl::sycl::item<dimensions, with_offset>(R, I);
  }

  template <int dimensions>
  static nd_item<dimensions>
  createNDItem(const cl::sycl::item<dimensions, true> &GL,
               const cl::sycl::item<dimensions, false> &L,
               const cl::sycl::group<dimensions> &GR) {
    return cl::sycl::nd_item<dimensions>(GL, L, GR);
  }
};

inline __spv::MemorySemanticsMask getSPIRVMemorySemanticsMask(memory_order) {
  return __spv::MemorySemanticsMask::None;
}

inline uint32_t
getSPIRVMemorySemanticsMask(access::fence_space AccessSpace,
                            __spv::MemorySemanticsMask LocalScopeMask =
                                __spv::MemorySemanticsMask::WorkgroupMemory) {
  uint32_t Flags =
      static_cast<uint32_t>(__spv::MemorySemanticsMask::SequentiallyConsistent);
  switch (AccessSpace) {
  case access::fence_space::global_space:
    Flags |=
        static_cast<uint32_t>(__spv::MemorySemanticsMask::CrossWorkgroupMemory);
    break;
  case access::fence_space::local_space:
    Flags |= static_cast<uint32_t>(LocalScopeMask);
    break;
  case access::fence_space::global_and_local:
  default:
    Flags |= static_cast<uint32_t>(
                 __spv::MemorySemanticsMask::CrossWorkgroupMemory) |
             static_cast<uint32_t>(LocalScopeMask);
    break;
  }

  return Flags;
}

} // namespace detail
} // namespace sycl
} // namespace cl
