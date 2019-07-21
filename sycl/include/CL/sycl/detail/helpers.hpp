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
#include <CL/sycl/detail/pi.hpp>

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cl {
namespace sycl {
class context;
class event;
template <int dimensions, bool with_offset> class item;
template <int dimensions> class group;
template <int dimensions> class range;
template <int dimensions> class id;
template <int dimensions> class nd_item;
enum class memory_order;
template <int dimensions> class h_item;

namespace detail {
class context_impl;
// The function returns list of events that can be passed to OpenCL API as
// dependency list and waits for others.
std::vector<RT::PiEvent>
getOrWaitEvents(std::vector<cl::sycl::event> DepEvents,
                std::shared_ptr<cl::sycl::detail::context_impl> Context);

void waitEvents(std::vector<cl::sycl::event> DepEvents);

class Builder {
public:
  Builder() = delete;
  template <int dimensions>
  static group<dimensions>
  createGroup(const cl::sycl::range<dimensions> &G,
              const cl::sycl::range<dimensions> &L,
              const cl::sycl::range<dimensions> &GroupRange,
              const cl::sycl::id<dimensions> &I) {
    return cl::sycl::group<dimensions>(G, L, GroupRange, I);
  }

  template <int dimensions>
  static group<dimensions> createGroup(const cl::sycl::range<dimensions> &G,
                                       const cl::sycl::range<dimensions> &L,
                                       const cl::sycl::id<dimensions> &I) {
    return cl::sycl::group<dimensions>(G, L, G / L, I);
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

  template <int dimensions>
  static h_item<dimensions>
  createHItem(const cl::sycl::item<dimensions, false> &GlobalItem,
              const cl::sycl::item<dimensions, false> &LocalItem) {
    return cl::sycl::h_item<dimensions>(GlobalItem, LocalItem);
  }

  template <int dimensions>
  static h_item<dimensions>
  createHItem(const cl::sycl::item<dimensions, false> &GlobalItem,
              const cl::sycl::item<dimensions, false> &LocalItem,
              const cl::sycl::range<dimensions> &FlexRange) {
    return cl::sycl::h_item<dimensions>(GlobalItem, LocalItem, FlexRange);
  }
};

inline constexpr
__spv::MemorySemanticsMask getSPIRVMemorySemanticsMask(memory_order) {
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
} // namespace sycl
} // namespace cl
