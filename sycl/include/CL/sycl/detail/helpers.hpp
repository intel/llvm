//==---------------- helpers.hpp - SYCL helpers ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_types.hpp>
#include <CL/__spirv/spirv_vars.hpp>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/type_traits.hpp>

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
class context;
class event;
template <int Dims, bool WithOffset> class item;
template <int Dims> class group;
template <int Dims> class range;
template <int Dims> class id;
template <int Dims> class nd_item;
template <int Dims> class h_item;
enum class memory_order;

namespace detail {
inline void memcpy(void *Dst, const void *Src, size_t Size) {
  char *Destination = reinterpret_cast<char *>(Dst);
  const char *Source = reinterpret_cast<const char *>(Src);
  for (size_t I = 0; I < Size; ++I) {
    Destination[I] = Source[I];
  }
}


class context_impl;
// The function returns list of events that can be passed to OpenCL API as
// dependency list and waits for others.
__SYCL_EXPORT std::vector<RT::PiEvent>
getOrWaitEvents(std::vector<cl::sycl::event> DepEvents,
                std::shared_ptr<cl::sycl::detail::context_impl> Context);

__SYCL_EXPORT void waitEvents(std::vector<cl::sycl::event> DepEvents);

template <typename T> T *declptr() { return static_cast<T *>(nullptr); }

// Function to get of store id, item, nd_item, group for the host implementation
// Pass nullptr to get stored object. Pass valid address to store object
template <typename T> T get_or_store(const T *obj) {
  static thread_local auto stored = *obj;
  if (obj != nullptr) {
    stored = *obj;
  }
  return stored;
}

class Builder {
public:
  Builder() = delete;

  template <int Dims>
  static group<Dims>
  createGroup(const range<Dims> &Global, const range<Dims> &Local,
              const range<Dims> &Group, const id<Dims> &Index) {
    return group<Dims>(Global, Local, Group, Index);
  }

  template <int Dims>
  static group<Dims> createGroup(const range<Dims> &Global,
                                 const range<Dims> &Local,
                                 const id<Dims> &Index) {
    return group<Dims>(Global, Local, Global / Local, Index);
  }

  template <int Dims, bool WithOffset>
  static detail::enable_if_t<WithOffset, item<Dims, WithOffset>>
  createItem(const range<Dims> &Extent, const id<Dims> &Index,
             const id<Dims> &Offset) {
    return item<Dims, WithOffset>(Extent, Index, Offset);
  }

  template <int Dims, bool WithOffset>
  static detail::enable_if_t<!WithOffset, item<Dims, WithOffset>>
  createItem(const range<Dims> &Extent, const id<Dims> &Index) {
    return item<Dims, WithOffset>(Extent, Index);
  }

  template <int Dims>
  static nd_item<Dims> createNDItem(const item<Dims, true> &Global,
                                    const item<Dims, false> &Local,
                                    const group<Dims> &Group) {
    return nd_item<Dims>(Global, Local, Group);
  }

  template <int Dims>
  static h_item<Dims> createHItem(const item<Dims, false> &Global,
                                  const item<Dims, false> &Local) {
    return h_item<Dims>(Global, Local);
  }

  template <int Dims>
  static h_item<Dims> createHItem(const item<Dims, false> &Global,
                                  const item<Dims, false> &Local,
                                  const range<Dims> &Flex) {
    return h_item<Dims>(Global, Local, Flex);
  }

  template <int Dims, bool WithOffset>
  static void updateItemIndex(cl::sycl::item<Dims, WithOffset> &Item,
                              const id<Dims> &NextIndex) {
    Item.MImpl.MIndex = NextIndex;
  }

#ifdef __SYCL_DEVICE_ONLY__

  template <int N>
  using is_valid_dimensions = std::integral_constant<bool, (N > 0) && (N < 4)>;

  template <int Dims> static const id<Dims> getElement(id<Dims> *) {
    static_assert(is_valid_dimensions<Dims>::value, "invalid dimensions");
    return __spirv::initGlobalInvocationId<Dims, id<Dims>>();
  }

  template <int Dims> static const group<Dims> getElement(group<Dims> *) {
    static_assert(is_valid_dimensions<Dims>::value, "invalid dimensions");
    range<Dims> GlobalSize{__spirv::initGlobalSize<Dims, range<Dims>>()};
    range<Dims> LocalSize{__spirv::initWorkgroupSize<Dims, range<Dims>>()};
    range<Dims> GroupRange{__spirv::initNumWorkgroups<Dims, range<Dims>>()};
    id<Dims> GroupId{__spirv::initWorkgroupId<Dims, id<Dims>>()};
    return createGroup<Dims>(GlobalSize, LocalSize, GroupRange, GroupId);
  }

  template <int Dims, bool WithOffset>
  static detail::enable_if_t<WithOffset, const item<Dims, WithOffset>>
  getItem() {
    static_assert(is_valid_dimensions<Dims>::value, "invalid dimensions");
    id<Dims> GlobalId{__spirv::initGlobalInvocationId<Dims, id<Dims>>()};
    range<Dims> GlobalSize{__spirv::initGlobalSize<Dims, range<Dims>>()};
    id<Dims> GlobalOffset{__spirv::initGlobalOffset<Dims, id<Dims>>()};
    return createItem<Dims, true>(GlobalSize, GlobalId, GlobalOffset);
  }

  template <int Dims, bool WithOffset>
  static detail::enable_if_t<!WithOffset, const item<Dims, WithOffset>>
  getItem() {
    static_assert(is_valid_dimensions<Dims>::value, "invalid dimensions");
    id<Dims> GlobalId{__spirv::initGlobalInvocationId<Dims, id<Dims>>()};
    range<Dims> GlobalSize{__spirv::initGlobalSize<Dims, range<Dims>>()};
    return createItem<Dims, false>(GlobalSize, GlobalId);
  }

  template <int Dims> static const nd_item<Dims> getElement(nd_item<Dims> *) {
    static_assert(is_valid_dimensions<Dims>::value, "invalid dimensions");
    range<Dims> GlobalSize{__spirv::initGlobalSize<Dims, range<Dims>>()};
    range<Dims> LocalSize{__spirv::initWorkgroupSize<Dims, range<Dims>>()};
    range<Dims> GroupRange{__spirv::initNumWorkgroups<Dims, range<Dims>>()};
    id<Dims> GroupId{__spirv::initWorkgroupId<Dims, id<Dims>>()};
    id<Dims> GlobalId{__spirv::initGlobalInvocationId<Dims, id<Dims>>()};
    id<Dims> LocalId{__spirv::initLocalInvocationId<Dims, id<Dims>>()};
    id<Dims> GlobalOffset{__spirv::initGlobalOffset<Dims, id<Dims>>()};
    group<Dims> Group =
        createGroup<Dims>(GlobalSize, LocalSize, GroupRange, GroupId);
    item<Dims, true> GlobalItem =
        createItem<Dims, true>(GlobalSize, GlobalId, GlobalOffset);
    item<Dims, false> LocalItem = createItem<Dims, false>(LocalSize, LocalId);
    return createNDItem<Dims>(GlobalItem, LocalItem, Group);
  }

  template <int Dims, bool WithOffset>
  static auto getElement(item<Dims, WithOffset> *)
      -> decltype(getItem<Dims, WithOffset>()) {
    return getItem<Dims, WithOffset>();
  }

  template <int Dims>
  static auto getNDItem() -> decltype(getElement(declptr<nd_item<Dims>>())) {
    return getElement(declptr<nd_item<Dims>>());
  }

#endif // __SYCL_DEVICE_ONLY__
};

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

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
