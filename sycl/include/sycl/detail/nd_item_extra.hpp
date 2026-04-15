//==-------- detail/nd_item_extra.hpp --- SYCL iteration nd_item extras ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/builder.hpp>
#include <sycl/detail/group_extra.hpp>
#include <sycl/detail/nd_item_core.hpp>
#include <sycl/device_event.hpp>
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/item.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/pointers.hpp>

#include <type_traits>

namespace sycl {
inline namespace _V1 {
template <int Dimensions>
id<Dimensions> nd_item<Dimensions>::get_offset() const {
  return get_offset_impl();
}

template <int Dimensions> auto nd_item<Dimensions>::get_nd_range() const {
  return nd_range<Dimensions>(get_global_range(), get_local_range(),
                              get_offset_impl());
}

template <int Dimensions>
template <access::mode accessMode>
void nd_item<Dimensions>::mem_fence(
    [[maybe_unused]]
    typename std::enable_if_t<accessMode == access::mode::read ||
                                  accessMode == access::mode::write ||
                                  accessMode == access::mode::read_write,
                              access::fence_space>
        accessSpace) const {
#if __SYCL_DEVICE_ONLY__
  uint32_t flags = detail::getSPIRVMemorySemanticsMask(accessSpace);
  __spirv_MemoryBarrier(__spv::Scope::Workgroup, flags);
#endif
}

template <int Dimensions>
template <typename dataT>
device_event nd_item<Dimensions>::async_work_group_copy(
    local_ptr<dataT> dest, global_ptr<dataT> src, size_t numElements,
    size_t srcStride) const {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_event_t E = __spirv_GroupAsyncCopy(
      __spv::Scope::Workgroup, detail::convertToOpenCLGroupAsyncCopyPtr(dest),
      detail::convertToOpenCLGroupAsyncCopyPtr(src), numElements, srcStride, 0);
  return device_event(E);
#else
  (void)dest;
  (void)src;
  (void)numElements;
  (void)srcStride;
  return nullptr;
#endif
}

template <int Dimensions>
template <typename dataT>
device_event nd_item<Dimensions>::async_work_group_copy(
    global_ptr<dataT> dest, local_ptr<dataT> src, size_t numElements,
    size_t destStride) const {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_event_t E = __spirv_GroupAsyncCopy(
      __spv::Scope::Workgroup, detail::convertToOpenCLGroupAsyncCopyPtr(dest),
      detail::convertToOpenCLGroupAsyncCopyPtr(src), numElements, destStride,
      0);
  return device_event(E);
#else
  (void)dest;
  (void)src;
  (void)numElements;
  (void)destStride;
  return nullptr;
#endif
}

template <int Dimensions>
template <typename DestDataT, typename SrcDataT>
std::enable_if_t<std::is_same_v<std::remove_const_t<SrcDataT>, DestDataT>,
                 device_event>
nd_item<Dimensions>::async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                                           decorated_global_ptr<SrcDataT> src,
                                           size_t numElements,
                                           size_t srcStride) const {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_event_t E = __spirv_GroupAsyncCopy(
      __spv::Scope::Workgroup, detail::convertToOpenCLGroupAsyncCopyPtr(dest),
      detail::convertToOpenCLGroupAsyncCopyPtr(src), numElements, srcStride, 0);
  return device_event(E);
#else
  (void)dest;
  (void)src;
  (void)numElements;
  (void)srcStride;
  return nullptr;
#endif
}

template <int Dimensions>
template <typename DestDataT, typename SrcDataT>
std::enable_if_t<std::is_same_v<std::remove_const_t<SrcDataT>, DestDataT>,
                 device_event>
nd_item<Dimensions>::async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                                           decorated_local_ptr<SrcDataT> src,
                                           size_t numElements,
                                           size_t destStride) const {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_event_t E = __spirv_GroupAsyncCopy(
      __spv::Scope::Workgroup, detail::convertToOpenCLGroupAsyncCopyPtr(dest),
      detail::convertToOpenCLGroupAsyncCopyPtr(src), numElements, destStride,
      0);
  return device_event(E);
#else
  (void)dest;
  (void)src;
  (void)numElements;
  (void)destStride;
  return nullptr;
#endif
}

template <int Dimensions>
template <typename dataT>
device_event nd_item<Dimensions>::async_work_group_copy(
    local_ptr<dataT> dest, global_ptr<dataT> src, size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

template <int Dimensions>
template <typename dataT>
device_event nd_item<Dimensions>::async_work_group_copy(
    global_ptr<dataT> dest, local_ptr<dataT> src, size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

template <int Dimensions>
template <typename DestDataT, typename SrcDataT>
typename std::enable_if_t<
    std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
nd_item<Dimensions>::async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                                           decorated_global_ptr<SrcDataT> src,
                                           size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

template <int Dimensions>
template <typename DestDataT, typename SrcDataT>
typename std::enable_if_t<
    std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
nd_item<Dimensions>::async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                                           decorated_local_ptr<SrcDataT> src,
                                           size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

template <int Dimensions>
template <typename... eventTN>
void nd_item<Dimensions>::wait_for(eventTN... events) const {
  waitForHelper(events...);
}

template <int Dimensions> void nd_item<Dimensions>::waitForHelper() const {}

template <int Dimensions>
void nd_item<Dimensions>::waitForHelper(device_event Event) const {
  Event.wait();
}

template <int Dimensions>
template <typename T, typename... Ts>
void nd_item<Dimensions>::waitForHelper(T E, Ts... Es) const {
  waitForHelper(E);
  waitForHelper(Es...);
}

template <int Dimensions>
ext::oneapi::experimental::root_group<Dimensions>
nd_item<Dimensions>::ext_oneapi_get_root_group() const {
  return ext::oneapi::experimental::root_group<Dimensions>{*this};
}

} // namespace _V1
} // namespace sycl