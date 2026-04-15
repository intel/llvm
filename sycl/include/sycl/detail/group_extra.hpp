//==---------- detail/group_extra.hpp --- SYCL work group extensions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/group_core.hpp>

#include <sycl/detail/builder.hpp>
#include <sycl/detail/fwd/half.hpp>
#include <sycl/detail/nd_loop.hpp>
#include <sycl/device_event.hpp>
#include <sycl/h_item.hpp>
#include <sycl/item.hpp>
#include <sycl/pointers.hpp>

#include <cstddef>
#include <memory>
#include <stdint.h>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
template <typename DataT, int NumElements> class vec;

namespace ext::oneapi {
class bfloat16;
}

namespace detail {
template <typename ElementType, access::address_space addressSpace>
struct DecoratedType;

inline void workGroupBarrier() {
#ifdef __SYCL_DEVICE_ONLY__
  constexpr uint32_t flags =
      static_cast<uint32_t>(
          __spv::MemorySemanticsMask::SequentiallyConsistent) |
      static_cast<uint32_t>(__spv::MemorySemanticsMask::WorkgroupMemory);
  __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
                         flags);
#endif // __SYCL_DEVICE_ONLY__
}

template <int Size>
using group_async_copy_fixed_width_unsigned = std::conditional_t<
    Size == 1, uint8_t,
    std::conditional_t<Size == 2, uint16_t,
                       std::conditional_t<Size == 4, uint32_t, uint64_t>>>;

template <int Size>
using group_async_copy_fixed_width_signed = std::conditional_t<
    Size == 1, int8_t,
    std::conditional_t<Size == 2, int16_t,
                       std::conditional_t<Size == 4, int32_t, int64_t>>>;

template <typename T, typename = void> struct group_async_copy_opencl_type {
  using type = T;
};

template <typename T>
struct group_async_copy_opencl_type<T,
                                    std::enable_if_t<std::is_integral_v<T>>> {
  using type =
      std::conditional_t<std::is_signed_v<T>,
                         group_async_copy_fixed_width_signed<sizeof(T)>,
                         group_async_copy_fixed_width_unsigned<sizeof(T)>>;
};

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <> struct group_async_copy_opencl_type<std::byte> {
  using type = uint8_t;
};
#endif

template <> struct group_async_copy_opencl_type<half> {
  using type = half_impl::BIsRepresentationT;
};

template <> struct group_async_copy_opencl_type<ext::oneapi::bfloat16> {
  using type = uint16_t;
};

template <typename ElementType, int NumElements>
struct group_async_copy_opencl_vec_type {
#ifdef __SYCL_DEVICE_ONLY__
  using type =
      std::conditional_t<NumElements == 1, ElementType,
                         ElementType
                         __attribute__((ext_vector_type(NumElements)))>;
#else
  using type = vec<ElementType, NumElements>;
#endif
};

template <typename T, int NumElements>
struct group_async_copy_opencl_type<vec<T, NumElements>> {
  using element_type = typename group_async_copy_opencl_type<T>::type;
  using type = typename group_async_copy_opencl_vec_type<element_type,
                                                         NumElements>::type;
};

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
auto convertToOpenCLGroupAsyncCopyPtr(
    multi_ptr<ElementType, Space, DecorateAddress> Ptr) {
  using converted_elem_type_no_cv = typename group_async_copy_opencl_type<
      std::remove_const_t<ElementType>>::type;
  using converted_elem_type =
      std::conditional_t<std::is_const_v<ElementType>,
                         const converted_elem_type_no_cv,
                         converted_elem_type_no_cv>;
  using result_type =
      typename DecoratedType<converted_elem_type, Space>::type *;
  return reinterpret_cast<result_type>(Ptr.get_decorated());
}

} // namespace detail

template <typename T, int Dimensions = 1>
class __SYCL_TYPE(private_memory) private_memory {
public:
  private_memory(const group<Dimensions> &G) {
#ifndef __SYCL_DEVICE_ONLY__
    Val.reset(new T[G.get_local_range().size()]);
#endif
    (void)G;
  }

  T &operator()(const h_item<Dimensions> &Id) {
#ifndef __SYCL_DEVICE_ONLY__
    size_t Ind = Id.get_physical_local().get_linear_id();
    return Val.get()[Ind];
#else
    (void)Id;
    return Val;
#endif
  }

private:
#ifdef __SYCL_DEVICE_ONLY__
  T Val;
#else
  std::unique_ptr<T[]> Val;
#endif
};

template <int Dimensions>
template <typename WorkItemFunctionT>
#ifdef __NativeCPU__
__attribute__((__libclc_call__))
#endif
void group<Dimensions>::parallel_for_work_item(WorkItemFunctionT Func) const {
  detail::workGroupBarrier();
#ifdef __SYCL_DEVICE_ONLY__
  range<Dimensions> GlobalSize{
      __spirv::initBuiltInGlobalSize<Dimensions, range<Dimensions>>()};
  range<Dimensions> LocalSize{
      __spirv::initBuiltInWorkgroupSize<Dimensions, range<Dimensions>>()};
  id<Dimensions> GlobalId{
      __spirv::initBuiltInGlobalInvocationId<Dimensions, id<Dimensions>>()};
  id<Dimensions> LocalId{
      __spirv::initBuiltInLocalInvocationId<Dimensions, id<Dimensions>>()};

  item<Dimensions, false> GlobalItem =
      detail::Builder::createItem<Dimensions, false>(GlobalSize, GlobalId);
  item<Dimensions, false> LocalItem =
      detail::Builder::createItem<Dimensions, false>(LocalSize, LocalId);
  h_item<Dimensions> HItem =
      detail::Builder::createHItem<Dimensions>(GlobalItem, LocalItem);

  Func(HItem);
#else
  id<Dimensions> GroupStartID = index * id<Dimensions>{localRange};

  detail::NDLoop<Dimensions>::iterate(
      localRange, [&](const id<Dimensions> &LocalID) {
        item<Dimensions, false> GlobalItem =
            detail::Builder::createItem<Dimensions, false>(
                globalRange, GroupStartID + LocalID);
        item<Dimensions, false> LocalItem =
            detail::Builder::createItem<Dimensions, false>(localRange, LocalID);
        h_item<Dimensions> HItem =
            detail::Builder::createHItem<Dimensions>(GlobalItem, LocalItem);
        Func(HItem);
      });
#endif
  detail::workGroupBarrier();
}

template <int Dimensions>
template <typename WorkItemFunctionT>
#ifdef __NativeCPU__
__attribute__((__libclc_call__))
#endif
void group<Dimensions>::parallel_for_work_item(range<Dimensions> flexibleRange,
                                               WorkItemFunctionT Func) const {
  detail::workGroupBarrier();
#ifdef __SYCL_DEVICE_ONLY__
  range<Dimensions> GlobalSize{
      __spirv::initBuiltInGlobalSize<Dimensions, range<Dimensions>>()};
  range<Dimensions> LocalSize{
      __spirv::initBuiltInWorkgroupSize<Dimensions, range<Dimensions>>()};
  id<Dimensions> GlobalId{
      __spirv::initBuiltInGlobalInvocationId<Dimensions, id<Dimensions>>()};
  id<Dimensions> LocalId{
      __spirv::initBuiltInLocalInvocationId<Dimensions, id<Dimensions>>()};

  item<Dimensions, false> GlobalItem =
      detail::Builder::createItem<Dimensions, false>(GlobalSize, GlobalId);
  item<Dimensions, false> LocalItem =
      detail::Builder::createItem<Dimensions, false>(LocalSize, LocalId);
  h_item<Dimensions> HItem = detail::Builder::createHItem<Dimensions>(
      GlobalItem, LocalItem, flexibleRange);

  detail::NDLoop<Dimensions>::iterate(
      LocalId, LocalSize, flexibleRange,
      [&](const id<Dimensions> &LogicalLocalID) {
        HItem.setLogicalLocalID(LogicalLocalID);
        Func(HItem);
      });
#else
  id<Dimensions> GroupStartID = index * localRange;

  detail::NDLoop<Dimensions>::iterate(
      localRange, [&](const id<Dimensions> &LocalID) {
        item<Dimensions, false> GlobalItem =
            detail::Builder::createItem<Dimensions, false>(
                globalRange, GroupStartID + LocalID);
        item<Dimensions, false> LocalItem =
            detail::Builder::createItem<Dimensions, false>(localRange, LocalID);
        h_item<Dimensions> HItem = detail::Builder::createHItem<Dimensions>(
            GlobalItem, LocalItem, flexibleRange);

        detail::NDLoop<Dimensions>::iterate(
            LocalID, localRange, flexibleRange,
            [&](const id<Dimensions> &LogicalLocalID) {
              HItem.setLogicalLocalID(LogicalLocalID);
              Func(HItem);
            });
      });
#endif
  detail::workGroupBarrier();
}

template <int Dimensions>
template <typename dataT>
device_event group<Dimensions>::async_work_group_copy(local_ptr<dataT> dest,
                                                      global_ptr<dataT> src,
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
template <typename dataT>
device_event group<Dimensions>::async_work_group_copy(global_ptr<dataT> dest,
                                                      local_ptr<dataT> src,
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
template <typename DestDataT, typename SrcDataT>
std::enable_if_t<std::is_same_v<std::remove_const_t<SrcDataT>, DestDataT>,
                 device_event>
group<Dimensions>::async_work_group_copy(decorated_local_ptr<DestDataT> dest,
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
group<Dimensions>::async_work_group_copy(decorated_global_ptr<DestDataT> dest,
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
device_event group<Dimensions>::async_work_group_copy(
    local_ptr<dataT> dest, global_ptr<dataT> src, size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

template <int Dimensions>
template <typename dataT>
device_event group<Dimensions>::async_work_group_copy(
    global_ptr<dataT> dest, local_ptr<dataT> src, size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

template <int Dimensions>
template <typename DestDataT, typename SrcDataT>
typename std::enable_if_t<
    std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
group<Dimensions>::async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                                         decorated_global_ptr<SrcDataT> src,
                                         size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

template <int Dimensions>
template <typename DestDataT, typename SrcDataT>
typename std::enable_if_t<
    std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
group<Dimensions>::async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                                         decorated_local_ptr<SrcDataT> src,
                                         size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

template <int Dimensions>
template <typename... eventTN>
void group<Dimensions>::wait_for(eventTN... Events) const {
  waitForHelper(Events...);
}

template <int Dimensions> void group<Dimensions>::waitForHelper() const {}

template <int Dimensions>
void group<Dimensions>::waitForHelper(device_event Event) const {
  Event.wait();
}

template <int Dimensions>
template <typename T, typename... Ts>
void group<Dimensions>::waitForHelper(T E, Ts... Es) const {
  waitForHelper(E);
  waitForHelper(Es...);
}

} // namespace _V1
} // namespace sycl