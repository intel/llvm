//==--------- nd_item.hpp --- SYCL iteration nd_item -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>           // for __spirv_ControlBarrier
#include <CL/__spirv/spirv_types.hpp>         // for Scope
#include <CL/__spirv/spirv_vars.hpp>          // for initLocalInvocationId
#include <sycl/access/access.hpp>             // for mode, fence_space
#include <sycl/detail/defines.hpp>            // for __SYCL_ASSUME_INT
#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED, __SY...
#include <sycl/detail/generic_type_traits.hpp> // for ConvertToOpenCLType_t
#include <sycl/detail/helpers.hpp>            // for getSPIRVMemorySemanticsMask
#include <sycl/detail/type_traits.hpp>        // for is_bool, change_base_...
#include <sycl/device_event.hpp>              // for device_event
#include <sycl/exception.hpp> // for make_error_code, errc, exce...
#include <sycl/group.hpp>     // for group
#include <sycl/id.hpp>        // for id
#include <sycl/item.hpp>      // for item
#include <sycl/nd_range.hpp>  // for nd_range
#include <sycl/pointers.hpp>  // for decorated_global_ptr, decor...
#include <sycl/range.hpp>     // for range
#include <sycl/sub_group.hpp> // for sub_group

#include <cstddef>     // for size_t
#include <stdint.h>    // for uint32_t
#include <type_traits> // for enable_if_t, remove_const_t

namespace sycl {
inline namespace _V1 {
namespace detail {
class Builder;
}

namespace ext::oneapi::experimental {
template <int Dimensions> class root_group;
}

#if __INTEL_PREVIEW_BREAKING_CHANGES
/// Identifies an instance of the function object executing at each point in an
/// nd_range.
///
/// \ingroup sycl_api
template <int Dimensions = 1> class nd_item {
public:
  static constexpr int dimensions = Dimensions;

  id<Dimensions> get_global_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initGlobalInvocationId<Dimensions, id<Dimensions>>();
#else
    assert(false && "nd_item methods can't be used on the host!");
    return {};
#endif
  }

  size_t __SYCL_ALWAYS_INLINE get_global_id(int Dimension) const {
    size_t Id = get_global_id()[Dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  size_t __SYCL_ALWAYS_INLINE get_global_linear_id() const {
    size_t LinId = 0;
    id<Dimensions> Index = get_global_id();
    range<Dimensions> Extent = get_global_range();
    id<Dimensions> Offset = get_offset();
    if (1 == Dimensions) {
      LinId = Index[0] - Offset[0];
    } else if (2 == Dimensions) {
      LinId = (Index[0] - Offset[0]) * Extent[1] + Index[1] - Offset[1];
    } else {
      LinId = (Index[0] - Offset[0]) * Extent[1] * Extent[2] +
              (Index[1] - Offset[1]) * Extent[2] + Index[2] - Offset[2];
    }
    __SYCL_ASSUME_INT(LinId);
    return LinId;
  }

  id<Dimensions> get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initLocalInvocationId<Dimensions, id<Dimensions>>();
#else
    assert(false && "nd_item methods can't be used on the host!");
    return {};
#endif
  }

  size_t __SYCL_ALWAYS_INLINE get_local_id(int Dimension) const {
    size_t Id = get_local_id()[Dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  size_t get_local_linear_id() const {
    size_t LinId = 0;
    id<Dimensions> Index = get_local_id();
    range<Dimensions> Extent = get_local_range();
    if (1 == Dimensions) {
      LinId = Index[0];
    } else if (2 == Dimensions) {
      LinId = Index[0] * Extent[1] + Index[1];
    } else {
      LinId =
          Index[0] * Extent[1] * Extent[2] + Index[1] * Extent[2] + Index[2];
    }
    __SYCL_ASSUME_INT(LinId);
    return LinId;
  }

  group<Dimensions> get_group() const {
    // TODO: ideally Group object should be stateless and have a contructor with
    // no arguments.
    return detail::Builder::createGroup(get_global_range(), get_local_range(),
                                        get_group_range(), get_group_id());
  }

  sub_group get_sub_group() const { return sub_group(); }

  size_t __SYCL_ALWAYS_INLINE get_group(int Dimension) const {
    size_t Id = get_group_id()[Dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  size_t __SYCL_ALWAYS_INLINE get_group_linear_id() const {
    size_t LinId = 0;
    id<Dimensions> Index = get_group_id();
    range<Dimensions> Extent = get_group_range();
    if (1 == Dimensions) {
      LinId = Index[0];
    } else if (2 == Dimensions) {
      LinId = Index[0] * Extent[1] + Index[1];
    } else {
      LinId =
          Index[0] * Extent[1] * Extent[2] + Index[1] * Extent[2] + Index[2];
    }
    __SYCL_ASSUME_INT(LinId);
    return LinId;
  }

  range<Dimensions> get_group_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initNumWorkgroups<Dimensions, range<Dimensions>>();
#else
    assert(false && "nd_item methods can't be used on the host!");
    return {};
#endif
  }

  size_t __SYCL_ALWAYS_INLINE get_group_range(int Dimension) const {
    size_t Range = get_group_range()[Dimension];
    __SYCL_ASSUME_INT(Range);
    return Range;
  }

  range<Dimensions> get_global_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initGlobalSize<Dimensions, range<Dimensions>>();
#else
    assert(false && "nd_item methods can't be used on the host!");
    return {};
#endif
  }

  size_t get_global_range(int Dimension) const {
    size_t Val = get_global_range()[Dimension];
    __SYCL_ASSUME_INT(Val);
    return Val;
  }

  range<Dimensions> get_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initWorkgroupSize<Dimensions, range<Dimensions>>();
#else
    assert(false && "nd_item methods can't be used on the host!");
    return {};
#endif
  }

  size_t get_local_range(int Dimension) const {
    size_t Id = get_local_range()[Dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL 2020")
  id<Dimensions> get_offset() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initGlobalOffset<Dimensions, id<Dimensions>>();
#else
    assert(false && "nd_item methods can't be used on the host!");
    return {};
#endif
  }

  nd_range<Dimensions> get_nd_range() const {
    return nd_range<Dimensions>(get_global_range(), get_local_range(),
                                get_offset());
  }

  void barrier(access::fence_space accessSpace =
                   access::fence_space::global_and_local) const {
    uint32_t flags = _V1::detail::getSPIRVMemorySemanticsMask(accessSpace);
    __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
                           flags);
  }

  /// Executes a work-group mem-fence with memory ordering on the local address
  /// space, global address space or both based on the value of \p accessSpace.
  template <access::mode accessMode = access::mode::read_write>
  __SYCL2020_DEPRECATED("use sycl::atomic_fence() free function instead")
  void mem_fence(
      typename std::enable_if_t<accessMode == access::mode::read ||
                                    accessMode == access::mode::write ||
                                    accessMode == access::mode::read_write,
                                access::fence_space>
          accessSpace = access::fence_space::global_and_local) const {
    uint32_t flags = detail::getSPIRVMemorySemanticsMask(accessSpace);
    // TODO: currently, there is no good way in SPIR-V to set the memory
    // barrier only for load operations or only for store operations.
    // The full read-and-write barrier is used and the template parameter
    // 'accessMode' is ignored for now. Either SPIR-V or SYCL spec may be
    // changed to address this discrepancy between SPIR-V and SYCL,
    // or if we decide that 'accessMode' is the important feature then
    // we can fix this later, for example, by using OpenCL 1.2 functions
    // read_mem_fence() and write_mem_fence().
    __spirv_MemoryBarrier(__spv::Scope::Workgroup, flags);
  }

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest
  /// with a source stride specified by \p srcStride, and returns a SYCL
  /// device_event which can be used to wait on the completion of the copy.
  /// Permitted types for dataT are all scalar and vector types, except boolean.
  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  std::enable_if_t<!detail::is_bool<dataT>::value,
                   device_event> async_work_group_copy(local_ptr<dataT> dest,
                                                       global_ptr<dataT> src,
                                                       size_t numElements,
                                                       size_t srcStride) const {
    using DestT = detail::ConvertToOpenCLType_t<decltype(dest)>;
    using SrcT = detail::ConvertToOpenCLType_t<decltype(src)>;

    __ocl_event_t E = __SYCL_OpGroupAsyncCopyGlobalToLocal(
        __spv::Scope::Workgroup, DestT(dest.get()), SrcT(src.get()),
        numElements, srcStride, 0);
    return device_event(E);
  }

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest with
  /// the destination stride specified by \p destStride, and returns a SYCL
  /// device_event which can be used to wait on the completion of the copy.
  /// Permitted types for dataT are all scalar and vector types, except boolean.
  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  std::enable_if_t<!detail::is_bool<dataT>::value,
                   device_event> async_work_group_copy(global_ptr<dataT> dest,
                                                       local_ptr<dataT> src,
                                                       size_t numElements,
                                                       size_t destStride)
      const {
    using DestT = detail::ConvertToOpenCLType_t<decltype(dest)>;
    using SrcT = detail::ConvertToOpenCLType_t<decltype(src)>;

    __ocl_event_t E = __SYCL_OpGroupAsyncCopyLocalToGlobal(
        __spv::Scope::Workgroup, DestT(dest.get()), SrcT(src.get()),
        numElements, destStride, 0);
    return device_event(E);
  }

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest
  /// with a source stride specified by \p srcStride, and returns a SYCL
  /// device_event which can be used to wait on the completion of the copy.
  /// Permitted types for DestDataT are all scalar and vector types, except
  /// boolean. SrcDataT must be either the same as DestDataT or const DestDataT.
  template <typename DestDataT, typename SrcDataT>
  std::enable_if_t<!detail::is_bool<DestDataT>::value &&
                       std::is_same_v<std::remove_const_t<SrcDataT>, DestDataT>,
                   device_event>
  async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                        decorated_global_ptr<SrcDataT> src, size_t numElements,
                        size_t srcStride) const {
    using DestT = detail::ConvertToOpenCLType_t<decltype(dest)>;
    using SrcT = detail::ConvertToOpenCLType_t<decltype(src)>;

    __ocl_event_t E = __SYCL_OpGroupAsyncCopyGlobalToLocal(
        __spv::Scope::Workgroup, DestT(dest.get()), SrcT(src.get()),
        numElements, srcStride, 0);
    return device_event(E);
  }

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest with
  /// the destination stride specified by \p destStride, and returns a SYCL
  /// device_event which can be used to wait on the completion of the copy.
  /// Permitted types for DestDataT are all scalar and vector types, except
  /// boolean. SrcDataT must be either the same as DestDataT or const DestDataT.
  template <typename DestDataT, typename SrcDataT>
  std::enable_if_t<!detail::is_bool<DestDataT>::value &&
                       std::is_same_v<std::remove_const_t<SrcDataT>, DestDataT>,
                   device_event>
  async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                        decorated_local_ptr<SrcDataT> src, size_t numElements,
                        size_t destStride) const {
    using DestT = detail::ConvertToOpenCLType_t<decltype(dest)>;
    using SrcT = detail::ConvertToOpenCLType_t<decltype(src)>;

    __ocl_event_t E = __SYCL_OpGroupAsyncCopyLocalToGlobal(
        __spv::Scope::Workgroup, DestT(dest.get()), SrcT(src.get()),
        numElements, destStride, 0);
    return device_event(E);
  }

  /// Specialization for scalar bool type.
  /// Asynchronously copies a number of elements specified by \p NumElements
  /// from the source pointed by \p Src to destination pointed by \p Dest
  /// with a stride specified by \p Stride, and returns a SYCL device_event
  /// which can be used to wait on the completion of the copy.
  template <typename T, access::address_space DestS, access::address_space SrcS>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  std::enable_if_t<
      detail::is_scalar_bool<T>::value,
      device_event> async_work_group_copy(multi_ptr<T, DestS,
                                                    access::decorated::legacy>
                                              Dest,
                                          multi_ptr<T, SrcS,
                                                    access::decorated::legacy>
                                              Src,
                                          size_t NumElements,
                                          size_t Stride) const {
    static_assert(sizeof(bool) == sizeof(uint8_t),
                  "Async copy to/from bool memory is not supported.");
    auto DestP = multi_ptr<uint8_t, DestS, access::decorated::legacy>(
        reinterpret_cast<uint8_t *>(Dest.get()));
    auto SrcP = multi_ptr<uint8_t, SrcS, access::decorated::legacy>(
        reinterpret_cast<uint8_t *>(Src.get()));
    return async_work_group_copy(DestP, SrcP, NumElements, Stride);
  }

  /// Specialization for vector bool type.
  /// Asynchronously copies a number of elements specified by \p NumElements
  /// from the source pointed by \p Src to destination pointed by \p Dest
  /// with a stride specified by \p Stride, and returns a SYCL device_event
  /// which can be used to wait on the completion of the copy.
  template <typename T, access::address_space DestS, access::address_space SrcS>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  std::enable_if_t<
      detail::is_vector_bool<T>::value,
      device_event> async_work_group_copy(multi_ptr<T, DestS,
                                                    access::decorated::legacy>
                                              Dest,
                                          multi_ptr<T, SrcS,
                                                    access::decorated::legacy>
                                              Src,
                                          size_t NumElements,
                                          size_t Stride) const {
    static_assert(sizeof(bool) == sizeof(uint8_t),
                  "Async copy to/from bool memory is not supported.");
    using VecT = detail::change_base_type_t<T, uint8_t>;
    auto DestP = address_space_cast<DestS, access::decorated::legacy>(
        reinterpret_cast<VecT *>(Dest.get()));
    auto SrcP = address_space_cast<SrcS, access::decorated::legacy>(
        reinterpret_cast<VecT *>(Src.get()));
    return async_work_group_copy(DestP, SrcP, NumElements, Stride);
  }

  /// Specialization for scalar bool type.
  /// Asynchronously copies a number of elements specified by \p NumElements
  /// from the source pointed by \p Src to destination pointed by \p Dest
  /// with a stride specified by \p Stride, and returns a SYCL device_event
  /// which can be used to wait on the completion of the copy.
  template <typename DestT, access::address_space DestS, typename SrcT,
            access::address_space SrcS>
  std::enable_if_t<detail::is_scalar_bool<DestT>::value &&
                       std::is_same_v<std::remove_const_t<SrcT>, DestT>,
                   device_event>
  async_work_group_copy(multi_ptr<DestT, DestS, access::decorated::yes> Dest,
                        multi_ptr<SrcT, SrcS, access::decorated::yes> Src,
                        size_t NumElements, size_t Stride) const {
    static_assert(sizeof(bool) == sizeof(uint8_t),
                  "Async copy to/from bool memory is not supported.");
    using QualSrcT =
        std::conditional_t<std::is_const_v<SrcT>, const uint8_t, uint8_t>;
    auto DestP = multi_ptr<uint8_t, DestS, access::decorated::yes>(
        detail::cast_AS<typename multi_ptr<uint8_t, DestS,
                                           access::decorated::yes>::pointer>(
            Dest.get_decorated()));
    auto SrcP = multi_ptr<QualSrcT, SrcS, access::decorated::yes>(
        detail::cast_AS<typename multi_ptr<QualSrcT, SrcS,
                                           access::decorated::yes>::pointer>(
            Src.get_decorated()));
    return async_work_group_copy(DestP, SrcP, NumElements, Stride);
  }

  /// Specialization for vector bool type.
  /// Asynchronously copies a number of elements specified by \p NumElements
  /// from the source pointed by \p Src to destination pointed by \p Dest
  /// with a stride specified by \p Stride, and returns a SYCL device_event
  /// which can be used to wait on the completion of the copy.
  template <typename DestT, access::address_space DestS, typename SrcT,
            access::address_space SrcS>
  std::enable_if_t<detail::is_vector_bool<DestT>::value &&
                       std::is_same_v<std::remove_const_t<SrcT>, DestT>,
                   device_event>
  async_work_group_copy(multi_ptr<DestT, DestS, access::decorated::yes> Dest,
                        multi_ptr<SrcT, SrcS, access::decorated::yes> Src,
                        size_t NumElements, size_t Stride) const {
    static_assert(sizeof(bool) == sizeof(uint8_t),
                  "Async copy to/from bool memory is not supported.");
    using VecT = detail::change_base_type_t<DestT, uint8_t>;
    using QualSrcVecT =
        std::conditional_t<std::is_const_v<SrcT>, std::add_const_t<VecT>, VecT>;
    auto DestP = multi_ptr<VecT, DestS, access::decorated::yes>(
        detail::cast_AS<
            typename multi_ptr<VecT, DestS, access::decorated::yes>::pointer>(
            Dest.get_decorated()));
    auto SrcP = multi_ptr<QualSrcVecT, SrcS, access::decorated::yes>(
        detail::cast_AS<typename multi_ptr<QualSrcVecT, SrcS,
                                           access::decorated::yes>::pointer>(
            Src.get_decorated()));
    return async_work_group_copy(DestP, SrcP, NumElements, Stride);
  }

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest and
  /// returns a SYCL device_event which can be used to wait on the completion
  /// of the copy.
  /// Permitted types for dataT are all scalar and vector types.
  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  device_event
      async_work_group_copy(local_ptr<dataT> dest, global_ptr<dataT> src,
                            size_t numElements) const {
    return async_work_group_copy(dest, src, numElements, 1);
  }

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest and
  /// returns a SYCL device_event which can be used to wait on the completion
  /// of the copy.
  /// Permitted types for dataT are all scalar and vector types.
  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  device_event
      async_work_group_copy(global_ptr<dataT> dest, local_ptr<dataT> src,
                            size_t numElements) const {
    return async_work_group_copy(dest, src, numElements, 1);
  }

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest and
  /// returns a SYCL device_event which can be used to wait on the completion
  /// of the copy.
  /// Permitted types for DestDataT are all scalar and vector types. SrcDataT
  /// must be either the same as DestDataT or const DestDataT.
  template <typename DestDataT, typename SrcDataT>
  typename std::enable_if_t<
      std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
  async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                        decorated_global_ptr<SrcDataT> src,
                        size_t numElements) const {
    return async_work_group_copy(dest, src, numElements, 1);
  }

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest and
  /// returns a SYCL device_event which can be used to wait on the completion
  /// of the copy.
  /// Permitted types for DestDataT are all scalar and vector types. SrcDataT
  /// must be either the same as DestDataT or const DestDataT.
  template <typename DestDataT, typename SrcDataT>
  typename std::enable_if_t<
      std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
  async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                        decorated_local_ptr<SrcDataT> src,
                        size_t numElements) const {
    return async_work_group_copy(dest, src, numElements, 1);
  }

  template <typename... eventTN> void wait_for(eventTN... events) const {
    waitForHelper(events...);
  }

  sycl::ext::oneapi::experimental::root_group<Dimensions>
  ext_oneapi_get_root_group() const {
    return sycl::ext::oneapi::experimental::root_group<Dimensions>{*this};
  }

  nd_item(const nd_item &rhs) = default;
  nd_item(nd_item &&rhs) = default;

  nd_item &operator=(const nd_item &rhs) = default;
  nd_item &operator=(nd_item &&rhs) = default;

  bool operator==(const nd_item &) const { return true; }
  bool operator!=(const nd_item &rhs) const { return !((*this) == rhs); }

protected:
  friend class detail::Builder;
  nd_item() {}
  nd_item(const item<Dimensions, true> &, const item<Dimensions, false> &,
          const group<Dimensions> &) {}

  void waitForHelper() const {}

  void waitForHelper(device_event Event) const { Event.wait(); }

  template <typename T, typename... Ts>
  void waitForHelper(T E, Ts... Es) const {
    waitForHelper(E);
    waitForHelper(Es...);
  }

  id<Dimensions> get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initWorkgroupId<Dimensions, id<Dimensions>>();
#else
    assert(false && "nd_item methods can't be used on the host!");
    return {};
#endif
  }
};
#else
/// Identifies an instance of the function object executing at each point in an
/// nd_range.
///
/// \ingroup sycl_api
template <int Dimensions = 1> class nd_item {
public:
  static constexpr int dimensions = Dimensions;

  nd_item() = delete;

  id<Dimensions> get_global_id() const { return globalItem.get_id(); }

  size_t __SYCL_ALWAYS_INLINE get_global_id(int Dimension) const {
    size_t Id = globalItem.get_id(Dimension);
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  size_t __SYCL_ALWAYS_INLINE get_global_linear_id() const {
    size_t Id = globalItem.get_linear_id();
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  id<Dimensions> get_local_id() const { return localItem.get_id(); }

  size_t __SYCL_ALWAYS_INLINE get_local_id(int Dimension) const {
    size_t Id = localItem.get_id(Dimension);
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  size_t get_local_linear_id() const {
    size_t Id = localItem.get_linear_id();
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  group<Dimensions> get_group() const { return Group; }

  sub_group get_sub_group() const { return sub_group(); }

  size_t __SYCL_ALWAYS_INLINE get_group(int Dimension) const {
    size_t Id = Group[Dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  size_t __SYCL_ALWAYS_INLINE get_group_linear_id() const {
    size_t Id = Group.get_linear_id();
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  range<Dimensions> get_group_range() const { return Group.get_group_range(); }

  size_t __SYCL_ALWAYS_INLINE get_group_range(int Dimension) const {
    size_t Range = Group.get_group_range(Dimension);
    __SYCL_ASSUME_INT(Range);
    return Range;
  }

  range<Dimensions> get_global_range() const { return globalItem.get_range(); }

  size_t get_global_range(int Dimension) const {
    return globalItem.get_range(Dimension);
  }

  range<Dimensions> get_local_range() const { return localItem.get_range(); }

  size_t get_local_range(int Dimension) const {
    return localItem.get_range(Dimension);
  }

  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL 2020")
  id<Dimensions> get_offset() const { return globalItem.get_offset(); }

  nd_range<Dimensions> get_nd_range() const {
    return nd_range<Dimensions>(get_global_range(), get_local_range(),
                                get_offset());
  }

  void barrier(access::fence_space accessSpace =
                   access::fence_space::global_and_local) const {
    uint32_t flags = detail::getSPIRVMemorySemanticsMask(accessSpace);
    __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
                           flags);
  }

  /// Executes a work-group mem-fence with memory ordering on the local address
  /// space, global address space or both based on the value of \p accessSpace.
  template <access::mode accessMode = access::mode::read_write>
  __SYCL2020_DEPRECATED("use sycl::atomic_fence() free function instead")
  void mem_fence(
      typename std::enable_if_t<accessMode == access::mode::read ||
                                    accessMode == access::mode::write ||
                                    accessMode == access::mode::read_write,
                                access::fence_space>
          accessSpace = access::fence_space::global_and_local) const {
    (void)accessSpace;
    Group.mem_fence();
  }

  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  device_event
      async_work_group_copy(local_ptr<dataT> dest, global_ptr<dataT> src,
                            size_t numElements) const {
    return Group.async_work_group_copy(dest, src, numElements);
  }

  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  device_event
      async_work_group_copy(global_ptr<dataT> dest, local_ptr<dataT> src,
                            size_t numElements) const {
    return Group.async_work_group_copy(dest, src, numElements);
  }

  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  device_event
      async_work_group_copy(local_ptr<dataT> dest, global_ptr<dataT> src,
                            size_t numElements, size_t srcStride) const {

    return Group.async_work_group_copy(dest, src, numElements, srcStride);
  }

  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  device_event
      async_work_group_copy(global_ptr<dataT> dest, local_ptr<dataT> src,
                            size_t numElements, size_t destStride) const {
    return Group.async_work_group_copy(dest, src, numElements, destStride);
  }

  template <typename DestDataT, typename SrcDataT>
  typename std::enable_if_t<
      std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
  async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                        decorated_global_ptr<SrcDataT> src,
                        size_t numElements) const {
    return Group.async_work_group_copy(dest, src, numElements);
  }

  template <typename DestDataT, typename SrcDataT>
  typename std::enable_if_t<
      std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
  async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                        decorated_local_ptr<SrcDataT> src,
                        size_t numElements) const {
    return Group.async_work_group_copy(dest, src, numElements);
  }

  template <typename DestDataT, typename SrcDataT>
  typename std::enable_if_t<
      std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
  async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                        decorated_global_ptr<SrcDataT> src, size_t numElements,
                        size_t srcStride) const {

    return Group.async_work_group_copy(dest, src, numElements, srcStride);
  }

  template <typename DestDataT, typename SrcDataT>
  typename std::enable_if_t<
      std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
  async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                        decorated_local_ptr<SrcDataT> src, size_t numElements,
                        size_t destStride) const {
    return Group.async_work_group_copy(dest, src, numElements, destStride);
  }

  template <typename... eventTN> void wait_for(eventTN... events) const {
    Group.wait_for(events...);
  }

  sycl::ext::oneapi::experimental::root_group<Dimensions>
  ext_oneapi_get_root_group() const {
    return sycl::ext::oneapi::experimental::root_group<Dimensions>{*this};
  }

  nd_item(const nd_item &rhs) = default;

  nd_item(nd_item &&rhs) = default;

  nd_item &operator=(const nd_item &rhs) = default;

  nd_item &operator=(nd_item &&rhs) = default;

  bool operator==(const nd_item &rhs) const {
    return (rhs.localItem == this->localItem) &&
           (rhs.globalItem == this->globalItem) && (rhs.Group == this->Group);
  }

  bool operator!=(const nd_item &rhs) const { return !((*this) == rhs); }

protected:
  friend class detail::Builder;
  nd_item(const item<Dimensions, true> &GL, const item<Dimensions, false> &L,
          const group<Dimensions> &GR)
      : globalItem(GL), localItem(L), Group(GR) {}

private:
  item<Dimensions, true> globalItem;
  item<Dimensions, false> localItem;
  group<Dimensions> Group;
};
#endif

template <int Dims>
__SYCL_DEPRECATED("use sycl::ext::oneapi::experimental::this_nd_item() instead")
nd_item<Dims> this_nd_item() {
#ifdef __SYCL_DEVICE_ONLY__
  return detail::Builder::getElement(detail::declptr<nd_item<Dims>>());
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host");
#endif
}

namespace ext::oneapi::experimental {
template <int Dims> nd_item<Dims> this_nd_item() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::Builder::getElement(
      sycl::detail::declptr<nd_item<Dims>>());
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host");
#endif
}
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
