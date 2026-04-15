//==--------- detail/nd_item_core.hpp --- SYCL iteration nd_item core -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_types.hpp>
#include <sycl/__spirv/spirv_vars.hpp>
#include <sycl/access/access_base.hpp>
#include <sycl/detail/defines.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/fwd/multi_ptr.hpp>
#include <sycl/detail/group_core.hpp>
#include <sycl/detail/spirv_memory_semantics.hpp>
#include <sycl/id.hpp>
#include <sycl/range.hpp>

#include <cstddef>
#include <stdint.h>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
class device_event;
struct sub_group;

namespace detail {
class Builder;
}

namespace ext::oneapi::experimental {
template <int Dimensions> class root_group;
}

/// Identifies an instance of the function object executing at each point in an
/// nd_range.
///
/// \ingroup sycl_api
template <int Dimensions = 1> class nd_item {
public:
  static constexpr int dimensions = Dimensions;

  id<Dimensions> get_global_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initBuiltInGlobalInvocationId<Dimensions, id<Dimensions>>();
#else
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
    id<Dimensions> Offset = get_offset_impl();
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
    return __spirv::initBuiltInLocalInvocationId<Dimensions, id<Dimensions>>();
#else
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
    return group<Dimensions>(get_global_range(), get_local_range(),
                             get_group_range(), get_group_id());
  }

  // Out-of-class definition in sub_group.hpp.
  sub_group get_sub_group() const;

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
    return __spirv::initBuiltInNumWorkgroups<Dimensions, range<Dimensions>>();
#else
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
    return __spirv::initBuiltInGlobalSize<Dimensions, range<Dimensions>>();
#else
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
    return __spirv::initBuiltInWorkgroupSize<Dimensions, range<Dimensions>>();
#else
    return {};
#endif
  }

  size_t get_local_range(int Dimension) const {
    size_t Id = get_local_range()[Dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL 2020")
  id<Dimensions> get_offset() const;

  auto get_nd_range() const;

  void barrier([[maybe_unused]] access::fence_space accessSpace =
                   access::fence_space::global_and_local) const {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t flags = _V1::detail::getSPIRVMemorySemanticsMask(accessSpace);
    __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
                           flags);
#endif
  }

  template <access::mode accessMode = access::mode::read_write>
  __SYCL2020_DEPRECATED("use sycl::atomic_fence() free function instead")
  void mem_fence(
      [[maybe_unused]]
      typename std::enable_if_t<accessMode == access::mode::read ||
                                    accessMode == access::mode::write ||
                                    accessMode == access::mode::read_write,
                                access::fence_space>
          accessSpace = access::fence_space::global_and_local) const;

  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  device_event async_work_group_copy(
      multi_ptr<dataT, access::address_space::local_space,
                access::decorated::legacy>
          dest,
      multi_ptr<dataT, access::address_space::global_space,
                access::decorated::legacy>
          src,
      size_t numElements, size_t srcStride) const;

  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  device_event async_work_group_copy(
      multi_ptr<dataT, access::address_space::global_space,
                access::decorated::legacy>
          dest,
      multi_ptr<dataT, access::address_space::local_space,
                access::decorated::legacy>
          src,
      size_t numElements, size_t destStride) const;

  template <typename DestDataT, typename SrcDataT>
  std::enable_if_t<std::is_same_v<std::remove_const_t<SrcDataT>, DestDataT>,
                   device_event>
  async_work_group_copy(multi_ptr<DestDataT, access::address_space::local_space,
                                  access::decorated::yes>
                            dest,
                        multi_ptr<SrcDataT, access::address_space::global_space,
                                  access::decorated::yes>
                            src,
                        size_t numElements, size_t srcStride) const;

  template <typename DestDataT, typename SrcDataT>
  std::enable_if_t<std::is_same_v<std::remove_const_t<SrcDataT>, DestDataT>,
                   device_event>
  async_work_group_copy(
      multi_ptr<DestDataT, access::address_space::global_space,
                access::decorated::yes>
          dest,
      multi_ptr<SrcDataT, access::address_space::local_space,
                access::decorated::yes>
          src,
      size_t numElements, size_t destStride) const;

  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  device_event async_work_group_copy(
      multi_ptr<dataT, access::address_space::local_space,
                access::decorated::legacy>
          dest,
      multi_ptr<dataT, access::address_space::global_space,
                access::decorated::legacy>
          src,
      size_t numElements) const;

  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  device_event async_work_group_copy(
      multi_ptr<dataT, access::address_space::global_space,
                access::decorated::legacy>
          dest,
      multi_ptr<dataT, access::address_space::local_space,
                access::decorated::legacy>
          src,
      size_t numElements) const;

  template <typename DestDataT, typename SrcDataT>
  typename std::enable_if_t<
      std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
  async_work_group_copy(multi_ptr<DestDataT, access::address_space::local_space,
                                  access::decorated::yes>
                            dest,
                        multi_ptr<SrcDataT, access::address_space::global_space,
                                  access::decorated::yes>
                            src,
                        size_t numElements) const;

  template <typename DestDataT, typename SrcDataT>
  typename std::enable_if_t<
      std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
  async_work_group_copy(
      multi_ptr<DestDataT, access::address_space::global_space,
                access::decorated::yes>
          dest,
      multi_ptr<SrcDataT, access::address_space::local_space,
                access::decorated::yes>
          src,
      size_t numElements) const;

  template <typename... eventTN> void wait_for(eventTN... events) const;

  ext::oneapi::experimental::root_group<Dimensions>
  ext_oneapi_get_root_group() const;

  nd_item(const nd_item &rhs) = default;
  nd_item(nd_item &&rhs) = default;

  nd_item &operator=(const nd_item &rhs) = default;
  nd_item &operator=(nd_item &&rhs) = default;

  bool operator==(const nd_item &) const { return true; }
  bool operator!=(const nd_item &rhs) const { return !((*this) == rhs); }

protected:
  friend class detail::Builder;
  nd_item() = default;

  void waitForHelper() const;
  void waitForHelper(device_event Event) const;

  template <typename T, typename... Ts> void waitForHelper(T E, Ts... Es) const;

  id<Dimensions> get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initBuiltInWorkgroupId<Dimensions, id<Dimensions>>();
#else
    return {};
#endif
  }

  id<Dimensions> get_offset_impl() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initBuiltInGlobalOffset<Dimensions, id<Dimensions>>();
#else
    return {};
#endif
  }
};
} // namespace _V1
} // namespace sycl