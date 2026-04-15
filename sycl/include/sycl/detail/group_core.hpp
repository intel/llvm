//==----------- detail/group_core.hpp --- SYCL work group core -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_types.hpp>
#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/__spirv/spirv_vars.hpp>
#endif
#include <sycl/access/access_base.hpp>
#include <sycl/detail/assert.hpp>
#include <sycl/detail/defines.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/fwd/multi_ptr.hpp>
#include <sycl/detail/spirv_memory_semantics.hpp>
#include <sycl/id.hpp>
#include <sycl/memory_enums.hpp>
#include <sycl/range.hpp>

#ifndef __SYCL_DEVICE_ONLY__
#include <sycl/exception.hpp>
#endif

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
template <int Dimensions> class h_item;
template <int Dimensions> class nd_item;
class device_event;

/// Encapsulates all functionality required to represent a particular work-group
/// within a parallel execution.
///
/// \ingroup sycl_api
template <int Dimensions = 1> class __SYCL_TYPE(group) group {
public:
#ifndef __DISABLE_SYCL_INTEL_GROUP_ALGORITHMS__
  using id_type = id<Dimensions>;
  using range_type = range<Dimensions>;
  using linear_id_type = size_t;
  static constexpr int dimensions = Dimensions;
#endif // __DISABLE_SYCL_INTEL_GROUP_ALGORITHMS__

  static constexpr sycl::memory_scope fence_scope =
      sycl::memory_scope::work_group;

  group() = delete;

  __SYCL2020_DEPRECATED("use sycl::group::get_group_id() instead")
  id<Dimensions> get_id() const { return index; }

  __SYCL2020_DEPRECATED("use sycl::group::get_group_id() instead")
  size_t get_id(int dimension) const { return index[dimension]; }

  id<Dimensions> get_group_id() const { return index; }

  size_t get_group_id(int dimension) const { return index[dimension]; }

  __SYCL2020_DEPRECATED("calculate sycl::group::get_group_range() * "
                        "sycl::group::get_max_local_range() instead")
  range<Dimensions> get_global_range() const { return globalRange; }

  size_t get_global_range(int dimension) const {
    return globalRange[dimension];
  }

  id<Dimensions> get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initBuiltInLocalInvocationId<Dimensions, id<Dimensions>>();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "get_local_id() is not implemented on host");
#endif
  }

  size_t get_local_id(int dimention) const { return get_local_id()[dimention]; }

  size_t get_local_linear_id() const {
    return get_local_linear_id_impl<Dimensions>();
  }

  range<Dimensions> get_local_range() const { return localRange; }

  size_t get_local_range(int dimension) const { return localRange[dimension]; }

  size_t get_local_linear_range() const {
    return get_local_linear_range_impl();
  }

  range<Dimensions> get_group_range() const { return groupRange; }

  size_t get_group_range(int dimension) const {
    return get_group_range()[dimension];
  }

  size_t get_group_linear_range() const {
    return get_group_linear_range_impl();
  }

  range<Dimensions> get_max_local_range() const { return get_local_range(); }

  size_t operator[](int dimension) const { return index[dimension]; }

  __SYCL2020_DEPRECATED("use sycl::group::get_group_linear_id() instead")
  size_t get_linear_id() const { return get_group_linear_id(); }

  size_t get_group_linear_id() const { return get_group_linear_id_impl(); }

  bool leader() const { return (get_local_linear_id() == 0); }

  template <typename WorkItemFunctionT>
#ifdef __NativeCPU__
  __attribute__((__libclc_call__))
#endif
  void parallel_for_work_item(WorkItemFunctionT Func) const;

  template <typename WorkItemFunctionT>
#ifdef __NativeCPU__
  __attribute__((__libclc_call__))
#endif
  void parallel_for_work_item(range<Dimensions> flexibleRange,
                              WorkItemFunctionT Func) const;

  template <access::mode accessMode = access::mode::read_write>
  void mem_fence(
      [[maybe_unused]]
      typename std::enable_if_t<accessMode == access::mode::read ||
                                    accessMode == access::mode::write ||
                                    accessMode == access::mode::read_write,
                                access::fence_space>
          accessSpace = access::fence_space::global_and_local) const {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t flags = detail::getSPIRVMemorySemanticsMask(accessSpace);
    __spirv_MemoryBarrier(__spv::Scope::Workgroup, flags);
#endif
  }

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

  template <typename... eventTN> void wait_for(eventTN... Events) const;

  bool operator==(const group<Dimensions> &rhs) const {
    bool Result = (rhs.globalRange == globalRange) &&
                  (rhs.localRange == localRange) && (rhs.index == index);
    __SYCL_ASSERT(rhs.groupRange == groupRange &&
                  "inconsistent group class fields");
    return Result;
  }

  bool operator!=(const group<Dimensions> &rhs) const {
    return !((*this) == rhs);
  }

private:
  range<Dimensions> globalRange;
  range<Dimensions> localRange;
  range<Dimensions> groupRange;
  id<Dimensions> index;

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 1), size_t>
  get_local_linear_id_impl() const {
    id<Dimensions> localId = get_local_id();
    return localId[0];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 2), size_t>
  get_local_linear_id_impl() const {
    id<Dimensions> localId = get_local_id();
    return localId[0] * localRange[1] + localId[1];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 3), size_t>
  get_local_linear_id_impl() const {
    id<Dimensions> localId = get_local_id();
    return (localId[0] * localRange[1] * localRange[2]) +
           (localId[1] * localRange[2]) + localId[2];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 1), size_t>
  get_local_linear_range_impl() const {
    auto localRange = get_local_range();
    return localRange[0];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 2), size_t>
  get_local_linear_range_impl() const {
    auto localRange = get_local_range();
    return localRange[0] * localRange[1];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 3), size_t>
  get_local_linear_range_impl() const {
    auto localRange = get_local_range();
    return localRange[0] * localRange[1] * localRange[2];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 1), size_t>
  get_group_linear_range_impl() const {
    auto groupRange = get_group_range();
    return groupRange[0];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 2), size_t>
  get_group_linear_range_impl() const {
    auto groupRange = get_group_range();
    return groupRange[0] * groupRange[1];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 3), size_t>
  get_group_linear_range_impl() const {
    auto groupRange = get_group_range();
    return groupRange[0] * groupRange[1] * groupRange[2];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 1), size_t>
  get_group_linear_id_impl() const {
    return index[0];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 2), size_t>
  get_group_linear_id_impl() const {
    return index[0] * groupRange[1] + index[1];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 3), size_t>
  get_group_linear_id_impl() const {
    return (index[0] * groupRange[1] * groupRange[2]) +
           (index[1] * groupRange[2]) + index[2];
  }

  void waitForHelper() const;
  void waitForHelper(device_event Event) const;

  template <typename T, typename... Ts> void waitForHelper(T E, Ts... Es) const;

protected:
  friend class detail::Builder;
  template <int dims> friend class nd_item;
  group(const range<Dimensions> &G, const range<Dimensions> &L,
        const range<Dimensions> GroupRange, const id<Dimensions> &I)
      : globalRange(G), localRange(L), groupRange(GroupRange), index(I) {}
};
} // namespace _V1
} // namespace sycl