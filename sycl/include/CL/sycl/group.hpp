//==-------------- group.hpp --- SYCL work group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_types.hpp>
#include <CL/__spirv/spirv_vars.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/device_event.hpp>
#include <CL/sycl/h_item.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/pointers.hpp>
#include <CL/sycl/range.hpp>
#include <stdexcept>
#include <type_traits>

namespace cl {
namespace sycl {
namespace detail {
class Builder;

// Implements a barrier accross work items within a work group.
static inline void workGroupBarrier() {
#ifdef __SYCL_DEVICE_ONLY__
  uint32_t flags =
      static_cast<uint32_t>(
          __spv::MemorySemanticsMask::SequentiallyConsistent) |
      static_cast<uint32_t>(__spv::MemorySemanticsMask::WorkgroupMemory);
  __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
                         flags);
#endif // __SYCL_DEVICE_ONLY__
}

} // namespace detail

template <int dimensions = 1> class group {
public:
  group() = delete;

  id<dimensions> get_id() const { return index; }

  size_t get_id(int dimension) const { return index[dimension]; }

  range<dimensions> get_global_range() const { return globalRange; }

  size_t get_global_range(int dimension) const {
    return globalRange[dimension];
  }

  range<dimensions> get_local_range() const { return localRange; }

  size_t get_local_range(int dimension) const { return localRange[dimension]; }

  range<dimensions> get_group_range() const { return localRange; }

  size_t get_group_range(int dimension) const { return localRange[dimension]; }

  size_t operator[](int dimension) const { return index[dimension]; }

  template <int dims = dimensions>
  typename std::enable_if<(dims == 1), size_t>::type get_linear_id() const {
    range<dimensions> groupNum = globalRange / localRange;
    return index[0];
  }

  template <int dims = dimensions>
  typename std::enable_if<(dims == 2), size_t>::type get_linear_id() const {
    range<dimensions> groupNum = globalRange / localRange;
    return index[1] * groupNum[0] + index[0];
  }

  template <int dims = dimensions>
  typename std::enable_if<(dims == 3), size_t>::type get_linear_id() const {
    range<dimensions> groupNum = globalRange / localRange;
    return (index[2] * groupNum[1] * groupNum[0]) + (index[1] * groupNum[0]) +
           index[0];
  }

  template <typename WorkItemFunctionT>
  void parallel_for_work_item(WorkItemFunctionT Func) const {
    // need barriers to enforce SYCL semantics for the work item loop -
    // compilers are expected to optimize when possible
    detail::workGroupBarrier();
#ifdef __SYCL_DEVICE_ONLY__
    range<dimensions> GlobalSize;
    range<dimensions> LocalSize;
    id<dimensions> GlobalId;
    id<dimensions> LocalId;

    __spirv::initGlobalSize<dimensions>(GlobalSize);
    __spirv::initWorkgroupSize<dimensions>(LocalSize);
    __spirv::initGlobalInvocationId<dimensions>(GlobalId);
    __spirv::initLocalInvocationId<dimensions>(LocalId);

    // no 'iterate' in the device code variant, because
    // (1) this code is already invoked by each work item as a part of the
    //     enclosing parallel_for_work_group kernel
    // (2) the range this pfwi iterates over matches work group size exactly
    item<dimensions, false> GlobalItem =
        detail::Builder::createItem<dimensions, false>(GlobalSize, GlobalId);
    item<dimensions, false> LocalItem =
        detail::Builder::createItem<dimensions, false>(LocalSize, LocalId);
    h_item<dimensions> HItem =
        detail::Builder::createHItem<dimensions>(GlobalItem, LocalItem);

    Func(HItem);
#else
    id<dimensions> GroupStartID = index * localRange;

    // ... host variant needs explicit 'iterate' because it is serial
    detail::NDLoop<dimensions>::iterate(
        localRange, [&](const id<dimensions> &LocalID) {
          item<dimensions, false> GlobalItem =
              detail::Builder::createItem<dimensions, false>(
                  globalRange, GroupStartID + LocalID);
          item<dimensions, false> LocalItem =
              detail::Builder::createItem<dimensions, false>(localRange,
                                                             LocalID);
          h_item<dimensions> HItem =
              detail::Builder::createHItem<dimensions>(GlobalItem, LocalItem);
          Func(HItem);
        });
#endif // __SYCL_DEVICE_ONLY__
    // Need both barriers here - before and after the parallel_for_work_item
    // (PFWI). There can be work group scope code after the PFWI which reads
    // work group local data written within this PFWI. Back Ends are expected to
    // optimize away unneeded barriers (e.g. two barriers in a row).
    detail::workGroupBarrier();
  }

  template <typename WorkItemFunctionT>
  void parallel_for_work_item(range<dimensions> flexibleRange,
                              WorkItemFunctionT Func) const {
    detail::workGroupBarrier();
#ifdef __SYCL_DEVICE_ONLY__
    range<dimensions> GlobalSize;
    range<dimensions> LocalSize;
    id<dimensions> GlobalId;
    id<dimensions> LocalId;

    __spirv::initGlobalSize<dimensions>(GlobalSize);
    __spirv::initWorkgroupSize<dimensions>(LocalSize);
    __spirv::initGlobalInvocationId<dimensions>(GlobalId);
    __spirv::initLocalInvocationId<dimensions>(LocalId);

    item<dimensions, false> GlobalItem =
        detail::Builder::createItem<dimensions, false>(GlobalSize, GlobalId);
    item<dimensions, false> LocalItem =
        detail::Builder::createItem<dimensions, false>(LocalSize, LocalId);
    h_item<dimensions> HItem =
        detail::Builder::createHItem<dimensions>(GlobalItem, LocalItem);

    // iterate over flexible range with work group size stride; each item
    // performs flexibleRange/LocalSize iterations (if the former is divisible
    // by the latter)
    detail::NDLoop<dimensions>::iterate(
        LocalId, LocalSize, flexibleRange,
        [&](const id<dimensions> &LogicalLocalID) {
          HItem.setLogicalLocalID(LogicalLocalID);
          Func(HItem);
        });
#else
    id<dimensions> GroupStartID = index * localRange;

    detail::NDLoop<dimensions>::iterate(
        localRange, [&](const id<dimensions> &LocalID) {
          item<dimensions, false> GlobalItem =
              detail::Builder::createItem<dimensions, false>(
                  globalRange, GroupStartID + LocalID);
          item<dimensions, false> LocalItem =
              detail::Builder::createItem<dimensions, false>(localRange,
                                                             LocalID);
          h_item<dimensions> HItem =
              detail::Builder::createHItem<dimensions>(GlobalItem, LocalItem);

          detail::NDLoop<dimensions>::iterate(
              LocalID, localRange, flexibleRange,
              [&](const id<dimensions> &LogicalLocalID) {
                HItem.setLogicalLocalID(LogicalLocalID);
                Func(HItem);
              });
        });
#endif // __SYCL_DEVICE_ONLY__
    detail::workGroupBarrier();
  }

  /// Executes a work-group mem-fence with memory ordering on the local address
  /// space, global address space or both based on the value of \p accessSpace.
  template <access::mode accessMode = access::mode::read_write>
  void mem_fence(typename std::enable_if<
                     accessMode == access::mode::read ||
                     accessMode == access::mode::write ||
                     accessMode == access::mode::read_write,
                     access::fence_space>::type accessSpace =
                     access::fence_space::global_and_local) const {
    uint32_t flags = detail::getSPIRVMemorySemanticsMask(accessSpace);
    // TODO: currently, there is no good way in SPIRV to set the memory
    // barrier only for load operations or only for store operations.
    // The full read-and-write barrier is used and the template parameter
    // 'accessMode' is ignored for now. Either SPIRV or SYCL spec may be
    // changed to address this discrepancy between SPIRV and SYCL,
    // or if we decide that 'accessMode' is the important feature then
    // we can fix this later, for example, by using OpenCL 1.2 functions
    // read_mem_fence() and write_mem_fence().
    __spirv_MemoryBarrier(__spv::Scope::Workgroup, flags);
  }

  template <typename dataT>
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src,
                                     size_t numElements) const {
    __ocl_event_t e =
        OpGroupAsyncCopyGlobalToLocal<dataT>(
            __spv::Scope::Workgroup,
            dest.get(), src.get(), numElements, 1, 0);
    return device_event(&e);
  }

  template <typename dataT>
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src,
                                     size_t numElements) const {
    __ocl_event_t e =
        OpGroupAsyncCopyLocalToGlobal<dataT>(
            __spv::Scope::Workgroup,
            dest.get(), src.get(), numElements, 1, 0);
    return device_event(&e);
  }

  template <typename dataT>
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src,
                                     size_t numElements,
                                     size_t srcStride) const {
    __ocl_event_t e =
        OpGroupAsyncCopyGlobalToLocal<dataT>(
            __spv::Scope::Workgroup,
            dest.get(), src.get(), numElements, srcStride, 0);
    return device_event(&e);
  }

  template <typename dataT>
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src,
                                     size_t numElements,
                                     size_t destStride) const {
    __ocl_event_t e =
        OpGroupAsyncCopyLocalToGlobal<dataT>(
            __spv::Scope::Workgroup,
            dest.get(), src.get(), numElements, destStride, 0);
    return device_event(&e);
  }

  template <typename... eventTN>
  void wait_for(eventTN... Events) const {
    waitForHelper(Events...);
  }

  bool operator==(const group<dimensions> &rhs) const {
    return (rhs.globalRange == this->globalRange) &&
           (rhs.localRange == this->localRange) && (rhs.index == this->index);
  }

  bool operator!=(const group<dimensions> &rhs) const {
    return !((*this) == rhs);
  }

private:
  range<dimensions> globalRange;
  range<dimensions> localRange;
  id<dimensions> index;

  void waitForHelper() const {}

  void waitForHelper(device_event Event) const {
    Event.wait();
  }

  template <typename T, typename... Ts>
  void waitForHelper(T E, Ts... Es) const {
    waitForHelper(E);
    waitForHelper(Es...);
  }

protected:
  friend class detail::Builder;
  group(const range<dimensions> &G, const range<dimensions> &L,
        const id<dimensions> &I)
      : globalRange(G), localRange(L), index(I) {}
};

} // namespace sycl
} // namespace cl
