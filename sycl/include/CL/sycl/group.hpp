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
#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/device_event.hpp>
#include <CL/sycl/h_item.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/pointers.hpp>
#include <CL/sycl/range.hpp>
#include <stdexcept>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
class Builder;

// Implements a barrier accross work items within a work group.
static inline void workGroupBarrier() {
#ifdef __SYCL_DEVICE_ONLY__
  constexpr uint32_t flags =
      static_cast<uint32_t>(
          __spv::MemorySemanticsMask::SequentiallyConsistent) |
      static_cast<uint32_t>(__spv::MemorySemanticsMask::WorkgroupMemory);
  __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
                         flags);
#endif // __SYCL_DEVICE_ONLY__
}

} // namespace detail

// SYCL 1.2.1rev5, section "4.8.5.3 Parallel For hierarchical invoke":
// Quote:
//   ... To guarantee use of private per-work-item memory, the private_memory
//   class can be used to wrap the data. This class very simply constructs
//   private data for a given group across the entire group.The id of the
//   current work-item is passed to any access to grab the correct data.
template <typename T, int Dimensions = 1> class private_memory {
public:
  // Construct based directly off the number of work-items
  private_memory(const group<Dimensions> &G) {
#ifndef __SYCL_DEVICE_ONLY__
    // serial host => one instance per work-group - allocate space for each WI
    // in the group:
    Val.reset(new T[G.get_local_range().size()]);
#endif // __SYCL_DEVICE_ONLY__
  }

  // Access the instance for the current work-item
  T &operator()(const h_item<Dimensions> &Id) {
#ifndef __SYCL_DEVICE_ONLY__
    // Calculate the linear index of current WI and return reference to the
    // corresponding spot in the value array:
    size_t Ind = Id.get_physical_local().get_linear_id();
    return Val.get()[Ind];
#else
    return Val;
#endif // __SYCL_DEVICE_ONLY__
  }

private:
#ifdef __SYCL_DEVICE_ONLY__
  // On SYCL device private_memory<T> instance is created per physical WI, so
  // there is 1:1 correspondence betwen this class instances and per-WI memory.
  T Val;
#else
  // On serial host there is one private_memory<T> instance per work group, so
  // it must have space to hold separate value per WI in the group.
  std::unique_ptr<T[]> Val;
#endif // #ifdef __SYCL_DEVICE_ONLY__
};

template <int Dimensions = 1> class group {
public:
#ifndef __DISABLE_SYCL_INTEL_GROUP_ALGORITHMS__
  using id_type = id<Dimensions>;
  using range_type = range<Dimensions>;
  using linear_id_type = size_t;
  static constexpr int dimensions = Dimensions;
#endif // __DISABLE_SYCL_INTEL_GROUP_ALGORITHMS__

  group() = delete;

  id<Dimensions> get_id() const { return index; }

  size_t get_id(int dimension) const { return index[dimension]; }

  range<Dimensions> get_global_range() const { return globalRange; }

  size_t get_global_range(int dimension) const {
    return globalRange[dimension];
  }

  range<Dimensions> get_local_range() const { return localRange; }

  size_t get_local_range(int dimension) const { return localRange[dimension]; }

  range<Dimensions> get_group_range() const { return groupRange; }

  size_t get_group_range(int dimension) const {
    return get_group_range()[dimension];
  }

  size_t operator[](int dimension) const { return index[dimension]; }

  template <int dims = Dimensions>
  typename std::enable_if<(dims == 1), size_t>::type get_linear_id() const {
    return index[0];
  }

  template <int dims = Dimensions>
  typename std::enable_if<(dims == 2), size_t>::type get_linear_id() const {
    return index[0] * groupRange[1] + index[1];
  }

  // SYCL specification 1.2.1rev5, section 4.7.6.5 "Buffer accessor":
  //    Whenever a multi-dimensional index is passed to a SYCL accessor the
  //    linear index is calculated based on the index {id1, id2, id3} provided
  //    and the range of the SYCL accessor {r1, r2, r3} according to row-major
  //    ordering as follows:
  //      id3 + (id2 · r3) + (id1 · r3 · r2)            (4.3)
  // section 4.8.1.8 "group class":
  //    size_t get_linear_id()const
  //    Get a linearized version of the work-group id. Calculating a linear
  //    work-group id from a multi-dimensional index follows the equation 4.3.
  template <int dims = Dimensions>
  typename std::enable_if<(dims == 3), size_t>::type get_linear_id() const {
    return (index[0] * groupRange[1] * groupRange[2]) +
           (index[1] * groupRange[2]) + index[2];
  }

  template <typename WorkItemFunctionT>
  void parallel_for_work_item(WorkItemFunctionT Func) const {
    // need barriers to enforce SYCL semantics for the work item loop -
    // compilers are expected to optimize when possible
    detail::workGroupBarrier();
#ifdef __SYCL_DEVICE_ONLY__
    range<Dimensions> GlobalSize{
        __spirv::initGlobalSize<Dimensions, range<Dimensions>>()};
    range<Dimensions> LocalSize{
        __spirv::initWorkgroupSize<Dimensions, range<Dimensions>>()};
    id<Dimensions> GlobalId{
        __spirv::initGlobalInvocationId<Dimensions, id<Dimensions>>()};
    id<Dimensions> LocalId{
        __spirv::initLocalInvocationId<Dimensions, id<Dimensions>>()};

    // no 'iterate' in the device code variant, because
    // (1) this code is already invoked by each work item as a part of the
    //     enclosing parallel_for_work_group kernel
    // (2) the range this pfwi iterates over matches work group size exactly
    item<Dimensions, false> GlobalItem =
        detail::Builder::createItem<Dimensions, false>(GlobalSize, GlobalId);
    item<Dimensions, false> LocalItem =
        detail::Builder::createItem<Dimensions, false>(LocalSize, LocalId);
    h_item<Dimensions> HItem =
        detail::Builder::createHItem<Dimensions>(GlobalItem, LocalItem);

    Func(HItem);
#else
    id<Dimensions> GroupStartID = index * localRange;

    // ... host variant needs explicit 'iterate' because it is serial
    detail::NDLoop<Dimensions>::iterate(
        localRange, [&](const id<Dimensions> &LocalID) {
          item<Dimensions, false> GlobalItem =
              detail::Builder::createItem<Dimensions, false>(
                  globalRange, GroupStartID + LocalID);
          item<Dimensions, false> LocalItem =
              detail::Builder::createItem<Dimensions, false>(localRange,
                                                             LocalID);
          h_item<Dimensions> HItem =
              detail::Builder::createHItem<Dimensions>(GlobalItem, LocalItem);
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
  void parallel_for_work_item(range<Dimensions> flexibleRange,
                              WorkItemFunctionT Func) const {
    detail::workGroupBarrier();
#ifdef __SYCL_DEVICE_ONLY__
    range<Dimensions> GlobalSize{
        __spirv::initGlobalSize<Dimensions, range<Dimensions>>()};
    range<Dimensions> LocalSize{
        __spirv::initWorkgroupSize<Dimensions, range<Dimensions>>()};
    id<Dimensions> GlobalId{
        __spirv::initGlobalInvocationId<Dimensions, id<Dimensions>>()};
    id<Dimensions> LocalId{
        __spirv::initLocalInvocationId<Dimensions, id<Dimensions>>()};

    item<Dimensions, false> GlobalItem =
        detail::Builder::createItem<Dimensions, false>(GlobalSize, GlobalId);
    item<Dimensions, false> LocalItem =
        detail::Builder::createItem<Dimensions, false>(LocalSize, LocalId);
    h_item<Dimensions> HItem = detail::Builder::createHItem<Dimensions>(
        GlobalItem, LocalItem, flexibleRange);

    // iterate over flexible range with work group size stride; each item
    // performs flexibleRange/LocalSize iterations (if the former is divisible
    // by the latter)
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
              detail::Builder::createItem<Dimensions, false>(localRange,
                                                             LocalID);
          h_item<Dimensions> HItem = detail::Builder::createHItem<Dimensions>(
              GlobalItem, LocalItem, flexibleRange);

          detail::NDLoop<Dimensions>::iterate(
              LocalID, localRange, flexibleRange,
              [&](const id<Dimensions> &LogicalLocalID) {
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
    using DestT = detail::ConvertToOpenCLType_t<decltype(dest)>;
    using SrcT = detail::ConvertToOpenCLType_t<decltype(src)>;

    __ocl_event_t e = OpGroupAsyncCopyGlobalToLocal(
        __spv::Scope::Workgroup, DestT(dest.get()), SrcT(src.get()),
        numElements, 1, 0);
    return device_event(&e);
  }

  template <typename dataT>
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src,
                                     size_t numElements) const {
    using DestT = detail::ConvertToOpenCLType_t<decltype(dest)>;
    using SrcT = detail::ConvertToOpenCLType_t<decltype(src)>;

    __ocl_event_t e = OpGroupAsyncCopyLocalToGlobal(
        __spv::Scope::Workgroup, DestT(dest.get()), SrcT(src.get()),
        numElements, 1, 0);
    return device_event(&e);
  }

  template <typename dataT>
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src,
                                     size_t numElements,
                                     size_t srcStride) const {
    using DestT = detail::ConvertToOpenCLType_t<decltype(dest)>;
    using SrcT = detail::ConvertToOpenCLType_t<decltype(src)>;

    __ocl_event_t e = OpGroupAsyncCopyGlobalToLocal(
        __spv::Scope::Workgroup, DestT(dest.get()), SrcT(src.get()),
        numElements, srcStride, 0);
    return device_event(&e);
  }

  template <typename dataT>
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src,
                                     size_t numElements,
                                     size_t destStride) const {
    using DestT = detail::ConvertToOpenCLType_t<decltype(dest)>;
    using SrcT = detail::ConvertToOpenCLType_t<decltype(src)>;

    __ocl_event_t e = OpGroupAsyncCopyLocalToGlobal(
        __spv::Scope::Workgroup, DestT(dest.get()), SrcT(src.get()),
        numElements, destStride, 0);
    return device_event(&e);
  }

  template <typename... eventTN>
  void wait_for(eventTN... Events) const {
    waitForHelper(Events...);
  }

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
  group(const range<Dimensions> &G, const range<Dimensions> &L,
        const range<Dimensions> GroupRange, const id<Dimensions> &I)
      : globalRange(G), localRange(L), groupRange(GroupRange), index(I) {
    // Make sure local range divides global without remainder:
    __SYCL_ASSERT(((G % L).size() == 0) &&
                  "global range is not multiple of local");
    __SYCL_ASSERT((((G / L) - GroupRange).size() == 0) &&
                  "inconsistent group constructor arguments");
  }
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
