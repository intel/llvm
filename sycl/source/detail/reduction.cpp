//==---------------- reduction.cpp - SYCL reduction ------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/queue_impl.hpp>
#include <sycl/ext/oneapi/reduction.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace detail {

// TODO: The algorithm of choosing the work-group size is definitely
// imperfect now and can be improved.
__SYCL_EXPORT size_t reduComputeWGSize(size_t NWorkItems, size_t MaxWGSize,
                                       size_t &NWorkGroups) {
  size_t WGSize = MaxWGSize;
  if (NWorkItems <= WGSize) {
    NWorkGroups = 1;
    WGSize = NWorkItems;
  } else {
    NWorkGroups = NWorkItems / WGSize;
    size_t Rem = NWorkItems % WGSize;
    if (Rem != 0) {
      // Let's suppose MaxWGSize = 128 and NWorkItems = (128+32).
      // It seems better to have 5 groups 32 work-items each than 2 groups with
      // 128 work-items in the 1st group and 32 work-items in the 2nd group.
      size_t NWorkGroupsAlt = NWorkItems / Rem;
      size_t RemAlt = NWorkItems % Rem;
      if (RemAlt == 0 && NWorkGroupsAlt <= MaxWGSize) {
        // Choose smaller uniform work-groups.
        // The condition 'NWorkGroupsAlt <= MaxWGSize' was checked to ensure
        // that choosing smaller groups will not cause the need in additional
        // invocations of the kernel.
        NWorkGroups = NWorkGroupsAlt;
        WGSize = Rem;
      } else {
        // Add 1 more group to process the remaining elements and proceed
        // with bigger non-uniform work-groups
        NWorkGroups++;
      }
    }
  }
  return WGSize;
}

// Returns the estimated number of physical threads on the device associated
// with the given queue.
__SYCL_EXPORT uint32_t reduGetMaxNumConcurrentWorkGroups(
    std::shared_ptr<sycl::detail::queue_impl> Queue) {
  device Dev = Queue->get_device();
  uint32_t NumThreads = Dev.get_info<info::device::max_compute_units>();
  // TODO: The heuristics here require additional tuning for various devices
  // and vendors. For now this code assumes that execution units have about
  // 8 working threads, which gives good results on some known/supported
  // GPU devices.
  if (Dev.is_gpu())
    NumThreads *= 8;
  return NumThreads;
}

__SYCL_EXPORT size_t
reduGetMaxWGSize(std::shared_ptr<sycl::detail::queue_impl> Queue,
                 size_t LocalMemBytesPerWorkItem) {
  device Dev = Queue->get_device();
  size_t MaxWGSize = Dev.get_info<info::device::max_work_group_size>();
  size_t WGSizePerMem = MaxWGSize * 2;
  size_t WGSize = MaxWGSize;
  if (LocalMemBytesPerWorkItem != 0) {
    size_t MemSize = Dev.get_info<info::device::local_mem_size>();
    WGSizePerMem = MemSize / LocalMemBytesPerWorkItem;

    // If the work group size is NOT power of two, then an additional element
    // in local memory is needed for the reduction algorithm and thus the real
    // work-group size requirement per available memory is stricter.
    if ((WGSizePerMem & (WGSizePerMem - 1)) != 0)
      WGSizePerMem--;
    WGSize = (std::min)(WGSizePerMem, WGSize);
  }
  // TODO: This is a temporary workaround for a big problem of detecting
  // the maximal usable work-group size. The detection method used above
  // is based on maximal work-group size possible on the device is too risky
  // as may return too big value. Even though it also tries using the memory
  // factor into consideration, it is too rough estimation. For example,
  // if (WGSize * LocalMemBytesPerWorkItem) is equal to local_mem_size, then
  // the reduction local accessor takes all available local memory for it needs
  // not leaving any local memory for other kernel needs (barriers,
  // builtin calls, etc), which often leads to crushes with CL_OUT_OF_RESOURCES
  // error, or in even worse cases it may cause silent writes/clobbers of
  // the local memory assigned to one work-group by code in another work-group.
  // It seems the only good solution for this work-group detection problem is
  // kernel precompilation and querying the kernel properties.
  if (WGSize >= 4) {
    // Let's return a twice smaller number, but... do that only if the kernel
    // is limited by memory, or the kernel uses opencl:cpu backend, which
    // surprisingly uses lots of resources to run the kernels with reductions
    // and often causes CL_OUT_OF_RESOURCES error even when reduction
    // does not use local accessors.
    if (WGSizePerMem < MaxWGSize * 2 ||
        (Queue->get_device().is_cpu() &&
         Queue->get_device().get_platform().get_backend() == backend::opencl))
      WGSize /= 2;
  }

  return WGSize;
}

} // namespace detail
} // namespace oneapi
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
  using namespace ext::oneapi;
  namespace detail {
  __SYCL_EXPORT size_t reduComputeWGSize(size_t NWorkItems, size_t MaxWGSize,
                                         size_t &NWorkGroups) {
    return ext::oneapi::detail::reduComputeWGSize(NWorkItems, MaxWGSize,
                                                  NWorkGroups);
  }

  __SYCL_EXPORT size_t
  reduGetMaxWGSize(shared_ptr_class<sycl::detail::queue_impl> Queue,
                   size_t LocalMemBytesPerWorkItem) {
    return ext::oneapi::detail::reduGetMaxWGSize(Queue,
                                                 LocalMemBytesPerWorkItem);
  }
  } // namespace detail
} // namespace ONEAPI
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
