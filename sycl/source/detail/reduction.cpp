//==---------------- reduction.cpp - SYCL reduction ------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/intel/reduction.hpp>
#include <detail/queue_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {
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
      // Let's say MaxWGSize = 128 and NWorkItems is (128+32).
      // It seems better to have 5 groups 32 work-items each than 2 groups with
      // 128 work-items in the 1st group and 32 work-items in the 2nd group.
      size_t NWorkGroupsAlt = NWorkItems / Rem;
      size_t RemAlt = NWorkItems % Rem;
      if (RemAlt == 0 && NWorkGroupsAlt <= MaxWGSize) {
        NWorkGroups = NWorkGroupsAlt;
        WGSize = Rem;
      }
    } else {
      NWorkGroups++;
    }
  }
  return WGSize;
}

__SYCL_EXPORT size_t
reduGetMaxWGSize(shared_ptr_class<sycl::detail::queue_impl> Queue,
                 size_t LocalMemBytesPerWorkItem) {
  device Dev = Queue->get_device();
  size_t WGSize = Dev.get_info<info::device::max_work_group_size>();
  if (LocalMemBytesPerWorkItem != 0) {
    size_t MemSize = Dev.get_info<info::device::local_mem_size>();
    size_t WGSizePerMem = MemSize / LocalMemBytesPerWorkItem;

    // If the work group size is not pow of two, then an additional element
    // in local memory is needed for the reduction algorithm and thus the real
    // work-group size requirement per available memory is stricter.
    if ((WGSize & (WGSize - 1)) == 0)
      WGSizePerMem--;
    WGSize = (std::min)(WGSizePerMem, WGSize);
  }
  return WGSize;
}

} // namespace detail
} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
