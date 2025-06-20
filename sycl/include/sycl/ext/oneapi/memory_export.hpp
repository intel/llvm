//==------------- memory_export.hpp --- SYCL memory export -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef> // for size_t

#include <sycl/context.hpp>       // for context
#include <sycl/detail/export.hpp> // for __SYCL_EXPORT
#include <sycl/device.hpp>        // for device
#include <sycl/queue.hpp>         // for queue

#include "common_interop_resource_types.hpp" // for external_mem_handle_type

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

__SYCL_EXPORT void *
alloc_exportable_memory(size_t alignment, size_t size,
                        external_mem_handle_type externalMemHandleType,
                        const sycl::device &syclDevice,
                        const sycl::context &syclContext);

inline void *
alloc_exportable_memory(size_t alignment, size_t size,
                        external_mem_handle_type externalMemHandleType,
                        const sycl::queue &syclQueue) {
  return alloc_exportable_memory(size, alignment, externalMemHandleType,
                                 syclQueue.get_device(),
                                 syclQueue.get_context());
}

__SYCL_EXPORT void free_exportable_memory(void *deviceMemory,
                                          const sycl::device &syclDevice,
                                          const sycl::context &syclContext);

inline void free_exportable_memory(void *deviceMemory,
                                   const sycl::queue &syclQueue) {
  free_exportable_memory(deviceMemory, syclQueue.get_device(),
                         syclQueue.get_context());
}

template <typename ResourceType>
__SYCL_EXPORT ResourceType
export_memory_handle(void *deviceMemory, const sycl::device &syclDevice,
                     const sycl::context &syclContext);

template <typename ResourceType>
inline ResourceType export_memory_handle(void *deviceMemory,
                                         const sycl::queue &syclQueue) {
  return export_memory_handle<ResourceType>(
      deviceMemory, syclQueue.get_device(), syclQueue.get_context());
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
