//==- virtual_mem.hpp - sycl_ext_oneapi_virtual_mem virtual mem free funcs -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/queue.hpp>

#include <optional>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {

__SYCL_EXPORT size_t get_minimum_mem_granularity(size_t numBytes,
                                                 const device &SyclDevice,
                                                 const context &SyclContext);

inline size_t get_minimum_mem_granularity(size_t numBytes,
                                          const queue &SyclQueue) {
  return get_minimum_mem_granularity(numBytes, SyclQueue.get_device(),
                                     SyclQueue.get_context());
}

inline size_t get_minimum_mem_granularity(size_t numBytes,
                                          const physical_mem &SyclPhysicalMem) {
  return get_minimum_mem_granularity(numBytes, SyclPhysicalMem.get_device(),
                                     SyclPhysicalMem.get_context());
}

__SYCL_EXPORT size_t get_recommended_mem_granularity(
    size_t numBytes, const device &SyclDevice, const context &SyclContext);

inline size_t get_recommended_mem_granularity(size_t numBytes,
                                              const queue &SyclQueue) {
  return get_recommended_mem_granularity(numBytes, SyclQueue.get_device(),
                                         SyclQueue.get_context());
}

inline size_t
get_recommended_mem_granularity(size_t numBytes,
                                const physical_mem &SyclPhysicalMem) {
  return get_recommended_mem_granularity(numBytes, SyclPhysicalMem.get_device(),
                                         SyclPhysicalMem.get_context());
}

__SYCL_EXPORT void *reserve_virtual_mem(const void *Start, size_t NumBytes,
                                        const context &SyclContext);

inline void *reserve_virtual_mem(size_t NumBytes, const context &SyclContext) {
  return reserve_virtual_mem(nullptr, NumBytes, SyclContext);
}

__SYCL_EXPORT void free_virtual_mem(const void *Ptr, size_t NumBytes,
                                    const context &SyclContext);

__SYCL_EXPORT void set_access_mode(const void *Ptr, size_t NumBytes,
                                   access_mode Mode,
                                   const context &SyclContext);

__SYCL_EXPORT void set_inaccessible(const void *Ptr, size_t NumBytes,
                                    const context &SyclContext);

__SYCL_EXPORT std::optional<access_mode>
get_access_mode(const void *Ptr, size_t NumBytes, const context &SyclContext);

__SYCL_EXPORT void unmap(const void *Ptr, size_t NumBytes,
                         const context &SyclContext);

} // Namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // Namespace sycl
