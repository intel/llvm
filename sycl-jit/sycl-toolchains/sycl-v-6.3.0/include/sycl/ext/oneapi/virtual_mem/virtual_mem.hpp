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
inline namespace _V1 {
namespace ext::oneapi::experimental {

enum class granularity_mode : char {
  minimum = 0,
  recommended = 1,
};

__SYCL_EXPORT size_t
get_mem_granularity(const device &SyclDevice, const context &SyclContext,
                    granularity_mode Mode = granularity_mode::recommended);

__SYCL_EXPORT size_t
get_mem_granularity(const context &SyclContext,
                    granularity_mode Mode = granularity_mode::recommended);

__SYCL_EXPORT uintptr_t reserve_virtual_mem(uintptr_t Start, size_t NumBytes,
                                            const context &SyclContext);

inline uintptr_t reserve_virtual_mem(size_t NumBytes,
                                     const context &SyclContext) {
  return reserve_virtual_mem(0, NumBytes, SyclContext);
}

__SYCL_EXPORT void free_virtual_mem(uintptr_t Ptr, size_t NumBytes,
                                    const context &SyclContext);

__SYCL_EXPORT void set_access_mode(const void *Ptr, size_t NumBytes,
                                   address_access_mode Mode,
                                   const context &SyclContext);

__SYCL_EXPORT address_access_mode get_access_mode(const void *Ptr,
                                                  size_t NumBytes,
                                                  const context &SyclContext);

__SYCL_EXPORT void unmap(const void *Ptr, size_t NumBytes,
                         const context &SyclContext);

} // Namespace ext::oneapi::experimental
} // namespace _V1
} // Namespace sycl
