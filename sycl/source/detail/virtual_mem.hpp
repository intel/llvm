//==---------------- virtual_mem.hpp ---------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

size_t
get_mem_granularity_for_allocation_size(const detail::device_impl &SyclDevice,
                                        const detail::context_impl &SyclContext,
                                        granularity_mode Mode,
                                        size_t AllocationSize);

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
