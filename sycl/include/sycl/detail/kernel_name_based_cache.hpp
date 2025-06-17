//==--------------------- kernel_name_based_cache.hpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/detail/export.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

struct KernelNameBasedCacheT;
__SYCL_EXPORT KernelNameBasedCacheT *createKernelNameBasedCache();

// Retrieves a cache pointer unique to a kernel name type that can be used to
// avoid kernel name based lookup in the runtime.
template <typename KernelName>
KernelNameBasedCacheT *getKernelNameBasedCache() {
  static KernelNameBasedCacheT *Instance = createKernelNameBasedCache();
  return Instance;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
