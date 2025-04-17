//==------------------------- kernel_cache_hint.hpp ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

namespace sycl {
inline namespace _V1 {
namespace detail {

// Retrieves a kernel name type based hint pointer that can be used by the
// runtime to avoid kernel name based lookup.
template <typename KernelName> void **getKernelCacheHint() {
  static void *Instance = nullptr;
  return &Instance;
}

} // namespace detail
} // namespace _V1
} // namespace sycl