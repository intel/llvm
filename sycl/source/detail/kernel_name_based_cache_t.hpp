//==-------------------- kernel_name_based_cache_t.hpp ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <detail/kernel_program_cache.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

struct KernelNameBasedCacheT {
  FastKernelSubcacheT *FastKernelSubcachePtr = nullptr;
};

} // namespace detail
} // namespace _V1
} // namespace sycl