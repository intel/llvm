//==--------------------- kernel_name_based_cache.cpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/global_handler.hpp>
#include <sycl/detail/kernel_name_based_cache.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

KernelNameBasedCacheT *createKernelNameBasedCache() {
  return GlobalHandler::instance().createKernelNameBasedCache();
}

} // namespace detail
} // namespace _V1
} // namespace sycl
