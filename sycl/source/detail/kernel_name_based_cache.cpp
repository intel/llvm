//==--------------------- kernel_name_based_cache.cpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/global_handler.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/detail/kernel_name_based_cache.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
KernelNameBasedCacheT *createKernelNameBasedCache() {
  return GlobalHandler::instance().createKernelNameBasedCache();
}
#endif

KernelNameBasedCacheT *
createKernelNameBasedCache(detail::ABINeutralKernelNameStrRefT KernelName) {
  return ProgramManager::getInstance().createKernelNameBasedCache(
      KernelName.data());
}

} // namespace detail
} // namespace _V1
} // namespace sycl
