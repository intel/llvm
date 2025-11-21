//==--- kernel_program_cache.cpp - Cache for kernel and program -*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/kernel_program_cache.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

void KernelProgramCache::traceKernelImpl(const char *Msg,
                                         KernelNameStrRefT KernelName,
                                         bool IsFastKernelCache) {
  std::string Identifier =
      "[IsFastCache: " + std::to_string(IsFastKernelCache) +
      "][Key:{Name = " + KernelName.data() + "}]: ";

  std::cerr << "[In-Memory Cache][Thread Id:" << std::this_thread::get_id()
            << "][Kernel Cache]" << Identifier << Msg << std::endl;
}

adapter_impl &KernelProgramCache::getAdapter() {
  return MParentContext.getAdapter();
}

ur_context_handle_t KernelProgramCache::getURContext() const {
  return MParentContext.getHandleRef();
}

} // namespace detail
} // namespace _V1
} // namespace sycl
